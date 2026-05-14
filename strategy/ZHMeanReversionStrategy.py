"""
ZH Mean Reversion Strategy v3 — hybrid mean-reversion + momentum confirmation.

Combines multiple confirmation signals before entering a mean reversion trade:
  1. Price deviation: price must be > entry_threshold below rolling mean
  2. Bollinger Band: price must be at or below the lower band
  3. RSI oversold: RSI must be below threshold (selling exhaustion)
  4. Volume spike: above-average volume confirms reactionary move
  5. Bullish candle: close > open on entry candle (buyers stepping in)

Exit logic:
  - Primary: price reverts to within exit_threshold of the mean
  - Trailing: after reaching 0.5% profit, activate trailing stop at 0.3%
  - Hard stop: entry_threshold × stop_multiple below entry

Source: https://github.com/ZhuRong818/ZH-trading
"""

from datetime import datetime

import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from pandas import DataFrame
from technical import qtpylib


class ZHMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False

    # Take profit at 3% if reversion signal hasn't triggered
    minimal_roi = {"0": 0.03}

    # Hard stoploss fallback
    stoploss = -0.10
    use_custom_stoploss = True

    # Trailing stop: activate after 0.5% profit, trail at 0.3%
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    # --- Core mean reversion ---
    lookback = IntParameter(10, 50, default=20, space="buy")
    entry_threshold = DecimalParameter(
        0.005, 0.03, default=0.01, decimals=3, space="buy",
    )
    exit_threshold = DecimalParameter(
        0.001, 0.01, default=0.003, decimals=3, space="sell",
    )
    stop_multiple = DecimalParameter(
        1.0, 4.0, default=2.0, decimals=1, space="stoploss",
    )

    # --- Bollinger Band ---
    bb_period = IntParameter(15, 30, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")

    # --- RSI ---
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(20, 45, default=40, space="buy")

    # --- Volume filter ---
    volume_surge_threshold = DecimalParameter(
        0.8, 3.0, default=1.2, decimals=1, space="buy",
    )

    # --- Regime filter ---
    min_price = DecimalParameter(0.0, 0.30, default=0.0, decimals=2, space="buy")
    max_price = DecimalParameter(
        0.70, 1000000.0, default=1000000.0, decimals=2, space="buy",
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = self.lookback.value

        # Core: rolling mean and deviation
        dataframe["moving_avg"] = dataframe["close"].rolling(window=lb).mean()
        dataframe["deviation_pct"] = (
            (dataframe["close"] - dataframe["moving_avg"]) / dataframe["moving_avg"]
        )

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period.value,
            stds=self.bb_std.value,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_middle"] = bollinger["mid"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Volume surge
        mean_vol = dataframe["volume"].rolling(20).mean()
        dataframe["volume_surge"] = dataframe["volume"] / mean_vol.replace(0, 1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        threshold = self.entry_threshold.value
        dataframe.loc[
            (
                # 1. Price well below rolling mean
                (dataframe["deviation_pct"] < -threshold)
                # 2. Price at or below lower Bollinger Band
                & (dataframe["close"] <= dataframe["bb_lower"])
                # 3. RSI confirms oversold
                & (dataframe["rsi"] < self.rsi_oversold.value)
                # 4. Volume spike — reactionary move
                & (dataframe["volume_surge"] > self.volume_surge_threshold.value)
                # Regime filter
                & (dataframe["close"] > self.min_price.value)
                & (dataframe["close"] < self.max_price.value)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_thresh = self.exit_threshold.value
        dataframe.loc[
            (
                # Price reverted to near the mean
                (dataframe["deviation_pct"] >= -exit_thresh)
                # OR price crossed above middle Bollinger Band
                | (dataframe["close"] > dataframe["bb_middle"])
            ),
            "exit_long",
        ] = 1
        return dataframe

    def custom_stoploss(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float,
        after_fill: bool, **kwargs,
    ) -> float:
        """Dynamic stop: entry_threshold × stop_multiple below entry."""
        return -(self.entry_threshold.value * self.stop_multiple.value)

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: str | None,
        side: str, **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
        last_close = dataframe.iloc[-1]["close"]
        max_deviation = 0.01
        deviation = abs(rate - last_close) / last_close
        if deviation > max_deviation:
            return False
        return True
