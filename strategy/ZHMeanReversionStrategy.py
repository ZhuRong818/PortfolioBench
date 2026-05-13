"""
ZH Mean Reversion Strategy — ported from ZH-trading/strategies/v2/meanrev.py

Buys when price drops significantly below its rolling mean, expecting
reversion. Exits when price reverts back to the mean or hits the stop.

Original logic:
  1. Compute rolling moving average over `lookback` candles
  2. Entry: price deviates below mean by > entry_threshold (percentage)
  3. Exit: price reverts to within exit_threshold of the mean
  4. Stop: entry_threshold × stop_multiple below entry price

Works on both regular assets (BTC, stocks) and prediction market
contracts. Set min_price/max_price to constrain the tradeable price
range (e.g., 0.20–0.80 for probability-bounded contracts).

Source: https://github.com/ZhuRong818/ZH-trading
"""

from datetime import datetime

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from pandas import DataFrame


class ZHMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False

    # No fixed ROI — exit via mean reversion signal only
    minimal_roi = {}

    # Fallback stoploss; actual stop is dynamic via custom_stoploss
    stoploss = -0.10
    use_custom_stoploss = True

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    # --- Tunable parameters (matching ZH-trading defaults) ---
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

    # Regime filter — widen for regular assets, narrow for prediction markets
    min_price = DecimalParameter(0.0, 0.30, default=0.0, decimals=2, space="buy")
    max_price = DecimalParameter(
        0.70, 1000000.0, default=1000000.0, decimals=2, space="buy",
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = self.lookback.value
        dataframe["moving_avg"] = dataframe["close"].rolling(window=lb).mean()
        dataframe["deviation"] = dataframe["close"] - dataframe["moving_avg"]
        dataframe["deviation_pct"] = (
            dataframe["deviation"] / dataframe["moving_avg"]
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        threshold = self.entry_threshold.value
        dataframe.loc[
            (
                # Price well below moving average
                (dataframe["deviation_pct"] < -threshold)
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
                # Price reverted back to within exit_threshold of the mean
                (dataframe["deviation_pct"] >= -exit_thresh)
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
