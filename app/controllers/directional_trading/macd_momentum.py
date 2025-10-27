from typing import List
import pandas_ta as ta
from pydantic import Field, validator
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class MacdMomentumControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "macd_momentum"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(
        default="5m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 3m, 5m, 15m): ",
            prompt_on_new=False
        )
    )
    macd_fast: int = Field(
        default=12,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True
        )
    )
    macd_slow: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True
        )
    )
    macd_signal: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True
        )
    )
    rsi_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI period: ",
            prompt_on_new=True
        )
    )

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class MacdMomentumController(DirectionalTradingControllerBase):
    def __init__(self, config: MacdMomentumControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # --- Calculate Indicators ---
        macd_cols = df.ta.macd(
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            append=True
        )
        rsi_col = df.ta.rsi(length=self.config.rsi_period, append=True)

        # Get actual column names
        macd_col = macd_cols.columns[0]
        macd_signal_col = macd_cols.columns[1]
        macd_hist_col = macd_cols.columns[2]
        rsi_colname = rsi_col.name

        # Ensure columns exist
        for col in [macd_col, macd_signal_col, macd_hist_col, rsi_colname]:
            if col not in df.columns:
                raise KeyError(f"Column {col} not found in DataFrame.")

        # --- Generate Signals ---
        df["signal"] = 0

        # Long (bullish momentum)
        long_condition = (
            (df[macd_col] > df[macd_signal_col]) &
            (df[macd_col].shift(1) <= df[macd_signal_col].shift(1)) &  # MACD crosses above signal
            (df[macd_hist_col] > df[macd_hist_col].shift(1)) &         # histogram expanding positively
            (df[rsi_colname] > 50)                                     # RSI filter
        )

        # Short (bearish momentum)
        short_condition = (
            (df[macd_col] < df[macd_signal_col]) &
            (df[macd_col].shift(1) >= df[macd_signal_col].shift(1)) &  # MACD crosses below signal
            (df[macd_hist_col] < df[macd_hist_col].shift(1)) &         # histogram expanding negatively
            (df[rsi_colname] < 50)                                     # RSI filter
        )

        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # --- Exits ---
        # Exit when MACD crosses zero line (momentum fade)
        df.loc[(df[macd_col] * df[macd_col].shift(1) < 0), "signal"] = 0

        # --- Update processed data for controller ---
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
