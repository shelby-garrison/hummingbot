from typing import List
import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator


class EMACrossoverControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "ema_crossover"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 3m, 5m): ",
            prompt_on_new=False
        )
    )
    ema_fast: int = Field(
        default=25,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the fast EMA period: ",
            prompt_on_new=True
        )
    )
    ema_slow: int = Field(
        default=50,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the slow EMA period: ",
            prompt_on_new=True
        )
    )
    volume_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the average volume period: ",
            prompt_on_new=True
        )
    )
    volume_multiplier: float = Field(
        default=1.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume multiplier (e.g., 1.5x): ",
            prompt_on_new=True
        )
    )
    adx_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ADX period for trend strength filter: ",
            prompt_on_new=True
        )
    )
    adx_threshold: int = Field(
        default=25,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ADX threshold for valid trades: ",
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


class EMACrossoverController(DirectionalTradingControllerBase):
    def __init__(self, config: EMACrossoverControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_trading_pair,
                    interval=config.interval,
                    max_records=self.max_records
                )
            ]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Calculate indicators
        ema_fast_col = df.ta.ema(length=self.config.ema_fast, append=True).name
        ema_slow_col = df.ta.ema(length=self.config.ema_slow, append=True).name
        adx_col = df.ta.adx(length=self.config.adx_period, append=True).iloc[:, 0].name
        df["avg_volume"] = df["volume"].rolling(window=self.config.volume_period).mean()

        # Validate indicator columns
        required_columns = [ema_fast_col, ema_slow_col, adx_col, "avg_volume"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column {col} not found in DataFrame.")

        fast_ema = df[ema_fast_col]
        slow_ema = df[ema_slow_col]
        adx = df[adx_col]
        close = df["close"]
        volume = df["volume"]

        # Entry logic
        long_condition = (
            (fast_ema > slow_ema)
            & (fast_ema.shift(1) <= slow_ema.shift(1))
            & (volume > df["avg_volume"] * self.config.volume_multiplier)
            & (adx > self.config.adx_threshold)
        )

        short_condition = (
            (fast_ema < slow_ema)
            & (fast_ema.shift(1) >= slow_ema.shift(1))
            & (volume > df["avg_volume"] * self.config.volume_multiplier)
            & (adx > self.config.adx_threshold)
        )

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Store output
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
