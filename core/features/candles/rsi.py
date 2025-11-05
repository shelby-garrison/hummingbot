"""RSI"""

from typing import Optional, TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class RSIConfig(FeatureConfig):
    """Configuration settings for the RSI feature."""

    name: str = "rsi"
    length: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    rolling_window: int = 500


class RSIFeature(FeatureBase[RSIConfig]):
    """Calculate RSI values and derive trading signals from them."""

    def calculate(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the candle frame annotated with RSI columns."""

        if "close" not in candles.columns:
            raise ValueError("Candles dataframe must contain 'close' column")

        df = candles.copy()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.config.length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.config.length, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["signal"] = 0
        df.loc[df["rsi"] < self.config.oversold, "signal"] = 1
        df.loc[df["rsi"] > self.config.overbought, "signal"] = -1

        df["signal_intensity"] = 0.0
        long_mask = df["signal"] == 1
        short_mask = df["signal"] == -1

        if long_mask.any():
            df.loc[long_mask, "signal_intensity"] = (
                (self.config.oversold - df.loc[long_mask, "rsi"]) / self.config.oversold
            ).clip(lower=0, upper=1)

        if short_mask.any():
            df.loc[short_mask, "signal_intensity"] = (
                (df.loc[short_mask, "rsi"] - self.config.overbought) / (100 - self.config.overbought)
            ).clip(lower=0, upper=1)

        return df

    def create_feature(self, candles: "Candles") -> Feature:
        """Create a `Feature` payload with the most recent RSI values."""

        df = self.calculate(candles.data)
        value = {
            "rsi": float(df["rsi"].iloc[-1]),
            "signal": int(df["signal"].iloc[-1]),
            "intensity": float(df["signal_intensity"].iloc[-1]),
            "price": float(df["close"].iloc[-1]),
        }

        return Feature(
            feature_name="rsi",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value=value,
            info={
                "length": self.config.length,
                "description": "Relative Strength Index feature",
                "interval": candles.interval,
            },
        )

    def create_signal(self, candles: "Candles", min_intensity: float = 0.6) -> Optional[Signal]:
        """Return a `Signal` if the latest RSI reading clears the thresholds."""

        df = self.calculate(candles.data)
        signal_dir = int(df["signal"].iloc[-1])
        intensity = float(df["signal_intensity"].iloc[-1])

        if abs(signal_dir) == 1 and intensity >= min_intensity:
            return Signal(
                signal_name=f"rsi_{self.config.length}",
                trading_pair=candles.trading_pair,
                category="rsi",
                value=signal_dir * intensity,
            )

        return None

    def add_to_fig(
        self,
        fig: go.Figure,
        candles: "Candles",
        row: Optional[int] = None,
        **kwargs,
    ) -> go.Figure:
        """Overlay RSI and bounds on a Plotly figure."""

        df = self.calculate(candles.data)

        rsi_trace = go.Scatter(
            x=df.index,
            y=df["rsi"],
            mode="lines",
            name="RSI",
            line=dict(color="purple", width=1.5),
        )
        overbought_trace = go.Scatter(
            x=df.index,
            y=[self.config.overbought] * len(df),
            mode="lines",
            name="Overbought",
            line=dict(color="red", dash="dash"),
        )
        oversold_trace = go.Scatter(
            x=df.index,
            y=[self.config.oversold] * len(df),
            mode="lines",
            name="Oversold",
            line=dict(color="green", dash="dash"),
        )

        traces = [rsi_trace, overbought_trace, oversold_trace]

        for trace in traces:
            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

        return fig

    def __str__(self):  # pragma: no cover - trivial string representation
        return f"RSIFeature(length={self.config.length})"
