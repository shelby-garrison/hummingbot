"""
Bollinger Bands
"""

import pandas as pd
from typing import Optional, TYPE_CHECKING
import plotly.graph_objects as go

from core.features.feature_base import FeatureBase, FeatureConfig
from core.features.models import Feature, Signal

if TYPE_CHECKING:
    from core.data_structures.candles import Candles


class BollingerConfig(FeatureConfig):
    """Configuration for Bollinger Bands feature"""
    name: str = "bollinger"
    length: int = 20
    mult: float = 2.0
    rolling_window: int = 500


class BollingerFeature(FeatureBase[BollingerConfig]):
    """
    Calculates Bollinger Bands and related signals.
    """

    def calculate(self, candles: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in candles.columns:
            raise ValueError("Candles dataframe must contain 'close' column")

        df = candles.copy()

        # Bollinger Band components
        ma = df['close'].rolling(self.config.length).mean()
        sd = df['close'].rolling(self.config.length).std(ddof=0)

        df['bb_mid'] = ma
        df['bb_upper'] = ma + self.config.mult * sd
        df['bb_lower'] = ma - self.config.mult * sd
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Band position (0 = lower band, 1 = upper band)
        df['band_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Signal logic (mean reversion bias)
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1   # price below lower band → long bias
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # price above upper band → short bias

        # Intensity: distance from nearest band
        dist_lower = (df['bb_mid'] - df['close']).clip(lower=0)
        dist_upper = (df['close'] - df['bb_mid']).clip(lower=0)
        df['signal_intensity'] = 0.0
        df.loc[df['signal'] == 1, 'signal_intensity'] = (dist_lower / (df['bb_upper'] - df['bb_lower'])).clip(0, 1)
        df.loc[df['signal'] == -1, 'signal_intensity'] = (dist_upper / (df['bb_upper'] - df['bb_lower'])).clip(0, 1)

        return df

    def create_feature(self, candles: "Candles") -> Feature:
        df = self.calculate(candles.data)
        value = {
            'bb_upper': float(df['bb_upper'].iloc[-1]),
            'bb_mid': float(df['bb_mid'].iloc[-1]),
            'bb_lower': float(df['bb_lower'].iloc[-1]),
            'bb_width': float(df['bb_width'].iloc[-1]),
            'band_pos': float(df['band_pos'].iloc[-1]),
            'signal': int(df['signal'].iloc[-1]),
            'intensity': float(df['signal_intensity'].iloc[-1]),
            'price': float(df['close'].iloc[-1])
        }

        return Feature(
            feature_name="bollinger",
            trading_pair=candles.trading_pair,
            connector_name=candles.connector_name,
            value=value,
            info={
                'length': self.config.length,
                'mult': self.config.mult,
                'description': "Bollinger Bands feature",
                'interval': candles.interval
            }
        )

    def create_signal(self, candles: "Candles", min_intensity: float = 0.6) -> Optional[Signal]:
        df = self.calculate(candles.data)
        signal_dir = int(df['signal'].iloc[-1])
        intensity = float(df['signal_intensity'].iloc[-1])

        if abs(signal_dir) == 1 and intensity >= min_intensity:
            return Signal(
                signal_name=f"bollinger_{self.config.length}_{self.config.mult}",
                trading_pair=candles.trading_pair,
                category='bb',
                value=signal_dir * intensity
            )

        return None

    def add_to_fig(self, fig: go.Figure, candles: "Candles", row: Optional[int] = None, **kwargs) -> go.Figure:
        df = self.calculate(candles.data)
        traces = [
            go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='orange', width=1), name='BB Upper'),
            go.Scatter(x=df.index, y=df['bb_mid'], line=dict(color='gray', width=1), name='BB Mid'),
            go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='orange', width=1), name='BB Lower')
        ]
        for t in traces:
            if row: fig.add_trace(t, row=row, col=1)
            else: fig.add_trace(t)
        return fig

    def __str__(self):
        return f"BollingerFeature(length={self.config.length}, mult={self.config.mult})"
