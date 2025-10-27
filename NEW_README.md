Things added--

-- Controller Files (`ema_crossover.py` and `macd_momentum.py`) in `app\controllers\directional_trading` to load OHLCV data, initialize strategy parameters and execute strategy runs.

 In `research_notebooks\eda_strategies\ema_crossover` and `research_notebooks\eda_strategies\macd_momentum`:

-- Strategy Design Files (`strategy_design.ipynb`)  that contain the core trading logic and indicator computations (e.g., EMA, MACD, RSI) to generate buy/sell signals based on defined crossover or momentum rules.


-- Backtester Files(`single_controller_backtest.ipynb`) that simulate trades using historical data based on generated signals and track entries, exits, P&L and key metrics.

