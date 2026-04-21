from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

env_file = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    # Application
    app_name: str = "forex-trading-system"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501
    worker_poll_interval: int = 60

    # Database
    database_url: str = "sqlite:///./data/market_data.db"
    trading_database_url: str = "sqlite:///./data/trades.db"
    portfolio_database_url: str = "sqlite:///./data/portfolio.db"

    # Trading
    default_account_balance: float = 100_000.0
    max_position_size: float = 0.02
    max_total_exposure: float = 0.10
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.05
    paper_trading: bool = True

    # Market Data
    data_retention_days: int = 365
    data_cache_ttl_hours: int = 1

    # yfinance
    yfinance_proxy: str = ""
    yfinance_timeout: int = 30

    # Risk Management
    risk_mode: str = "conservative"

    # Forex pairs (instrument -> yahoo finance ticker)
    forex_pairs: dict[str, str] = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPYUSD=X",
        "USD/CHF": "CHFUSD=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CADUSD=X",
    }

    # Index tickers
    index_tickers: dict[str, str] = {
        "DAX": "^DAX",
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC",
        "FTSE": "^FTSE",
    }

    class Config:
        env_file = str(env_file)
        env_file_encoding = "utf-8"


settings = Settings()
