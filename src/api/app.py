from fastapi import FastAPI
from .routes import health, market, strategies, orders, portfolio, backtest, metrics
from .middleware import error_middleware
from config.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Forex Trading System API",
    )

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(market.router, prefix="/api/v1", tags=["market"])
    app.include_router(strategies.router, prefix="/api/v1", tags=["strategies"])
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(backtest.router, prefix="/api/v1", tags=["backtest"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

    app = error_middleware(app)
    return app
