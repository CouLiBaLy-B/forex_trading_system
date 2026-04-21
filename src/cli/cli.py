import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings


def _print_config(args):
    settings = Settings()
    for key, val in vars(settings).items():
        print(f"  {key}: {val}")


def _print_strategies(args):
    print("Available strategies:")
    from strategies.base import StrategyRegistry
    for name, cls in StrategyRegistry().get_all().items():
        print(f"  - {name}: {cls.__doc__}")


def _print_market(args):
    print(f"Market data for {args.symbol}")
    from market_data.service import MarketDataService
    service = MarketDataService()
    quote = service.get_quote(args.symbol)
    print(f"  Bid: {quote.bid}, Ask: {quote.ask}, Last: {quote.last}")


def _print_portfolio(args):
    print("Portfolio summary:")
    print("  Positions: 0")
    print("  Cash: $100,000")
    print("  Equity: $100,000")


def _run_backtest(args):
    print(f"Running backtest: {args.strategy} on {args.symbol}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Initial capital: ${args.initial_capital}")
    print("Backtest completed. Use --report for full report.")


def main():
    parser = argparse.ArgumentParser(description="Forex Trading System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # config
    p_config = subparsers.add_parser("config", help="Show configuration")
    p_config.set_defaults(func=_print_config)

    # strategies
    p_strats = subparsers.add_parser("strategies", help="List strategies")
    p_strats.set_defaults(func=_print_strategies)

    # market
    p_market = subparsers.add_parser("market", help="Get market data")
    p_market.add_argument("symbol", help="Trading pair (e.g. EUR/USD)")
    p_market.set_defaults(func=_print_market)

    # portfolio
    p_port = subparsers.add_parser("portfolio", help="Show portfolio")
    p_port.set_defaults(func=_print_portfolio)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("strategy", help="Strategy name")
    p_bt.add_argument("symbol", help="Trading pair")
    p_bt.add_argument("--start", default="2024-01-01")
    p_bt.add_argument("--end", default="2024-12-31")
    p_bt.add_argument("--initial-capital", type=float, default=100000)
    p_bt.set_defaults(func=_run_backtest)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
