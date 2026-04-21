"""PortfolioManager – position lifecycle, P&L calculation, and snapshots."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from .models import (
    AggregatedPosition,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    PortfolioState,
    Position,
    PositionSide,
    PositionSummary,
    TradeRecord,
    TradeStatus,
)


def _compute_unrealized_pnl(
    side: PositionSide,
    entry_price: float,
    current_price: float,
    quantity: float,
) -> float:
    """Calculate unrealized P&L for a single position.

    Args:
        side: Long or short direction.
        entry_price: Price at entry.
        current_price: Latest market price.
        quantity: Position size.

    Returns:
        Unrealized profit/loss (positive = gain).
    """
    price_change = current_price - entry_price
    if side == PositionSide.LONG:
        return price_change * quantity
    else:
        return -price_change * quantity


class _PositionHolder(BaseModel):
    """Internal container that tracks a live position plus its peak/drawdown.

    Attributes:
        position: The position model.
        peak_unrealized_pnl: Highest unrealized P&L seen.
        worst_unrealized_pnl: Lowest unrealized P&L seen.
    """

    position: Position
    peak_unrealized_pnl: float = 0.0
    worst_unrealized_pnl: float = 0.0


class PortfolioManager:
    """Manages trading positions, P&L, and portfolio snapshots.

    Responsibilities:
        * **Position lifecycle** – open, close, modify SL/TP.
        * **P&L calculation** – real-time unrealized and cumulative realized P&L.
        * **Portfolio snapshot** – full state export with equity, drawdown, etc.
        * **Position aggregation** – combine sub-positions by strategy/instrument.

    All prices and quantities use standard forex conventions (e.g. EUR/USD).
    """

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        """Initialize the portfolio manager.

        Args:
            initial_equity: Starting account equity.
        """
        self._initial_equity = initial_equity
        self._peak_equity = initial_equity
        self._total_realized_pnl: float = 0.0
        self._positions: dict[str, _PositionHolder] = {}
        self._trade_history: list[TradeRecord] = []
        self._next_trade_id: int = 1

    # ------------------------------------------------------------------
    #  Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        strategy: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> Position:
        """Open a new trading position.

        Args:
            symbol: Trading pair (e.g. ``"EUR/USD"``).
            side: Long or short.
            quantity: Number of units / lots.
            entry_price: Execution price.
            strategy: Strategy name that triggered the order.
            stop_loss: Optional stop-loss price.
            take_profit: Optional take-profit price.
            order_type: Type of order (market/limit/stop).

        Returns:
            The newly created Position model.

        Raises:
            ValueError: If quantity or entry_price is not positive.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        position_id = str(uuid.uuid4())
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc),
            strategy=strategy,
            order_type=order_type,
        )
        self._positions[position_id] = _PositionHolder(
            position=position,
            peak_unrealized_pnl=0.0,
            worst_unrealized_pnl=0.0,
        )
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> TradeRecord:
        """Close an open position at the given price.

        Args:
            position_id: ID of the position to close.
            exit_price: Execution price at exit.
            reason: Human-readable reason for closing (e.g. ``"stop_loss"``).

        Returns:
            TradeRecord describing the closed trade.

        Raises:
            KeyError: If the position_id is not found among open positions.
        """
        holder = self._positions.pop(position_id, None)
        if holder is None:
            raise KeyError(f"Position {position_id} not found among open positions")

        entry_price = holder.position.entry_price
        side = holder.position.side
        quantity = holder.position.quantity

        # Calculate realized P&L
        price_diff = exit_price - entry_price
        realized_pnl = price_diff * quantity if side == PositionSide.LONG else -price_diff * quantity
        notional = entry_price * quantity
        realized_pnl_pct = realized_pnl / notional if notional != 0 else 0.0

        # Determine trigger
        stop_loss_hit = holder.position.stop_loss is not None and exit_price == holder.position.stop_loss
        take_profit_hit = holder.position.take_profit is not None and exit_price == holder.position.take_profit

        trade = TradeRecord(
            trade_id=str(self._next_trade_id),
            symbol=holder.position.symbol,
            side=side.value,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            stop_loss_hit=stop_loss_hit,
            take_profit_hit=take_profit_hit,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            entry_time=holder.position.entry_time,
            exit_time=datetime.now(timezone.utc),
            strategy=holder.position.strategy,
            reason=reason,
            max_unrealized_pnl=holder.peak_unrealized_pnl,
            max_drawdown=abs(holder.worst_unrealized_pnl),
        )
        self._next_trade_id += 1
        self._trade_history.append(trade)
        self._total_realized_pnl += realized_pnl
        return trade

    def modify_stop_loss(self, position_id: str, new_sl: float) -> Position:
        """Update the stop-loss price for an open position.

        Args:
            position_id: ID of the position.
            new_sl: New stop-loss price.

        Returns:
            Updated Position model.

        Raises:
            KeyError: If the position is not open.
        """
        holder = self._positions.get(position_id)
        if holder is None:
            raise KeyError(f"Position {position_id} not found among open positions")
        holder.position.stop_loss = new_sl
        return holder.position

    def modify_take_profit(self, position_id: str, new_tp: float) -> Position:
        """Update the take-profit price for an open position.

        Args:
            position_id: ID of the position.
            new_tp: New take-profit price.

        Returns:
            Updated Position model.

        Raises:
            KeyError: If the position is not open.
        """
        holder = self._positions.get(position_id)
        if holder is None:
            raise KeyError(f"Position {position_id} not found among open positions")
        holder.position.take_profit = new_tp
        return holder.position

    # ------------------------------------------------------------------
    #  P&L
    # ------------------------------------------------------------------

    def update_mark_to_market(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for all open positions of a symbol.

        Iterates every open position; if its symbol matches, recomputes
        unrealized P&L and tracks peak/drawdown for the position.

        Args:
            symbol: Trading pair to update.
            current_price: Latest market price for the symbol.
        """
        for holder in self._positions.values():
            if holder.position.symbol == symbol:
                upnl = _compute_unrealized_pnl(
                    holder.position.side,
                    holder.position.entry_price,
                    current_price,
                    holder.position.quantity,
                )
                holder.position.unrealized_pnl = upnl
                holder.peak_unrealized_pnl = max(holder.peak_unrealized_pnl, upnl)
                holder.worst_unrealized_pnl = min(holder.worst_unrealized_pnl, upnl)

    def update_mark_to_market_all(self, quotes: dict[str, float]) -> None:
        """Update unrealized P&L for all open positions using a quote map.

        Args:
            quotes: Mapping of symbol -> latest price.
        """
        for symbol, price in quotes.items():
            self.update_mark_to_market(symbol, price)

    def get_realized_pnl(self) -> float:
        """Return cumulative realized profit/loss across all closed trades.

        Returns:
            Total realized P&L (positive = net profit).
        """
        return self._total_realized_pnl

    def get_unrealized_pnl(self) -> float:
        """Return total unrealized P&L across all open positions.

        Returns:
            Sum of unrealized P&L for every open position.
        """
        return sum(
            holder.position.unrealized_pnl for holder in self._positions.values()
        )

    # ------------------------------------------------------------------
    #  Portfolio snapshot
    # ------------------------------------------------------------------

    def get_portfolio_state(self) -> PortfolioState:
        """Return a full snapshot of the current portfolio state.

        Computes equity, exposure, drawdown, and aggregates open/closed
        positions into a serializable DTO.

        Returns:
            PortfolioState with up-to-date values.
        """
        total_unrealized = self.get_unrealized_pnl()
        current_equity = self._initial_equity + self._total_realized_pnl + total_unrealized

        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Update max drawdown
        if self._peak_equity > 0:
            current_dd = 1.0 - (current_equity / self._peak_equity)
        else:
            current_dd = 0.0

        # Track global max drawdown
        if not hasattr(self, "_global_max_dd") or current_dd > self._global_max_dd:  # type: ignore[attr-defined]
            self._global_max_dd = current_dd  # type: ignore[attr-defined]

        open_list = [holder.position for holder in self._positions.values()]
        margin_used = sum(p.entry_price * p.quantity for p in open_list)

        return PortfolioState(
            equity=current_equity,
            cash=self._initial_equity + self._total_realized_pnl,
            margin_used=margin_used,
            total_realized_pnl=self._total_realized_pnl,
            total_unrealized_pnl=total_unrealized,
            open_positions=open_list,
            closed_positions=list(self._trade_history),
            peak_equity=self._peak_equity,
            max_drawdown=self._global_max_dd if hasattr(self, "_global_max_dd") else 0.0,  # type: ignore[attr-defined]
            trade_count=len(self._trade_history),
        )

    # ------------------------------------------------------------------
    #  Position aggregation
    # ------------------------------------------------------------------

    def aggregate_positions_by_instrument(
        self,
        current_prices: dict[str, float],
    ) -> list[AggregatedPosition]:
        """Aggregate all open positions by (symbol, side).

        For each symbol+side combination that has multiple sub-positions,
        merges them into a single aggregated view with volume-weighted
        average entry price.

        Args:
            current_prices: Latest market prices for symbol lookups.

        Returns:
            List of AggregatedPosition DTOs.
        """
        groups: dict[tuple[str, str], list[_PositionHolder]] = {}
        for holder in self._positions.values():
            key = (holder.position.symbol, holder.position.side.value)
            groups.setdefault(key, []).append(holder)

        result: list[AggregatedPosition] = []
        for (symbol, side), holders in groups.items():
            total_qty = sum(h.position.quantity for h in holders)
            avg_entry = (
                sum(h.position.entry_price * h.position.quantity for h in holders)
                / total_qty
                if total_qty != 0
                else 0.0
            )
            total_upnl = sum(h.position.unrealized_pnl for h in holders)
            cps = current_prices.get(symbol, holders[0].position.entry_price)

            # SL: nearest to market for longs, farthest for shorts
            sl_prices = [h.position.stop_loss for h in holders if h.position.stop_loss is not None]
            tp_prices = [h.position.take_profit for h in holders if h.position.take_profit is not None]
            sl = min(sl_prices) if sl_prices else None
            tp = max(tp_prices) if tp_prices else None

            # Most frequent strategy
            strategy_counts: dict[str, int] = {}
            for h in holders:
                strategy_counts[h.position.strategy] = strategy_counts.get(h.position.strategy, 0) + 1
            primary_strategy = max(strategy_counts, key=strategy_counts.get)

            result.append(
                AggregatedPosition(
                    symbol=symbol,
                    side=side,
                    total_quantity=total_qty,
                    average_entry_price=avg_entry,
                    unrealized_pnl=total_upnl,
                    stop_loss=sl,
                    take_profit=tp,
                    position_ids=[h.position.position_id for h in holders],
                    strategy=primary_strategy,
                )
            )
        return result

    def aggregate_positions_by_strategy(
        self,
        current_prices: dict[str, float],
    ) -> dict[str, list[AggregatedPosition]]:
        """Aggregate all open positions by strategy name.

        Args:
            current_prices: Latest market prices for symbol lookups.

        Returns:
            Mapping of strategy name -> list of aggregated positions.
        """
        strategy_map: dict[str, dict[tuple[str, str], list[_PositionHolder]]] = {}
        for holder in self._positions.values():
            strategy_map.setdefault(holder.position.strategy, {}).setdefault(
                (holder.position.symbol, holder.position.side.value), []
            ).append(holder)

        result: dict[str, list[AggregatedPosition]] = {}
        for strategy, groups in strategy_map.items():
            result[strategy] = self._aggregate_from_groups(groups, current_prices)
        return result

    def _aggregate_from_groups(
        self,
        groups: dict[tuple[str, str], list[_PositionHolder]],
        current_prices: dict[str, float],
    ) -> list[AggregatedPosition]:
        """Helper: convert a dict of position groups to AggregatedPosition list."""
        result: list[AggregatedPosition] = []
        for (symbol, side), holders in groups.items():
            total_qty = sum(h.position.quantity for h in holders)
            avg_entry = (
                sum(h.position.entry_price * h.position.quantity for h in holders)
                / total_qty
                if total_qty != 0
                else 0.0
            )
            total_upnl = sum(h.position.unrealized_pnl for h in holders)

            sl_prices = [h.position.stop_loss for h in holders if h.position.stop_loss is not None]
            tp_prices = [h.position.take_profit for h in holders if h.position.take_profit is not None]
            sl = min(sl_prices) if sl_prices else None
            tp = max(tp_prices) if tp_prices else None

            strategy_counts: dict[str, int] = {}
            for h in holders:
                strategy_counts[h.position.strategy] = strategy_counts.get(h.position.strategy, 0) + 1
            primary_strategy = max(strategy_counts, key=strategy_counts.get)

            result.append(
                AggregatedPosition(
                    symbol=symbol,
                    side=side,
                    total_quantity=total_qty,
                    average_entry_price=avg_entry,
                    unrealized_pnl=total_upnl,
                    stop_loss=sl,
                    take_profit=tp,
                    position_ids=[h.position.position_id for h in holders],
                    strategy=primary_strategy,
                )
            )
        return result

    # ------------------------------------------------------------------
    #  Position summaries
    # ------------------------------------------------------------------

    def get_position_summaries(
        self,
        current_prices: dict[str, float],
    ) -> list[PositionSummary]:
        """Build a list of position summaries with current market data.

        Args:
            current_prices: Mapping of symbol -> latest price.

        Returns:
            List of PositionSummary DTOs with live P&L values.
        """
        summaries: list[PositionSummary] = []
        for holder in self._positions.values():
            p = holder.position
            cp = current_prices.get(p.symbol, p.entry_price)
            upnl = _compute_unrealized_pnl(p.side, p.entry_price, cp, p.quantity)
            notional = p.entry_price * p.quantity
            upnl_pct = upnl / notional if notional != 0 else 0.0
            days_open = (datetime.now(timezone.utc) - p.entry_time).days
            summaries.append(
                PositionSummary(
                    position_id=p.position_id,
                    symbol=p.symbol,
                    side=p.side.value,
                    entry_price=p.entry_price,
                    current_price=cp,
                    quantity=p.quantity,
                    unrealized_pnl=upnl,
                    unrealized_pnl_pct=upnl_pct,
                    stop_loss=p.stop_loss,
                    take_profit=p.take_profit,
                    days_open=days_open,
                )
            )
        return summaries

    # ------------------------------------------------------------------
    #  Query helpers
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        """Return all currently open positions.

        Returns:
            List of Position models.
        """
        return [h.position for h in self._positions.values()]

    def get_position(self, position_id: str) -> Optional[Position]:
        """Retrieve a single open position by ID.

        Args:
            position_id: UUID string of the position.

        Returns:
            Position model or None if not found.
        """
        holder = self._positions.get(position_id)
        return holder.position if holder else None

    def get_trade_history(self) -> list[TradeRecord]:
        """Return the full trade history (most recent first).

        Returns:
            List of TradeRecord models in reverse chronological order.
        """
        return list(reversed(self._trade_history))

    @property
    def open_count(self) -> int:
        """Number of currently open positions."""
        return len(self._positions)

    @property
    def peak_equity(self) -> float:
        """Highest equity value ever reached."""
        return self._peak_equity
