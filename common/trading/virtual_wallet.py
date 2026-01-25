"""
VirtualWallet - Track Proteus scan signals as if actually purchased.

Simulates a paper trading portfolio:
- Opens positions based on scan signals (strength >= min_signal_strength)
- Tracks positions with current prices
- Applies exit logic (stop loss, profit targets, max hold days, trailing stops)
- Records trade history and daily snapshots
- Calculates performance metrics (win rate, Sharpe, max drawdown)
- Sends daily email summaries via Mailjet
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from mailjet_rest import Client
except ImportError:
    Client = None


class VirtualWallet:
    """Virtual paper trading wallet for tracking Proteus signals."""

    DATA_DIR = Path('features/simulation/data/virtual_wallet')
    STATE_FILE = DATA_DIR / 'wallet_state.json'
    HISTORY_FILE = DATA_DIR / 'trade_history.json'
    SNAPSHOTS_FILE = DATA_DIR / 'daily_snapshots.json'
    CONFIG_FILE = Path('features/simulation/config.json')
    EMAIL_CONFIG_FILE = Path('email_config.json')
    SCANS_DIR = Path('features/daily_picks/data/smart_scans')
    # Sync positions to scanner's expected location
    SCANNER_POSITIONS_FILE = Path('features/simulation/data/portfolio/active_positions.json')

    def __init__(self, initial_capital: float = 100000, min_signal_strength: float = 65):
        """
        Initialize virtual wallet.

        Args:
            initial_capital: Starting capital
            min_signal_strength: Minimum signal strength to open positions
        """
        self.initial_capital = initial_capital
        self.min_signal_strength = min_signal_strength

        # Ensure data directory exists
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()
        self.portfolio_constraints = self.config.get('portfolio_constraints', {})
        self.max_positions = self.portfolio_constraints.get('max_positions', 6)
        self.max_portfolio_heat = self.portfolio_constraints.get('max_portfolio_heat_pct', 15)

        # Build ticker -> tier lookup
        self.ticker_tiers: Dict[str, str] = {}
        for tier_name, tier_data in self.config.get('stock_tiers', {}).items():
            if tier_name.startswith('_'):
                continue
            for ticker in tier_data.get('tickers', []):
                self.ticker_tiers[ticker] = tier_name

        # Load state
        self.state = self._load_state()
        self.trade_history = self._load_trade_history()
        self.daily_snapshots = self._load_snapshots()

    def _load_config(self) -> dict:
        """Load unified configuration."""
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE) as f:
                return json.load(f)
        return {}

    def _load_state(self) -> dict:
        """Load wallet state from file."""
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)

        # Initialize new state
        return {
            'last_updated': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': self.initial_capital,
            'positions': {},
            'total_equity': self.initial_capital
        }

    def _save_state(self) -> None:
        """Save wallet state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        # Sync to scanner's position file
        self._sync_to_scanner()

    def _sync_to_scanner(self) -> None:
        """Sync positions to scanner's active_positions.json for position awareness."""
        # Ensure directory exists
        self.SCANNER_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Convert to scanner's expected format
        scanner_positions = {}
        for ticker, pos in self.state.get('positions', {}).items():
            scanner_positions[ticker] = {
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'shares': pos['shares'],
                'cost_basis': pos['cost_basis'],
                'tier': pos.get('tier', 'average'),
                'signal_strength': pos.get('signal_strength', 70),
                'high_since_entry': pos.get('high_since_entry', pos['entry_price']),
                'source': 'virtual_wallet'
            }

        scanner_data = {
            'last_updated': datetime.now().isoformat(),
            'source': 'virtual_wallet',
            'portfolio_value': self.state.get('total_equity', self.initial_capital),
            'positions': scanner_positions
        }

        with open(self.SCANNER_POSITIONS_FILE, 'w') as f:
            json.dump(scanner_data, f, indent=2)

    def _load_trade_history(self) -> dict:
        """Load trade history from file."""
        if self.HISTORY_FILE.exists():
            with open(self.HISTORY_FILE) as f:
                return json.load(f)
        return {
            'trades': [],
            'summary': {
                'total_trades': 0,
                'wins': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
        }

    def _save_trade_history(self) -> None:
        """Save trade history to file."""
        # Update summary
        trades = self.trade_history['trades']
        if trades:
            wins = sum(1 for t in trades if t['outcome'] == 'win')
            total_pnl = sum(t['pnl_dollars'] for t in trades)
            self.trade_history['summary'] = {
                'total_trades': len(trades),
                'wins': wins,
                'win_rate': round(wins / len(trades) * 100, 1) if trades else 0,
                'total_pnl': round(total_pnl, 2)
            }

        with open(self.HISTORY_FILE, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

    def _load_snapshots(self) -> dict:
        """Load daily snapshots from file."""
        if self.SNAPSHOTS_FILE.exists():
            with open(self.SNAPSHOTS_FILE) as f:
                return json.load(f)
        return {'snapshots': []}

    def _save_snapshots(self) -> None:
        """Save daily snapshots to file."""
        with open(self.SNAPSHOTS_FILE, 'w') as f:
            json.dump(self.daily_snapshots, f, indent=2)

    def get_stock_tier(self, ticker: str) -> str:
        """Get tier for a ticker."""
        return self.ticker_tiers.get(ticker, 'average')

    def get_exit_params(self, ticker: str, signal_strength: float = 70) -> dict:
        """Get exit parameters for a ticker based on tier."""
        tier = self.get_stock_tier(ticker)
        exit_config = self.config.get('exit_strategy', {}).get(tier, {})

        # Default parameters
        params = {
            'stop_loss': exit_config.get('stop_loss', -2.5),
            'profit_target_1': exit_config.get('profit_target_1', 2.0),
            'profit_target_2': exit_config.get('profit_target_2', 3.0),
            'max_hold_days': exit_config.get('max_hold_days', 3),
            'use_trailing': exit_config.get('use_trailing', True),
            'trailing_distance': exit_config.get('trailing_distance', 1.0)
        }

        # Adjust for strong signals
        if signal_strength >= 80:
            params['max_hold_days'] = min(params['max_hold_days'] + 1, 5)
            params['profit_target_1'] += 0.5
            if params['profit_target_2']:
                params['profit_target_2'] += 0.5

        return params

    def fetch_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for tickers using yfinance."""
        if not yf or not tickers:
            return {}

        prices = {}
        try:
            # Batch download
            data = yf.download(tickers, period='1d', progress=False)
            if not data.empty:
                # Handle single vs multiple tickers
                if len(tickers) == 1:
                    if 'Close' in data.columns:
                        price = data['Close'].iloc[-1]
                        if not math.isnan(price):
                            prices[tickers[0]] = float(price)
                else:
                    if 'Close' in data.columns:
                        close_data = data['Close']
                        for ticker in tickers:
                            if ticker in close_data.columns:
                                price = close_data[ticker].iloc[-1]
                                if not math.isnan(price):
                                    prices[ticker] = float(price)
        except Exception as e:
            print(f"Error fetching prices: {e}")

        return prices

    def get_position_size(self, ticker: str, price: float, signal_strength: float) -> Tuple[int, float]:
        """
        Calculate position size based on tier and constraints.

        Returns:
            Tuple of (shares, cost_basis)
        """
        tier = self.get_stock_tier(ticker)
        tier_config = self.config.get('stock_tiers', {}).get(tier, {})
        position_size_pct = tier_config.get('position_size_pct', 7.5)

        # Calculate max position value
        max_position_value = self.state['total_equity'] * (position_size_pct / 100)

        # Check portfolio heat constraint
        current_heat = self._calculate_portfolio_heat()
        remaining_heat = self.max_portfolio_heat - current_heat
        if remaining_heat <= 0:
            return 0, 0

        # Limit by remaining cash
        available = min(max_position_value, self.state['cash'],
                       self.state['total_equity'] * (remaining_heat / 100))

        # Calculate shares
        shares = int(available / price)
        if shares <= 0:
            return 0, 0

        cost_basis = shares * price
        return shares, cost_basis

    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (% of equity at risk)."""
        if not self.state['positions']:
            return 0

        total_at_risk = 0
        for ticker, pos in self.state['positions'].items():
            # Risk = position value * stop loss percentage
            pos_value = pos['shares'] * pos.get('current_price', pos['entry_price'])
            stop_loss = abs(pos.get('stop_loss', -2.5))
            total_at_risk += pos_value * (stop_loss / 100)

        return (total_at_risk / self.state['total_equity']) * 100 if self.state['total_equity'] > 0 else 0

    def open_position(self, signal: Dict) -> Optional[Dict]:
        """
        Open a new position based on a signal.

        Args:
            signal: Signal dict from scan with ticker, adjusted_strength, current_price, etc.

        Returns:
            Position dict if opened, None otherwise
        """
        ticker = signal.get('ticker')
        strength = signal.get('adjusted_strength', signal.get('raw_strength', 0))
        price = signal.get('current_price', 0)

        # Validation
        if not ticker or not price or price <= 0:
            return None

        if strength < self.min_signal_strength:
            return None

        # Check if already holding
        if ticker in self.state['positions']:
            return None

        # Check max positions
        if len(self.state['positions']) >= self.max_positions:
            return None

        # Check for skip_trade flag
        if signal.get('skip_trade'):
            return None

        # Get position size
        shares, cost_basis = self.get_position_size(ticker, price, strength)
        if shares <= 0:
            return None

        # Get exit parameters
        exit_params = self.get_exit_params(ticker, strength)
        tier = self.get_stock_tier(ticker)

        # Create position
        position = {
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'entry_price': price,
            'shares': shares,
            'cost_basis': round(cost_basis, 2),
            'stop_loss': exit_params['stop_loss'],
            'profit_target_1': exit_params['profit_target_1'],
            'profit_target_2': exit_params['profit_target_2'],
            'max_hold_days': exit_params['max_hold_days'],
            'tier': tier,
            'signal_strength': strength,
            'current_price': price,
            'current_pnl_pct': 0,
            'days_held': 0,
            'high_since_entry': price,
            'trailing_stop_active': False,
            'partial_exit_done': False
        }

        # Update state
        self.state['positions'][ticker] = position
        self.state['cash'] -= cost_basis
        self._update_total_equity()

        return position

    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update all positions with current prices."""
        for ticker, pos in self.state['positions'].items():
            if ticker in prices:
                current_price = prices[ticker]
                entry_price = pos['entry_price']

                # Update current price and P&L
                pos['current_price'] = current_price
                pos['current_pnl_pct'] = round(((current_price / entry_price) - 1) * 100, 2)

                # Update high since entry
                if current_price > pos.get('high_since_entry', entry_price):
                    pos['high_since_entry'] = current_price

                # Update days held
                entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
                pos['days_held'] = (datetime.now() - entry_date).days

        self._update_total_equity()

    def _update_total_equity(self) -> None:
        """Update total equity value."""
        positions_value = sum(
            pos['shares'] * pos.get('current_price', pos['entry_price'])
            for pos in self.state['positions'].values()
        )
        self.state['total_equity'] = round(self.state['cash'] + positions_value, 2)

    def check_exits(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Check all positions for exit conditions.

        Returns:
            List of exit actions taken
        """
        exits = []
        EPSILON = 0.001  # Tolerance for floating point comparison

        for ticker, pos in list(self.state['positions'].items()):
            current_price = prices.get(ticker, pos.get('current_price', pos['entry_price']))
            entry_price = pos['entry_price']
            high_since_entry = pos.get('high_since_entry', entry_price)
            days_held = pos.get('days_held', 0)

            current_return = ((current_price / entry_price) - 1) * 100
            max_return = ((high_since_entry / entry_price) - 1) * 100

            exit_reason = None
            profit_target_1 = pos.get('profit_target_1', 2.0)
            profit_target_2 = pos.get('profit_target_2')

            # Check stop-loss (highest priority)
            if current_return <= pos.get('stop_loss', -2.5):
                exit_reason = f"STOP-LOSS at {current_return:.2f}%"

            # Check trailing stop (if first target hit)
            elif pos.get('trailing_stop_active') or pos.get('partial_exit_done'):
                trailing_distance = self.get_exit_params(ticker).get('trailing_distance', 1.0)
                trailing_trigger = max_return - trailing_distance
                if current_return < trailing_trigger - EPSILON and max_return >= profit_target_1 - EPSILON:
                    exit_reason = f"TRAILING STOP at {current_return:.2f}% (max was {max_return:.2f}%)"

            # Check profit target 1 (activate trailing if profit_target_2 exists)
            elif current_return >= profit_target_1 - EPSILON:
                if profit_target_2:
                    # Activate trailing stop instead of immediate exit
                    pos['trailing_stop_active'] = True
                    pos['partial_exit_done'] = True
                else:
                    exit_reason = f"PROFIT TARGET at {current_return:.2f}%"

            # Check profit target 2
            elif profit_target_2 and current_return >= profit_target_2 - EPSILON:
                exit_reason = f"PROFIT TARGET 2 at {current_return:.2f}%"

            # Check max hold days
            elif days_held >= pos.get('max_hold_days', 3):
                exit_reason = f"MAX HOLD ({pos.get('max_hold_days', 3)}d) at {current_return:.2f}%"

            if exit_reason:
                exit_result = self.close_position(ticker, current_price, exit_reason)
                if exit_result:
                    exits.append(exit_result)

        return exits

    def close_position(self, ticker: str, price: float, reason: str) -> Optional[Dict]:
        """
        Close a position.

        Args:
            ticker: Stock ticker
            price: Exit price
            reason: Exit reason

        Returns:
            Trade record if closed, None otherwise
        """
        if ticker not in self.state['positions']:
            return None

        pos = self.state['positions'][ticker]

        # Calculate P&L
        entry_price = pos['entry_price']
        shares = pos['shares']
        pnl_pct = ((price / entry_price) - 1) * 100
        pnl_dollars = (price - entry_price) * shares

        # Create trade record
        trade = {
            'ticker': ticker,
            'entry_date': pos['entry_date'],
            'exit_date': datetime.now().strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'exit_price': round(price, 2),
            'shares': shares,
            'pnl_pct': round(pnl_pct, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'exit_reason': reason,
            'hold_days': pos.get('days_held', 0),
            'outcome': 'win' if pnl_pct > 0 else 'loss',
            'tier': pos.get('tier', 'average'),
            'signal_strength': pos.get('signal_strength', 0)
        }

        # Update trade history
        self.trade_history['trades'].append(trade)

        # Update cash
        self.state['cash'] += shares * price

        # Remove position
        del self.state['positions'][ticker]

        self._update_total_equity()

        return trade

    def record_daily_snapshot(self) -> Dict:
        """Record a daily equity snapshot for metrics calculation."""
        snapshot = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'equity': self.state['total_equity'],
            'cash': self.state['cash'],
            'num_positions': len(self.state['positions']),
            'positions_value': self.state['total_equity'] - self.state['cash']
        }

        # Avoid duplicate snapshots for same day
        existing_dates = {s['date'] for s in self.daily_snapshots['snapshots']}
        if snapshot['date'] not in existing_dates:
            self.daily_snapshots['snapshots'].append(snapshot)
            self._save_snapshots()

        return snapshot

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        trades = self.trade_history['trades']
        snapshots = self.daily_snapshots['snapshots']

        metrics = {
            'total_trades': len(trades),
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'avg_hold_days': 0
        }

        if not trades:
            return metrics

        # Win/loss stats
        wins = [t for t in trades if t['outcome'] == 'win']
        losses = [t for t in trades if t['outcome'] == 'loss']

        metrics['wins'] = len(wins)
        metrics['losses'] = len(losses)
        metrics['win_rate'] = round(len(wins) / len(trades) * 100, 1)

        # Average win/loss
        if wins:
            metrics['avg_win'] = round(np.mean([t['pnl_pct'] for t in wins]), 2)
        if losses:
            metrics['avg_loss'] = round(np.mean([t['pnl_pct'] for t in losses]), 2)

        # Profit factor
        gross_profit = sum(t['pnl_dollars'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_dollars'] for t in losses)) if losses else 0
        if gross_loss > 0:
            metrics['profit_factor'] = round(gross_profit / gross_loss, 2)

        # Total return
        if self.initial_capital > 0:
            metrics['total_return_pct'] = round(
                ((self.state['total_equity'] / self.initial_capital) - 1) * 100, 2
            )

        # Average hold days
        metrics['avg_hold_days'] = round(np.mean([t['hold_days'] for t in trades]), 1)

        # Sharpe ratio and max drawdown from snapshots
        if len(snapshots) >= 2:
            equities = [s['equity'] for s in snapshots]
            returns = np.diff(equities) / equities[:-1]

            if len(returns) > 1 and np.std(returns) > 0:
                # Annualized Sharpe (assuming 252 trading days)
                metrics['sharpe_ratio'] = round(
                    np.mean(returns) / np.std(returns) * np.sqrt(252), 2
                )

            # Max drawdown
            peak = equities[0]
            max_dd = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            metrics['max_drawdown_pct'] = round(max_dd, 2)

        return metrics

    def process_daily_scan(self, scan_file: Optional[str] = None) -> Dict:
        """
        Main daily operation: update positions, check exits, process new signals.

        Args:
            scan_file: Path to scan file (defaults to latest_scan.json)

        Returns:
            Daily summary dict
        """
        # Load scan file
        if scan_file is None:
            scan_path = self.SCANS_DIR / 'latest_scan.json'
        else:
            scan_path = Path(scan_file)

        scan_data = {}
        if scan_path.exists():
            with open(scan_path) as f:
                scan_data = json.load(f)

        signals = scan_data.get('signals', [])

        # Get all tickers we need prices for
        held_tickers = list(self.state['positions'].keys())
        signal_tickers = [s['ticker'] for s in signals if s.get('adjusted_strength', 0) >= self.min_signal_strength]
        all_tickers = list(set(held_tickers + signal_tickers))

        # Fetch current prices
        prices = self.fetch_prices(all_tickers)

        # Update existing positions
        self.update_positions(prices)

        # Check for exits
        exits = self.check_exits(prices)

        # Process new signals (sorted by strength, descending)
        qualified_signals = [
            s for s in signals
            if s.get('adjusted_strength', 0) >= self.min_signal_strength
            and not s.get('skip_trade')
            and s.get('ticker') not in self.state['positions']
        ]
        qualified_signals.sort(key=lambda x: x.get('adjusted_strength', 0), reverse=True)

        entries = []
        for signal in qualified_signals:
            if len(self.state['positions']) >= self.max_positions:
                break

            # Update signal with current price
            if signal['ticker'] in prices:
                signal['current_price'] = prices[signal['ticker']]

            result = self.open_position(signal)
            if result:
                entries.append({
                    'ticker': signal['ticker'],
                    'price': signal['current_price'],
                    'signal_strength': signal.get('adjusted_strength', 0),
                    'shares': result['shares'],
                    'tier': result['tier']
                })

        # Record daily snapshot
        snapshot = self.record_daily_snapshot()

        # Save state
        self._save_state()
        self._save_trade_history()

        # Build summary
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'total_equity': self.state['total_equity'],
            'cash': self.state['cash'],
            'return_pct': round(((self.state['total_equity'] / self.initial_capital) - 1) * 100, 2),
            'positions': self.state['positions'],
            'num_positions': len(self.state['positions']),
            'entries': entries,
            'exits': exits,
            'regime': scan_data.get('regime', 'unknown'),
            'metrics': self.calculate_metrics()
        }

        return summary

    def _format_regime(self, regime) -> str:
        """Format regime value (handles both string and dict)."""
        if isinstance(regime, dict):
            return regime.get('type', 'unknown').upper()
        return str(regime).upper()

    def format_email_html(self, summary: Dict) -> str:
        """Format summary as HTML email."""
        positions_html = ""
        for ticker, pos in summary['positions'].items():
            pnl_color = '#10b981' if pos['current_pnl_pct'] >= 0 else '#ef4444'
            pnl_sign = '+' if pos['current_pnl_pct'] >= 0 else ''
            pnl_dollars = (pos['current_price'] - pos['entry_price']) * pos['shares']
            positions_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>{ticker}</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">${pos['entry_price']:.2f}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">${pos['current_price']:.2f}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; color: {pnl_color};">
                    {pnl_sign}${pnl_dollars:.0f} ({pnl_sign}{pos['current_pnl_pct']:.1f}%)
                </td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">Day {pos['days_held']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">{pos['tier'].upper()}</td>
            </tr>
            """

        if not positions_html:
            positions_html = '<tr><td colspan="6" style="padding: 8px; text-align: center; color: #666;">No open positions</td></tr>'

        # Entries
        entries_html = ""
        for entry in summary.get('entries', []):
            entries_html += f"<div style='margin: 5px 0;'>+ {entry['ticker']} @ ${entry['price']:.2f} (signal {entry['signal_strength']:.0f})</div>"
        if not entries_html:
            entries_html = "<div style='color: #666;'>None</div>"

        # Exits
        exits_html = ""
        for exit_trade in summary.get('exits', []):
            pnl_color = '#10b981' if exit_trade['pnl_pct'] >= 0 else '#ef4444'
            pnl_sign = '+' if exit_trade['pnl_pct'] >= 0 else ''
            exits_html += f"<div style='margin: 5px 0; color: {pnl_color};'>- {exit_trade['ticker']} @ ${exit_trade['exit_price']:.2f} ({pnl_sign}{exit_trade['pnl_pct']:.1f}%) - {exit_trade['exit_reason']}</div>"
        if not exits_html:
            exits_html = "<div style='color: #666;'>None</div>"

        # Metrics
        metrics = summary.get('metrics', {})

        # Return color
        return_color = '#10b981' if summary['return_pct'] >= 0 else '#ef4444'
        return_sign = '+' if summary['return_pct'] >= 0 else ''

        html = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; max-width: 700px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin-bottom: 25px;">
        <h1 style="margin: 0;">Proteus Virtual Wallet</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Daily Portfolio Summary</p>
        <p style="font-size: 0.9em; opacity: 0.8;">{summary['date']}</p>
    </div>

    <div style="background: #f0fdf4; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid {return_color};">
        <h2 style="margin-top: 0; color: #374151;">Portfolio Summary</h2>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="margin: 10px 0;">
                <span style="color: #666;">Initial:</span>
                <strong>${summary['initial_capital']:,.0f}</strong>
            </div>
            <div style="margin: 10px 0;">
                <span style="color: #666;">Current:</span>
                <strong style="color: {return_color};">${summary['total_equity']:,.0f} ({return_sign}{summary['return_pct']:.2f}%)</strong>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="margin: 10px 0;">
                <span style="color: #666;">Cash:</span>
                <strong>${summary['cash']:,.0f}</strong>
            </div>
            <div style="margin: 10px 0;">
                <span style="color: #666;">Positions:</span>
                <strong>${summary['total_equity'] - summary['cash']:,.0f}</strong>
            </div>
        </div>
        <div style="margin: 10px 0;">
            <span style="color: #666;">Regime:</span>
            <strong>{self._format_regime(summary.get('regime', 'unknown'))}</strong>
        </div>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;">
        <h3 style="margin-top: 0; color: #374151;">Open Positions ({summary['num_positions']})</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #e5e7eb;">
                <th style="padding: 8px; text-align: left;">Ticker</th>
                <th style="padding: 8px; text-align: left;">Entry</th>
                <th style="padding: 8px; text-align: left;">Current</th>
                <th style="padding: 8px; text-align: left;">P&L</th>
                <th style="padding: 8px; text-align: left;">Days</th>
                <th style="padding: 8px; text-align: left;">Tier</th>
            </tr>
            {positions_html}
        </table>
    </div>

    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;">
        <h3 style="margin-top: 0; color: #374151;">Today's Activity</h3>
        <div style="margin-bottom: 15px;">
            <strong>Entries:</strong>
            {entries_html}
        </div>
        <div>
            <strong>Exits:</strong>
            {exits_html}
        </div>
    </div>

    <div style="background: #e0f2fe; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #0284c7;">
        <h3 style="margin-top: 0; color: #0284c7;">Performance ({metrics.get('total_trades', 0)} trades)</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div><span style="color: #666;">Win Rate:</span> <strong>{metrics.get('win_rate', 0):.1f}%</strong></div>
            <div><span style="color: #666;">Sharpe:</span> <strong>{metrics.get('sharpe_ratio', 0):.2f}</strong></div>
            <div><span style="color: #666;">Max Drawdown:</span> <strong>{metrics.get('max_drawdown_pct', 0):.1f}%</strong></div>
            <div><span style="color: #666;">Profit Factor:</span> <strong>{metrics.get('profit_factor', 0):.2f}</strong></div>
            <div><span style="color: #666;">Avg Win:</span> <strong style="color: #10b981;">+{metrics.get('avg_win', 0):.1f}%</strong></div>
            <div><span style="color: #666;">Avg Loss:</span> <strong style="color: #ef4444;">{metrics.get('avg_loss', 0):.1f}%</strong></div>
        </div>
    </div>

    <div style="color: #666; font-size: 0.9em; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
        <p><em>Automated report from Proteus Virtual Wallet</em></p>
        <p><em>This is simulated paper trading - no real money involved</em></p>
    </div>
</body>
</html>
"""
        return html

    def send_daily_email(self, summary: Dict) -> bool:
        """
        Send daily summary email via Mailjet.

        Returns:
            True if sent successfully, False otherwise
        """
        if Client is None:
            print("Mailjet not installed. Run: pip install mailjet_rest")
            return False

        if not self.EMAIL_CONFIG_FILE.exists():
            print(f"Email config not found: {self.EMAIL_CONFIG_FILE}")
            return False

        with open(self.EMAIL_CONFIG_FILE) as f:
            email_config = json.load(f)

        # Check for required Mailjet keys
        api_key = email_config.get('mailjet_api_key')
        secret_key = email_config.get('mailjet_secret_key')

        if not api_key or not secret_key:
            print("Mailjet API keys not configured in email_config.json")
            return False

        # Format email
        html_content = self.format_email_html(summary)
        return_sign = '+' if summary['return_pct'] >= 0 else ''
        subject = f"Proteus Virtual Wallet - {summary['date']} ({return_sign}{summary['return_pct']:.2f}%)"

        # Send via Mailjet
        mailjet = Client(auth=(api_key, secret_key), version='v3.1')

        data = {
            'Messages': [
                {
                    'From': {
                        'Email': email_config.get('sender_email', 'noreply@example.com'),
                        'Name': 'Proteus Virtual Wallet'
                    },
                    'To': [
                        {
                            'Email': email_config.get('recipient_email'),
                            'Name': 'Trader'
                        }
                    ],
                    'Subject': subject,
                    'HTMLPart': html_content
                }
            ]
        }

        try:
            result = mailjet.send.create(data=data)
            if result.status_code == 200:
                print(f"Email sent successfully!")
                return True
            else:
                print(f"Email failed: {result.status_code} - {result.json()}")
                return False
        except Exception as e:
            print(f"Email error: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current wallet status."""
        return {
            'last_updated': self.state.get('last_updated'),
            'initial_capital': self.initial_capital,
            'total_equity': self.state['total_equity'],
            'cash': self.state['cash'],
            'return_pct': round(((self.state['total_equity'] / self.initial_capital) - 1) * 100, 2),
            'num_positions': len(self.state['positions']),
            'positions': list(self.state['positions'].keys()),
            'total_trades': len(self.trade_history['trades']),
            'win_rate': self.trade_history['summary'].get('win_rate', 0)
        }

    def run_scanner_and_process(self) -> Dict:
        """
        Run the smart scanner and automatically process results.

        This is the recommended way to update the wallet - it:
        1. Syncs current positions to scanner (so it knows what we hold)
        2. Runs the scanner with position awareness
        3. Processes the scan results
        4. Returns the summary

        Returns:
            Daily summary dict
        """
        # Sync positions first so scanner knows what we're holding
        self._sync_to_scanner()

        # Run scanner
        try:
            from trading.smart_scanner_v2 import SmartScannerV2
            print("[WALLET] Running SmartScannerV2...")
            scanner = SmartScannerV2(portfolio_value=self.state['total_equity'])
            scan_result = scanner.scan()

            # Save the scan result
            print("[WALLET] Processing scan results...")

        except Exception as e:
            print(f"[WALLET] Scanner error: {e}")
            print("[WALLET] Falling back to latest_scan.json...")

        # Process using latest scan (scanner saves to this file)
        return self.process_daily_scan()


def demo():
    """Demonstrate VirtualWallet usage."""
    print("=" * 60)
    print("VIRTUAL WALLET DEMO")
    print("=" * 60)

    wallet = VirtualWallet(initial_capital=100000, min_signal_strength=65)

    print(f"\nInitial State:")
    status = wallet.get_status()
    print(f"  Capital: ${status['total_equity']:,.0f}")
    print(f"  Cash: ${status['cash']:,.0f}")
    print(f"  Positions: {status['num_positions']}")

    print(f"\nProcessing daily scan...")
    summary = wallet.process_daily_scan()

    print(f"\nAfter Processing:")
    print(f"  Equity: ${summary['total_equity']:,.0f} ({summary['return_pct']:+.2f}%)")
    print(f"  Positions: {summary['num_positions']}")
    print(f"  Entries: {len(summary['entries'])}")
    print(f"  Exits: {len(summary['exits'])}")

    if summary['positions']:
        print(f"\nOpen Positions:")
        for ticker, pos in summary['positions'].items():
            print(f"  {ticker}: ${pos['entry_price']:.2f} -> ${pos['current_price']:.2f} ({pos['current_pnl_pct']:+.1f}%)")

    metrics = summary['metrics']
    if metrics['total_trades'] > 0:
        print(f"\nPerformance ({metrics['total_trades']} trades):")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
