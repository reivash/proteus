"""
Daily Paper Trading Runner

Orchestrates daily paper trading workflow:
1. Check market regime
2. Scan for new signals
3. Process entries
4. Update positions and check exits
5. Track performance
6. Generate daily report

Usage:
    python src/trading/daily_runner.py

    Or for historical testing:
    python src/trading/daily_runner.py --date 2025-11-14
"""

import argparse
from datetime import datetime
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.trading.signal_scanner import SignalScanner
from src.trading.paper_trader import PaperTrader
from src.trading.performance_tracker import PerformanceTracker
from src.data.fetchers.rate_limited_yahoo import RateLimitedYahooFinanceFetcher


class DailyRunner:
    """
    Orchestrates daily paper trading workflow.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 0.1,
        max_positions: int = 5
    ):
        """
        Initialize daily runner.

        Args:
            initial_capital: Starting capital for paper trading
            position_size: Position size as fraction of capital
            max_positions: Maximum concurrent positions
        """
        # Q4-ONLY FILTER DEPLOYED (2025-11-18)
        # Trade only top 25% quality signals based on EXP-091 research
        # Expected: +14.1pp win rate (63.7% → 77.8%)
        # Threshold: 65.0 (conservative estimate, exact value pending EXP-093 validation)
        self.scanner = SignalScanner(lookback_days=90, min_signal_strength=65.0)

        # DYNAMIC POSITION SIZING DEPLOYED (2025-11-18)
        # EXP-096: Signal-strength-based tiered position sizing
        # ELITE (90-100): 1.5x, STRONG (80-89): 1.25x, GOOD (70-79): 1.0x, ACCEPTABLE (65-69): 0.75x
        # Portfolio heat limit: 50% max deployed capital for risk management
        self.trader = PaperTrader(
            initial_capital=initial_capital,
            profit_target=2.0,
            stop_loss=-2.0,
            max_hold_days=2,
            position_size=position_size,
            max_positions=max_positions,
            use_limit_orders=True,           # EXP-080: +29% improvement
            use_dynamic_sizing=True,         # EXP-096: Signal-strength tiered sizing
            max_portfolio_heat=0.50          # EXP-096: Max 50% capital deployed
        )
        self.tracker = PerformanceTracker()
        self.fetcher = RateLimitedYahooFinanceFetcher()

    def run_daily_workflow(self, date: str = None) -> Dict:
        """
        Run complete daily workflow.

        Args:
            date: Date to run workflow for (default: today)

        Returns:
            Workflow results dictionary
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        print("=" * 70)
        print(f"PAPER TRADING DAILY WORKFLOW - {date}")
        print("=" * 70)
        print()

        results = {
            'date': date,
            'regime': None,
            'signals_found': 0,
            'entries_executed': 0,
            'exits_executed': 0,
            'performance': None,
            'report': None
        }

        # Step 1: Check market regime
        print("Step 1: Checking market regime...")
        regime = self.scanner.get_market_regime()
        results['regime'] = regime
        print(f"[REGIME] {regime}")
        print()

        if regime == 'BEAR':
            print("[ALERT] Market is in BEAR regime - Trading disabled!")
            print("No signals will be scanned in bear markets.")
            print()
            results['report'] = self._generate_bear_market_report(date)
            return results

        # Step 2: Scan for signals
        print(f"Step 2: Scanning for signals on {date}...")
        signals = self.scanner.scan_all_stocks(date)
        results['signals_found'] = len(signals)
        print(f"[SIGNALS] Found {len(signals)} signal(s)")
        print()

        if signals:
            print("Signals details:")
            for signal in signals:
                print(f"  {signal['ticker']}: {signal['signal_type']} @ ${signal['price']:.2f} "
                      f"(Z: {signal['z_score']:.2f}, RSI: {signal['rsi']:.1f}, "
                      f"Expected: {signal['expected_return']:.2f}%)")
            print()

        # Step 3: Process entries
        if signals:
            print("Step 3: Processing entry signals...")
            entries = self.trader.process_signals(signals, date)
            results['entries_executed'] = len(entries)
            print(f"[ENTRIES] Executed {len(entries)} entry trade(s)")
            print()

            # Log trades with tracker
            for entry in entries:
                print(f"  Entered {entry['ticker']}: {entry['shares']:.2f} shares @ ${entry['entry_price']:.2f}")
        else:
            print("Step 3: No entry signals to process")
            print()

        # Step 4: Update positions and check exits
        print("Step 4: Checking exits for open positions...")
        current_prices = self._fetch_current_prices(date)

        if current_prices:
            # Update all positions with current prices
            self.trader.update_daily_equity(current_prices, date)

            # Check for exits
            exits = self.trader.check_exits(current_prices, date)
            results['exits_executed'] = len(exits)
            print(f"[EXITS] Executed {len(exits)} exit trade(s)")

            # Log exits with tracker
            for exit_trade in exits:
                self.tracker.log_trade(exit_trade)
                print(f"  Exited {exit_trade['ticker']}: {exit_trade['exit_reason']} "
                      f"@ ${exit_trade['exit_price']:.2f}, "
                      f"P&L: ${exit_trade['pnl']:+.2f} ({exit_trade['return']:+.2f}%)")
            print()
        else:
            print("[WARN] Could not fetch current prices for exit checks")
            print()

        # Step 5: Update performance tracking
        print("Step 5: Updating performance tracking...")
        self.tracker.update(self.trader)
        print("[OK] Performance updated")
        print()

        # Step 6: Generate daily report
        print("Step 6: Generating daily report...")
        report = self.tracker.generate_daily_report()
        results['performance'] = self.tracker.get_stats_summary()
        results['report'] = report
        print()

        # Save states
        self.trader.save()
        self.tracker.export_to_csv()

        return results

    def _fetch_current_prices(self, date: str) -> Dict[str, float]:
        """
        Fetch current prices for all tickers.

        Args:
            date: Date to fetch prices for

        Returns:
            Dictionary of {ticker: price}
        """
        prices = {}

        # Get all tickers (from open positions + watchlist)
        tickers = set()
        for pos in self.trader.get_open_positions():
            tickers.add(pos['ticker'])

        if not tickers:
            return prices

        print(f"Fetching prices for {len(tickers)} ticker(s)...")

        for ticker in tickers:
            try:
                data = self.fetcher.fetch_stock_data(
                    ticker,
                    start_date=date,
                    end_date=date
                )
                if not data.empty:
                    prices[ticker] = float(data['Close'].iloc[-1])
                    print(f"  {ticker}: ${prices[ticker]:.2f}")
            except Exception as e:
                print(f"  [ERROR] Could not fetch {ticker}: {e}")

        return prices

    def _generate_bear_market_report(self, date: str) -> str:
        """Generate report for bear market days."""
        report = []
        report.append("=" * 70)
        report.append(f"PAPER TRADING REPORT - {date}")
        report.append("=" * 70)
        report.append("")
        report.append("[ALERT] BEAR MARKET DETECTED - TRADING DISABLED")
        report.append("")
        report.append("Strategy rules prohibit trading during bear market regimes.")
        report.append("The system will resume scanning when market returns to BULL or SIDEWAYS.")
        report.append("")
        report.append("=" * 70)
        return "\n".join(report)

    def print_workflow_summary(self, results: Dict):
        """
        Print workflow summary.

        Args:
            results: Workflow results dictionary
        """
        print()
        print("=" * 70)
        print("WORKFLOW SUMMARY")
        print("=" * 70)
        print(f"Date:                 {results['date']}")
        print(f"Market Regime:        {results['regime']}")
        print(f"Signals Found:        {results['signals_found']}")
        print(f"Entries Executed:     {results['entries_executed']}")
        print(f"Exits Executed:       {results['exits_executed']}")

        if results['performance']:
            perf = results['performance']
            print()
            print("CURRENT PERFORMANCE")
            print("-" * 70)
            print(f"Total Equity:         ${perf.get('current_equity', 0):,.2f}")
            print(f"Total Return:         {perf.get('total_return', 0):.2f}%")
            print(f"Total Trades:         {perf.get('total_trades', 0)}")
            print(f"Win Rate:             {perf.get('win_rate', 0):.1f}%")
            if perf.get('sharpe_ratio'):
                print(f"Sharpe Ratio:         {perf['sharpe_ratio']:.2f}")
            if perf.get('max_drawdown'):
                print(f"Max Drawdown:         {perf['max_drawdown']:.2f}%")

        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run daily paper trading workflow')
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date to run workflow for (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as fraction of capital (default: 0.1)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions (default: 5)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Only generate report without trading'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = DailyRunner(
        initial_capital=args.capital,
        position_size=args.position_size,
        max_positions=args.max_positions
    )

    if args.report_only:
        # Just generate report
        print("Generating report...")
        report = runner.tracker.generate_daily_report()
        print(report)
        return

    # Run workflow
    try:
        results = runner.run_daily_workflow(date=args.date)

        # Print summary
        runner.print_workflow_summary(results)

        # Print full report
        if results['report']:
            print()
            print(results['report'])

    except Exception as e:
        print(f"[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
