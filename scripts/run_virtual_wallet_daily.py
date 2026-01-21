#!/usr/bin/env python
"""
Daily Virtual Wallet Update Script

Entry point for daily virtual wallet operations:
- Updates positions with current prices
- Checks exit conditions
- Opens new positions from scan signals
- Records daily snapshot
- Sends email summary via Mailjet

Usage:
    python scripts/run_virtual_wallet_daily.py               # Update and send email
    python scripts/run_virtual_wallet_daily.py --no-email    # Skip email
    python scripts/run_virtual_wallet_daily.py --scan-first  # Run scanner first
    python scripts/run_virtual_wallet_daily.py --status      # Show current status
    python scripts/run_virtual_wallet_daily.py --reset       # Reset wallet to initial state
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading.virtual_wallet import VirtualWallet


def run_scanner():
    """Run the smart scanner to generate latest_scan.json."""
    try:
        from trading.scanner_runner import ScannerRunner
        print("Running smart scanner with health checks...")
        runner = ScannerRunner()

        # Run health checks first
        health = runner.run_health_checks()
        if not health.healthy:
            print(f"Health checks failed: {health.errors}")
            return False

        if health.warnings:
            for warn in health.warnings:
                print(f"  Warning: {warn}")

        # Run scan
        result, error = runner.run_safe_scan(skip_health_checks=True)
        if error:
            print(f"Scanner error: {error}")
            return False

        print(f"Scan complete: {len(result.get('signals', []))} signals")
        return True
    except Exception as e:
        print(f"Scanner error: {e}")
        return False


def show_status(wallet: VirtualWallet):
    """Display current wallet status."""
    status = wallet.get_status()

    print("\n" + "=" * 50)
    print("VIRTUAL WALLET STATUS")
    print("=" * 50)
    print(f"Last Updated: {status['last_updated']}")
    print(f"\nCapital:")
    print(f"  Initial:  ${status['initial_capital']:>12,.0f}")
    print(f"  Current:  ${status['total_equity']:>12,.0f} ({status['return_pct']:+.2f}%)")
    print(f"  Cash:     ${status['cash']:>12,.0f}")

    print(f"\nPositions: {status['num_positions']}")
    if status['positions']:
        for ticker in status['positions']:
            pos = wallet.state['positions'][ticker]
            pnl_sign = '+' if pos['current_pnl_pct'] >= 0 else ''
            print(f"  {ticker:6s}  ${pos['entry_price']:>8.2f} -> ${pos['current_price']:>8.2f}  "
                  f"({pnl_sign}{pos['current_pnl_pct']:.1f}%)  Day {pos['days_held']}")

    print(f"\nTrade History:")
    print(f"  Total Trades: {status['total_trades']}")
    print(f"  Win Rate:     {status['win_rate']:.1f}%")

    if wallet.trade_history['trades']:
        recent = wallet.trade_history['trades'][-5:]
        print(f"\nRecent Trades:")
        for trade in reversed(recent):
            pnl_sign = '+' if trade['pnl_pct'] >= 0 else ''
            outcome = 'W' if trade['outcome'] == 'win' else 'L'
            print(f"  [{outcome}] {trade['ticker']:6s}  {pnl_sign}{trade['pnl_pct']:>5.1f}%  "
                  f"${trade['pnl_dollars']:>7.0f}  {trade['exit_reason'][:20]}")

    print("=" * 50)


def reset_wallet():
    """Reset wallet to initial state."""
    wallet_dir = Path('data/virtual_wallet')

    # Confirm
    response = input("\nThis will delete all wallet data. Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    # Remove files
    files = ['wallet_state.json', 'trade_history.json', 'daily_snapshots.json']
    for f in files:
        path = wallet_dir / f
        if path.exists():
            path.unlink()
            print(f"Deleted: {path}")

    print("\nWallet reset complete. Run again to initialize fresh wallet.")


def main():
    parser = argparse.ArgumentParser(
        description='Virtual Wallet Daily Update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_virtual_wallet_daily.py               # Normal daily update
  python scripts/run_virtual_wallet_daily.py --no-email    # Skip email notification
  python scripts/run_virtual_wallet_daily.py --scan-first  # Run scanner then update
  python scripts/run_virtual_wallet_daily.py --status      # View current status
        """
    )

    parser.add_argument('--no-email', action='store_true',
                       help='Skip sending email notification')
    parser.add_argument('--scan-first', action='store_true',
                       help='Run smart scanner before processing')
    parser.add_argument('--full', action='store_true',
                       help='Full cycle: sync positions, run scanner, process, email')
    parser.add_argument('--status', action='store_true',
                       help='Show current wallet status and exit')
    parser.add_argument('--reset', action='store_true',
                       help='Reset wallet to initial state')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    parser.add_argument('--min-strength', type=float, default=65,
                       help='Minimum signal strength (default: 65)')
    parser.add_argument('--scan-file', type=str,
                       help='Path to specific scan file to process')

    args = parser.parse_args()

    # Handle reset
    if args.reset:
        reset_wallet()
        return

    # Initialize wallet
    wallet = VirtualWallet(
        initial_capital=args.capital,
        min_signal_strength=args.min_strength
    )

    # Handle status
    if args.status:
        show_status(wallet)
        return

    print("=" * 60)
    print("PROTEUS VIRTUAL WALLET - DAILY UPDATE")
    print("=" * 60)

    # Full cycle mode (recommended for daily use)
    if args.full:
        print("\n[FULL CYCLE] Syncing positions, running scanner, processing...")
        summary = wallet.run_scanner_and_process()
    else:
        # Run scanner if requested
        if args.scan_first:
            if not run_scanner():
                print("Warning: Scanner failed, using existing scan file")

        # Process daily scan
        print("\nProcessing daily scan...")
        summary = wallet.process_daily_scan(scan_file=args.scan_file)

    # Display results
    return_sign = '+' if summary['return_pct'] >= 0 else ''
    print(f"\n{'PORTFOLIO SUMMARY':=^50}")
    print(f"Initial: ${summary['initial_capital']:,.0f}  |  "
          f"Current: ${summary['total_equity']:,.0f} ({return_sign}{summary['return_pct']:.2f}%)")
    print(f"Cash: ${summary['cash']:,.0f}  |  "
          f"Positions: ${summary['total_equity'] - summary['cash']:,.0f}")
    regime = summary.get('regime', 'unknown')
    if isinstance(regime, dict):
        regime = regime.get('type', 'unknown')
    print(f"Regime: {regime.upper()}")

    # Positions
    print(f"\n{'OPEN POSITIONS':=^50}")
    if summary['positions']:
        print(f"{'Ticker':<8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'Days':>6}")
        print("-" * 50)
        for ticker, pos in summary['positions'].items():
            pnl_sign = '+' if pos['current_pnl_pct'] >= 0 else ''
            pnl_dollars = (pos['current_price'] - pos['entry_price']) * pos['shares']
            print(f"{ticker:<8} ${pos['entry_price']:>8.2f} ${pos['current_price']:>8.2f} "
                  f"{pnl_sign}${pnl_dollars:>7.0f} ({pnl_sign}{pos['current_pnl_pct']:.1f}%) "
                  f"{pos['days_held']:>4}")
    else:
        print("No open positions")

    # Activity
    activity_header = "TODAY'S ACTIVITY"
    print(f"\n{activity_header:=^50}")
    if summary['entries']:
        print("Entries:")
        for entry in summary['entries']:
            print(f"  + {entry['ticker']} @ ${entry['price']:.2f} "
                  f"(signal {entry['signal_strength']:.0f}, {entry['shares']} shares)")
    else:
        print("Entries: None")

    if summary['exits']:
        print("Exits:")
        for exit_trade in summary['exits']:
            pnl_sign = '+' if exit_trade['pnl_pct'] >= 0 else ''
            print(f"  - {exit_trade['ticker']} @ ${exit_trade['exit_price']:.2f} "
                  f"({pnl_sign}{exit_trade['pnl_pct']:.1f}%) - {exit_trade['exit_reason']}")
    else:
        print("Exits: None")

    # Performance
    metrics = summary.get('metrics', {})
    if metrics.get('total_trades', 0) > 0:
        print(f"\n{'PERFORMANCE':=^50}")
        print(f"Total Trades: {metrics['total_trades']}  |  "
              f"Win Rate: {metrics['win_rate']:.1f}%  |  "
              f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%  |  "
              f"Profit Factor: {metrics['profit_factor']:.2f}  |  "
              f"Avg Hold: {metrics['avg_hold_days']:.1f}d")

    print("=" * 60)

    # Send email
    if not args.no_email:
        print("\nSending email...")
        if wallet.send_daily_email(summary):
            print("Email sent successfully!")
        else:
            print("Email failed - check configuration")
    else:
        print("\nEmail skipped (--no-email flag)")

    print("\nDone!")


if __name__ == '__main__':
    main()
