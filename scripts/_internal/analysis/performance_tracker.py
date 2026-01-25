"""
Performance Tracking & Attribution System
=========================================

Tracks investment decisions and actual outcomes to validate the system.

Features:
1. Investment Decision Logging (buy/sell/hold with rationale)
2. Performance Tracking (actual returns vs expected)
3. Hit Rate Analysis (% of recommendations that worked)
4. Attribution Analysis (which layers were most predictive)
5. Stop Loss / Target Tracking
6. Confidence Calibration (are 98% confidence recs actually right 98% of the time?)

Usage:
    python performance_tracker.py --log               # Log a new investment decision
    python performance_tracker.py --update            # Update positions with current prices
    python performance_tracker.py --report            # Generate performance report
    python performance_tracker.py --attribution       # Analyze which layers predict best
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os


class PerformanceTracker:
    """Tracks investment decisions and validates system performance."""

    def __init__(self):
        self.data_dir = 'data'
        self.results_dir = 'results'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.decisions_file = os.path.join(self.data_dir, 'investment_decisions.json')
        self.performance_file = os.path.join(self.data_dir, 'performance_history.json')

        # Load existing data
        self.decisions = self.load_decisions()
        self.performance = self.load_performance()

    def load_decisions(self) -> List[Dict]:
        """Load investment decision history."""
        if os.path.exists(self.decisions_file):
            try:
                with open(self.decisions_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def load_performance(self) -> Dict:
        """Load performance tracking history."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_decisions(self):
        """Save investment decisions to file."""
        with open(self.decisions_file, 'w') as f:
            json.dump(self.decisions, f, indent=2, default=str)

    def save_performance(self):
        """Save performance data to file."""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance, f, indent=2, default=str)

    def log_decision(self,
                     ticker: str,
                     action: str,  # BUY, SELL, HOLD, TRIM
                     price: float,
                     confidence: int,
                     rationale: str,
                     validation_scores: Dict = None,
                     expected_return: float = None,
                     stop_loss: float = None,
                     targets: List[float] = None) -> Dict:
        """Log a new investment decision."""

        decision = {
            'id': len(self.decisions) + 1,
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'action': action,
            'entry_price': price,
            'confidence': confidence,
            'rationale': rationale,
            'expected_return_pct': expected_return,
            'stop_loss': stop_loss,
            'targets': targets or [],
            'validation_scores': validation_scores or {},
            'status': 'ACTIVE',  # ACTIVE, CLOSED, STOPPED_OUT
            'outcome': None,  # Will be filled when closed
            'actual_return_pct': None,
            'exit_price': None,
            'exit_date': None,
        }

        self.decisions.append(decision)
        self.save_decisions()

        print(f"\n[+] Decision logged: {action} {ticker} at ${price}")
        print(f"  Confidence: {confidence}%")
        print(f"  Expected Return: {expected_return:+.1f}%" if expected_return else "")
        print(f"  Decision ID: {decision['id']}")

        return decision

    def update_position(self, decision_id: int, current_price: float = None,
                       exit_price: float = None, status: str = None):
        """Update a position with current price or close it."""

        for decision in self.decisions:
            if decision['id'] == decision_id:
                if exit_price:
                    # Close the position
                    decision['exit_price'] = exit_price
                    decision['exit_date'] = datetime.now().isoformat()
                    decision['status'] = status or 'CLOSED'

                    # Calculate actual return
                    if decision['action'] in ['BUY', 'HOLD']:
                        actual_return = ((exit_price - decision['entry_price']) /
                                       decision['entry_price']) * 100
                    else:  # SELL or TRIM
                        actual_return = ((decision['entry_price'] - exit_price) /
                                       decision['entry_price']) * 100

                    decision['actual_return_pct'] = actual_return

                    # Determine outcome
                    expected = decision.get('expected_return_pct', 0)
                    if actual_return >= expected * 0.8:  # Within 20% of expected
                        decision['outcome'] = 'SUCCESS'
                    elif actual_return >= 0:
                        decision['outcome'] = 'PARTIAL_SUCCESS'
                    else:
                        decision['outcome'] = 'LOSS'

                    print(f"\n[+] Position closed: {decision['ticker']}")
                    print(f"  Entry: ${decision['entry_price']:.2f}")
                    print(f"  Exit: ${exit_price:.2f}")
                    print(f"  Return: {actual_return:+.2f}%")
                    print(f"  Expected: {expected:+.1f}%")
                    print(f"  Outcome: {decision['outcome']}")

                elif current_price:
                    # Update with current price (for tracking)
                    decision['current_price'] = current_price
                    decision['current_return_pct'] = ((current_price - decision['entry_price']) /
                                                     decision['entry_price']) * 100
                    decision['last_updated'] = datetime.now().isoformat()

                self.save_decisions()
                return decision

        print(f"Decision ID {decision_id} not found")
        return None

    def update_all_active_positions(self):
        """Update all active positions with current prices."""
        print("\nUpdating all active positions...")

        updated = 0
        for decision in self.decisions:
            if decision['status'] == 'ACTIVE':
                try:
                    ticker = decision['ticker']
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')

                    if current_price:
                        decision['current_price'] = current_price
                        decision['current_return_pct'] = ((current_price - decision['entry_price']) /
                                                         decision['entry_price']) * 100
                        decision['last_updated'] = datetime.now().isoformat()

                        # Check stop loss
                        if decision.get('stop_loss') and current_price <= decision['stop_loss']:
                            print(f"\n[!] STOP LOSS HIT: {ticker} at ${current_price:.2f} (stop: ${decision['stop_loss']:.2f})")

                        # Check targets
                        if decision.get('targets'):
                            for i, target in enumerate(decision['targets'], 1):
                                if current_price >= target:
                                    print(f"\n[+] TARGET {i} HIT: {ticker} at ${current_price:.2f} (target: ${target:.2f})")

                        updated += 1

                except Exception as e:
                    print(f"Error updating {decision['ticker']}: {e}")

        self.save_decisions()
        print(f"\n[+] Updated {updated} active positions")

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""

        closed_positions = [d for d in self.decisions if d['status'] in ['CLOSED', 'STOPPED_OUT']]
        active_positions = [d for d in self.decisions if d['status'] == 'ACTIVE']

        if not closed_positions:
            print("\nNo closed positions yet to analyze.")
            return {
                'total_decisions': len(self.decisions),
                'active_positions': len(active_positions),
                'closed_positions': 0,
            }

        # Calculate metrics
        total_closed = len(closed_positions)
        successes = [d for d in closed_positions if d['outcome'] == 'SUCCESS']
        partial_successes = [d for d in closed_positions if d['outcome'] == 'PARTIAL_SUCCESS']
        losses = [d for d in closed_positions if d['outcome'] == 'LOSS']

        hit_rate = (len(successes) / total_closed * 100) if total_closed > 0 else 0
        success_rate = ((len(successes) + len(partial_successes)) / total_closed * 100) if total_closed > 0 else 0

        # Average returns
        actual_returns = [d['actual_return_pct'] for d in closed_positions if d['actual_return_pct'] is not None]
        expected_returns = [d['expected_return_pct'] for d in closed_positions if d.get('expected_return_pct') is not None]

        avg_actual = sum(actual_returns) / len(actual_returns) if actual_returns else 0
        avg_expected = sum(expected_returns) / len(expected_returns) if expected_returns else 0

        # Best and worst
        best_trade = max(closed_positions, key=lambda x: x['actual_return_pct'] or 0) if closed_positions else None
        worst_trade = min(closed_positions, key=lambda x: x['actual_return_pct'] or 0) if closed_positions else None

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_decisions': len(self.decisions),
            'active_positions': len(active_positions),
            'closed_positions': total_closed,
            'hit_rate_pct': round(hit_rate, 1),
            'success_rate_pct': round(success_rate, 1),
            'avg_actual_return_pct': round(avg_actual, 2),
            'avg_expected_return_pct': round(avg_expected, 2),
            'expectation_accuracy_pct': round((avg_actual / avg_expected * 100) if avg_expected != 0 else 0, 1),
            'successes': len(successes),
            'partial_successes': len(partial_successes),
            'losses': len(losses),
            'best_trade': {
                'ticker': best_trade['ticker'],
                'return': best_trade['actual_return_pct'],
                'entry': best_trade['entry_price'],
                'exit': best_trade['exit_price'],
            } if best_trade else None,
            'worst_trade': {
                'ticker': worst_trade['ticker'],
                'return': worst_trade['actual_return_pct'],
                'entry': worst_trade['entry_price'],
                'exit': worst_trade['exit_price'],
            } if worst_trade else None,
        }

        return report

    def display_performance_report(self):
        """Display performance report in formatted output."""
        report = self.generate_performance_report()

        print("\n" + "="*80)
        print("PERFORMANCE TRACKING REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        print(f"\nOVERVIEW:")
        print(f"  Total Decisions Logged: {report['total_decisions']}")
        print(f"  Active Positions: {report['active_positions']}")
        print(f"  Closed Positions: {report['closed_positions']}")

        if report['closed_positions'] > 0:
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Hit Rate: {report['hit_rate_pct']:.1f}% (met/exceeded expectations)")
            print(f"  Success Rate: {report['success_rate_pct']:.1f}% (positive returns)")
            print(f"  Average Actual Return: {report['avg_actual_return_pct']:+.2f}%")
            print(f"  Average Expected Return: {report['avg_expected_return_pct']:+.2f}%")
            print(f"  Expectation Accuracy: {report['expectation_accuracy_pct']:.1f}%")

            print(f"\nOUTCOME BREAKDOWN:")
            print(f"  Successes: {report['successes']} (met expectations)")
            print(f"  Partial Successes: {report['partial_successes']} (positive but below target)")
            print(f"  Losses: {report['losses']} (negative returns)")

            if report['best_trade']:
                print(f"\nBEST TRADE:")
                bt = report['best_trade']
                print(f"  {bt['ticker']}: ${bt['entry']:.2f} → ${bt['exit']:.2f} ({bt['return']:+.2f}%)")

            if report['worst_trade']:
                print(f"\nWORST TRADE:")
                wt = report['worst_trade']
                print(f"  {wt['ticker']}: ${wt['entry']:.2f} → ${wt['exit']:.2f} ({wt['return']:+.2f}%)")

        # Active positions
        active_positions = [d for d in self.decisions if d['status'] == 'ACTIVE']
        if active_positions:
            print(f"\nACTIVE POSITIONS ({len(active_positions)}):")
            print(f"{'Ticker':<8} {'Entry':<10} {'Current':<10} {'Return':<12} {'vs Expected':<15} {'Days Held':<10}")
            print("-" * 80)

            for pos in active_positions:
                current = pos.get('current_price', pos['entry_price'])
                current_return = pos.get('current_return_pct', 0)
                expected = pos.get('expected_return_pct', 0)

                entry_date = datetime.fromisoformat(pos['timestamp'])
                days_held = (datetime.now() - entry_date).days

                vs_expected = f"{current_return:.1f}% / {expected:.1f}%" if expected else f"{current_return:.1f}%"

                print(f"{pos['ticker']:<8} ${pos['entry_price']:<9.2f} ${current:<9.2f} {current_return:>+10.2f}% {vs_expected:<15} {days_held:<10}")

        print("\n" + "="*80)

    def analyze_attribution(self) -> Dict:
        """Analyze which validation layers are most predictive."""

        closed_positions = [d for d in self.decisions if d['status'] in ['CLOSED', 'STOPPED_OUT']]

        if not closed_positions:
            print("\nInsufficient data for attribution analysis (need closed positions).")
            return {}

        # Group by confidence level
        confidence_buckets = {
            '95-100%': [],
            '90-94%': [],
            '80-89%': [],
            '70-79%': [],
            '<70%': [],
        }

        for pos in closed_positions:
            conf = pos.get('confidence', 0)
            ret = pos.get('actual_return_pct', 0)

            if conf >= 95:
                confidence_buckets['95-100%'].append(ret)
            elif conf >= 90:
                confidence_buckets['90-94%'].append(ret)
            elif conf >= 80:
                confidence_buckets['80-89%'].append(ret)
            elif conf >= 70:
                confidence_buckets['70-79%'].append(ret)
            else:
                confidence_buckets['<70%'].append(ret)

        print("\n" + "="*80)
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print("="*80)

        for bucket, returns in confidence_buckets.items():
            if returns:
                avg_return = sum(returns) / len(returns)
                win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                print(f"\n{bucket} Confidence ({len(returns)} trades):")
                print(f"  Average Return: {avg_return:+.2f}%")
                print(f"  Win Rate: {win_rate:.1f}%")

        print("\n" + "="*80)

        return confidence_buckets


def main():
    """Main function with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Proteus Performance Tracking System')
    parser.add_argument('--log', action='store_true', help='Log a new investment decision')
    parser.add_argument('--update', action='store_true', help='Update all active positions')
    parser.add_argument('--close', type=int, help='Close a position (by decision ID)')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--attribution', action='store_true', help='Analyze attribution')

    # For logging new decisions
    parser.add_argument('--ticker', type=str, help='Ticker symbol')
    parser.add_argument('--action', type=str, choices=['BUY', 'SELL', 'HOLD', 'TRIM'], help='Action')
    parser.add_argument('--price', type=float, help='Entry/exit price')
    parser.add_argument('--confidence', type=int, help='Confidence level (0-100)')
    parser.add_argument('--expected-return', type=float, help='Expected return %')
    parser.add_argument('--stop-loss', type=float, help='Stop loss price')

    args = parser.parse_args()

    tracker = PerformanceTracker()

    if args.log:
        # Log new decision
        if not all([args.ticker, args.action, args.price, args.confidence]):
            print("Error: --log requires --ticker, --action, --price, --confidence")
            return

        tracker.log_decision(
            ticker=args.ticker,
            action=args.action,
            price=args.price,
            confidence=args.confidence,
            rationale=f"Logged via CLI on {datetime.now().strftime('%Y-%m-%d')}",
            expected_return=args.expected_return,
            stop_loss=args.stop_loss,
        )

    elif args.update:
        tracker.update_all_active_positions()

    elif args.close:
        if not args.price:
            print("Error: --close requires --price (exit price)")
            return
        tracker.update_position(args.close, exit_price=args.price)

    elif args.report:
        tracker.display_performance_report()

    elif args.attribution:
        tracker.analyze_attribution()

    else:
        # Default: show current status
        print("\nProteus Performance Tracker")
        print("="*80)
        print(f"Total Decisions: {len(tracker.decisions)}")
        print(f"Active Positions: {len([d for d in tracker.decisions if d['status'] == 'ACTIVE'])}")
        print(f"Closed Positions: {len([d for d in tracker.decisions if d['status'] in ['CLOSED', 'STOPPED_OUT']])}")
        print("\nUse --help for available commands")


if __name__ == "__main__":
    main()
