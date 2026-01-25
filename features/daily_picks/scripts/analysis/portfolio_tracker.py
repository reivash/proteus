#!/usr/bin/env python3
"""
Portfolio Tracking Tool
Tracks AI/ML IPO portfolio positions with rebalancing alerts and performance metrics.
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import yfinance as yf


class PortfolioTracker:
    """Track portfolio positions and generate rebalancing alerts."""

    REBALANCING_RULES = {
        'mandatory_trim': 0.40,  # Rule 1: Trim if position > 40%
        'urgent_trim': 0.50,     # Rule 2: Urgent trim if > 50%
        'reversion_threshold': 0.05,  # Rule 4: Exit or add if < 5%
        'drift_threshold': 0.05  # Threshold rebalancing: act if drift > 5%
    }

    RISK_SCORES = {
        'PLTR': 6.5,
        'ALAB': 7.5,
        'CRWV': 8.0,
        'ARM': 6.0,
        'TEM': 7.0,
        'IONQ': 8.5,
        'RBRK': 7.0,
        'SNOW': 6.5,
        'DDOG': 5.5,
        'CRWD': 5.0,
        'ZS': 5.5
    }

    def __init__(self):
        self.score_history = self._load_score_history()

    def _load_score_history(self) -> Dict:
        """Load latest scores from history."""
        history_path = Path(__file__).parent / 'data' / 'score_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return {}

    def create_tracking_template(self,
                                 strategy: str = 'balanced',
                                 initial_budget: float = 100000) -> Dict:
        """
        Create portfolio tracking template based on strategy.

        Args:
            strategy: 'conservative', 'balanced', or 'aggressive'
            initial_budget: Initial investment amount

        Returns:
            Dictionary with portfolio allocation
        """
        # Define target allocations by strategy
        allocations = self._get_strategy_allocations(strategy)

        # Build portfolio template
        portfolio = {
            'metadata': {
                'strategy': strategy.upper(),
                'initial_budget': initial_budget,
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'last_rebalance': datetime.now().strftime('%Y-%m-%d'),
                'next_rebalance': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
            },
            'positions': [],
            'cash': 0.0,
            'total_value': initial_budget
        }

        # Get current prices
        for ticker, target_pct in allocations.items():
            target_value = initial_budget * target_pct

            # Fetch current price using yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                company_name = info.get('longName', ticker)

                if current_price > 0:
                    shares = int(target_value / current_price)
                    actual_value = shares * current_price

                    # Get score and risk
                    score = self._get_latest_score(ticker)
                    risk_score = self.RISK_SCORES.get(ticker, 7.0)

                    position = {
                        'ticker': ticker,
                        'company_name': company_name,
                        'target_allocation_pct': target_pct * 100,
                        'target_value': target_value,
                        'purchase_price': current_price,
                        'current_price': current_price,
                        'shares': shares,
                        'cost_basis': actual_value,
                        'current_value': actual_value,
                        'unrealized_gain_loss': 0.0,
                        'unrealized_gain_loss_pct': 0.0,
                        'current_allocation_pct': (actual_value / initial_budget) * 100,
                        'drift_pct': 0.0,
                        'advanced_yc_score': score,
                        'risk_score': risk_score,
                        'days_held': 0,
                        'rebalancing_action': 'HOLD',
                        'notes': ''
                    }

                    portfolio['positions'].append(position)
                else:
                    print(f"Warning: Could not fetch valid price for {ticker}")
            except Exception as e:
                print(f"Warning: Error fetching {ticker}: {e}")

        # Calculate remaining cash
        invested = sum(p['cost_basis'] for p in portfolio['positions'])
        portfolio['cash'] = initial_budget - invested

        return portfolio

    def _get_strategy_allocations(self, strategy: str) -> Dict[str, float]:
        """Get target allocations by strategy."""
        if strategy.lower() == 'conservative':
            return {
                'PLTR': 0.38,
                'ALAB': 0.33,
                'CRWV': 0.29
            }
        elif strategy.lower() == 'balanced':
            return {
                'PLTR': 0.25,
                'ALAB': 0.20,
                'CRWV': 0.15,
                'ARM': 0.15,
                'TEM': 0.10,
                'IONQ': 0.15
            }
        elif strategy.lower() == 'aggressive':
            return {
                'PLTR': 0.20,
                'ALAB': 0.15,
                'CRWV': 0.15,
                'ARM': 0.12,
                'TEM': 0.10,
                'IONQ': 0.10,
                'RBRK': 0.08,
                'DDOG': 0.10
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def update_portfolio(self, portfolio: Dict) -> Dict:
        """
        Update portfolio with current prices and calculate metrics.

        Args:
            portfolio: Portfolio dictionary from create_tracking_template

        Returns:
            Updated portfolio with current values
        """
        total_current_value = 0.0

        for position in portfolio['positions']:
            ticker = position['ticker']

            # Fetch current price using yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                if current_price > 0:
                    position['current_price'] = current_price
                    position['current_value'] = position['shares'] * position['current_price']
                    position['unrealized_gain_loss'] = position['current_value'] - position['cost_basis']
                    position['unrealized_gain_loss_pct'] = (
                        (position['unrealized_gain_loss'] / position['cost_basis']) * 100
                        if position['cost_basis'] > 0 else 0.0
                    )

                    total_current_value += position['current_value']
                else:
                    total_current_value += position['current_value']
            except Exception as e:
                print(f"Warning: Error updating {ticker}: {e}")
                total_current_value += position['current_value']

        # Update total portfolio value
        portfolio['total_value'] = total_current_value + portfolio['cash']

        # Calculate current allocations and drift
        for position in portfolio['positions']:
            position['current_allocation_pct'] = (
                position['current_value'] / portfolio['total_value']
            ) * 100
            position['drift_pct'] = (
                position['current_allocation_pct'] - position['target_allocation_pct']
            )

            # Determine rebalancing action
            position['rebalancing_action'] = self._get_rebalancing_action(position)

        # Update latest scores
        for position in portfolio['positions']:
            position['advanced_yc_score'] = self._get_latest_score(position['ticker'])

        return portfolio

    def _get_rebalancing_action(self, position: Dict) -> str:
        """Determine rebalancing action based on rules."""
        current_pct = position['current_allocation_pct'] / 100
        target_pct = position['target_allocation_pct'] / 100
        drift = abs(position['drift_pct'] / 100)

        # Rule 2: Urgent trim (>50%)
        if current_pct >= self.REBALANCING_RULES['urgent_trim']:
            return 'URGENT TRIM'

        # Rule 1: Mandatory trim (>40%)
        if current_pct >= self.REBALANCING_RULES['mandatory_trim']:
            return 'TRIM'

        # Rule 4: Reversion threshold (<5% of target)
        if current_pct < self.REBALANCING_RULES['reversion_threshold']:
            return 'ADD OR EXIT'

        # Threshold rebalancing (>5% drift)
        if drift >= self.REBALANCING_RULES['drift_threshold']:
            if position['drift_pct'] > 0:
                return 'TRIM'
            else:
                return 'ADD'

        return 'HOLD'

    def _get_latest_score(self, ticker: str) -> Optional[float]:
        """Get latest advanced YC score from history."""
        if ticker in self.score_history:
            scores = self.score_history[ticker].get('scores', [])
            if scores:
                return scores[-1].get('advanced_yc_score', 0.0)
        return None

    def generate_report(self, portfolio: Dict) -> str:
        """Generate markdown report of portfolio status."""
        report = []
        report.append("# PORTFOLIO TRACKING REPORT")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Portfolio Summary
        meta = portfolio['metadata']
        report.append("## PORTFOLIO SUMMARY\n")
        report.append(f"**Strategy:** {meta['strategy']}")
        report.append(f"**Initial Budget:** ${meta['initial_budget']:,.2f}")
        report.append(f"**Current Value:** ${portfolio['total_value']:,.2f}")

        total_gain_loss = portfolio['total_value'] - meta['initial_budget']
        total_return_pct = (total_gain_loss / meta['initial_budget']) * 100
        report.append(f"**Total Return:** ${total_gain_loss:,.2f} ({total_return_pct:+.2f}%)")
        report.append(f"**Cash:** ${portfolio['cash']:,.2f}")
        report.append(f"**Last Rebalance:** {meta['last_rebalance']}")
        report.append(f"**Next Rebalance:** {meta['next_rebalance']}\n")

        # Positions requiring action
        action_positions = [p for p in portfolio['positions']
                          if p['rebalancing_action'] != 'HOLD']

        if action_positions:
            report.append("## REBALANCING ALERTS\n")
            for pos in action_positions:
                report.append(f"### {pos['ticker']} - {pos['rebalancing_action']}")
                report.append(f"- Current Allocation: {pos['current_allocation_pct']:.1f}%")
                report.append(f"- Target Allocation: {pos['target_allocation_pct']:.1f}%")
                report.append(f"- Drift: {pos['drift_pct']:+.1f}%")
                report.append(f"- Action: {self._get_action_description(pos)}\n")
        else:
            report.append("## REBALANCING ALERTS\n")
            report.append("*No rebalancing required at this time.*\n")

        # Holdings Table
        report.append("## CURRENT HOLDINGS\n")
        report.append("| Ticker | Shares | Cost Basis | Current Value | Gain/Loss | Return % | Allocation | Target | Drift | Score | Risk | Action |")
        report.append("|--------|--------|------------|---------------|-----------|----------|------------|--------|-------|-------|------|--------|")

        for pos in sorted(portfolio['positions'],
                         key=lambda x: x['current_value'], reverse=True):
            report.append(
                f"| {pos['ticker']} | "
                f"{pos['shares']:,} | "
                f"${pos['cost_basis']:,.0f} | "
                f"${pos['current_value']:,.0f} | "
                f"${pos['unrealized_gain_loss']:+,.0f} | "
                f"{pos['unrealized_gain_loss_pct']:+.1f}% | "
                f"{pos['current_allocation_pct']:.1f}% | "
                f"{pos['target_allocation_pct']:.1f}% | "
                f"{pos['drift_pct']:+.1f}% | "
                f"{pos['advanced_yc_score']:.1f} | "
                f"{pos['risk_score']:.1f} | "
                f"{pos['rebalancing_action']} |"
            )

        # Performance Metrics
        report.append("\n## PERFORMANCE METRICS\n")

        # Best/worst performers
        best_performer = max(portfolio['positions'],
                            key=lambda x: x['unrealized_gain_loss_pct'])
        worst_performer = min(portfolio['positions'],
                             key=lambda x: x['unrealized_gain_loss_pct'])

        report.append(f"**Best Performer:** {best_performer['ticker']} "
                     f"({best_performer['unrealized_gain_loss_pct']:+.1f}%)")
        report.append(f"**Worst Performer:** {worst_performer['ticker']} "
                     f"({worst_performer['unrealized_gain_loss_pct']:+.1f}%)")

        # Average score
        avg_score = sum(p['advanced_yc_score'] for p in portfolio['positions']) / len(portfolio['positions'])
        report.append(f"**Average Score:** {avg_score:.1f}/100")

        # Weighted risk
        weighted_risk = sum(
            (p['current_value'] / portfolio['total_value']) * p['risk_score']
            for p in portfolio['positions']
        )
        report.append(f"**Weighted Risk:** {weighted_risk:.1f}/10\n")

        # Exit Trigger Checks
        report.append("## EXIT TRIGGER CHECKS\n")

        exit_warnings = []
        for pos in portfolio['positions']:
            score = pos['advanced_yc_score']
            # Check for tier downgrades (would need historical comparison)
            # For now, check if score is in warning zones
            if score < 35:
                exit_warnings.append(
                    f"- **{pos['ticker']}**: Score {score:.1f} in WEAK tier - Consider exit"
                )
            elif score < 50:
                exit_warnings.append(
                    f"- **{pos['ticker']}**: Score {score:.1f} in MODERATE tier - Monitor closely"
                )

        if exit_warnings:
            report.extend(exit_warnings)
        else:
            report.append("*No exit triggers detected.*")

        return "\n".join(report)

    def _get_action_description(self, position: Dict) -> str:
        """Get detailed action description."""
        action = position['rebalancing_action']
        current_pct = position['current_allocation_pct']
        target_pct = position['target_allocation_pct']
        current_value = position['current_value']

        if action == 'URGENT TRIM':
            target_value = (position['current_value'] / current_pct) * 40  # Trim to 40%
            sell_amount = current_value - target_value
            return f"URGENT: Sell ${sell_amount:,.0f} to reduce to 40%"

        elif action == 'TRIM':
            if current_pct >= 40:
                target_value = (position['current_value'] / current_pct) * 35  # Trim to 35%
            else:
                target_value = (position['current_value'] / current_pct) * target_pct
            sell_amount = current_value - target_value
            return f"Sell ${sell_amount:,.0f} to reach target"

        elif action == 'ADD':
            target_value = (position['current_value'] / current_pct) * target_pct
            buy_amount = target_value - current_value
            return f"Buy ${buy_amount:,.0f} to reach target"

        elif action == 'ADD OR EXIT':
            return f"Position < 5% - Either add ${current_value:,.0f} more or exit completely"

        return "No action required"

    def save_portfolio(self, portfolio: Dict, filename: str = None):
        """Save portfolio to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'portfolio_snapshot_{timestamp}.json'

        filepath = Path(__file__).parent / 'data' / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(portfolio, f, indent=2)

        print(f"Portfolio saved to: {filepath}")
        return filepath

    def export_to_csv(self, portfolio: Dict, filename: str = None):
        """Export portfolio to CSV for Excel/Google Sheets."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'portfolio_tracking_{timestamp}.csv'

        filepath = Path(__file__).parent / 'results' / filename
        filepath.parent.mkdir(exist_ok=True)

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header rows
            writer.writerow(['PORTFOLIO TRACKING TEMPLATE'])
            writer.writerow([f"Strategy: {portfolio['metadata']['strategy']}"])
            writer.writerow([f"Initial Budget: ${portfolio['metadata']['initial_budget']:,.2f}"])
            writer.writerow([f"Current Value: ${portfolio['total_value']:,.2f}"])
            writer.writerow([f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Column headers
            writer.writerow([
                'Ticker', 'Company', 'Shares', 'Purchase Price', 'Current Price',
                'Cost Basis', 'Current Value', 'Gain/Loss $', 'Gain/Loss %',
                'Current Alloc %', 'Target Alloc %', 'Drift %',
                'Score', 'Risk', 'Days Held', 'Action', 'Notes'
            ])

            # Position data
            for pos in portfolio['positions']:
                writer.writerow([
                    pos['ticker'],
                    pos['company_name'],
                    pos['shares'],
                    f"${pos['purchase_price']:.2f}",
                    f"${pos['current_price']:.2f}",
                    f"${pos['cost_basis']:.2f}",
                    f"${pos['current_value']:.2f}",
                    f"${pos['unrealized_gain_loss']:.2f}",
                    f"{pos['unrealized_gain_loss_pct']:.2f}%",
                    f"{pos['current_allocation_pct']:.2f}%",
                    f"{pos['target_allocation_pct']:.2f}%",
                    f"{pos['drift_pct']:.2f}%",
                    pos['advanced_yc_score'],
                    pos['risk_score'],
                    pos['days_held'],
                    pos['rebalancing_action'],
                    pos['notes']
                ])

            # Summary rows
            writer.writerow([])
            writer.writerow(['SUMMARY'])
            writer.writerow(['Total Invested', f"${sum(p['cost_basis'] for p in portfolio['positions']):.2f}"])
            writer.writerow(['Current Value', f"${sum(p['current_value'] for p in portfolio['positions']):.2f}"])
            writer.writerow(['Cash', f"${portfolio['cash']:.2f}"])
            writer.writerow(['Total Portfolio Value', f"${portfolio['total_value']:.2f}"])

        print(f"CSV exported to: {filepath}")
        return filepath


def main():
    """Demo portfolio tracking."""
    print("=" * 80)
    print("PORTFOLIO TRACKING TOOL")
    print("=" * 80)
    print()

    tracker = PortfolioTracker()

    # Example: Create and track Balanced portfolio
    print("Creating BALANCED portfolio with $100,000 initial budget...")
    print()

    portfolio = tracker.create_tracking_template(
        strategy='balanced',
        initial_budget=100000
    )

    # Simulate some time passing and update prices
    print("Updating portfolio with current prices...")
    print()
    portfolio = tracker.update_portfolio(portfolio)

    # Generate report
    report = tracker.generate_report(portfolio)
    print(report)
    print()

    # Save portfolio snapshot
    json_path = tracker.save_portfolio(portfolio)
    print()

    # Export to CSV
    csv_path = tracker.export_to_csv(portfolio)
    print()

    print("=" * 80)
    print("PORTFOLIO TRACKING COMPLETE")
    print("=" * 80)
    print()
    print("FILES CREATED:")
    print(f"1. JSON Snapshot: {json_path}")
    print(f"2. CSV Template: {csv_path}")
    print()
    print("NEXT STEPS:")
    print("1. Open CSV in Excel/Google Sheets")
    print("2. Update 'Current Price' column manually or with formulas")
    print("3. Review 'Action' column for rebalancing needs")
    print("4. Run quarterly for systematic rebalancing")
    print()
    print("AUTOMATION:")
    print("- Run: python portfolio_tracker.py")
    print("- Schedule quarterly with Task Scheduler")
    print("- Email reports automatically")


if __name__ == '__main__':
    main()
