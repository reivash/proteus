"""
Trade Retrospective Analyzer

Analyzes completed trades to learn patterns:
1. What conditions led to winning vs losing trades?
2. Were there missed signals that would have been profitable?
3. How did conviction scores correlate with actual returns?
4. What exit timing would have been optimal?

Run weekly to continuously improve the prediction system.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import yfinance as yf


@dataclass
class TradeAnalysis:
    """Analysis of a single trade."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int

    # Entry conditions
    entry_z_score: float
    entry_rsi: float
    entry_signal_strength: float
    entry_conviction: Optional[float]

    # What actually happened
    max_favorable_excursion: float  # Best unrealized profit
    max_adverse_excursion: float    # Worst unrealized loss
    optimal_exit_return: float      # Return if exited at best point
    exit_reason: str

    # Learning
    was_profitable: bool
    could_have_been_better: bool
    key_lesson: str


@dataclass
class RetrospectiveReport:
    """Complete retrospective analysis report."""
    report_date: str
    analysis_period_days: int
    total_trades_analyzed: int

    # Performance breakdown
    win_rate: float
    avg_winner: float
    avg_loser: float
    profit_factor: float

    # Entry analysis
    avg_z_score_winners: float
    avg_z_score_losers: float
    avg_signal_strength_winners: float
    avg_signal_strength_losers: float

    # Conviction analysis
    high_conviction_win_rate: float
    medium_conviction_win_rate: float
    low_conviction_win_rate: float

    # Exit timing analysis
    avg_mfe_winners: float  # How much more could winners have made
    avg_mae_losers: float   # How much worse losers got before exit
    pct_exited_too_early: float
    pct_held_too_long: float

    # Patterns discovered
    best_entry_conditions: Dict
    worst_entry_conditions: Dict
    optimal_hold_period: int

    # Recommendations
    recommendations: List[str]
    parameter_suggestions: Dict


class TradeRetrospective:
    """
    Analyzes historical trades to improve future predictions.
    """

    def __init__(self,
                 trades_file: str = "data/paper_trading/trade_history.json",
                 output_dir: str = "data/trade_retrospectives"):
        self.trades_file = trades_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_trades(self, days_back: int = 90) -> List[Dict]:
        """Load completed trades from history."""
        if not os.path.exists(self.trades_file):
            print(f"[WARN] Trade history not found: {self.trades_file}")
            return []

        with open(self.trades_file, 'r') as f:
            all_trades = json.load(f)

        cutoff = datetime.now() - timedelta(days=days_back)
        recent_trades = []

        for trade in all_trades:
            try:
                exit_date = datetime.strptime(trade.get('exit_date', ''), '%Y-%m-%d')
                if exit_date >= cutoff:
                    recent_trades.append(trade)
            except:
                pass

        return recent_trades

    def get_price_path(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """Get daily prices for a ticker between dates."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            return hist['Close']
        except:
            return pd.Series()

    def analyze_trade(self, trade: Dict) -> TradeAnalysis:
        """Perform deep analysis on a single trade."""
        ticker = trade['ticker']
        entry_date = trade.get('entry_date', '')
        exit_date = trade.get('exit_date', '')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)

        # Calculate return
        return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        # Calculate hold days
        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            hold_days = (exit_dt - entry_dt).days
        except:
            hold_days = 0

        # Get price path for MFE/MAE analysis
        prices = self.get_price_path(ticker, entry_date, exit_date)

        if len(prices) > 0:
            returns_from_entry = ((prices / entry_price) - 1) * 100
            mfe = returns_from_entry.max()
            mae = returns_from_entry.min()
            optimal_return = mfe
        else:
            mfe = return_pct
            mae = return_pct
            optimal_return = return_pct

        # Extract entry conditions
        entry_z_score = trade.get('z_score', 0)
        entry_rsi = trade.get('rsi', 50)
        entry_signal_strength = trade.get('signal_strength', 0)
        entry_conviction = trade.get('conviction_score')

        # Determine if profitable
        was_profitable = return_pct > 0

        # Could it have been better?
        could_have_been_better = mfe > return_pct + 0.5  # At least 0.5% better possible

        # Key lesson
        if was_profitable:
            if could_have_been_better:
                lesson = f"Profitable but left {mfe - return_pct:.1f}% on table. Consider later exit."
            else:
                lesson = "Good execution - captured most of the move."
        else:
            if mae < return_pct - 1:
                lesson = f"Loss was smaller than worst point ({mae:.1f}%). Exit timing reasonable."
            else:
                lesson = f"Entry signal may have been weak (Z: {entry_z_score:.2f}, RSI: {entry_rsi:.0f})."

        return TradeAnalysis(
            ticker=ticker,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=round(return_pct, 2),
            hold_days=hold_days,
            entry_z_score=entry_z_score,
            entry_rsi=entry_rsi,
            entry_signal_strength=entry_signal_strength,
            entry_conviction=entry_conviction,
            max_favorable_excursion=round(mfe, 2),
            max_adverse_excursion=round(mae, 2),
            optimal_exit_return=round(optimal_return, 2),
            exit_reason=trade.get('exit_reason', 'unknown'),
            was_profitable=was_profitable,
            could_have_been_better=could_have_been_better,
            key_lesson=lesson
        )

    def run_retrospective(self, days_back: int = 90) -> RetrospectiveReport:
        """Run complete retrospective analysis."""
        print("="*70)
        print("PROTEUS TRADE RETROSPECTIVE")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis Period: Last {days_back} days")

        # Load trades
        print("\n1. Loading trade history...")
        trades = self.load_trades(days_back)
        print(f"   Found {len(trades)} trades")

        if len(trades) == 0:
            print("\n[WARN] No trades to analyze")
            return self._empty_report(days_back)

        # Analyze each trade
        print("\n2. Analyzing individual trades...")
        analyses = []
        for trade in trades:
            analysis = self.analyze_trade(trade)
            analyses.append(analysis)
            print(f"   {analysis.ticker}: {analysis.return_pct:+.1f}% "
                  f"(MFE: {analysis.max_favorable_excursion:+.1f}%, "
                  f"MAE: {analysis.max_adverse_excursion:+.1f}%)")

        # Calculate aggregate stats
        print("\n3. Calculating aggregate statistics...")

        winners = [a for a in analyses if a.was_profitable]
        losers = [a for a in analyses if not a.was_profitable]

        win_rate = len(winners) / len(analyses) * 100 if analyses else 0
        avg_winner = np.mean([w.return_pct for w in winners]) if winners else 0
        avg_loser = np.mean([l.return_pct for l in losers]) if losers else 0

        gross_profit = sum(w.return_pct for w in winners)
        gross_loss = abs(sum(l.return_pct for l in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Entry condition analysis
        avg_z_winners = np.mean([w.entry_z_score for w in winners]) if winners else 0
        avg_z_losers = np.mean([l.entry_z_score for l in losers]) if losers else 0
        avg_str_winners = np.mean([w.entry_signal_strength for w in winners]) if winners else 0
        avg_str_losers = np.mean([l.entry_signal_strength for l in losers]) if losers else 0

        # Conviction analysis
        high_conv = [a for a in analyses if a.entry_conviction and a.entry_conviction >= 70]
        med_conv = [a for a in analyses if a.entry_conviction and 50 <= a.entry_conviction < 70]
        low_conv = [a for a in analyses if a.entry_conviction and a.entry_conviction < 50]

        high_conv_wr = len([t for t in high_conv if t.was_profitable]) / len(high_conv) * 100 if high_conv else 0
        med_conv_wr = len([t for t in med_conv if t.was_profitable]) / len(med_conv) * 100 if med_conv else 0
        low_conv_wr = len([t for t in low_conv if t.was_profitable]) / len(low_conv) * 100 if low_conv else 0

        # Exit timing analysis
        avg_mfe_winners = np.mean([w.max_favorable_excursion - w.return_pct for w in winners]) if winners else 0
        avg_mae_losers = np.mean([l.max_adverse_excursion for l in losers]) if losers else 0

        exited_early = [a for a in analyses if a.could_have_been_better]
        pct_exited_early = len(exited_early) / len(analyses) * 100 if analyses else 0

        held_too_long = [a for a in analyses if a.max_favorable_excursion > a.return_pct + 1]
        pct_held_long = len(held_too_long) / len(analyses) * 100 if analyses else 0

        # Find patterns
        best_conditions = {
            'z_score_range': f"{np.percentile([w.entry_z_score for w in winners], 25):.2f} to {np.percentile([w.entry_z_score for w in winners], 75):.2f}" if winners else "N/A",
            'rsi_range': f"{np.percentile([w.entry_rsi for w in winners], 25):.0f} to {np.percentile([w.entry_rsi for w in winners], 75):.0f}" if winners else "N/A",
            'signal_strength_avg': f"{avg_str_winners:.1f}" if winners else "N/A"
        }

        worst_conditions = {
            'z_score_range': f"{np.percentile([l.entry_z_score for l in losers], 25):.2f} to {np.percentile([l.entry_z_score for l in losers], 75):.2f}" if losers else "N/A",
            'rsi_range': f"{np.percentile([l.entry_rsi for l in losers], 25):.0f} to {np.percentile([l.entry_rsi for l in losers], 75):.0f}" if losers else "N/A",
            'signal_strength_avg': f"{avg_str_losers:.1f}" if losers else "N/A"
        }

        optimal_hold = int(np.mean([a.hold_days for a in winners])) if winners else 2

        # Generate recommendations
        recommendations = []

        if avg_str_winners > avg_str_losers + 5:
            recommendations.append(f"Raise min_signal_strength to {int(avg_str_losers + (avg_str_winners - avg_str_losers)/2)} to filter weak signals")

        if high_conv_wr > med_conv_wr + 10:
            recommendations.append("High conviction trades outperform - increase position size on HIGH tier")

        if pct_exited_early > 50:
            recommendations.append(f"Consider extending profit target - {pct_exited_early:.0f}% of trades had more upside")

        if avg_z_winners < avg_z_losers:
            recommendations.append(f"Winners had lower Z-scores ({avg_z_winners:.2f}) - extreme readings may be less reliable")

        if profit_factor < 1.5:
            recommendations.append("Profit factor low - consider tighter entry criteria or better exit timing")

        # Parameter suggestions
        param_suggestions = {}

        if avg_str_winners > 55:
            param_suggestions['min_signal_strength'] = int(avg_str_winners - 10)

        if optimal_hold != 2:
            param_suggestions['max_hold_days'] = optimal_hold

        if avg_mfe_winners > 1:
            param_suggestions['profit_target'] = round(avg_winner + avg_mfe_winners * 0.5, 1)

        # Build report
        report = RetrospectiveReport(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            analysis_period_days=days_back,
            total_trades_analyzed=len(analyses),
            win_rate=round(win_rate, 1),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            profit_factor=round(profit_factor, 2),
            avg_z_score_winners=round(avg_z_winners, 2),
            avg_z_score_losers=round(avg_z_losers, 2),
            avg_signal_strength_winners=round(avg_str_winners, 1),
            avg_signal_strength_losers=round(avg_str_losers, 1),
            high_conviction_win_rate=round(high_conv_wr, 1),
            medium_conviction_win_rate=round(med_conv_wr, 1),
            low_conviction_win_rate=round(low_conv_wr, 1),
            avg_mfe_winners=round(avg_mfe_winners, 2),
            avg_mae_losers=round(avg_mae_losers, 2),
            pct_exited_too_early=round(pct_exited_early, 1),
            pct_held_too_long=round(pct_held_long, 1),
            best_entry_conditions=best_conditions,
            worst_entry_conditions=worst_conditions,
            optimal_hold_period=optimal_hold,
            recommendations=recommendations,
            parameter_suggestions=param_suggestions
        )

        # Save report
        self._save_report(report, analyses)

        # Print summary
        self._print_summary(report)

        return report

    def _empty_report(self, days_back: int) -> RetrospectiveReport:
        """Return empty report when no trades available."""
        return RetrospectiveReport(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            analysis_period_days=days_back,
            total_trades_analyzed=0,
            win_rate=0, avg_winner=0, avg_loser=0, profit_factor=0,
            avg_z_score_winners=0, avg_z_score_losers=0,
            avg_signal_strength_winners=0, avg_signal_strength_losers=0,
            high_conviction_win_rate=0, medium_conviction_win_rate=0, low_conviction_win_rate=0,
            avg_mfe_winners=0, avg_mae_losers=0,
            pct_exited_too_early=0, pct_held_too_long=0,
            best_entry_conditions={}, worst_entry_conditions={},
            optimal_hold_period=2,
            recommendations=["Not enough trades to analyze - run system longer"],
            parameter_suggestions={}
        )

    def _save_report(self, report: RetrospectiveReport, analyses: List[TradeAnalysis]):
        """Save report and analyses to files."""
        # Save report
        report_file = os.path.join(
            self.output_dir,
            f"retrospective_{report.report_date}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\n[SAVED] {report_file}")

        # Save detailed analyses
        analyses_file = os.path.join(
            self.output_dir,
            f"trade_analyses_{report.report_date}.json"
        )
        with open(analyses_file, 'w') as f:
            json.dump([asdict(a) for a in analyses], f, indent=2)
        print(f"[SAVED] {analyses_file}")

    def _print_summary(self, report: RetrospectiveReport):
        """Print summary of retrospective."""
        print("\n" + "="*70)
        print("RETROSPECTIVE SUMMARY")
        print("="*70)

        print(f"\nTrades Analyzed: {report.total_trades_analyzed}")
        print(f"Win Rate: {report.win_rate:.1f}%")
        print(f"Avg Winner: {report.avg_winner:+.2f}%")
        print(f"Avg Loser: {report.avg_loser:+.2f}%")
        print(f"Profit Factor: {report.profit_factor:.2f}")

        print("\nEntry Analysis:")
        print(f"  Winners Z-Score: {report.avg_z_score_winners:.2f}")
        print(f"  Losers Z-Score: {report.avg_z_score_losers:.2f}")
        print(f"  Winners Signal Strength: {report.avg_signal_strength_winners:.1f}")
        print(f"  Losers Signal Strength: {report.avg_signal_strength_losers:.1f}")

        print("\nConviction Impact:")
        print(f"  HIGH conviction win rate: {report.high_conviction_win_rate:.1f}%")
        print(f"  MEDIUM conviction win rate: {report.medium_conviction_win_rate:.1f}%")
        print(f"  LOW conviction win rate: {report.low_conviction_win_rate:.1f}%")

        print("\nExit Timing:")
        print(f"  Exited too early: {report.pct_exited_too_early:.0f}% of trades")
        print(f"  Held too long: {report.pct_held_too_long:.0f}% of trades")
        print(f"  Optimal hold period: {report.optimal_hold_period} days")

        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

        if report.parameter_suggestions:
            print("\nSuggested Parameter Changes:")
            for param, value in report.parameter_suggestions.items():
                print(f"  {param}: {value}")


def run_trade_retrospective():
    """Main entry point."""
    retro = TradeRetrospective()
    report = retro.run_retrospective(days_back=90)
    return report


if __name__ == "__main__":
    run_trade_retrospective()
