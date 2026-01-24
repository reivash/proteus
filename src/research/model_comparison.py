"""
Model Comparison Framework - GPU (MLP) vs LSTM V2
=================================================

A/B test both models on the same data to determine which performs better.
This is the critical validation step before deploying LSTM V2.

Expected: LSTM V2 should show +3-5% accuracy improvement over MLP baseline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance not installed")
    sys.exit(1)


@dataclass
class ModelPrediction:
    """Single prediction from a model."""
    ticker: str
    date: str
    prob: float
    expected_return: float
    confidence: float
    signal_strength: float


@dataclass
class BacktestTrade:
    """Single backtest trade result."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int
    signal_strength: float
    model: str
    won: bool


@dataclass
class ComparisonResult:
    """Comparison result between two models."""
    mlp_trades: int
    mlp_win_rate: float
    mlp_avg_return: float
    mlp_sharpe: float
    lstm_trades: int
    lstm_win_rate: float
    lstm_avg_return: float
    lstm_sharpe: float
    improvement_win_rate: float
    improvement_avg_return: float
    improvement_sharpe: float
    winner: str


class ModelComparison:
    """
    A/B test framework for comparing GPU (MLP) and LSTM models.
    """

    PROTEUS_TICKERS = [
        "NVDA", "V", "MA", "AVGO", "AXP", "KLAC", "ORCL", "MRVL", "ABBV", "SYK",
        "EOG", "TXN", "GILD", "INTU", "MSFT", "QCOM", "JPM", "JNJ", "PFE", "WMT",
        "AMAT", "ADI", "NOW", "MLM", "IDXX", "EXR", "ROAD", "INSM", "SCHW", "AIG",
        "USB", "CVS", "LOW", "LMT", "COP", "SLB", "APD", "MS", "PNC", "CRM",
        "ADBE", "TGT", "CAT", "XOM", "MPC", "ECL", "NEE", "HCA", "CMCSA", "TMUS",
        "META", "ETN", "HD", "SHW"
    ]

    def __init__(self, output_dir: str = "data/research"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.mlp_model = None
        self.lstm_model = None

    def load_models(self):
        """Load both models for comparison."""
        print("\n" + "="*70)
        print("LOADING MODELS FOR A/B TEST")
        print("="*70)

        # Load GPU (MLP) model
        try:
            from src.models.gpu_signal_model import GPUSignalModel
            self.mlp_model = GPUSignalModel()
            print("[OK] GPU Signal Model (MLP) loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load MLP model: {e}")
            return False

        # Load LSTM model
        try:
            from src.models.lstm_signal_model import LSTMSignalModel
            self.lstm_model = LSTMSignalModel()
            print("[OK] LSTM Signal Model (V2) loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load LSTM model: {e}")
            return False

        return True

    def fetch_historical_data(self, ticker: str, days_back: int = 365) -> Optional[pd.DataFrame]:
        """Fetch historical data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 60)
            df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'))
            if len(df) < 100:
                return None
            return df
        except Exception as e:
            return None

    def generate_predictions_at_date(self, ticker: str, df: pd.DataFrame,
                                     idx: int) -> Tuple[Optional[dict], Optional[dict]]:
        """Generate predictions from both models at a specific date index."""
        # Get data up to this date
        df_slice = df.iloc[:idx+1].copy()

        if len(df_slice) < 90:  # Need enough history
            return None, None

        mlp_pred = None
        lstm_pred = None

        # MLP prediction
        try:
            mlp_signal = self.mlp_model.predict(ticker, df_slice)
            if mlp_signal:
                mlp_pred = {
                    'prob': mlp_signal.mean_reversion_prob,
                    'expected_return': mlp_signal.expected_return,
                    'confidence': mlp_signal.confidence,
                    'signal_strength': mlp_signal.signal_strength
                }
        except Exception as e:
            pass

        # LSTM prediction
        try:
            lstm_signal = self.lstm_model.predict(ticker, df_slice)
            if lstm_signal:
                lstm_pred = {
                    'prob': lstm_signal.mean_reversion_prob,
                    'expected_return': lstm_signal.expected_return,
                    'confidence': lstm_signal.confidence,
                    'signal_strength': lstm_signal.signal_strength
                }
        except Exception as e:
            pass

        return mlp_pred, lstm_pred

    def backtest_model(self, trades: List[BacktestTrade]) -> Dict:
        """Calculate performance metrics for a list of trades."""
        if not trades:
            return {
                'trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe': 0,
                'profit_factor': 0
            }

        returns = [t.return_pct for t in trades]
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]

        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.01

        # Annualized Sharpe (assuming 3-day avg hold)
        sharpe = (avg_return / std_return) * np.sqrt(252 / 3) if std_return > 0 else 0

        # Profit factor
        gross_profit = sum(t.return_pct for t in wins) if wins else 0
        gross_loss = abs(sum(t.return_pct for t in losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        return {
            'trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100,
            'avg_return': avg_return,
            'total_return': sum(returns),
            'sharpe': sharpe,
            'profit_factor': profit_factor
        }

    def run_comparison(self, days_back: int = 180, signal_threshold: float = 60.0,
                      hold_days: int = 2) -> ComparisonResult:
        """
        Run A/B comparison backtest on both models.

        Args:
            days_back: How many days to backtest
            signal_threshold: Minimum signal strength to trade
            hold_days: Days to hold each position
        """
        print(f"\n" + "="*70)
        print(f"RUNNING A/B COMPARISON BACKTEST")
        print(f"="*70)
        print(f"Period: Last {days_back} days")
        print(f"Signal threshold: {signal_threshold}")
        print(f"Hold period: {hold_days} days")
        print()

        mlp_trades = []
        lstm_trades = []

        for ticker in self.PROTEUS_TICKERS:
            print(f"Processing {ticker}...", end=" ")

            df = self.fetch_historical_data(ticker, days_back + 60)
            if df is None:
                print("SKIP (no data)")
                continue

            # Find start index for backtest period
            start_idx = max(90, len(df) - days_back)

            ticker_mlp_trades = 0
            ticker_lstm_trades = 0

            # Walk through each day
            i = start_idx
            while i < len(df) - hold_days:
                mlp_pred, lstm_pred = self.generate_predictions_at_date(ticker, df, i)

                entry_date = df.index[i].strftime('%Y-%m-%d')
                exit_idx = min(i + hold_days, len(df) - 1)
                exit_date = df.index[exit_idx].strftime('%Y-%m-%d')
                entry_price = df['Close'].iloc[i]
                exit_price = df['Close'].iloc[exit_idx]
                actual_return = (exit_price / entry_price - 1) * 100

                # MLP trade
                if mlp_pred and mlp_pred['signal_strength'] >= signal_threshold:
                    trade = BacktestTrade(
                        ticker=ticker,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=actual_return,
                        hold_days=hold_days,
                        signal_strength=mlp_pred['signal_strength'],
                        model='MLP',
                        won=actual_return > 0
                    )
                    mlp_trades.append(trade)
                    ticker_mlp_trades += 1

                # LSTM trade
                if lstm_pred and lstm_pred['signal_strength'] >= signal_threshold:
                    trade = BacktestTrade(
                        ticker=ticker,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        return_pct=actual_return,
                        hold_days=hold_days,
                        signal_strength=lstm_pred['signal_strength'],
                        model='LSTM',
                        won=actual_return > 0
                    )
                    lstm_trades.append(trade)
                    ticker_lstm_trades += 1

                i += 1

            print(f"MLP: {ticker_mlp_trades}, LSTM: {ticker_lstm_trades}")

        # Calculate metrics
        print("\n" + "-"*70)
        print("CALCULATING METRICS...")

        mlp_metrics = self.backtest_model(mlp_trades)
        lstm_metrics = self.backtest_model(lstm_trades)

        # Determine winner
        mlp_score = (mlp_metrics['win_rate'] * 0.4 +
                    mlp_metrics['avg_return'] * 20 +
                    mlp_metrics['sharpe'] * 0.3)
        lstm_score = (lstm_metrics['win_rate'] * 0.4 +
                     lstm_metrics['avg_return'] * 20 +
                     lstm_metrics['sharpe'] * 0.3)

        winner = "LSTM" if lstm_score > mlp_score else "MLP"

        result = ComparisonResult(
            mlp_trades=mlp_metrics['trades'],
            mlp_win_rate=mlp_metrics['win_rate'],
            mlp_avg_return=mlp_metrics['avg_return'],
            mlp_sharpe=mlp_metrics['sharpe'],
            lstm_trades=lstm_metrics['trades'],
            lstm_win_rate=lstm_metrics['win_rate'],
            lstm_avg_return=lstm_metrics['avg_return'],
            lstm_sharpe=lstm_metrics['sharpe'],
            improvement_win_rate=lstm_metrics['win_rate'] - mlp_metrics['win_rate'],
            improvement_avg_return=lstm_metrics['avg_return'] - mlp_metrics['avg_return'],
            improvement_sharpe=lstm_metrics['sharpe'] - mlp_metrics['sharpe'],
            winner=winner
        )

        return result, mlp_trades, lstm_trades

    def print_results(self, result: ComparisonResult):
        """Print comparison results."""
        print("\n" + "="*70)
        print("A/B COMPARISON RESULTS: MLP vs LSTM V2")
        print("="*70)

        print(f"\n{'Metric':<25} {'MLP (Current)':<18} {'LSTM V2':<18} {'Diff':<12}")
        print("-"*70)
        print(f"{'Total Trades':<25} {result.mlp_trades:<18} {result.lstm_trades:<18}")
        print(f"{'Win Rate':<25} {result.mlp_win_rate:.1f}%{'':<13} {result.lstm_win_rate:.1f}%{'':<13} {result.improvement_win_rate:+.1f}%")
        print(f"{'Avg Return/Trade':<25} {result.mlp_avg_return:.3f}%{'':<12} {result.lstm_avg_return:.3f}%{'':<12} {result.improvement_avg_return:+.3f}%")
        print(f"{'Sharpe Ratio':<25} {result.mlp_sharpe:.2f}{'':<14} {result.lstm_sharpe:.2f}{'':<14} {result.improvement_sharpe:+.2f}")

        print("\n" + "="*70)
        if result.winner == "LSTM":
            print(f"WINNER: LSTM V2 (+{result.improvement_win_rate:.1f}% win rate)")
            print("RECOMMENDATION: Deploy LSTM V2 as primary model")
        else:
            print(f"WINNER: MLP (Current model)")
            print("RECOMMENDATION: Keep MLP, investigate LSTM improvements")
        print("="*70)

    def save_results(self, result: ComparisonResult, mlp_trades: List, lstm_trades: List):
        """Save comparison results to JSON."""
        output = {
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': asdict(result),
            'mlp_trades_sample': [asdict(t) for t in mlp_trades[:100]],
            'lstm_trades_sample': [asdict(t) for t in lstm_trades[:100]]
        }

        output_file = os.path.join(self.output_dir, 'model_comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[SAVED] Results saved to {output_file}")


def run_model_comparison():
    """Main entry point for model comparison."""
    comparison = ModelComparison()

    if not comparison.load_models():
        print("[ERROR] Failed to load models - cannot run comparison")
        return None

    result, mlp_trades, lstm_trades = comparison.run_comparison(
        days_back=180,
        signal_threshold=55.0,
        hold_days=2
    )

    comparison.print_results(result)
    comparison.save_results(result, mlp_trades, lstm_trades)

    return result


if __name__ == "__main__":
    run_model_comparison()
