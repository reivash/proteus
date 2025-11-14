"""
Mean Reversion Strategy - Detect and trade stock overcorrections.

Based on user observation: "When stock jumps down or up suddenly,
it tends to normalize in 1-2 days because the jump made no sense."

Strategy: Buy panic sells, sell panic buys, hold 1-2 days.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MeanReversionDetector:
    """
    Detects overcorrections in stock prices using multiple signals.
    """

    def __init__(
        self,
        z_score_threshold=2.0,
        rsi_oversold=30,
        rsi_overbought=70,
        volume_multiplier=2.0,
        price_drop_threshold=-3.0,
        price_jump_threshold=3.0
    ):
        """
        Initialize mean reversion detector.

        Args:
            z_score_threshold: Std deviations from mean to consider overcorrection
            rsi_oversold: RSI level below which considered oversold
            rsi_overbought: RSI level above which considered overbought
            volume_multiplier: Volume must be X times average to signal anomaly
            price_drop_threshold: % drop to consider for panic sell
            price_jump_threshold: % jump to consider for panic buy
        """
        self.z_score_threshold = z_score_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_multiplier = volume_multiplier
        self.price_drop_threshold = price_drop_threshold
        self.price_jump_threshold = price_jump_threshold

    def calculate_z_score(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Calculate rolling z-score of price."""
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        z_score = (df['Close'] - rolling_mean) / rolling_std
        return z_score

    def calculate_rsi(self, df: pd.DataFrame, window=14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_volume_spike(self, df: pd.DataFrame, window=20) -> pd.Series:
        """Detect unusual volume."""
        avg_volume = df['Volume'].rolling(window=window).mean()
        volume_spike = (df['Volume'] > avg_volume * self.volume_multiplier).astype(int)
        return volume_spike

    def calculate_daily_return(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily percentage return."""
        return df['Close'].pct_change() * 100

    def detect_overcorrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential overcorrections (both oversold and overbought).

        Returns DataFrame with overcorrection signals.
        """
        data = df.copy()

        # Calculate indicators
        data['z_score'] = self.calculate_z_score(data)
        data['rsi'] = self.calculate_rsi(data)
        data['volume_spike'] = self.detect_volume_spike(data)
        data['daily_return'] = self.calculate_daily_return(data)

        # Detect PANIC SELL (Oversold) - BUY opportunity
        data['panic_sell'] = (
            (data['z_score'] < -self.z_score_threshold) &  # Price well below average
            (data['rsi'] < self.rsi_oversold) &             # RSI oversold
            (data['volume_spike'] == 1) &                   # Unusual volume
            (data['daily_return'] < self.price_drop_threshold)  # Big drop
        ).astype(int)

        # Detect PANIC BUY (Overbought) - SELL opportunity
        data['panic_buy'] = (
            (data['z_score'] > self.z_score_threshold) &    # Price well above average
            (data['rsi'] > self.rsi_overbought) &           # RSI overbought
            (data['volume_spike'] == 1) &                   # Unusual volume
            (data['daily_return'] > self.price_jump_threshold)  # Big jump
        ).astype(int)

        # Combined signal
        data['overcorrection'] = data['panic_sell'] + data['panic_buy']

        return data

    def calculate_reversion_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate expected reversion targets.

        For panic sells: Expect +2-3% recovery
        For panic buys: Expect -2-3% correction
        """
        data = df.copy()

        # Target is ~20-day moving average (mean reversion)
        data['ma_20'] = data['Close'].rolling(window=20).mean()

        data['reversion_target'] = np.where(
            data['panic_sell'] == 1,
            data['ma_20'],  # Revert to 20-day average
            np.where(
                data['panic_buy'] == 1,
                data['ma_20'],  # Revert to 20-day average
                data['Close']   # No reversion expected
            )
        )

        data['expected_return'] = (data['reversion_target'] / data['Close'] - 1) * 100

        return data


class MeanReversionBacktester:
    """
    Backtest mean reversion trading strategy.
    """

    def __init__(
        self,
        initial_capital=10000,
        profit_target=2.0,  # % gain to take profit
        stop_loss=-2.0,     # % loss to cut losses
        max_hold_days=2,    # Maximum days to hold position
        position_size_multiplier=1.0  # Position size multiplier (0.5-1.0 for dynamic sizing)
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            profit_target: % gain to exit with profit
            stop_loss: % loss to exit with loss
            max_hold_days: Max days to hold before forced exit
            position_size_multiplier: Position size multiplier (default 1.0 for full position)
        """
        self.initial_capital = initial_capital
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.position_size_multiplier = position_size_multiplier
        self.max_hold_days = max_hold_days

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        position = None
        trades = []

        data = df.reset_index(drop=True)

        for i in range(len(data) - self.max_hold_days):
            row = data.iloc[i]

            # ENTRY: Look for panic sell signals (BUY opportunity)
            if row['panic_sell'] == 1 and position is None:
                entry_price = row['Close']
                # Apply position sizing multiplier
                position_capital = capital * self.position_size_multiplier
                shares = position_capital / entry_price
                position = {
                    'entry_idx': i,
                    'entry_date': row['Date'],
                    'entry_price': entry_price,
                    'shares': shares,
                    'type': 'long',  # Buying the dip
                    'position_size': self.position_size_multiplier
                }

            # EXIT: Check exit conditions if in position
            elif position is not None:
                current_price = data.iloc[i]['Close']
                entry_price = position['entry_price']
                days_held = i - position['entry_idx']

                # Calculate return
                return_pct = (current_price / entry_price - 1) * 100

                # Exit conditions
                hit_profit = return_pct >= self.profit_target
                hit_stop = return_pct <= self.stop_loss
                timeout = days_held >= self.max_hold_days

                if hit_profit or hit_stop or timeout:
                    # Close position
                    profit = (current_price - entry_price) * position['shares']
                    capital += profit

                    # Record trade
                    exit_reason = (
                        'profit' if hit_profit else
                        'stop_loss' if hit_stop else
                        'timeout'
                    )

                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': data.iloc[i]['Date'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'profit': profit,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })

                    position = None

        # Calculate metrics
        total_trades = len(trades)

        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'total_return': 0,
                'final_capital': capital,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': []
            }

        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]

        win_rate = len(winning_trades) / total_trades * 100

        avg_gain = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0

        total_return = (capital / self.initial_capital - 1) * 100

        # Calculate Sharpe ratio
        returns = [t['return_pct'] for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / self.max_hold_days) if np.std(returns) > 0 else 0

        # Calculate max drawdown
        cumulative = [self.initial_capital]
        for trade in trades:
            cumulative.append(cumulative[-1] + trade['profit'])

        cumulative_array = np.array(cumulative)
        running_max = np.maximum.accumulate(cumulative_array)
        drawdown = (cumulative_array - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'final_capital': capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(sum(t['profit'] for t in winning_trades) / sum(t['profit'] for t in losing_trades)) if losing_trades else float('inf'),
            'trades': trades
        }


if __name__ == "__main__":
    print("Mean Reversion Strategy module loaded successfully")
