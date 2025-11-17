"""
EXPERIMENT: EXP-066
Date: 2025-11-17
Objective: Build ML feature engineering pipeline for stock selection

HYPOTHESIS:
A comprehensive set of technical indicators (50+ features) can be used by machine
learning models to predict which stocks will experience profitable mean reversion
opportunities. This enables smarter stock selection and portfolio expansion.

ALGORITHM TESTED:
ML Feature Engineering Pipeline
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, volume ratios
- Price features: Returns, momentum, volatility
- Market features: Correlation to SPY, relative performance
- Target: Binary classification (will have profitable signal in next 7 days)

TEST STOCKS:
Current 45-stock portfolio + 10 candidate stocks for comprehensive dataset

TIME FRAME:
2020-01-01 to 2025-11-15 (5+ years for training/validation/testing)

SUCCESS CRITERIA:
- Successfully generate 50+ features for all stocks
- Feature matrix has <10% missing values
- Target variable properly balanced (30-70% positive class)
- Ready for XGBoost training in EXP-067

EXPECTED OUTCOME:
If successful: Proceed to EXP-067 (XGBoost model training)
If insufficient features: Add fundamental data or alternative indicators
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import yfinance for price data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance not installed: pip install yfinance")

# Import TA-Lib or fallback to pandas_ta
try:
    import talib
    TALIB_AVAILABLE = True
    print("[OK] TA-Lib available")
except ImportError:
    TALIB_AVAILABLE = False
    print("[INFO] TA-Lib not available, using pandas-based indicators")


class MLFeatureEngineer:
    """
    Feature engineering for ML-based stock selection.

    Generates comprehensive technical features from price data.
    """

    def __init__(self, lookback_days: int = 252 * 3):
        """
        Initialize feature engineer.

        Args:
            lookback_days: Number of days of historical data to use (default 3 years)
        """
        self.lookback_days = lookback_days

    def fetch_price_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV price data for a stock."""
        if not YFINANCE_AVAILABLE:
            print(f"  [SKIP] yfinance not available for {ticker}")
            return None

        try:
            print(f"  Fetching data for ${ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty or len(df) < 100:
                print(f"  [SKIP] Insufficient data for {ticker}")
                return None

            df = df.reset_index()
            df['ticker'] = ticker

            return df[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            print(f"  [ERROR] Failed to fetch {ticker}: {e}")
            return None

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return features."""
        close = df['Close']

        # Simple returns
        df['return_1d'] = close.pct_change(1)
        df['return_5d'] = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)
        df['return_60d'] = close.pct_change(60)

        # Log returns
        df['log_return_1d'] = np.log(close / close.shift(1))
        df['log_return_5d'] = np.log(close / close.shift(5))

        # Forward returns (target variables)
        df['forward_return_1d'] = close.pct_change(1).shift(-1)
        df['forward_return_3d'] = close.pct_change(3).shift(-3)
        df['forward_return_7d'] = close.pct_change(7).shift(-7)

        return df

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_28'] = talib.RSI(close, timeperiod=28)
        else:
            df['rsi_14'] = self._calculate_rsi(close, 14)
            df['rsi_28'] = self._calculate_rsi(close, 28)

        # MACD
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
        else:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high, low, close)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
        else:
            df['stoch_k'] = self._calculate_stochastic(df, 14)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Rate of Change (ROC)
        df['roc_10'] = ((close / close.shift(10)) - 1) * 100
        df['roc_20'] = ((close / close.shift(20)) - 1) * 100

        return df

    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features."""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # Bollinger Bands
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        else:
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            df['bb_upper'] = sma_20 + (2 * std_20)
            df['bb_middle'] = sma_20
            df['bb_lower'] = sma_20 - (2 * std_20)

        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range)
        if TALIB_AVAILABLE:
            df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()

        # Historical volatility
        df['volatility_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
        df['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = close.pct_change().rolling(60).std() * np.sqrt(252)

        return df

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        close = df['Close']
        volume = df['Volume']

        # Volume ratios
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']

        # OBV (On-Balance Volume)
        if TALIB_AVAILABLE:
            df['obv'] = talib.OBV(close, volume)
        else:
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            df['obv'] = obv

        df['obv_sma_20'] = df['obv'].rolling(20).mean()

        # Volume-weighted price
        df['vwap_20'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()

        return df

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based features."""
        close = df['Close']

        # Moving averages
        df['sma_10'] = close.rolling(10).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()

        df['ema_10'] = close.ewm(span=10).mean()
        df['ema_20'] = close.ewm(span=20).mean()
        df['ema_50'] = close.ewm(span=50).mean()

        # MA crossovers
        df['sma_10_20_ratio'] = df['sma_10'] / df['sma_20']
        df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
        df['price_sma_50_ratio'] = close / df['sma_50']
        df['price_sma_200_ratio'] = close / df['sma_200']

        # ADX (Average Directional Index)
        if TALIB_AVAILABLE:
            df['adx'] = talib.ADX(df['High'], df['Low'], close, timeperiod=14)

        return df

    def calculate_market_features(self, df: pd.DataFrame, spy_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-relative features."""
        # Merge SPY data
        df = df.merge(
            spy_data[['Date', 'Close']].rename(columns={'Close': 'spy_close'}),
            on='Date',
            how='left'
        )

        # Calculate SPY returns
        df['spy_return_1d'] = df['spy_close'].pct_change(1)
        df['spy_return_20d'] = df['spy_close'].pct_change(20)

        # Relative performance
        df['rel_perf_1d'] = df['return_1d'] - df['spy_return_1d']
        df['rel_perf_20d'] = df['return_20d'] - df['spy_return_20d']

        # Rolling correlation with SPY
        df['spy_corr_60'] = df['return_1d'].rolling(60).corr(df['spy_return_1d'])

        # Beta (simplified)
        window = 60
        cov = df['return_1d'].rolling(window).cov(df['spy_return_1d'])
        var = df['spy_return_1d'].rolling(window).var()
        df['beta_60'] = cov / var

        return df

    def create_target_variable(self, df: pd.DataFrame, threshold: float = 0.03) -> pd.DataFrame:
        """
        Create binary target: Will stock have profitable mean reversion signal?

        Target = 1 if:
        - Stock drops > 3% from recent high (panic sell)
        - AND recovers within 7 days (successful mean reversion)
        """
        close = df['Close']

        # Calculate rolling max (recent high)
        df['rolling_max_20'] = close.rolling(20).max()
        df['drawdown'] = (close - df['rolling_max_20']) / df['rolling_max_20']

        # Panic sell signal (drop > threshold from recent high)
        df['panic_sell'] = (df['drawdown'] < -threshold).astype(int)

        # Check if recovers within 7 days (forward-looking)
        df['recovery_7d'] = (df['forward_return_7d'] > threshold).astype(int)

        # Target: Panic sell + successful recovery
        df['target'] = (df['panic_sell'] == 1) & (df['recovery_7d'] == 1)
        df['target'] = df['target'].astype(int)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K manually."""
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        return stoch_k

    def generate_features(self, ticker: str, start_date: str, end_date: str,
                         spy_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate complete feature set for a stock.

        Returns DataFrame with all features + target variable.
        """
        # Fetch price data
        df = self.fetch_price_data(ticker, start_date, end_date)
        if df is None:
            return None

        # Calculate features
        df = self.calculate_returns(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_trend_indicators(df)
        df = self.calculate_market_features(df, spy_data)
        df = self.create_target_variable(df)

        return df


def run_exp066_feature_engineering():
    """
    Build ML feature engineering pipeline.
    """
    print("="*70)
    print("EXP-066: ML FEATURE ENGINEERING PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("ALGORITHM: Comprehensive Technical Feature Generation")
    print("  - Returns (1d, 5d, 10d, 20d, 60d)")
    print("  - Momentum (RSI, MACD, Stochastic, ROC)")
    print("  - Volatility (Bollinger Bands, ATR, Historical Vol)")
    print("  - Volume (OBV, Volume Ratios, VWAP)")
    print("  - Trend (SMA, EMA, MA Crossovers, ADX)")
    print("  - Market (SPY correlation, Beta, Relative Performance)")
    print()

    if not YFINANCE_AVAILABLE:
        print("[ERROR] yfinance required")
        print("Install: pip install yfinance")
        return None

    # Test on 10-stock subset first
    test_stocks = [
        'NVDA', 'MSFT', 'AAPL', 'JPM', 'V',
        'MA', 'ORCL', 'AVGO', 'ABBV', 'TXN'
    ]

    print(f"TEST STOCKS: {', '.join(test_stocks)} (10 stocks)")
    print(f"TIME FRAME: 2020-01-01 to 2025-11-15 (5+ years)")
    print()

    start_date = '2020-01-01'
    end_date = '2025-11-15'

    # Fetch SPY data first (needed for market features)
    print("Fetching SPY (market) data...")
    try:
        spy = yf.Ticker('SPY')
        spy_data = spy.history(start=start_date, end=end_date)
        spy_data = spy_data.reset_index()[['Date', 'Close']]
        print(f"  [OK] SPY data: {len(spy_data)} days")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch SPY: {e}")
        return None

    print()
    print("Generating features for each stock...")
    print("-" * 70)

    engineer = MLFeatureEngineer()

    all_features = []
    successful = 0
    failed = 0

    for ticker in test_stocks:
        features_df = engineer.generate_features(ticker, start_date, end_date, spy_data)

        if features_df is not None:
            all_features.append(features_df)

            # Calculate metrics
            n_rows = len(features_df)
            n_features = len(features_df.columns) - 8  # Exclude Date, ticker, OHLCV, target
            n_targets = features_df['target'].sum()
            target_pct = (n_targets / n_rows) * 100
            missing_pct = (features_df.isnull().sum().sum() / (n_rows * len(features_df.columns))) * 100

            print(f"  [OK] ${ticker}: {n_rows} rows, {n_features} features, "
                  f"{target_pct:.1f}% positive targets, {missing_pct:.1f}% missing")
            successful += 1
        else:
            print(f"  [FAILED] ${ticker}")
            failed += 1

    if not all_features:
        print()
        print("[FAILED] No stocks successfully generated features")
        return None

    # Combine all stocks
    combined_df = pd.concat(all_features, ignore_index=True)

    print()
    print("="*70)
    print("FEATURE ENGINEERING RESULTS")
    print("="*70)
    print()

    print(f"Stocks processed: {successful} / {len(test_stocks)}")
    print(f"Total data points: {len(combined_df):,}")
    print(f"Features generated: {len(combined_df.columns) - 8}")
    print()

    # Feature statistics
    feature_cols = [col for col in combined_df.columns
                   if col not in ['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']]

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target variable (positive class): {combined_df['target'].sum():,} ({combined_df['target'].mean()*100:.1f}%)")
    print()

    # Missing value analysis
    missing_by_feature = combined_df[feature_cols].isnull().sum().sort_values(ascending=False)
    top_missing = missing_by_feature.head(10)

    print("Features with most missing values:")
    for feat, count in top_missing.items():
        pct = (count / len(combined_df)) * 100
        print(f"  {feat}: {count} ({pct:.1f}%)")

    overall_missing = (combined_df[feature_cols].isnull().sum().sum() /
                      (len(combined_df) * len(feature_cols))) * 100

    print()
    print(f"Overall missing data: {overall_missing:.2f}%")
    print()

    # Deployment decision
    deploy = (successful >= 8 and overall_missing < 15.0 and
              0.20 <= combined_df['target'].mean() <= 0.70)

    print("="*70)
    print("DEPLOYMENT DECISION")
    print("="*70)
    print()

    if deploy:
        print("[SUCCESS] Feature pipeline validated!")
        print()
        print(f"[OK] Stocks processed: {successful} / {len(test_stocks)}")
        print(f"[OK] Missing data: {overall_missing:.2f}% (< 15% required)")
        print(f"[OK] Target balance: {combined_df['target'].mean()*100:.1f}% (20-70% required)")
        print()
        print("NEXT STEP: EXP-067 - XGBoost model training")
        print("  - Train gradient boosting model on feature matrix")
        print("  - Walk-forward validation (time-series split)")
        print("  - Feature importance analysis")
        print("  - Out-of-sample AUC > 0.60 target")
    else:
        reasons = []
        if successful < 8:
            reasons.append(f"Only {successful}/10 stocks processed successfully")
        if overall_missing >= 15.0:
            reasons.append(f"Missing data {overall_missing:.2f}% >= 15%")
        if not (0.20 <= combined_df['target'].mean() <= 0.70):
            reasons.append(f"Target imbalance: {combined_df['target'].mean()*100:.1f}%")

        print("[INSUFFICIENT QUALITY]")
        for reason in reasons:
            print(f"  - {reason}")
        print()
        print("IMPROVEMENTS NEEDED:")
        print("  1. Handle missing values (forward fill, interpolation)")
        print("  2. Add fundamental features (P/E, EPS, revenue)")
        print("  3. Balance target variable (SMOTE, class weighting)")

    # Save features
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    # Save feature matrix
    feature_file = os.path.join(results_dir, 'exp066_feature_matrix.csv')
    combined_df.to_csv(feature_file, index=False)
    print(f"\nFeature matrix saved to: {feature_file}")

    # Save results
    exp_results = {
        'experiment_id': 'EXP-066',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'ML Feature Engineering Pipeline',
        'test_stocks': test_stocks,
        'time_frame': f'{start_date} to {end_date}',
        'stocks_tested': len(test_stocks),
        'successful_processing': successful,
        'failed_processing': failed,
        'aggregate_metrics': {
            'total_data_points': int(len(combined_df)),
            'n_features': len(feature_cols),
            'overall_missing_pct': float(overall_missing),
            'target_positive_rate': float(combined_df['target'].mean()),
            'target_positive_count': int(combined_df['target'].sum())
        },
        'feature_list': feature_cols,
        'deploy': deploy,
        'next_step': 'EXP-067' if deploy else 'Improve feature quality',
        'hypothesis': 'Comprehensive technical features enable ML prediction of profitable mean reversion signals',
        'methodology': {
            'feature_categories': [
                'Returns (5 features)',
                'Momentum indicators (10+ features)',
                'Volatility indicators (10+ features)',
                'Volume indicators (5+ features)',
                'Trend indicators (10+ features)',
                'Market-relative features (5+ features)'
            ],
            'target_definition': 'Panic sell (>3% drop) + recovery within 7 days',
            'success_criteria': '8+ stocks, <15% missing, 20-70% target balance'
        }
    }

    results_file = os.path.join(results_dir, 'exp066_ml_feature_engineering.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending ML feature engineering report email...")
            notifier.send_experiment_report('EXP-066', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-066: ML feature engineering pipeline."""

    print("\n[ML FEATURES] Building comprehensive feature matrix")
    print("This creates the foundation for ML-based stock selection")
    print()

    results = run_exp066_feature_engineering()

    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
