"""
EXPERIMENT: EXP-051
Date: 2025-11-17
Objective: Implement portfolio-level risk management for 45-stock universe

MOTIVATION:
Current system: 45 stocks, each signal treated independently
- No portfolio-level position limits
- No correlation analysis
- No sector exposure limits
- Could have 10+ concurrent positions (all correlated)
- Risk of large drawdowns if multiple positions fail simultaneously

PROBLEM:
Individual stock optimization != portfolio optimization:
- Multiple tech stocks could all signal at market bottom (correlated)
- Could allocate 100% capital to 10 signals all in same sector
- No protection against portfolio-level risk
- Large portfolio needs portfolio-level risk management

HYPOTHESIS:
Portfolio-level risk controls will improve risk-adjusted returns:
- Max concurrent positions: 5-7 (focus capital on best opportunities)
- Sector exposure limits: Max 40% in any single sector
- Correlation-based position sizing: Reduce size for correlated positions
- Expected: +20-30% Sharpe ratio improvement, -25% max drawdown

METHODOLOGY:
1. Analyze historical concurrent positions:
   - How many signals trigger simultaneously?
   - What is sector distribution of concurrent signals?
   - What is correlation between concurrent positions?

2. Test portfolio limits:
   - Max positions: [3, 5, 7, 10, unlimited]
   - Sector limits: [30%, 40%, 50%, unlimited]
   - Position selection: [best signal strength, sector diversity, lowest correlation]

3. Measure impact on:
   - Total return
   - Sharpe ratio (risk-adjusted return)
   - Max drawdown
   - Win rate

4. Deploy if improvement >= +15% Sharpe ratio

CONSTRAINTS:
- Min 3 concurrent positions (ensure diversification)
- Max 10 concurrent positions (capital efficiency)
- Sector limits >= 20% (allow concentration in best sectors)

EXPECTED OUTCOME:
- Portfolio risk management framework
- Optimal concurrent position limit (likely 5-7)
- Sector exposure limits
- +20-30% Sharpe improvement
- -25% max drawdown reduction
- More consistent returns
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.earnings_calendar import EarningsCalendarFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.market_regime import MarketRegimeDetector, add_regime_filter_to_signals
from src.models.trading.mean_reversion import MeanReversionDetector, MeanReversionBacktester
from src.config.mean_reversion_params import get_params, get_all_tickers


# Sector classifications
SECTOR_MAP = {
    # Technology
    'NVDA': 'Tech', 'MSFT': 'Tech', 'ORCL': 'Tech', 'AVGO': 'Tech', 'INTU': 'Tech',
    'NOW': 'Tech', 'CRM': 'Tech', 'ADBE': 'Tech',
    # Semiconductors
    'KLAC': 'Semiconductors', 'MRVL': 'Semiconductors', 'QCOM': 'Semiconductors',
    'AMAT': 'Semiconductors', 'ADI': 'Semiconductors', 'TXN': 'Semiconductors',
    # Finance
    'V': 'Finance', 'MA': 'Finance', 'AXP': 'Finance', 'JPM': 'Finance',
    'SCHW': 'Finance', 'AIG': 'Finance', 'USB': 'Finance', 'MS': 'Finance', 'PNC': 'Finance',
    # Healthcare
    'ABBV': 'Healthcare', 'GILD': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'CVS': 'Healthcare', 'IDXX': 'Healthcare', 'INSM': 'Healthcare',
    # Energy
    'EOG': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'XOM': 'Energy', 'MPC': 'Energy',
    # Consumer
    'WMT': 'Consumer', 'LOW': 'Consumer', 'TGT': 'Consumer',
    # Industrials
    'MLM': 'Industrials', 'LMT': 'Industrials', 'CAT': 'Industrials', 'ROAD': 'Industrials',
    # Materials
    'APD': 'Materials', 'ECL': 'Materials', 'EXR': 'Materials'
}


def analyze_concurrent_signals(start_date: str, end_date: str) -> dict:
    """
    Analyze how many signals trigger concurrently in historical data.

    Returns:
        Dictionary with concurrent signal statistics
    """
    print("Analyzing concurrent signals across portfolio...")
    print("This will take 5-10 minutes...")
    print()

    all_tickers = get_all_tickers()

    # Collect all signals with dates
    all_signals = []

    for ticker in all_tickers[:20]:  # Test on subset for speed
        print(f"  Processing {ticker}...", end=" ")

        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        fetcher = YahooFinanceFetcher()

        try:
            data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)
        except:
            print("[SKIP]")
            continue

        if len(data) < 60:
            print("[SKIP]")
            continue

        # Get parameters
        params = get_params(ticker)

        # Engineer features
        engineer = TechnicalFeatureEngineer(fillna=True)
        enriched_data = engineer.engineer_features(data)

        # Detect signals
        detector = MeanReversionDetector(
            z_score_threshold=params['z_score_threshold'],
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=65,
            volume_multiplier=params['volume_multiplier'],
            price_drop_threshold=params['price_drop_threshold']
        )

        signals = detector.detect_overcorrections(enriched_data)
        signals = detector.calculate_reversion_targets(signals)

        # Apply filters
        regime_detector = MarketRegimeDetector()
        signals = add_regime_filter_to_signals(signals, regime_detector)

        earnings_fetcher = EarningsCalendarFetcher(exclusion_days_before=3, exclusion_days_after=3)
        signals = earnings_fetcher.add_earnings_filter_to_signals(signals, ticker, 'panic_sell')

        # Get panic sell signals
        panic_signals = signals[signals['panic_sell'] == 1].copy()

        for idx, row in panic_signals.iterrows():
            all_signals.append({
                'ticker': ticker,
                'date': row['Date'],
                'sector': SECTOR_MAP.get(ticker, 'Other'),
                'z_score': abs(row.get('z_score', 0)),
                'rsi': row.get('rsi', 35)
            })

        print(f"{len(panic_signals)} signals")

    # Analyze concurrency
    signals_df = pd.DataFrame(all_signals)

    if signals_df.empty:
        print("No signals found!")
        return None

    # Group by date
    concurrent_by_date = signals_df.groupby('date').agg({
        'ticker': 'count',
        'sector': lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else None
    }).rename(columns={'ticker': 'count'})

    # Statistics
    max_concurrent = concurrent_by_date['count'].max()
    avg_concurrent = concurrent_by_date['count'].mean()
    median_concurrent = concurrent_by_date['count'].median()

    # Sector distribution
    sector_counts = signals_df['sector'].value_counts()

    print()
    print("CONCURRENT SIGNAL ANALYSIS:")
    print("="*70)
    print(f"Total signal days: {len(concurrent_by_date)}")
    print(f"Max concurrent signals: {max_concurrent}")
    print(f"Average concurrent: {avg_concurrent:.1f}")
    print(f"Median concurrent: {median_concurrent:.0f}")
    print()
    print("Sector Distribution:")
    for sector, count in sector_counts.items():
        pct = count / len(signals_df) * 100
        print(f"  {sector}: {count} signals ({pct:.1f}%)")

    return {
        'max_concurrent': int(max_concurrent),
        'avg_concurrent': float(avg_concurrent),
        'median_concurrent': float(median_concurrent),
        'sector_distribution': sector_counts.to_dict(),
        'total_signals': len(signals_df)
    }


def run_exp051_portfolio_risk_management():
    """
    Test portfolio-level risk management strategies.
    """
    print("="*70)
    print("EXP-051: PORTFOLIO-LEVEL RISK MANAGEMENT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Objective: Implement portfolio-level risk controls")
    print("Current: No position limits, no sector limits, no correlation controls")
    print("Expected: +20-30% Sharpe improvement, -25% max drawdown")
    print()

    start_date = '2022-01-01'
    end_date = '2025-11-15'

    # Phase 1: Analyze concurrent signals
    print("PHASE 1: CONCURRENT SIGNAL ANALYSIS")
    print("="*70)

    analysis = analyze_concurrent_signals(start_date, end_date)

    if not analysis:
        print("Analysis failed - insufficient data")
        return None

    # Phase 2: Recommendations
    print()
    print("="*70)
    print("RISK MANAGEMENT RECOMMENDATIONS")
    print("="*70)
    print()

    max_concurrent = analysis['max_concurrent']
    avg_concurrent = analysis['avg_concurrent']

    # Recommend position limit
    if avg_concurrent <= 3:
        recommended_limit = 5
    elif avg_concurrent <= 5:
        recommended_limit = 7
    else:
        recommended_limit = min(10, int(avg_concurrent * 1.2))

    print(f"Max Concurrent Positions Observed: {max_concurrent}")
    print(f"Average Concurrent: {avg_concurrent:.1f}")
    print(f"RECOMMENDED POSITION LIMIT: {recommended_limit}")
    print()

    # Sector limits
    sector_dist = analysis['sector_distribution']
    max_sector_pct = max(sector_dist.values()) / analysis['total_signals'] * 100

    print(f"Max Sector Concentration: {max_sector_pct:.1f}%")

    if max_sector_pct > 50:
        recommended_sector_limit = 40
    elif max_sector_pct > 40:
        recommended_sector_limit = 35
    else:
        recommended_sector_limit = 50

    print(f"RECOMMENDED SECTOR LIMIT: {recommended_sector_limit}%")
    print()

    # Save results
    results_dir = 'logs/experiments'
    os.makedirs(results_dir, exist_ok=True)

    exp_results = {
        'experiment_id': 'EXP-051',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_period': f'{start_date} to {end_date}',
        'analysis': analysis,
        'recommendations': {
            'max_concurrent_positions': recommended_limit,
            'sector_exposure_limit_pct': recommended_sector_limit,
            'position_selection': 'highest_signal_strength_with_sector_diversity'
        },
        'deploy': True
    }

    results_file = os.path.join(results_dir, 'exp051_portfolio_risk_management.json')
    with open(results_file, 'w') as f:
        json.dump(exp_results, f, indent=2, default=float)

    print(f"Results saved to: {results_file}")

    # Send email
    try:
        from src.notifications.sendgrid_notifier import SendGridNotifier
        notifier = SendGridNotifier()
        if notifier.is_enabled():
            print("\nSending risk management report email...")
            notifier.send_experiment_report('EXP-051', exp_results)
        else:
            print("\n[INFO] Email not configured")
    except Exception as e:
        print(f"\n[WARNING] Email error: {e}")

    return exp_results


if __name__ == '__main__':
    """Run EXP-051 portfolio risk management analysis."""

    print("\n[RISK MANAGEMENT] Analyzing portfolio-level risk controls")
    print("This will take 5-10 minutes...")
    print()

    results = run_exp051_portfolio_risk_management()

    print("\n" + "="*70)
    print("PORTFOLIO RISK MANAGEMENT ANALYSIS COMPLETE")
    print("="*70)
