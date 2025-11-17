"""
EXP-083: Sentiment Features Validation (PHASE 3)
Part of 6-week feature engineering program

LEARNING FROM EXP-082:
- Microstructure features FAILED: -2.6pp AUC
- Need higher-impact features

RESEARCH BASIS:
"Sentiment analysis shows +3-6pp AUC improvement in ML trading systems"
- 2024 Journal of Financial Markets
- Twitter/Reddit capture emotion-driven panics
- News sentiment captures fundamental issues
- Divergence between social/news = trading edge

HYPOTHESIS:
Sentiment features will significantly improve panic sell detection by:
1. Identifying emotion-driven vs fundamental-driven drops
2. Measuring panic intensity via social sentiment velocity
3. Using sentiment divergence as confidence filter

FEATURES TESTED (5 new features):
1. social_sentiment: Twitter + Reddit combined (-1 to +1)
2. news_sentiment: Professional news sentiment (-1 to +1)
3. sentiment_divergence: |social - news| (panic detector)
4. sentiment_velocity: Rate of sentiment change
5. confidence_level: Trading signal quality (HIGH/MEDIUM/LOW)

METHODOLOGY:
- Test on 5 diverse stocks (due to API rate limits)
- Compare: Baseline (technical only) vs Baseline + Sentiment
- Success criteria: +3pp AUC improvement, 80%+ stocks improved

EXPECTED IMPACT: +3-6pp AUC (highest ROI of all feature groups)

NOTE: Sentiment collection may be slow due to API rate limits.
This is a proof-of-concept validation. Full production deployment
would require optimized data collection and caching.
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple
import json

sys.path.insert(0, os.path.abspath('.'))

from src.data.data_fetcher import YahooFinanceFetcher
from src.data.features.technical_features import TechnicalFeatureEngineer
from src.data.sentiment.sentiment_features import SentimentFeaturePipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

# Notification
from src.notifications.sendgrid_notifier import SendGridNotifier


def prepare_ml_dataset(ticker: str,
                       start_date: str = '2020-01-01',
                       end_date: str = '2024-11-17',
                       include_sentiment: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare ML dataset with optional sentiment features.

    Returns: (features_df, labels_df)
    """
    # Add buffer for technical indicators
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    buffer_start = (start_dt - timedelta(days=100)).strftime('%Y-%m-%d')

    print(f"\n[{ticker}] Fetching data from {buffer_start}...")

    # 1. Fetch price data
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_stock_data(ticker, start_date=buffer_start, end_date=end_date)

    if data is None or len(data) < 100:
        print(f"  [ERROR] Insufficient data for {ticker}")
        return None, None

    print(f"  [OK] Fetched {len(data)} days of price data")

    # 2. Add baseline technical features
    print(f"  [INFO] Engineering technical features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)

    # 3. Add sentiment features if requested
    if include_sentiment:
        print(f"  [INFO] Collecting sentiment features...")
        sentiment_pipeline = SentimentFeaturePipeline()

        if sentiment_pipeline.is_ready():
            try:
                # Collect sentiment for date range
                sentiment_data = sentiment_pipeline.collect_all_sentiment(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )

                if not sentiment_data.empty:
                    # Merge sentiment with price data
                    enriched_data = enriched_data.merge(
                        sentiment_data,
                        left_on='Date',
                        right_index=True,
                        how='left'
                    )
                    print(f"  [OK] Added {len(sentiment_data.columns)} sentiment features")

                    # Fill missing sentiment with neutral (0)
                    sentiment_cols = ['social_sentiment', 'news_sentiment', 'sentiment_divergence',
                                     'sentiment_velocity']
                    for col in sentiment_cols:
                        if col in enriched_data.columns:
                            enriched_data[col] = enriched_data[col].fillna(0)

                    if 'confidence_level' in enriched_data.columns:
                        enriched_data['confidence_level'] = enriched_data['confidence_level'].fillna('MEDIUM')
                        # Convert to numeric
                        enriched_data['confidence_numeric'] = enriched_data['confidence_level'].map({
                            'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.0
                        }).fillna(0.5)
                else:
                    print(f"  [WARN] No sentiment data collected, using technical only")
            except Exception as e:
                print(f"  [ERROR] Sentiment collection failed: {e}")
                print(f"  [INFO] Continuing with technical features only")
        else:
            print(f"  [WARN] Sentiment pipeline not ready, using technical only")

    # 4. Create labels (forward returns)
    enriched_data['forward_return_3d'] = enriched_data['Close'].shift(-3) / enriched_data['Close'] - 1
    enriched_data['win'] = (enriched_data['forward_return_3d'] > 0.01).astype(int)

    # 5. Filter to analysis period and valid signals
    enriched_data = enriched_data[enriched_data['Date'] >= start_date].copy()

    # Panic sell signals (for labeling, not features)
    enriched_data['panic_sell'] = (
        (enriched_data['rsi'] < 35) &
        (enriched_data['z_score'] < -1.5) &
        (enriched_data['volume_spike'] > 1.3) &
        (enriched_data['price_drop_pct'] < -1.5)
    )

    valid_signals = enriched_data[enriched_data['panic_sell']].copy()

    if len(valid_signals) < 30:
        print(f"  [ERROR] Insufficient panic sell signals: {len(valid_signals)}")
        return None, None

    # 6. Prepare features and labels
    exclude_cols = ['forward_return_3d', 'win', 'panic_sell', 'Date', 'confidence_level']
    feature_cols = [col for col in valid_signals.columns
                    if col not in exclude_cols and
                    valid_signals[col].dtype in ['int64', 'float64', 'bool']]

    X = valid_signals[feature_cols].copy()
    y = valid_signals['win'].copy()

    # Handle any remaining NaN
    X = X.fillna(0)

    print(f"  [OK] Prepared {len(X)} samples with {len(feature_cols)} features")

    return X, y


def test_single_stock(ticker: str) -> dict:
    """Test baseline vs sentiment-enhanced features on single stock."""
    print(f"\n{'='*70}")
    print(f"Testing {ticker}...")
    print(f"{'='*70}")

    # Test baseline (technical only)
    print(f"\n[BASELINE] Technical features only...")
    X_baseline, y = prepare_ml_dataset(ticker, include_sentiment=False)

    if X_baseline is None:
        return {'ticker': ticker, 'error': 'Insufficient data'}

    # Train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train baseline model
    model_baseline = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model_baseline.fit(X_train, y_train)

    # Evaluate baseline
    y_pred_proba_baseline = model_baseline.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_proba_baseline)

    print(f"  Baseline AUC: {baseline_auc:.3f}")

    # Test with sentiment
    print(f"\n[SENTIMENT] Technical + Sentiment features...")
    X_sentiment, y = prepare_ml_dataset(ticker, include_sentiment=True)

    if X_sentiment is None:
        return {
            'ticker': ticker,
            'baseline_auc': baseline_auc,
            'sentiment_auc': baseline_auc,
            'improvement': 0.0,
            'samples': len(X_test),
            'note': 'Sentiment collection failed'
        }

    # Train/test split (same random state for comparability)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sentiment, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train sentiment-enhanced model
    model_sentiment = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model_sentiment.fit(X_train_s, y_train_s)

    # Evaluate sentiment model
    y_pred_proba_sentiment = model_sentiment.predict_proba(X_test_s)[:, 1]
    sentiment_auc = roc_auc_score(y_test_s, y_pred_proba_sentiment)

    improvement = sentiment_auc - baseline_auc

    print(f"  Sentiment AUC: {sentiment_auc:.3f}")
    print(f"  Improvement: {improvement:+.3f}")

    return {
        'ticker': ticker,
        'baseline_auc': baseline_auc,
        'sentiment_auc': sentiment_auc,
        'improvement': improvement,
        'samples': len(X_test),
        'baseline_features': len(X_baseline.columns),
        'sentiment_features': len(X_sentiment.columns)
    }


def main():
    """Run sentiment features validation experiment."""
    print("\n" + "="*70)
    print("EXP-083: SENTIMENT FEATURES VALIDATION (PHASE 3)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Validate sentiment features for panic sell detection")
    print("RESEARCH: 'Sentiment analysis shows +3-6pp AUC improvement'")
    print("TARGET: +3pp AUC improvement, 80%+ stocks improved")
    print()
    print("Testing on 5 diverse stocks (API rate limit consideration)...")
    print()

    # Test stocks (diverse sectors, different volatility profiles)
    test_stocks = [
        'NVDA',  # Tech - high volatility
        'JPM',   # Finance - moderate volatility
        'JNJ',   # Healthcare - low volatility
        'TSLA',  # Tech - extreme volatility
        'WMT'    # Consumer - stable
    ]

    results = []

    # Test each stock
    for i, ticker in enumerate(test_stocks, 1):
        print(f"\n[{i}/{len(test_stocks)}] {ticker}")
        result = test_single_stock(ticker)
        results.append(result)

        if 'error' not in result:
            print(f"  Baseline AUC: {result['baseline_auc']:.3f}, " +
                  f"Sentiment AUC: {result['sentiment_auc']:.3f}, " +
                  f"Improvement: {result['improvement']:+.3f}")

    # Calculate aggregate statistics
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("\n[ERROR] No valid results")
        return

    avg_baseline = np.mean([r['baseline_auc'] for r in valid_results])
    avg_sentiment = np.mean([r['sentiment_auc'] for r in valid_results])
    avg_improvement = avg_sentiment - avg_baseline

    improved_count = sum(1 for r in valid_results if r['improvement'] > 0)
    improved_pct = (improved_count / len(valid_results)) * 100

    # Print summary
    print("\n" + "="*70)
    print("SENTIMENT FEATURES VALIDATION RESULTS")
    print("="*70)
    print()
    print(f"Stocks Tested: {len(valid_results)}/{len(test_stocks)}")
    print(f"Stocks Improved: {improved_count}/{len(valid_results)} ({improved_pct:.1f}%)")
    print()
    print(f"{'Metric':<30} {'Baseline':<15} {'+ Sentiment':<15} {'Improvement'}")
    print("-"*70)
    print(f"{'Average AUC':<30} {avg_baseline:.3f}          {avg_sentiment:.3f}          {avg_improvement:+.3f}")
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-"*70)
    print(f"{'Ticker':<10} {'Samples':<10} {'Base AUC':<12} {'Sent AUC':<12} {'Improvement'}")
    print("-"*70)

    # Sort by improvement
    sorted_results = sorted(valid_results, key=lambda x: x['improvement'], reverse=True)

    for r in sorted_results:
        print(f"{r['ticker']:<10} {r['samples']:<10} " +
              f"{r['baseline_auc']:.3f}       {r['sentiment_auc']:.3f}       " +
              f"{r['improvement']:+.3f}")

    # Validation decision
    print()
    print("="*70)
    print("VALIDATION DECISION")
    print("="*70)
    print()

    meets_improvement = avg_improvement >= 0.03
    meets_coverage = improved_pct >= 80
    meets_auc = avg_sentiment >= 0.70

    print(f"  [{'PASS' if meets_improvement else 'FAIL'}] avg_improvement >= +0.03 (actual: {avg_improvement:+.3f})")
    print(f"  [{'PASS' if meets_coverage else 'FAIL'}] improved_count >= 80% (actual: {improved_pct:.1f}%)")
    print(f"  [{'PASS' if meets_auc else 'FAIL'}] avg_sentiment_auc >= 0.70 (actual: {avg_sentiment:.3f})")
    print()

    if meets_improvement and meets_coverage:
        print("[SUCCESS] Sentiment features validated for production deployment!")
        print()
        print("Next steps:")
        print("1. Integrate sentiment features into production XGBoost model")
        print("2. Set up automated sentiment data collection pipeline")
        print("3. Monitor production performance vs validation results")
        decision = 'DEPLOY'
    elif meets_improvement:
        print(f"[MARGINAL] {avg_improvement:+.3f} improvement but only {improved_pct:.1f}% coverage")
        print()
        print("Sentiment shows promise but needs refinement:")
        print("- Improve sentiment data quality")
        print("- Add more sentiment sources")
        print("- Optimize feature engineering")
        decision = 'NEEDS_WORK'
    else:
        print(f"[NEEDS WORK] {avg_improvement:+.3f} improvement too small")
        print()
        print("Sentiment features may not provide significant edge:")
        print("- Review sentiment collection methodology")
        print("- Test different sentiment indicators")
        print("- Consider alternative data sources")
        decision = 'NEEDS_WORK'

    # Save results
    experiment_results = {
        'experiment_id': 'EXP-083-PHASE-3',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Validate sentiment features for panic sell ML prediction',
        'hypothesis': 'Sentiment features will improve AUC by +3-6pp',
        'methodology': 'Compare baseline (technical) vs technical + sentiment on 5 stocks',
        'test_stocks': test_stocks,
        'results': {
            'stocks_tested': len(valid_results),
            'stocks_improved': improved_count,
            'improvement_pct': improved_pct,
            'avg_baseline_auc': avg_baseline,
            'avg_sentiment_auc': avg_sentiment,
            'avg_improvement': avg_improvement,
            'detailed_results': valid_results
        },
        'validation_decision': decision,
        'meets_criteria': {
            'improvement_target': meets_improvement,
            'coverage_target': meets_coverage,
            'auc_target': meets_auc
        },
        'conclusion': 'Sentiment features validated for production' if decision == 'DEPLOY'
                     else 'Sentiment features need refinement before deployment',
        'next_step': 'Integrate into production XGBoost model' if decision == 'DEPLOY'
                    else 'Refine sentiment features and re-validate'
    }

    # Save to JSON
    os.makedirs('logs/experiments', exist_ok=True)
    result_file = f"logs/experiments/exp083_sentiment_features_phase3.json"
    with open(result_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    # Send email report
    print("\nSending sentiment features validation report...")
    notifier = SendGridNotifier()
    if notifier.is_enabled():
        notifier.send_experiment_report('EXP-083-PHASE-3', experiment_results)

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print("\nSentiment validation results documented for resumption.")
    print("Agent-owl will nudge you to continue with next phase.")


if __name__ == '__main__':
    main()
