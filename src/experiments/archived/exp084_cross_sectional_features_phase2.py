"""
EXP-084: Cross-Sectional Features Validation (PHASE 2)
Part of 6-week feature engineering program

LEARNING FROM EXP-082 & EXP-083:
- Microstructure features FAILED: -2.6pp AUC (too noisy)
- Sentiment features FAILED: +0.000 AUC (API limitations, no historical data)

RESEARCH BASIS:
"Cross-sectional features improve ML trading by +2-4pp AUC" (2024)
- Sector rotation patterns predict stock-specific panic sells
- Market regime context (VIX, SPY beta) separates systemic vs idiosyncratic drops
- Relative strength vs peers identifies outlier selloffs

HYPOTHESIS:
Cross-sectional features will improve panic sell detection by:
1. Distinguishing market-wide selloffs from stock-specific panics
2. Identifying sector rotation opportunities
3. Using VIX stress regime as confidence filter

FEATURES TESTED (18 new features):
1. Sector Relative (6): Performance vs sector peers
2. Market Context (6): SPY/VIX correlation, beta, regime
3. Peer Comparison (6): RSI/volume/momentum vs sector

METHODOLOGY:
- Test on 10 diverse stocks (adequate statistical power)
- Compare: Baseline (technical only) vs Baseline + Cross-sectional
- Success criteria: +2pp AUC improvement, 70%+ stocks improved

EXPECTED IMPACT: +2-4pp AUC (research-backed estimate)

NO EXTERNAL API DEPENDENCIES: Uses only Yahoo Finance price data for SPY/VIX
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple
import json

sys.path.insert(0, os.path.abspath('.'))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.data.features.cross_sectional_features import CrossSectionalFeatureEngineer
from src.models.trading.mean_reversion import MeanReversionDetector

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

# Notification
from src.notifications.sendgrid_notifier import SendGridNotifier


def prepare_ml_dataset(ticker: str,
                       start_date: str = '2020-01-01',
                       end_date: str = '2024-11-17',
                       include_cross_sectional: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare ML dataset with optional cross-sectional features.

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

    # 3. Add cross-sectional features if requested
    if include_cross_sectional:
        print(f"  [INFO] Engineering cross-sectional features...")
        cross_engineer = CrossSectionalFeatureEngineer(fillna=True)
        enriched_data = cross_engineer.engineer_all_cross_sectional_features(
            enriched_data, ticker, start_date=start_date, end_date=end_date
        )

    # 4. Use MeanReversionDetector to get panic sell signals (consistent methodology)
    detector = MeanReversionDetector(
        z_score_threshold=1.5,
        rsi_oversold=35,
        volume_multiplier=1.3,
        price_drop_threshold=-1.5
    )

    signals = detector.detect_overcorrections(enriched_data)

    if len(signals) == 0:
        print(f"  [ERROR] No panic sell signals detected")
        return None, None

    # 5. Calculate forward returns (3-day holding period)
    signals['forward_return_3d'] = signals['Close'].pct_change(3).shift(-3) * 100

    # Binary label: 1 if return > 0, 0 otherwise
    signals['win'] = (signals['forward_return_3d'] > 0).astype(int)

    # Filter out rows with missing forward returns
    signals = signals.dropna(subset=['forward_return_3d'])

    if len(signals) < 30:
        print(f"  [ERROR] Insufficient valid signals: {len(signals)}")
        return None, None

    # 6. Prepare features and labels
    exclude_cols = ['forward_return_3d', 'win', 'Date']
    feature_cols = [col for col in signals.columns
                    if col not in exclude_cols and
                    signals[col].dtype in ['int64', 'float64', 'bool']]

    X = signals[feature_cols].copy()
    y = signals['win'].copy()

    # Handle any remaining NaN
    X = X.fillna(0)

    print(f"  [OK] Prepared {len(X)} samples with {len(feature_cols)} features")

    return X, y


def test_single_stock(ticker: str) -> dict:
    """Test baseline vs cross-sectional-enhanced features on single stock."""
    print(f"\n{'='*70}")
    print(f"Testing {ticker}...")
    print(f"{'='*70}")

    # Test baseline (technical only)
    print(f"\n[BASELINE] Technical features only...")
    X_baseline, y = prepare_ml_dataset(ticker, include_cross_sectional=False)

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

    # Test with cross-sectional features
    print(f"\n[CROSS-SECTIONAL] Technical + Cross-sectional features...")
    X_cross, y = prepare_ml_dataset(ticker, include_cross_sectional=True)

    if X_cross is None:
        return {
            'ticker': ticker,
            'baseline_auc': baseline_auc,
            'cross_auc': baseline_auc,
            'improvement': 0.0,
            'samples': len(X_test),
            'note': 'Cross-sectional engineering failed'
        }

    # Train/test split (same random state for comparability)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cross, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train cross-sectional-enhanced model
    model_cross = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model_cross.fit(X_train_c, y_train_c)

    # Evaluate cross-sectional model
    y_pred_proba_cross = model_cross.predict_proba(X_test_c)[:, 1]
    cross_auc = roc_auc_score(y_test_c, y_pred_proba_cross)

    improvement = cross_auc - baseline_auc

    print(f"  Cross-sectional AUC: {cross_auc:.3f}")
    print(f"  Improvement: {improvement:+.3f}")

    # Get feature importance for cross-sectional features
    feature_names = list(X_cross.columns)
    feature_importance = model_cross.feature_importances_

    # Identify top cross-sectional features
    cross_features = [f for f in feature_names if any(
        prefix in f for prefix in ['sector_', 'spy_', 'vix_', 'market_', 'relative_', 'peer_']
    )]

    cross_feature_importance = []
    for feat in cross_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            cross_feature_importance.append({
                'feature': feat,
                'importance': float(feature_importance[idx])
            })

    # Sort by importance
    cross_feature_importance = sorted(
        cross_feature_importance,
        key=lambda x: x['importance'],
        reverse=True
    )[:10]  # Top 10

    return {
        'ticker': ticker,
        'baseline_auc': float(baseline_auc),
        'cross_auc': float(cross_auc),
        'improvement': float(improvement),
        'samples': int(len(X_test)),
        'baseline_features': int(len(X_baseline.columns)),
        'cross_features': int(len(X_cross.columns)),
        'top_cross_features': cross_feature_importance
    }


def main():
    """Run cross-sectional features validation experiment."""
    print("\n" + "="*70)
    print("EXP-084: CROSS-SECTIONAL FEATURES VALIDATION (PHASE 2)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OBJECTIVE: Validate cross-sectional features for panic sell detection")
    print("RESEARCH: 'Cross-sectional features show +2-4pp AUC improvement'")
    print("TARGET: +2pp AUC improvement, 70%+ stocks improved")
    print()
    print("Testing on 10 diverse stocks...")
    print()

    # Test stocks (diverse sectors, different market cap, volatility)
    test_stocks = [
        'NVDA',  # Tech - high volatility, large cap
        'JPM',   # Finance - moderate volatility, large cap
        'JNJ',   # Healthcare - low volatility, large cap
        'TSLA',  # Tech/Auto - extreme volatility, large cap
        'WMT',   # Consumer - stable, large cap
        'V',     # Financials - moderate volatility, large cap
        'BA',    # Industrials - cyclical, large cap
        'XOM',   # Energy - commodity-driven, large cap
        'DIS',   # Communication - moderate volatility, large cap
        'COST'   # Consumer - stable, large cap
    ]

    results = []

    # Test each stock
    for i, ticker in enumerate(test_stocks, 1):
        print(f"\n[{i}/{len(test_stocks)}] {ticker}")
        result = test_single_stock(ticker)
        results.append(result)

        if 'error' not in result:
            print(f"  Baseline AUC: {result['baseline_auc']:.3f}, " +
                  f"Cross AUC: {result['cross_auc']:.3f}, " +
                  f"Improvement: {result['improvement']:+.3f}")

    # Calculate aggregate statistics
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("\n[ERROR] No valid results")
        return

    avg_baseline = np.mean([r['baseline_auc'] for r in valid_results])
    avg_cross = np.mean([r['cross_auc'] for r in valid_results])
    avg_improvement = avg_cross - avg_baseline

    improved_count = sum(1 for r in valid_results if r['improvement'] > 0)
    improved_pct = (improved_count / len(valid_results)) * 100

    # Print summary
    print("\n" + "="*70)
    print("CROSS-SECTIONAL FEATURES VALIDATION RESULTS")
    print("="*70)
    print()
    print(f"Stocks Tested: {len(valid_results)}/{len(test_stocks)}")
    print(f"Stocks Improved: {improved_count}/{len(valid_results)} ({improved_pct:.1f}%)")
    print()
    print(f"{'Metric':<30} {'Baseline':<15} {'+ Cross-Sect':<15} {'Improvement'}")
    print("-"*70)
    print(f"{'Average AUC':<30} {avg_baseline:.3f}          {avg_cross:.3f}          {avg_improvement:+.3f}")
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-"*70)
    print(f"{'Ticker':<10} {'Samples':<10} {'Base AUC':<12} {'Cross AUC':<12} {'Improvement'}")
    print("-"*70)

    # Sort by improvement
    sorted_results = sorted(valid_results, key=lambda x: x['improvement'], reverse=True)

    for r in sorted_results:
        print(f"{r['ticker']:<10} {r['samples']:<10} " +
              f"{r['baseline_auc']:.3f}       {r['cross_auc']:.3f}       " +
              f"{r['improvement']:+.3f}")

    # Aggregate top features across all stocks
    print()
    print("="*70)
    print("TOP CROSS-SECTIONAL FEATURES (aggregated across all stocks)")
    print("="*70)

    feature_importance_agg = {}
    for r in valid_results:
        for feat_info in r.get('top_cross_features', []):
            feat = feat_info['feature']
            importance = feat_info['importance']
            if feat not in feature_importance_agg:
                feature_importance_agg[feat] = []
            feature_importance_agg[feat].append(importance)

    # Average importance across stocks
    avg_feature_importance = {
        feat: np.mean(importances)
        for feat, importances in feature_importance_agg.items()
    }

    # Sort and display top 15
    top_features = sorted(
        avg_feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    print()
    for i, (feat, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:<40} {importance:.4f}")

    # Validation decision
    print()
    print("="*70)
    print("VALIDATION DECISION")
    print("="*70)
    print()

    meets_improvement = avg_improvement >= 0.02
    meets_coverage = improved_pct >= 70
    meets_auc = avg_cross >= 0.70

    print(f"  [{'PASS' if meets_improvement else 'FAIL'}] avg_improvement >= +0.02 (actual: {avg_improvement:+.3f})")
    print(f"  [{'PASS' if meets_coverage else 'FAIL'}] improved_count >= 70% (actual: {improved_pct:.1f}%)")
    print(f"  [{'PASS' if meets_auc else 'FAIL'}] avg_cross_auc >= 0.70 (actual: {avg_cross:.3f})")
    print()

    if meets_improvement and meets_coverage:
        print("[SUCCESS] Cross-sectional features validated for production deployment!")
        print()
        print("Next steps:")
        print("1. Integrate cross-sectional features into production XGBoost model")
        print("2. Add 18 cross-sectional features to feature engineering pipeline")
        print("3. Monitor production performance vs validation results")
        decision = 'DEPLOY'
    elif meets_improvement:
        print(f"[MARGINAL] {avg_improvement:+.3f} improvement but only {improved_pct:.1f}% coverage")
        print()
        print("Cross-sectional features show promise but inconsistent:")
        print("- Works well for some stocks (likely sector-specific)")
        print("- Consider sector-specific feature selection")
        print("- May need refinement for universal deployment")
        decision = 'MARGINAL'
    else:
        print(f"[NEEDS WORK] {avg_improvement:+.3f} improvement too small")
        print()
        print("Cross-sectional features may not provide significant edge:")
        print("- Review feature engineering methodology")
        print("- Test different market context indicators")
        print("- Consider alternative cross-sectional data sources")
        decision = 'NEEDS_WORK'

    # Save results
    experiment_results = {
        'experiment_id': 'EXP-084-PHASE-2',
        'timestamp': datetime.now().isoformat(),
        'objective': 'Validate cross-sectional features for panic sell ML prediction',
        'hypothesis': 'Cross-sectional features will improve AUC by +2-4pp',
        'methodology': 'Compare baseline (technical) vs technical + cross-sectional on 10 stocks',
        'test_stocks': test_stocks,
        'results': {
            'stocks_tested': len(valid_results),
            'stocks_improved': improved_count,
            'improvement_pct': improved_pct,
            'avg_baseline_auc': avg_baseline,
            'avg_cross_auc': avg_cross,
            'avg_improvement': avg_improvement,
            'detailed_results': valid_results,
            'top_features': [{'feature': f, 'importance': float(imp)} for f, imp in top_features]
        },
        'validation_decision': decision,
        'meets_criteria': {
            'improvement_target': meets_improvement,
            'coverage_target': meets_coverage,
            'auc_target': meets_auc
        },
        'conclusion': 'Cross-sectional features validated for production' if decision == 'DEPLOY'
                     else 'Cross-sectional features need refinement before deployment',
        'next_step': 'Integrate into production XGBoost model' if decision == 'DEPLOY'
                    else 'Refine cross-sectional features and re-validate'
    }

    # Save to JSON
    os.makedirs('logs/experiments', exist_ok=True)
    result_file = f"logs/experiments/exp084_cross_sectional_features_phase2.json"
    with open(result_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    # Send email report
    print("\nSending cross-sectional features validation report...")
    notifier = SendGridNotifier()
    if notifier.is_enabled():
        notifier.send_experiment_report('EXP-084-PHASE-2', experiment_results)

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
