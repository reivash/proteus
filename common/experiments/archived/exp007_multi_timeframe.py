"""
EXPERIMENT: EXP-007
Date: 2025-11-14
Objective: Test prediction accuracy across multiple time horizons

METHOD:
- Horizons: 1-day, 5-day, 20-day, 60-day predictions
- Model: XGBoost (best from EXP-002)
- Same features for all horizons (36 technical indicators)
- Hypothesis: Medium timeframes (5-20 days) more predictable than daily

User Observation:
"Explore different prediction timeframes. For example how good are you
at predicting change within a day vs a week vs a month vs long term"

This tests which prediction horizon is most accurate.
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.data.fetchers.yahoo_finance import YahooFinanceFetcher
from common.data.features.technical_indicators import TechnicalFeatureEngineer
from common.data.features.multi_timeframe_targets import MultiTimeframeTargetGenerator
from common.models.ml.xgboost_classifier import XGBoostStockClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_multi_timeframe_experiment(ticker="^GSPC", ticker_name="S&P 500"):
    """Run multi-timeframe prediction experiment."""

    print("=" * 70)
    print(f"PROTEUS - EXPERIMENT EXP-007: MULTI-TIMEFRAME PREDICTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    period = "5y"  # Need more data for longer horizons
    horizons = [1, 5, 20, 60]

    print(f"Configuration:")
    print(f"  Ticker: {ticker} ({ticker_name})")
    print(f"  Period: {period}")
    print(f"  Model: XGBoost (from EXP-002)")
    print(f"  Horizons: {horizons} days")
    print()

    # Step 1: Fetch Data
    print("[1/6] Fetching data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error: {str(e)}")
        return None

    # Step 2: Engineer Features
    print("\n[2/6] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data)
    print(f"      [OK] Added {len(enriched_data.columns) - len(data.columns)} technical indicators")

    # Step 3: Generate Multi-Timeframe Targets
    print("\n[3/6] Generating multi-timeframe targets...")
    target_generator = MultiTimeframeTargetGenerator(horizons=horizons)
    enriched_data = target_generator.generate_targets(enriched_data)

    print(f"      [OK] Generated targets for horizons: {horizons} days")

    # Analyze target distributions
    target_stats = target_generator.analyze_target_distribution(enriched_data)
    print(f"\n      Target Distribution:")
    for horizon_name, stats in target_stats.items():
        print(f"      {horizon_name:>6}: {stats['up_days_pct']:.1f}% up, "
              f"{stats['down_days_pct']:.1f}% down, "
              f"avg return: {stats['avg_return']:+.2f}%, "
              f"samples: {stats['samples']}")

    # Get feature columns (all except OHLCV, Date, and targets)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Date']
    exclude_cols += [col for col in enriched_data.columns if 'direction_' in col or 'return_' in col or 'volatility_' in col]
    feature_cols = [col for col in enriched_data.columns if col not in exclude_cols]

    # Clean data: replace inf with NaN, then fill with 0
    enriched_data[feature_cols] = enriched_data[feature_cols].replace([np.inf, -np.inf], np.nan)
    enriched_data[feature_cols] = enriched_data[feature_cols].fillna(0)

    print(f"\n      Using {len(feature_cols)} features for prediction")

    # Step 4: Train and Evaluate Models for Each Horizon
    print("\n[4/6] Training models for each horizon...")

    results_by_horizon = {}

    for horizon in horizons:
        print(f"\n      ---- Training {horizon}-day prediction model ----")

        # Prepare data for this horizon
        split_data = target_generator.prepare_data_for_horizon(
            enriched_data,
            horizon,
            feature_cols
        )

        X_train = split_data['X_train']
        X_test = split_data['X_test']
        y_train = split_data['y_train']
        y_test = split_data['y_test']

        print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train XGBoost model
        model = XGBoostStockClassifier()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, zero_division=0) * 100
        recall = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100

        print(f"      Accuracy: {accuracy:.2f}%")
        print(f"      Precision: {precision:.2f}%")
        print(f"      Recall: {recall:.2f}%")
        print(f"      F1-Score: {f1:.2f}%")

        # Store results
        results_by_horizon[horizon] = {
            'horizon_days': horizon,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'baseline_accuracy': target_stats[f'{horizon}d']['up_days_pct'],
            'avg_return': target_stats[f'{horizon}d']['avg_return'],
            'model': model
        }

    # Step 5: Compare Horizons
    print("\n[5/6] Comparing prediction accuracy across horizons...")
    print()
    print("=" * 70)
    print("MULTI-TIMEFRAME PREDICTION RESULTS:")
    print("=" * 70)
    print(f"{'Horizon':<12} {'Accuracy':>10} {'Baseline':>10} {'Lift':>8} {'F1-Score':>10}")
    print("-" * 70)

    for horizon in horizons:
        r = results_by_horizon[horizon]
        baseline = r['baseline_accuracy']
        lift = r['accuracy'] - baseline

        print(f"{horizon}-day{' '*(8-len(str(horizon)))} "
              f"{r['accuracy']:>9.2f}% "
              f"{baseline:>9.2f}% "
              f"{lift:>+7.2f}% "
              f"{r['f1_score']:>9.2f}%")

    print("=" * 70)

    # Find best performing horizon
    best_horizon = max(results_by_horizon.keys(), key=lambda h: results_by_horizon[h]['accuracy'])
    best_accuracy = results_by_horizon[best_horizon]['accuracy']

    print(f"\nBest Performing Horizon: {best_horizon}-day ({best_accuracy:.2f}% accuracy)")

    # Step 6: Analyze Feature Importance for Best Horizon
    print(f"\n[6/6] Analyzing feature importance for {best_horizon}-day predictions...")

    best_model = results_by_horizon[best_horizon]['model']
    # feature_importance is a DataFrame with 'feature' and 'importance' columns, already sorted
    feature_importance = best_model.feature_importance

    print(f"\n      Top 10 Most Important Features:")
    try:
        for i in range(min(10, len(feature_importance))):
            feature = feature_importance.iloc[i]['feature']
            importance = feature_importance.iloc[i]['importance']
            if feature is not None:
                print(f"      {i+1:2}. {feature:<30} {importance:.4f}")
    except Exception as e:
        print(f"      [WARN] Could not display feature importance: {e}")

    # Prepare results for saving
    experiment_results = {
        'experiment_id': 'EXP-007',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Multi-Timeframe Prediction',
        'ticker': ticker,
        'ticker_name': ticker_name,
        'period': period,
        'horizons_tested': horizons,
        'results_by_horizon': {
            f'{h}d': {
                'horizon_days': h,
                'accuracy': float(results_by_horizon[h]['accuracy']),
                'precision': float(results_by_horizon[h]['precision']),
                'recall': float(results_by_horizon[h]['recall']),
                'f1_score': float(results_by_horizon[h]['f1_score']),
                'train_samples': results_by_horizon[h]['train_samples'],
                'test_samples': results_by_horizon[h]['test_samples'],
                'baseline_accuracy': float(results_by_horizon[h]['baseline_accuracy']),
                'lift_over_baseline': float(results_by_horizon[h]['accuracy'] - results_by_horizon[h]['baseline_accuracy'])
            }
            for h in horizons
        },
        'best_horizon': {
            'days': best_horizon,
            'accuracy': float(best_accuracy),
            'lift': float(best_accuracy - results_by_horizon[best_horizon]['baseline_accuracy'])
        },
        'top_features': [
            {
                'feature': feature_importance.iloc[i]['feature'],
                'importance': float(feature_importance.iloc[i]['importance'])
            }
            for i in range(min(20, len(feature_importance)))
        ],
        'comparison_to_exp002': {
            'exp002_1day_accuracy': 58.94,
            'exp007_1day_accuracy': float(results_by_horizon[1]['accuracy']),
            'best_timeframe_accuracy': float(best_accuracy),
            'improvement_over_daily': float(best_accuracy - results_by_horizon[1]['accuracy'])
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(
        results_dir,
        f'exp007_{ticker.lower().replace("^", "")}_multi_timeframe_results.json'
    )

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    # Test on S&P 500
    print("\n" + "=" * 70)
    print(" " * 15 + "MULTI-TIMEFRAME PREDICTION TEST")
    print("=" * 70 + "\n")

    results = run_multi_timeframe_experiment("^GSPC", "S&P 500")

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)

        best = results['best_horizon']
        print(f"[+] Best Horizon: {best['days']} days")
        print(f"[+] Best Accuracy: {best['accuracy']:.2f}%")
        print(f"[+] Lift over Baseline: +{best['lift']:.2f}%")
        print()

        # Show all horizons
        print("All Horizons:")
        for h_name, h_data in results['results_by_horizon'].items():
            print(f"  {h_name:>4}: {h_data['accuracy']:.2f}% "
                  f"(+{h_data['lift_over_baseline']:.2f}% vs baseline)")

        # Interpretation
        best_days = best['days']
        if best_days == 1:
            print("\n[INSIGHT] Daily predictions are most accurate.")
            print("          Short-term noise is predictable with our features.")
        elif best_days == 5:
            print("\n[INSIGHT] Weekly predictions are most accurate.")
            print("          Optimal for swing trading strategies.")
        elif best_days == 20:
            print("\n[INSIGHT] Monthly predictions are most accurate.")
            print("          Optimal for position trading strategies.")
        elif best_days >= 60:
            print("\n[INSIGHT] Quarterly+ predictions are most accurate.")
            print("          Optimal for long-term investing strategies.")

        if best['accuracy'] >= 65:
            print("\n[SUCCESS] Strong predictive signal found!")
        elif best['accuracy'] >= 60:
            print("\n[GOOD] Moderate predictive signal found.")
        elif best['accuracy'] >= 55:
            print("\n[MIXED] Weak but tradeable signal.")
        else:
            print("\n[CAUTION] Limited predictive power across all timeframes.")

        print("=" * 70)
    else:
        print("\n[FAILED] Experiment failed.")
