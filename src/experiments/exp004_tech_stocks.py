"""
EXPERIMENT: EXP-004
Date: 2025-11-14
Objective: Test XGBoost on tech stocks (AI sector focus)

METHOD:
- Approach: XGBoost (best performer from EXP-002)
- Data: NVIDIA (NVDA) - leading AI/GPU company
- Features: Technical indicators + market context (VIX, QQQ)
- Rationale: User noted tech/AI sector profitability
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.fetchers.market_context import MarketContextFetcher
from src.data.features.technical_indicators import TechnicalFeatureEngineer
from src.models.ml.xgboost_classifier import XGBoostStockClassifier


def run_tech_stock_experiment(ticker="NVDA", ticker_name="NVIDIA"):
    """Run XGBoost on a tech stock."""

    print("=" * 70)
    print(f"PROTEUS - EXPERIMENT EXP-004: TECH STOCKS ({ticker_name})")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    period = "2y"

    print(f"Configuration:")
    print(f"  Ticker: {ticker} ({ticker_name})")
    print(f"  Period: {period}")
    print(f"  Model: XGBoost (best from EXP-002)")
    print(f"  Context: VIX + QQQ (tech sector)")
    print()

    # Step 1: Fetch Stock Data
    print("[1/6] Fetching stock data...")
    fetcher = YahooFinanceFetcher()

    try:
        data = fetcher.fetch_stock_data(ticker, period=period)
        print(f"      [OK] Fetched {len(data)} rows")
        print(f"      [OK] Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"      [FAIL] Error fetching data: {str(e)}")
        return None

    # Step 2: Fetch Market Context
    print("\n[2/6] Fetching market context...")
    context_fetcher = MarketContextFetcher()
    context_data = context_fetcher.fetch_market_context(period=period)

    # Merge context
    data_with_context = context_fetcher.merge_context_with_stock(
        data,
        context_data,
        features_to_include=['VIX', 'QQQ']  # VIX for volatility, QQQ for tech sector
    )
    print(f"      [OK] Added market context features")

    # Step 3: Feature Engineering
    print("\n[3/6] Engineering features...")
    engineer = TechnicalFeatureEngineer(fillna=True)
    enriched_data = engineer.engineer_features(data_with_context)
    new_features = enriched_data.shape[1] - data.shape[1]
    print(f"      [OK] Created {new_features} new features")
    print(f"      [OK] Total features: {enriched_data.shape[1]}")

    # Step 4: Prepare Target
    print("\n[4/6] Preparing target...")
    enriched_data['Actual_Direction'] = (enriched_data['Close'].shift(-1) > enriched_data['Close']).astype(int)
    enriched_data = enriched_data[:-1].dropna()
    print(f"      [OK] Valid samples: {len(enriched_data)}")

    # Step 5: Train-Test Split
    print("\n[5/6] Training model...")
    classifier = XGBoostStockClassifier()

    X, y = classifier.prepare_data(enriched_data)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"      [OK] Train set: {len(X_train)} samples")
    print(f"      [OK] Test set: {len(X_test)} samples")

    classifier.fit(X_train, y_train, validation_split=0.2, verbose=False)
    print(f"      [OK] Model trained")

    # Step 6: Evaluate
    print("\n[6/6] Evaluating model...")
    results = classifier.evaluate(X_test, y_test, verbose=False)
    print(f"      [OK] Accuracy: {results['accuracy']:.2f}%")

    # Display Results
    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Stock: {ticker} ({ticker_name})")
    print(f"Directional Accuracy:  {results['accuracy']:.2f}%")
    print(f"Precision:             {results['precision']:.3f}")
    print(f"Recall:                {results['recall']:.3f}")
    print(f"F1 Score:              {results['f1_score']:.3f}")
    print()
    print("Top 10 Most Important Features:")
    top_features = classifier.get_top_features(10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:20s} {row['importance']:.4f}")
    print("=" * 70)

    # Prepare results
    experiment_results = {
        'experiment_id': 'EXP-004',
        'ticker': ticker,
        'ticker_name': ticker_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'XGBoostClassifier',
        'period': period,
        'metrics': results,
        'num_features': int(X.shape[1]),
        'top_features': top_features.head(10).to_dict('records'),
        'comparison': {
            'sp500_baseline': 53.49,
            'sp500_xgboost': 58.94,
            'tech_stock_accuracy': float(results['accuracy'])
        }
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'experiments')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'exp004_{ticker.lower()}_results.json')

    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return experiment_results


if __name__ == "__main__":
    # Test on NVIDIA (AI/GPU leader)
    results = run_tech_stock_experiment("NVDA", "NVIDIA Corporation")

    if results:
        print(f"\n[SUCCESS] Tech stock experiment complete!")
        print(f"[SUCCESS] {results['ticker_name']} Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"[SUCCESS] S&P 500 XGBoost: 58.94%")
        diff = results['metrics']['accuracy'] - 58.94
        if diff > 0:
            print(f"[SUCCESS] Tech stock performs +{diff:.2f}% better!")
        else:
            print(f"[INFO] Tech stock performs {diff:.2f}% vs S&P 500")
    else:
        print("\n[FAILED] Tech stock experiment failed!")
