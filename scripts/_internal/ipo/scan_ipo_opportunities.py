"""
Quick Start: IPO Niche Market Screener

Scans for recent IPOs dominating small, fast-growing niche markets.
Uses Y Combinator principles to identify breakout candidates.

Usage:
    python scan_ipo_opportunities.py [--strict|--balanced|--exploratory]

Examples:
    python scan_ipo_opportunities.py                    # Balanced (default)
    python scan_ipo_opportunities.py --strict           # Highest quality only
    python scan_ipo_opportunities.py --exploratory      # More candidates
"""

import sys
import argparse
from datetime import datetime

sys.path.insert(0, 'src')
from common.trading.ipo_niche_screener import IPONicheScreener


def run_scan(mode='balanced'):
    """
    Run IPO screening with preset configurations.

    Args:
        mode: 'strict', 'balanced', or 'exploratory'
    """
    print("=" * 80)
    print("IPO NICHE MARKET OPPORTUNITY SCANNER")
    print("Y Combinator Principles: Dominate Small -> Grow Massive")
    print("=" * 80)
    print(f"\nMode: {mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration presets
    configs = {
        'strict': {
            'max_ipo_age_months': 12,
            'min_yc_score': 70.0,
            'min_market_cap_b': 2.0,
            'max_market_cap_b': 15.0,
            'description': 'Strictest criteria - Only exceptional recent IPOs'
        },
        'balanced': {
            'max_ipo_age_months': 24,
            'min_yc_score': 60.0,
            'min_market_cap_b': 1.0,
            'max_market_cap_b': 30.0,
            'description': 'Balanced criteria - Strong candidates with growth room'
        },
        'exploratory': {
            'max_ipo_age_months': 36,
            'min_yc_score': 50.0,
            'min_market_cap_b': 0.5,
            'max_market_cap_b': 100.0,
            'description': 'Exploratory criteria - Cast wider net, more candidates'
        }
    }

    config = configs[mode]
    print(f"Configuration: {config['description']}")
    print("-" * 80)

    # Initialize screener
    screener = IPONicheScreener(
        max_ipo_age_months=config['max_ipo_age_months'],
        min_yc_score=config['min_yc_score'],
        min_market_cap_b=config['min_market_cap_b'],
        max_market_cap_b=config['max_market_cap_b']
    )

    # Run screen
    candidates = screener.screen_ipo_universe()

    # Generate report
    if not candidates.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"results/ipo_scan_{mode}_{timestamp}.txt"

        report = screener.generate_report(candidates, filename=report_filename)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Candidates Found: {len(candidates)}")
        print(f"Average YC Score: {candidates['yc_score'].mean():.1f}/100")
        print(f"Top Candidate: {candidates['ticker'].iloc[0]} (Score: {candidates['yc_score'].iloc[0]:.1f})")
        print(f"\nReport saved: {report_filename}")

        print("\n" + "=" * 80)
        print("TOP CANDIDATES")
        print("=" * 80)
        top_candidates = candidates[['ticker', 'company_name', 'yc_score',
                                     'category', 'market_cap_b', 'revenue_growth_yoy']].head(10)
        print(top_candidates.to_string(index=False))

        print("\n" + "=" * 80)
        print("INVESTMENT THESIS (Top 3)")
        print("=" * 80)
        for idx, row in candidates.head(3).iterrows():
            print(f"\n{row['ticker']} - {row['company_name']}")
            print(f"  YC Score: {row['yc_score']:.1f}/100")
            print(f"  Category: {row['category']} (TAM: ${row['estimated_tam_billions']:.1f}B)")
            print(f"  IPO: {row['ipo_date'].strftime('%Y-%m-%d')} ({row['months_since_ipo']:.0f}mo ago)")
            print(f"  Market Cap: ${row['market_cap_b']:.2f}B")
            print(f"  Revenue Growth: {(row['revenue_growth_yoy'] or 0)*100:.1f}% YoY")
            print(f"  Gross Margin: {(row.get('gross_margin', 0) or 0)*100:.1f}%")
            print(f"  Signals: Niche={row['is_niche']}, Emerging={row['is_emerging']}, Inflection={row['is_at_inflection']}")

    else:
        print("\n[INFO] No candidates found with current criteria")
        print(f"Try running with --exploratory mode for more results")

    print("\n" + "=" * 80)
    print("Scan Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scan for IPO niche market opportunities using Y Combinator principles'
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='balanced',
        choices=['strict', 'balanced', 'exploratory'],
        help='Scanning mode (default: balanced)'
    )

    # Alternative flags
    parser.add_argument('--strict', action='store_const', const='strict', dest='mode_flag',
                       help='Use strict criteria (highest quality only)')
    parser.add_argument('--balanced', action='store_const', const='balanced', dest='mode_flag',
                       help='Use balanced criteria (recommended)')
    parser.add_argument('--exploratory', action='store_const', const='exploratory', dest='mode_flag',
                       help='Use exploratory criteria (more candidates)')

    args = parser.parse_args()

    # Use flag if provided, otherwise use positional argument
    mode = args.mode_flag if args.mode_flag else args.mode

    run_scan(mode)
