"""Run all reporting steps: tables + statistics + framing."""
import sys
from ablation.reporting import tables, statistics, framing

def main():
    print("=" * 70)
    print("STEP 1: tables")
    print("=" * 70)
    tables.main()
    print()
    print("=" * 70)
    print("STEP 2: statistics")
    print("=" * 70)
    statistics.main()
    print()
    print("=" * 70)
    print("STEP 3: framing")
    print("=" * 70)
    framing.main()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())