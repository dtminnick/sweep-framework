
# cli/run_sweep.py

import argparse
from sweep_framework.registry import SweepRegistry
from sweep_framework.analysis.sweep_analyzer import SweepAnalyzer
from sweep_framework.analysis.report_builder import ReportBuilder

def main():
    parser = argparse.ArgumentParser(description="Run sweep diagnostics and export summary.")
    parser.add_argument("--sweep", type=str, required=True, help="Name of the sweep to analyze")
    parser.add_argument("--metric", type=str, default="macro_f1", help="Metric to rank by")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top configs to show")
    args = parser.parse_args()

    registry = SweepRegistry()
    sweep = registry.get_sweep(args.sweep)
    if not sweep:
        print(f"Sweep '{args.sweep}' not found.")
        return

    analyzer = SweepAnalyzer(registry)
    report = ReportBuilder(analyzer)

    summary = report.build_summary(metric=args.metric, top_k=args.top_k)
    print(f"\nTop {args.top_k} configs ranked by {args.metric}:\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

