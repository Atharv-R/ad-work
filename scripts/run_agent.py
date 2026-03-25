# scripts/run_agent.py

"""
Run the Ad-Work optimization agent.

Usage:
    uv run python scripts/run_agent.py

Windows:
    .venv\\Scripts\\python scripts\\run_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    from adwork.pipeline.daily_loop import run_daily_optimization

    print("=" * 60)
    print("  Ad-Work Optimization Agent")
    print("=" * 60)
    print()

    result = run_daily_optimization()

    # Print agent log
    print("Agent Log:")
    for entry in result.get("agent_log", []):
        print(f"  {entry}")
    print()

    # Print errors if any
    errors = result.get("errors", [])
    if errors:
        print("⚠️  Errors:")
        for err in errors:
            print(f"  - {err}")
        print()

    # Print daily summary
    print("-" * 60)
    print("DAILY SUMMARY")
    print("-" * 60)
    print(result.get("daily_summary", "No summary generated."))
    print()

    # Print recommendations
    recs = result.get("final_recommendations", [])
    if recs:
        print("-" * 60)
        print(f"RECOMMENDATIONS ({len(recs)})")
        print("-" * 60)
        for rec in sorted(recs, key=lambda r: r.get("priority", 99)):
            priority = rec.get("priority", "?")
            confidence = rec.get("confidence", "?").upper()
            print(f"\n  #{priority} [{confidence}] {rec.get('campaign_name', 'N/A')}")
            print(f"  Action: {rec.get('action_summary', 'N/A')}")
            print(f"  Reason: {rec.get('reasoning', 'N/A')}")

    print()
    print("=" * 60)
    print("  Agent run complete. Results saved to data/processed/agent_output.json")
    print("=" * 60)


if __name__ == "__main__":
    main()