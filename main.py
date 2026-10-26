"""Command-line entry point for the multi-agent crop management system.

Examples
  python main.py --demo                       # runs 6 scenarios (valid, invalid, revision, ...)
  python main.py --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203
  python main.py --question "How should I irrigate banana?"
  python main.py --demo --no-llm              # run without Ollama (rule-based fallbacks)
"""
import argparse
import json

from crop_agents.config import FEATURES, OUTPUT_DIR
from crop_agents.data import DatasetError
from crop_agents.orchestrator import build_system

VALID = {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82, "ph": 6.5, "rainfall": 203}

DEMO_SCENARIOS = [
    ("1. Valid measurements -> full pipeline",
     {"measurements": VALID, "question": "What irrigation and fertilizer should I plan?"}, False),
    ("2. INVALID input (pH 17, humidity 150, missing rainfall) -> safe response",
     {"measurements": {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 150, "ph": 17}, "question": ""}, False),
    ("3. OUT-OF-SCOPE question -> unsupported route",
     {"measurements": None, "question": "Who won the cricket match yesterday?"}, False),
    ("4. Knowledge-only question -> RAG route (no prediction)",
     {"measurements": None, "question": "How should I irrigate banana and what nutrients does it need?"}, False),
    ("5. Valid input with a FORCED review-triggered revision (--demo-revision behaviour)",
     {"measurements": VALID, "question": ""}, True),
    ("6. Extreme but plausible values (outside training range) -> warnings",
     {"measurements": {"N": 20, "P": 20, "K": 20, "temperature": 47, "humidity": 12, "ph": 9.8, "rainfall": 600}, "question": ""}, False),
]


def run_one(system, user_input, demo_revision=False, show_trace=False):
    state = system.run(user_input, demo_revision=demo_revision)
    print(state["report_text"])
    if show_trace:
        print("\nTRACE (shared-state log)")
        for line in state["trace"]:
            print("  " + line)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run the built-in demonstration scenarios")
    for f in FEATURES:
        ap.add_argument(f"--{f}", type=str, default=None)
    ap.add_argument("--question", default="", help="optional farmer question")
    ap.add_argument("--demo-revision", action="store_true", help="force the first draft to fail review once")
    ap.add_argument("--no-llm", action="store_true", help="do not call Ollama (rule-based fallbacks)")
    ap.add_argument("--rebuild-kb", action="store_true", help="rebuild the Chroma vector store")
    ap.add_argument("--trace", action="store_true", help="print the agent trace")
    args = ap.parse_args()

    try:
        system = build_system(force_offline=args.no_llm, rebuild_kb=args.rebuild_kb)
    except DatasetError as exc:
        raise SystemExit(f"\n{exc}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    states = []
    if args.demo:
        for title, ui, force_rev in DEMO_SCENARIOS:
            print("\n\n" + "#" * 100 + f"\n# {title}\n" + "#" * 100)
            states.append(run_one(system, ui, force_rev, show_trace=True))
    else:
        raw = {f: getattr(args, f) for f in FEATURES if getattr(args, f) is not None}
        states.append(run_one(system, {"measurements": raw or None, "question": args.question},
                              args.demo_revision, show_trace=args.trace))

    (OUTPUT_DIR / "last_run.json").write_text(json.dumps(states, indent=2, default=str))
    (OUTPUT_DIR / "last_report.txt").write_text("\n\n".join(s["report_text"] for s in states))
    print(f"\n[saved] outputs/last_report.txt and outputs/last_run.json (full shared state)")


if __name__ == "__main__":
    main()
