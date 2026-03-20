# ─────────────────────────────────────────────────────────
# main.py — Entry point for the MIMIC-IV Multi-Agent Pipeline
#
# Usage:
#   python main.py                    # full pipeline, all papers
#   python main.py --paper 1          # Paper 1 only
#   python main.py --paper 3          # Paper 3 only
#   python main.py --paper 1 3        # both papers explicitly
#   python main.py --sample 5000      # dev mode with row limit
#   python main.py --paper 3 --sample 10000  # Paper 3, 10K rows
# ─────────────────────────────────────────────────────────

from agents.orchestrator import Orchestrator, parse_args

if __name__ == "__main__":
    args = parse_args()
    orchestrator = Orchestrator()
    orchestrator.run(
        papers=args.paper,
        limit=args.sample
    )