"""Convenience entry point.

Rebuilds the analysis end-to-end. For selective runs use:
    python -m src.pressure.pipeline --no-figs
    python -m src.pressure.pipeline --from-wp
"""

from src.pressure.pipeline import main

if __name__ == "__main__":
    main()
