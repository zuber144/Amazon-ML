#!/usr/bin/env python3
"""
run_pipeline.py
===============
Entry point for the Business Entity Resolution pipeline.

Usage:
    python run_pipeline.py --mode train
    python run_pipeline.py --mode validate
    python run_pipeline.py --mode predict
    python run_pipeline.py --mode all

    python run_pipeline.py --mode train --model gradient_boosting
    python run_pipeline.py --mode train --blocking exact_name token_name tfidf_name
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pipeline import run_train, run_validate, run_predict, run_all


# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure logging based on config settings."""
    handlers = []
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler(sys.stdout))
    if config.LOG_TO_FILE:
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "validate", "predict", "all"],
        default="all",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
        default=None,
        help="Model type to use. Overrides config.MODEL_TYPE.",
    )
    parser.add_argument(
        "--blocking",
        nargs="+",
        choices=["exact_name", "token_name", "tfidf_name", "token_address", "country_aware"],
        default=None,
        help="Blocking strategies to use. Overrides config.BLOCKING_STRATEGIES.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (skips tuning).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("run_pipeline")

    logger.info("=" * 60)
    logger.info("Business Entity Resolution Pipeline")
    logger.info("Mode: %s", args.mode)
    if args.model:
        config.MODEL_TYPE = args.model
        logger.info("Model type override: %s", args.model)
    if args.blocking:
        config.BLOCKING_STRATEGIES = args.blocking
        logger.info("Blocking strategies override: %s", args.blocking)
    if args.threshold is not None:
        config.DEFAULT_THRESHOLD = args.threshold
        logger.info("Threshold override: %.4f", args.threshold)
    logger.info("=" * 60)

    t0 = time.time()

    try:
        if args.mode == "train":
            run_train(blocking_strategies=args.blocking, model_type=args.model)
        elif args.mode == "validate":
            run_validate(blocking_strategies=args.blocking)
        elif args.mode == "predict":
            run_predict(blocking_strategies=args.blocking)
        elif args.mode == "all":
            run_all(blocking_strategies=args.blocking, model_type=args.model)
        else:
            logger.error("Unknown mode: %s", args.mode)
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Please ensure all dataset files are in the dataset/ directory.")
        sys.exit(1)
    except ValueError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)

    elapsed = time.time() - t0
    logger.info("Pipeline finished in %.1f seconds (%.1f minutes).", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
