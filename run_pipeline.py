import sys
import time
from pipeline.logger import setup_logger
from pipeline.ingestion import run_ingestion
from pipeline.aggregation import run_aggregation
from pipeline.decay import run_decay_scoring
from pipeline.flagging import run_flagging
from pipeline.backtest import run_backtest

def main():
    logger = setup_logger()
    logger.info("Starting End-to-End Pipeline")
    start_time = time.time()

    try:
        run_ingestion()
        run_aggregation()
        run_decay_scoring()
        run_flagging()
        run_backtest()

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
