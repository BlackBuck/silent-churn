# Silent Churn — Leading Indicator Dashboard

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Adjust parameters in `config/config.yaml`.

## Running the Pipeline
Run the full data pipeline end-to-end to generate or ingest data, aggregate it, score decay, flag users, and backtest results.
```bash
python run_pipeline.py
```
This single command handles everything, ensuring idempotent execution, logs, and outputs to the `/data` directory.

## Launching the Dashboard
Start the interactive Streamlit dashboard to view insights:
```bash
streamlit run dashboard/app.py
```

## Architecture
- `config/` holds the YAML configuration.
- `pipeline/` contains data engineering and scoring logic:
  - `ingestion.py`: Normalizes data, generates synthetic data if needed.
  - `aggregation.py`: Summarizes events weekly per user.
  - `decay.py`: Computes rolling trends and half-lives.
  - `flagging.py`: Identifies at-risk users based on configurable thresholds.
  - `backtest.py`: Computes lead-time and accuracy metrics.
- `dashboard/` contains the presentation layer.
- `logs/` contains time-stamped execution logs.
