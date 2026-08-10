import pandas as pd
import yaml
import os
import json
from pipeline.logger import logger

def run_backtest(config_path="config/config.yaml"):
    logger.info("Starting backtesting...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    flags_path = config["data"]["flags_path"]
    users_path = config["data"]["users_path"]
    backtest_results_path = config["data"]["backtest_results_path"]

    if not os.path.exists(flags_path) or not os.path.exists(users_path):
        logger.error("Required data for backtesting not found.")
        return

    df_flags = pd.read_csv(flags_path)
    df_users = pd.read_csv(users_path)

    df_flags["week_start"] = pd.to_datetime(df_flags["week_start"])
    df_users["churn_date"] = pd.to_datetime(df_users["churn_date"])

    # 1. For users with a known churn_date, compute first flagged date.
    first_flags = df_flags[df_flags["is_flagged"] == True].groupby("user_id")["week_start"].min().reset_index()
    first_flags = first_flags.rename(columns={"week_start": "first_flagged_date"})

    # Merge with users
    eval_df = pd.merge(df_users, first_flags, on="user_id", how="left")

    # 2. Compute lead time for churned, flagged users
    eval_df["is_churned"] = eval_df["churn_date"].notna()
    eval_df["is_flagged_ever"] = eval_df["first_flagged_date"].notna()

    # Lead time = churn_date - first_flagged_date
    eval_df["lead_time_days"] = (eval_df["churn_date"] - eval_df["first_flagged_date"]).dt.days

    # Filter to only positive lead times (if flagged after churn, it's not lead time)
    # Actually, a negative lead time is a false positive / late flag, keep it to see stats.
    # But for "lead time" we usually average over > 0. Let's keep all for now.
    true_positives = eval_df[(eval_df["is_churned"] == True) & (eval_df["is_flagged_ever"] == True)]

    # 3. Compute metrics
    # recall = true_positives / all_churned
    # precision = true_positives / all_flagged

    all_churned = eval_df["is_churned"].sum()
    all_flagged = eval_df["is_flagged_ever"].sum()
    tp_count = len(true_positives)

    recall = tp_count / all_churned if all_churned > 0 else 0
    precision = tp_count / all_flagged if all_flagged > 0 else 0

    # Only valid lead times (flagged before churn)
    valid_lead_times = true_positives[true_positives["lead_time_days"] >= 0]["lead_time_days"]

    mean_lead_time = valid_lead_times.mean() if len(valid_lead_times) > 0 else 0
    median_lead_time = valid_lead_times.median() if len(valid_lead_times) > 0 else 0

    results = {
        "metric": ["total_users", "total_churned", "total_flagged", "true_positives", "recall", "precision", "mean_lead_time_days", "median_lead_time_days"],
        "value": [len(eval_df), all_churned, all_flagged, tp_count, recall, precision, mean_lead_time, median_lead_time]
    }

    df_results = pd.DataFrame(results)
    df_results.to_csv(backtest_results_path, index=False)

    logger.info("Backtesting Complete.")
    logger.info(f"Recall: {recall:.2%} | Precision: {precision:.2%}")
    logger.info(f"Mean Lead Time: {mean_lead_time:.1f} days")

if __name__ == "__main__":
    from pipeline.logger import setup_logger
    setup_logger()
    run_backtest()
