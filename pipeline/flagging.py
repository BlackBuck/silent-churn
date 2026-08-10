import pandas as pd
import numpy as np

def evaluate_risk_flag(decay_series, min_consecutive_negative=3, baseline_ratio_threshold=0.5):
    """
    Evaluates risk flags based on decay series.
    Returns: DataFrame with 'is_flagged' column.
    """
    df = pd.DataFrame(decay_series).sort_values("week_start").reset_index(drop=True)

    if len(df) == 0:
        return df

    df["is_flagged"] = False

    # We need to find consecutive negative trends
    df["is_negative_trend"] = df["trend_slope"] < 0
    df["consecutive_negative"] = df["is_negative_trend"].groupby(
        (~df["is_negative_trend"]).cumsum()
    ).cumsum()

    for i in range(len(df)):
        consecutive_neg = df.loc[i, "consecutive_negative"]
        baseline = df.loc[i, "baseline"]
        current_activity = df.loc[i, "event_count"]

        # Risk condition:
        if baseline > 0:
            ratio = current_activity / baseline
            if consecutive_neg >= min_consecutive_negative and ratio < baseline_ratio_threshold:
                df.loc[i, "is_flagged"] = True
        else:
            # If baseline is 0 and they're active, not risk.
            # If baseline is 0 and they stay 0, they never really started.
            df.loc[i, "is_flagged"] = True

    df = df.drop(columns=["is_negative_trend", "consecutive_negative"])
    return df


def run_flagging(config_path="config/config.yaml"):
    import yaml
    import os
    from pipeline.logger import logger

    logger.info("Starting flagging...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    decay_scores_path = config["data"]["decay_scores_path"]
    flags_path = config["data"]["flags_path"]

    min_consecutive_negative = config["flagging"]["min_consecutive_negative_trend_weeks"]
    baseline_ratio_threshold = config["flagging"]["baseline_ratio_threshold"]

    if not os.path.exists(decay_scores_path):
        logger.error("Decay scores data not found.")
        return

    df_scores = pd.read_csv(decay_scores_path)

    results = []

    for user_id, group in df_scores.groupby("user_id"):
        flagged = evaluate_risk_flag(group, min_consecutive_negative=min_consecutive_negative, baseline_ratio_threshold=baseline_ratio_threshold)
        results.append(flagged)

    df_flags = pd.concat(results, ignore_index=True)
    df_flags.to_csv(flags_path, index=False)

    logger.info(f"Flagging complete. Saved flags for {len(df_flags)} records.")

if __name__ == "__main__":
    from pipeline.logger import setup_logger
    setup_logger()
    run_flagging()
