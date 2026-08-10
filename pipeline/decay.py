import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_user_decay(weekly_series, window=8, baseline_weeks=4):
    """
    Computes baseline, rolling trend slope, and half-life for a user's activity.
    Returns: DataFrame with trend and half-life columns appended.
    """
    df = pd.DataFrame(weekly_series).sort_values("week_start").reset_index(drop=True)

    if len(df) == 0:
        return df

    # Baseline: average of first min(len, baseline_weeks)
    baseline = df.head(baseline_weeks)["event_count"].mean()
    df["baseline"] = baseline

    df["trend_slope"] = np.nan
    df["half_life_weeks"] = np.nan

    for i in range(len(df)):
        start_idx = max(0, i - window + 1)
        window_data = df.iloc[start_idx:i+1]

        # Calculate trend slope
        if len(window_data) > 1:
            X = np.arange(len(window_data)).reshape(-1, 1)
            y = window_data["event_count"].values

            # Simple linear regression for slope
            if np.all(y == y[0]):
                slope = 0.0 # Constant activity
            else:
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]

            df.loc[i, "trend_slope"] = slope

            # Calculate half-life
            # How many weeks until dropping below 50% of baseline?
            if slope < -1e-6 and baseline > 0: # significant negative slope
                current_value = window_data["event_count"].iloc[-1]
                target_value = 0.5 * baseline

                if current_value <= target_value:
                    df.loc[i, "half_life_weeks"] = 0
                else:
                    weeks_to_target = (target_value - current_value) / slope
                    df.loc[i, "half_life_weeks"] = max(0, weeks_to_target)
            elif baseline == 0:
                df.loc[i, "half_life_weeks"] = 0 # No baseline, effectively half-life is 0
            else:
                df.loc[i, "half_life_weeks"] = np.inf # Not decaying
        else:
            df.loc[i, "trend_slope"] = 0.0
            if baseline > 0:
                df.loc[i, "half_life_weeks"] = np.inf
            else:
                df.loc[i, "half_life_weeks"] = 0.0

    return df

def run_decay_scoring(config_path="config/config.yaml"):
    import yaml
    import os
    from pipeline.logger import logger

    logger.info("Starting decay scoring...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    weekly_activity_path = config["data"]["weekly_activity_path"]
    decay_scores_path = config["data"]["decay_scores_path"]

    window = config["pipeline"]["rolling_window_weeks"]
    baseline_weeks = config["pipeline"]["baseline_weeks"]

    if not os.path.exists(weekly_activity_path):
        logger.error("Weekly activity data not found.")
        return

    df_weekly = pd.read_csv(weekly_activity_path)

    # Process each user
    results = []

    for user_id, group in df_weekly.groupby("user_id"):
        scored = compute_user_decay(group, window=window, baseline_weeks=baseline_weeks)
        results.append(scored)

    df_scores = pd.concat(results, ignore_index=True)
    df_scores.to_csv(decay_scores_path, index=False)

    logger.info(f"Decay scoring complete. Saved scores for {len(df_scores)} records.")

if __name__ == "__main__":
    from pipeline.logger import setup_logger
    setup_logger()
    run_decay_scoring()
