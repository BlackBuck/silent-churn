import pandas as pd
import numpy as np
import yaml
import os
from pipeline.logger import logger
from datetime import timedelta, datetime

def generate_synthetic_data(config):
    """Generate synthetic data if raw data is not present."""
    logger.info("Generating synthetic data...")
    seed = config.get("synthetic_data", {}).get("seed", 42)
    np.random.seed(seed)

    num_users = config.get("synthetic_data", {}).get("num_users", 2000)
    num_events = config.get("synthetic_data", {}).get("num_events", 50000)

    # Generate users
    users_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for i in range(num_users):
        user_id = f"user_{i}"
        signup_offset = np.random.randint(0, 300)
        signup_date = start_date + timedelta(days=signup_offset)
        plan_tier = np.random.choice(["basic", "pro", "enterprise"], p=[0.6, 0.3, 0.1])
        users_data.append({
            "user_id": user_id,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "plan_tier": plan_tier
        })

    df_users = pd.DataFrame(users_data)

    # Generate events
    events_data = []
    event_types = ["login", "feature_use", "export", "setting_change"]

    user_ids = df_users["user_id"].values
    signup_dates = pd.to_datetime(df_users["signup_date"]).values

    # We want some users to churn, so we'll give them an activity window
    user_lifespans = np.random.gamma(shape=2.0, scale=30.0, size=num_users)
    user_lifespans = np.clip(user_lifespans, 1, 365)

    # Generate random events
    random_user_idx = np.random.randint(0, num_users, size=num_events)
    for idx in random_user_idx:
        user_id = user_ids[idx]
        signup = signup_dates[idx]
        lifespan = user_lifespans[idx]

        # Event time is between signup and signup + lifespan
        event_offset = np.random.randint(0, int(lifespan) + 1)
        event_time = signup + np.timedelta64(event_offset, 'D') + np.timedelta64(np.random.randint(0, 24), 'h')

        # Make sure event time is not in future
        if pd.Timestamp(event_time) > pd.Timestamp(end_date):
            continue

        events_data.append({
            "user_id": user_id,
            "event_timestamp": pd.Timestamp(event_time).strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": np.random.choice(event_types),
            "event_weight": 1.0
        })

    df_events = pd.DataFrame(events_data)

    os.makedirs("data", exist_ok=True)
    raw_users_path = config["data"]["raw_users_path"]
    raw_events_path = config["data"]["raw_events_path"]

    df_users.to_csv(raw_users_path, index=False)
    df_events.to_csv(raw_events_path, index=False)

    logger.info(f"Generated {len(df_users)} users and {len(df_events)} events.")

def run_ingestion(config_path="config/config.yaml"):
    logger.info("Starting ingestion...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_users_path = config["data"]["raw_users_path"]
    raw_events_path = config["data"]["raw_events_path"]

    if not os.path.exists(raw_users_path) or not os.path.exists(raw_events_path):
        generate_synthetic_data(config)

    # Read raw data
    df_users = pd.read_csv(raw_users_path)
    df_events = pd.read_csv(raw_events_path)

    logger.info(f"Read {len(df_users)} users and {len(df_events)} events from raw data.")

    # Process events
    df_events["event_timestamp"] = pd.to_datetime(df_events["event_timestamp"])

    # Deduplicate events
    original_events_count = len(df_events)
    df_events = df_events.drop_duplicates(subset=["user_id", "event_timestamp", "event_type"])
    logger.info(f"Dropped {original_events_count - len(df_events)} duplicate events.")

    # Process users
    df_users["signup_date"] = pd.to_datetime(df_users["signup_date"]).dt.date
    df_users["cohort_month"] = pd.to_datetime(df_users["signup_date"]).dt.to_period("M").dt.start_time.dt.date

    # Determine churn date
    # Churn date = last activity + threshold
    threshold_days = config["pipeline"]["inactivity_churn_threshold_days"]
    last_activity = df_events.groupby("user_id")["event_timestamp"].max().reset_index()
    last_activity["last_activity_date"] = last_activity["event_timestamp"].dt.date

    df_users = df_users.merge(last_activity[["user_id", "last_activity_date"]], on="user_id", how="left")

    # Determine if churned based on threshold to current max date in dataset
    max_date = df_events["event_timestamp"].max().date()

    df_users["churn_date"] = None

    def get_churn_date(row):
        if pd.isna(row["last_activity_date"]):
            return row["signup_date"] # Churned immediately

        days_inactive = (max_date - row["last_activity_date"]).days
        if days_inactive >= threshold_days:
            return row["last_activity_date"] + timedelta(days=threshold_days)
        return None

    df_users["churn_date"] = df_users.apply(get_churn_date, axis=1)

    # Validate churn_date > signup_date
    df_users["churn_date"] = pd.to_datetime(df_users["churn_date"]).dt.date

    invalid_churn = (df_users["churn_date"].notna()) & (df_users["churn_date"] < df_users["signup_date"])
    if invalid_churn.any():
        logger.warning(f"Found {invalid_churn.sum()} users with churn_date < signup_date. Fixing.")
        df_users.loc[invalid_churn, "churn_date"] = df_users.loc[invalid_churn, "signup_date"]

    # Save normalized data
    activity_events_path = config["data"]["activity_events_path"]
    users_path = config["data"]["users_path"]

    df_events.to_csv(activity_events_path, index=False)
    # Drop intermediate columns
    df_users = df_users.drop(columns=["last_activity_date"])
    df_users.to_csv(users_path, index=False)

    logger.info(f"Ingestion complete. Saved {len(df_users)} users and {len(df_events)} events.")

if __name__ == "__main__":
    from pipeline.logger import setup_logger
    setup_logger()
    run_ingestion()
