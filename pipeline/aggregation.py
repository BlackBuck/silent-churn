import pandas as pd
import yaml
import os
from pipeline.logger import logger

def run_aggregation(config_path="config/config.yaml"):
    logger.info("Starting weekly aggregation...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    activity_events_path = config["data"]["activity_events_path"]
    users_path = config["data"]["users_path"]
    weekly_activity_path = config["data"]["weekly_activity_path"]

    if not os.path.exists(activity_events_path) or not os.path.exists(users_path):
        logger.error("Normalized data not found. Run ingestion first.")
        return

    df_events = pd.read_csv(activity_events_path)
    df_users = pd.read_csv(users_path)

    df_events["event_timestamp"] = pd.to_datetime(df_events["event_timestamp"])
    df_events["event_date"] = df_events["event_timestamp"].dt.date

    # Calculate week start (Monday)
    df_events["week_start"] = df_events["event_timestamp"].dt.to_period('W').dt.start_time.dt.date

    # Aggregate events per user per week
    weekly_agg = df_events.groupby(["user_id", "week_start"]).agg(
        event_count=("event_type", "count"),
        distinct_features=("event_type", "nunique"),
        last_activity_in_week=("event_date", "max")
    ).reset_index()

    # We need to zero-fill for weeks without activity.
    # We zero-fill from their signup week up to either their churn week or the max week in dataset

    df_users["signup_date"] = pd.to_datetime(df_users["signup_date"])
    df_users["churn_date"] = pd.to_datetime(df_users["churn_date"])

    max_date = df_events["event_timestamp"].max().date()
    # Get global start of max week
    max_week_start = pd.Timestamp(max_date).to_period('W').start_time.date()

    all_user_weeks = []

    for _, user_row in df_users.iterrows():
        user_id = user_row["user_id"]
        signup_week_start = user_row["signup_date"].to_period('W').start_time.date()

        if pd.notna(user_row["churn_date"]):
            end_week_start = user_row["churn_date"].to_period('W').start_time.date()
        else:
            end_week_start = max_week_start

        # Ensure we don't go past max_week_start globally
        if end_week_start > max_week_start:
            end_week_start = max_week_start

        # Create date range of weeks
        if signup_week_start <= end_week_start:
            weeks = pd.date_range(start=signup_week_start, end=end_week_start, freq='W-MON').date
            if len(weeks) == 0 or weeks[0] > signup_week_start:
                # pandas date_range with W-MON might not include the start date if it's a Monday itself depending on bounds
                weeks = pd.date_range(start=signup_week_start, end=end_week_start + pd.Timedelta(days=6), freq='W-MON').date

            # Filter strictly by bounds
            weeks = [w for w in weeks if w >= signup_week_start and w <= end_week_start]

            # If for some reason weeks is empty (e.g. signup and churn in same week)
            if not weeks:
                weeks = [signup_week_start]

            for w in weeks:
                all_user_weeks.append({"user_id": user_id, "week_start": w})

    df_all_weeks = pd.DataFrame(all_user_weeks)

    # Merge with actual aggregated data
    df_weekly = pd.merge(df_all_weeks, weekly_agg, on=["user_id", "week_start"], how="left")

    # Fill NAs
    df_weekly["event_count"] = df_weekly["event_count"].fillna(0)
    df_weekly["distinct_features"] = df_weekly["distinct_features"].fillna(0)

    # Calculate days since last activity globally per user (cumulative)
    # To do this, we track the last active date and propagate it forward
    df_weekly = df_weekly.sort_values(["user_id", "week_start"])

    # We only have last_activity_in_week if they were active
    df_weekly['last_active_date'] = df_weekly['last_activity_in_week'].copy()

    # Forward fill the last active date for each user
    df_weekly['last_active_date'] = df_weekly.groupby('user_id')['last_active_date'].ffill()

    # If they were never active before a week, we use signup date
    df_users_signup = df_users[["user_id", "signup_date"]].copy()
    df_users_signup["signup_date"] = df_users_signup["signup_date"].dt.date
    df_weekly = df_weekly.merge(df_users_signup, on="user_id", how="left")

    df_weekly['last_active_date'] = df_weekly['last_active_date'].fillna(df_weekly['signup_date'])

    # Days since last activity (measured from end of the week, which is week_start + 6 days)
    df_weekly['week_end'] = pd.to_datetime(df_weekly['week_start']) + pd.Timedelta(days=6)
    df_weekly['days_since_last_activity'] = (df_weekly['week_end'] - pd.to_datetime(df_weekly['last_active_date'])).dt.days

    # Cleanup columns
    df_weekly = df_weekly.drop(columns=["last_activity_in_week", "last_active_date", "signup_date", "week_end"])

    df_weekly.to_csv(weekly_activity_path, index=False)
    logger.info(f"Aggregation complete. Saved {len(df_weekly)} weekly records for {df_weekly['user_id'].nunique()} users.")

if __name__ == "__main__":
    from pipeline.logger import setup_logger
    setup_logger()
    run_aggregation()
