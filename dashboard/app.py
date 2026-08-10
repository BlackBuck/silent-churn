import streamlit as st
import pandas as pd
import yaml
import os

# --- Layout and config ---
st.set_page_config(page_title="Silent Churn Leading Indicator", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    users = pd.read_csv(config["data"]["users_path"])
    flags = pd.read_csv(config["data"]["flags_path"])
    backtest = pd.read_csv(config["data"]["backtest_results_path"])

    users["cohort_month"] = pd.to_datetime(users["cohort_month"]).dt.strftime('%Y-%m')
    flags["week_start"] = pd.to_datetime(flags["week_start"])

    return config, users, flags, backtest

try:
    config, df_users, df_flags, df_backtest = load_data()
except Exception as e:
    st.error(f"Error loading data. Have you run the pipeline? ({e})")
    st.stop()

# --- Merge latest state for At-Risk and Cost of Waiting ---
# Find the latest week for each user
latest_flags = df_flags.sort_values("week_start").groupby("user_id").tail(1)
latest_state = pd.merge(latest_flags, df_users, on="user_id", how="inner")

# Currently at risk are those flagged in their latest week AND not yet churned
# If churn_date is filled, they are already churned.
latest_state["churn_date"] = pd.to_datetime(latest_state["churn_date"])
latest_state["is_churned"] = latest_state["churn_date"].notna()
currently_at_risk = latest_state[(latest_state["is_flagged"] == True) & (latest_state["is_churned"] == False)]

# Calculate weeks since flagged
# Find first time they were flagged
first_flags = df_flags[df_flags["is_flagged"] == True].groupby("user_id")["week_start"].min().reset_index()
first_flags = first_flags.rename(columns={"week_start": "first_flagged_date"})

currently_at_risk = pd.merge(currently_at_risk, first_flags, on="user_id", how="left")
max_date = df_flags["week_start"].max()
currently_at_risk["weeks_since_flagged"] = ((max_date - currently_at_risk["first_flagged_date"]).dt.days / 7).round(0).astype(int)

# --- Sidebar Filters ---
st.sidebar.title("Filters")
selected_cohorts = st.sidebar.multiselect("Cohort Month", options=sorted(df_users["cohort_month"].dropna().unique()), default=None)
plan_tiers_available = "plan_tier" in df_users.columns
if plan_tiers_available:
    selected_plans = st.sidebar.multiselect("Plan Tier", options=df_users["plan_tier"].dropna().unique(), default=None)

# Apply filters
filtered_users = df_users.copy()
filtered_risk = currently_at_risk.copy()

if selected_cohorts:
    filtered_users = filtered_users[filtered_users["cohort_month"].isin(selected_cohorts)]
    filtered_risk = filtered_risk[filtered_risk["cohort_month"].isin(selected_cohorts)]

if plan_tiers_available and selected_plans:
    filtered_users = filtered_users[filtered_users["plan_tier"].isin(selected_plans)]
    filtered_risk = filtered_risk[filtered_risk["plan_tier"].isin(selected_plans)]

# --- Tab Navigation ---
tab1, tab2, tab3 = st.tabs(["Cohort Overview", "At-Risk Users", "Cost of Waiting"])

# --- Tab 1: Cohort Overview ---
with tab1:
    st.header("Cohort Overview")

    # Calculate churn rate and lead time per cohort
    df_eval = pd.merge(filtered_users, first_flags, on="user_id", how="left")
    df_eval["is_churned"] = df_eval["churn_date"].notna()
    df_eval["is_flagged_ever"] = df_eval["first_flagged_date"].notna()
    df_eval["lead_time_days"] = (df_eval["churn_date"] - df_eval["first_flagged_date"]).dt.days

    cohort_stats = df_eval.groupby("cohort_month").agg(
        total_users=("user_id", "count"),
        churned_users=("is_churned", "sum"),
        avg_lead_time_days=("lead_time_days", lambda x: x[x >= 0].mean())
    ).reset_index()

    cohort_stats["churn_rate"] = (cohort_stats["churned_users"] / cohort_stats["total_users"]).map('{:.2%}'.format)
    cohort_stats["avg_lead_time_days"] = cohort_stats["avg_lead_time_days"].fillna(0).round(1)

    st.dataframe(cohort_stats, use_container_width=True)

    # Show pipeline backtest overall results
    st.subheader("Global Backtest Performance")
    backtest_metrics = dict(zip(df_backtest["metric"], df_backtest["value"]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall (Flagged Churns)", f"{backtest_metrics.get('recall', 0):.1%}")
    col2.metric("Precision (Accurate Flags)", f"{backtest_metrics.get('precision', 0):.1%}")
    col3.metric("Median Lead Time (Days)", f"{backtest_metrics.get('median_lead_time_days', 0):.1f}")


# --- Tab 2: At-Risk Users ---
with tab2:
    st.header("Currently At-Risk Users")
    st.write(f"Total at-risk users currently active: {len(filtered_risk)}")

    if len(filtered_risk) > 0:
        display_cols = ["user_id", "cohort_month", "half_life_weeks", "trend_slope", "weeks_since_flagged", "event_count"]
        if plan_tiers_available:
            display_cols.insert(2, "plan_tier")

        view_df = filtered_risk[display_cols].copy()
        view_df = view_df.sort_values("half_life_weeks") # Sort by severity (lowest half life)

        st.dataframe(view_df, use_container_width=True)
    else:
        st.info("No active users are currently flagged as at-risk.")

# --- Tab 3: Cost of Waiting ---
with tab3:
    st.header("Cost of Inaction / Waiting")

    per_user_value = config.get("business", {}).get("per_user_value", 100)

    st.write(f"Assumed Value per User: **${per_user_value:,.2f}**")

    if len(filtered_risk) > 0:
        cost_df = filtered_risk.groupby("weeks_since_flagged").agg(
            users_at_risk=("user_id", "count")
        ).reset_index()

        cost_df["value_at_risk"] = cost_df["users_at_risk"] * per_user_value

        st.bar_chart(data=cost_df, x="weeks_since_flagged", y="value_at_risk", use_container_width=True)

        total_risk = cost_df["value_at_risk"].sum()
        st.metric("Total Value Currently at Risk", f"${total_risk:,.2f}")
    else:
        st.info("No cost at risk currently.")
