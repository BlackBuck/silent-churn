# Methodology Summary

## Approach
This project detects silent churn by converting raw event logs into a time-series decay model. Instead of waiting for a user to cross a hard inactivity threshold (e.g., 60 days of zero activity), the system monitors a rolling 8-week window.

1. **Baseline Setting:** We establish a user's normal engagement level using their first 4 weeks of activity.
2. **Trend Analysis:** We use linear regression to compute the slope of their activity over a rolling window.
3. **Flagging Logic:** A user is flagged as "at risk" if their activity drops below 50% of their baseline AND they show a negative trend for at least 3 consecutive weeks.

## Backtested Results
Based on a synthetic dataset of 2,000 users and ~50,000 events:
- **Mean Lead Time:** 55.0 days
- **Recall (Flagged Churns):** 100%
- **Precision (Accurate Flags):** 85.4%

*(Note: These numbers reflect the synthetic generated dataset. Actual results will vary with production data.)*

## Business Recommendation
**Proactive Intervention based on the 55-day window:**
The system provides on average nearly two months of warning before a user officially churns according to the 60-day inactivity rule. We recommend routing flagged users into an automated re-engagement sequence immediately upon flagging, as the "Cost of Waiting" view demonstrates that delaying action leaves significant revenue unaddressed while the user is still theoretically reachable.