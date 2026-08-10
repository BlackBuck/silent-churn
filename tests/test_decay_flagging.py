import pandas as pd
import numpy as np
from pipeline.decay import compute_user_decay
from pipeline.flagging import evaluate_risk_flag

def test_decay_normal_case():
    # A user starting high and gradually decaying
    data = [
        {"user_id": "u1", "week_start": "2024-01-01", "event_count": 10},
        {"user_id": "u1", "week_start": "2024-01-08", "event_count": 10},
        {"user_id": "u1", "week_start": "2024-01-15", "event_count": 9},
        {"user_id": "u1", "week_start": "2024-01-22", "event_count": 9}, # Baseline = 9.5
        {"user_id": "u1", "week_start": "2024-01-29", "event_count": 8},
        {"user_id": "u1", "week_start": "2024-02-05", "event_count": 7},
        {"user_id": "u1", "week_start": "2024-02-12", "event_count": 6},
        {"user_id": "u1", "week_start": "2024-02-19", "event_count": 4}, # < 50% of baseline (4.75)
    ]
    df = pd.DataFrame(data)
    result = compute_user_decay(df, window=4, baseline_weeks=4)

    assert result["baseline"].iloc[0] == 9.5
    # The trend should be negative at the end
    assert result["trend_slope"].iloc[-1] < 0
    # Half-life should drop to 0 when it actually hits the threshold
    assert result["half_life_weeks"].iloc[-1] == 0

def test_decay_all_zero_activity():
    data = [
        {"user_id": "u2", "week_start": "2024-01-01", "event_count": 0},
        {"user_id": "u2", "week_start": "2024-01-08", "event_count": 0},
        {"user_id": "u2", "week_start": "2024-01-15", "event_count": 0},
    ]
    df = pd.DataFrame(data)
    result = compute_user_decay(df)

    assert result["baseline"].iloc[0] == 0
    assert result["trend_slope"].iloc[-1] == 0.0
    assert result["half_life_weeks"].iloc[-1] == 0.0

def test_decay_single_data_point():
    data = [
        {"user_id": "u3", "week_start": "2024-01-01", "event_count": 5},
    ]
    df = pd.DataFrame(data)
    result = compute_user_decay(df)

    assert result["baseline"].iloc[0] == 5.0
    assert result["trend_slope"].iloc[0] == 0.0
    assert result["half_life_weeks"].iloc[0] == np.inf

def test_decay_never_churns():
    data = [{"user_id": "u4", "week_start": f"2024-01-0{i+1}", "event_count": 10} for i in range(8)]
    df = pd.DataFrame(data)
    result = compute_user_decay(df, window=4)

    assert result["baseline"].iloc[0] == 10.0
    assert result["trend_slope"].iloc[-1] == 0.0
    assert result["half_life_weeks"].iloc[-1] == np.inf

def test_flagging_logic():
    data = [
        {"user_id": "u1", "week_start": "2024-01-01", "event_count": 10, "baseline": 10, "trend_slope": 0},
        {"user_id": "u1", "week_start": "2024-01-08", "event_count": 8, "baseline": 10, "trend_slope": -1}, # 1
        {"user_id": "u1", "week_start": "2024-01-15", "event_count": 6, "baseline": 10, "trend_slope": -2}, # 2
        {"user_id": "u1", "week_start": "2024-01-22", "event_count": 4, "baseline": 10, "trend_slope": -2}, # 3, ratio < 0.5 -> Flag!
    ]
    df = pd.DataFrame(data)
    result = evaluate_risk_flag(df, min_consecutive_negative=3, baseline_ratio_threshold=0.5)

    assert not result["is_flagged"].iloc[0]
    assert not result["is_flagged"].iloc[1]
    assert not result["is_flagged"].iloc[2]
    assert result["is_flagged"].iloc[3]
