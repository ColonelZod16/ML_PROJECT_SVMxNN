import pandas as pd
import numpy as np

train_path = "train.csv"
test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Basic composite scores ----
    # Overall engagement: hobby + physical + creative
    df["engagement_total"] = (
        df["hobby_engagement_level"]
        + df["physical_activity_index"]
        + df["creative_expression_index"]
    )

    # Support + upbringing (environmental influence)
    df["support_plus_upbringing"] = (
        df["support_environment_score"] + df["upbringing_influence"]
    )

    # Internal drive: focus + consistency + altruism
    df["internal_drive"] = (
        df["focus_intensity"] + df["consistency_score"] + df["altruism_score"]
    )

    # ---- Simple interactions ----
    # Guidance x support: do they use guidance *and* have support?
    df["guidance_x_support"] = (
        df["external_guidance_usage"] * df["support_environment_score"]
    )

    # Age interactions
    df["age_x_focus"] = df["age_group"] * df["focus_intensity"]
    df["age_x_consistency"] = df["age_group"] * df["consistency_score"]

    # Activity balance
    df["activity_balance"] = (
        df["physical_activity_index"] - df["creative_expression_index"]
    )

    # Hobby x creative (creative hobbies specifically)
    df["hobby_x_creative"] = (
        df["hobby_engagement_level"] * df["creative_expression_index"]
    )

    # Focus/nonlinearity
    df["focus_sq"] = df["focus_intensity"] ** 2
    df["consistency_sq"] = df["consistency_score"] ** 2

    # Ratio-style feature (guarded)
    df["focus_per_consistency"] = df["focus_intensity"] / (
        df["consistency_score"] + 1.0
    )

    return df

# ---------- 1) Add engineered features ----------
train_fe = add_features(train)
test_fe = add_features(test)

# ---------- 2) Scale all non-ID, non-target columns ----------
feature_cols = [
    c
    for c in train_fe.columns
    if c not in ["participant_id", "personality_cluster"]
]

means = train_fe[feature_cols].mean()
stds = train_fe[feature_cols].std().replace(0, 1.0)  # avoid div by 0

train_scaled = train_fe.copy()
test_scaled = test_fe.copy()

train_scaled[feature_cols] = (train_fe[feature_cols] - means) / stds
test_scaled[feature_cols] = (test_fe[feature_cols] - means) / stds

# ---------- 3) Save out ----------
train_out = "train_preprocessed_feature_engg.csv"
test_out = "test_preprocessed_feature_engg.csv"

train_scaled.to_csv(train_out, index=False)
test_scaled.to_csv(test_out, index=False)

print("Saved:", train_out, "shape:", train_scaled.shape)
print("Saved:", test_out, "shape:", test_scaled.shape)
