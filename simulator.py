import logging
import numpy as np
import pandas as pd
from config import RANDOM_STATE

logger = logging.getLogger(__name__)


def generate_users(n=50000, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)

    age = rng.normal(35, 12, n).clip(18, 70).astype(int)
    is_male = rng.binomial(1, 0.52, n)
    days_since_signup = rng.exponential(180, n).clip(1, 1000).astype(int)
    past_purchases = rng.poisson(3, n)
    total_spent = (past_purchases * rng.lognormal(3.5, 1.0, n)).clip(0, 10000).round(2)
    device_mobile = rng.binomial(1, 0.6, n)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.2, 0.35, 0.25, 0.2])
    channel = rng.choice(["organic", "paid", "email", "referral"], n, p=[0.35, 0.3, 0.2, 0.15])
    is_subscriber = rng.binomial(1, 0.15, n)
    sessions_last_week = rng.poisson(2, n)

    users = pd.DataFrame({
        "user_id": np.arange(n),
        "age": age,
        "is_male": is_male,
        "days_since_signup": days_since_signup,
        "past_purchases": past_purchases,
        "total_spent": total_spent,
        "device_mobile": device_mobile,
        "region": region,
        "channel": channel,
        "is_subscriber": is_subscriber,
        "sessions_last_week": sessions_last_week,
    })

    return users


def assign_treatment(users, assignment="random", seed=RANDOM_STATE):
    rng = np.random.RandomState(seed + 1)
    n = len(users)

    if assignment == "random":
        users["treatment"] = rng.binomial(1, 0.5, n)
    elif assignment == "biased":
        propensity = 0.3 + 0.3 * (users["past_purchases"] > 3).astype(float)
        propensity += 0.1 * users["is_subscriber"]
        propensity = propensity.clip(0.1, 0.9)
        users["treatment"] = rng.binomial(1, propensity)
        users["true_propensity"] = propensity
    else:
        raise ValueError(f"Unknown assignment: {assignment}")

    logger.info(f"  Treatment rate: {users['treatment'].mean():.3f}")
    return users


def generate_outcomes(users, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed + 2)
    n = len(users)

    base_rate = (
        0.04
        + 0.015 * (users["past_purchases"] > 2).astype(float)
        + 0.02 * users["is_subscriber"]
        + 0.005 * (users["sessions_last_week"] > 3).astype(float)
        - 0.01 * (users["days_since_signup"] > 365).astype(float)
        + 0.008 * (users["channel"] == "email").astype(float)
        + rng.normal(0, 0.005, n)
    ).clip(0.01, 0.25)

    treatment_effect = np.zeros(n)

    treatment_effect += 0.015
    treatment_effect += 0.025 * (users["past_purchases"] >= 3).astype(float)
    treatment_effect += 0.02 * users["is_subscriber"]
    treatment_effect -= 0.01 * (users["age"] < 25).astype(float)
    treatment_effect += 0.015 * (users["channel"] == "email").astype(float)
    treatment_effect -= 0.005 * users["device_mobile"]
    treatment_effect += 0.01 * (users["region"] == "south").astype(float)
    treatment_effect -= 0.01 * (users["days_since_signup"] < 30).astype(float)

    treatment_effect = treatment_effect.clip(-0.02, 0.10)
    users["true_treatment_effect"] = treatment_effect

    conversion_prob = base_rate + users["treatment"] * treatment_effect
    conversion_prob = conversion_prob.clip(0.005, 0.5)
    users["converted"] = rng.binomial(1, conversion_prob)

    base_revenue = rng.lognormal(3.0, 0.8, n)
    revenue_uplift = 1.0 + users["treatment"] * (
        0.1 + 0.15 * (users["past_purchases"] >= 3).astype(float)
    )
    users["revenue"] = (users["converted"] * base_revenue * revenue_uplift).round(2)

    pages_viewed = rng.poisson(3, n) + users["treatment"] * rng.poisson(1, n)
    users["pages_viewed"] = pages_viewed.clip(0, 30)

    time_on_site = rng.exponential(120, n) + users["treatment"] * rng.exponential(20, n)
    users["time_on_site"] = time_on_site.clip(5, 1200).round(0)

    logger.info(f"  Overall conversion: {users['converted'].mean():.4f}")
    logger.info(f"  Control conversion: {users.loc[users['treatment']==0, 'converted'].mean():.4f}")
    logger.info(f"  Treatment conversion: {users.loc[users['treatment']==1, 'converted'].mean():.4f}")

    return users


def generate_experiment(n=50000, assignment="random", seed=RANDOM_STATE):
    logger.info(f"Generating experiment data (n={n}, assignment={assignment})...")
    users = generate_users(n, seed)
    users = assign_treatment(users, assignment, seed)
    users = generate_outcomes(users, seed)
    return users


def generate_sequential_data(n_per_day=500, n_days=30, seed=RANDOM_STATE):
    logger.info(f"Generating sequential data ({n_days} days, {n_per_day}/day)...")
    rng = np.random.RandomState(seed + 10)

    all_days = []
    cumulative_control = []
    cumulative_treatment = []
    p_values = []

    n_ctrl_conversions = 0
    n_treat_conversions = 0
    n_ctrl_total = 0
    n_treat_total = 0

    for day in range(n_days):
        novelty_factor = max(0.3, 1.0 - day * 0.02)

        for _ in range(n_per_day):
            is_treatment = rng.binomial(1, 0.5)
            base_p = 0.05
            effect = 0.015 * novelty_factor if is_treatment else 0
            converted = rng.binomial(1, base_p + effect)

            if is_treatment:
                n_treat_total += 1
                n_treat_conversions += converted
            else:
                n_ctrl_total += 1
                n_ctrl_conversions += converted

        ctrl_rate = n_ctrl_conversions / max(n_ctrl_total, 1)
        treat_rate = n_treat_conversions / max(n_treat_total, 1)

        from scipy.stats import norm
        pooled_rate = (n_ctrl_conversions + n_treat_conversions) / max(n_ctrl_total + n_treat_total, 1)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/max(n_ctrl_total,1) + 1/max(n_treat_total,1)))
        z = (treat_rate - ctrl_rate) / max(se, 1e-10)
        p = 2 * (1 - norm.cdf(abs(z)))

        all_days.append(day + 1)
        cumulative_control.append(ctrl_rate)
        cumulative_treatment.append(treat_rate)
        p_values.append(p)

    return pd.DataFrame({
        "day": all_days,
        "control_rate": cumulative_control,
        "treatment_rate": cumulative_treatment,
        "p_value": p_values,
        "n_control": [n_ctrl_total] * n_days,
        "n_treatment": [n_treat_total] * n_days,
    })
