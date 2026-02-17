import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2_contingency, mannwhitneyu
from config import ASSETS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)
ASSETS_DIR.mkdir(exist_ok=True)
sns.set_style("whitegrid")


def check_balance(df):
    logger.info("Checking covariate balance...")

    numeric_cols = ["age", "days_since_signup", "past_purchases", "total_spent",
                    "sessions_last_week"]
    cat_cols = ["is_male", "device_mobile", "region", "channel", "is_subscriber"]

    balance = []
    ctrl = df[df["treatment"] == 0]
    treat = df[df["treatment"] == 1]

    for col in numeric_cols:
        ctrl_mean = ctrl[col].mean()
        treat_mean = treat[col].mean()
        pooled_std = df[col].std()
        smd = (treat_mean - ctrl_mean) / max(pooled_std, 1e-10)
        t_stat, p_val = stats.ttest_ind(ctrl[col], treat[col])
        balance.append({
            "feature": col, "control_mean": round(ctrl_mean, 2),
            "treatment_mean": round(treat_mean, 2),
            "smd": round(smd, 4), "p_value": round(p_val, 4),
        })

    for col in cat_cols:
        if df[col].nunique() <= 2:
            ctrl_mean = ctrl[col].mean()
            treat_mean = treat[col].mean()
            smd = (treat_mean - ctrl_mean) / max(df[col].std(), 1e-10)
            _, p_val = stats.ttest_ind(ctrl[col], treat[col])
            balance.append({
                "feature": col, "control_mean": round(ctrl_mean, 3),
                "treatment_mean": round(treat_mean, 3),
                "smd": round(smd, 4), "p_value": round(p_val, 4),
            })
        else:
            contingency = pd.crosstab(df["treatment"], df[col])
            _, p_val, _, _ = chi2_contingency(contingency)
            balance.append({
                "feature": col, "control_mean": "—", "treatment_mean": "—",
                "smd": "—", "p_value": round(p_val, 4),
            })

    result = pd.DataFrame(balance)
    imbalanced = result[
        result["p_value"].apply(lambda x: isinstance(x, float) and x < 0.05)
    ]
    if len(imbalanced) > 0:
        logger.info(f"  Warning: {len(imbalanced)} features have p < 0.05")
    else:
        logger.info("  All covariates balanced (p > 0.05)")

    return result


def plot_balance(balance_df):
    numeric_balance = balance_df[balance_df["smd"] != "—"].copy()
    numeric_balance["smd"] = numeric_balance["smd"].astype(float).abs()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(numeric_balance["feature"], numeric_balance["smd"], color="#2c3e50")
    ax.axvline(0.1, color="#e74c3c", linestyle="--", alpha=0.7, label="SMD = 0.1 threshold")
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate Balance Check")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "balance_check.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved balance_check.png")


def frequentist_test(df, metric="converted"):
    logger.info(f"Frequentist test on '{metric}'...")

    ctrl = df[df["treatment"] == 0]
    treat = df[df["treatment"] == 1]

    n_c, n_t = len(ctrl), len(treat)
    p_c = ctrl[metric].mean()
    p_t = treat[metric].mean()
    lift = (p_t - p_c) / max(p_c, 1e-10)

    pooled = (ctrl[metric].sum() + treat[metric].sum()) / (n_c + n_t)
    se = np.sqrt(pooled * (1 - pooled) * (1/n_c + 1/n_t))
    z = (p_t - p_c) / max(se, 1e-10)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    ci_se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    ci_lower = (p_t - p_c) - 1.96 * ci_se
    ci_upper = (p_t - p_c) + 1.96 * ci_se

    result = {
        "metric": metric,
        "control_rate": p_c, "treatment_rate": p_t,
        "absolute_diff": p_t - p_c, "relative_lift": lift,
        "z_stat": z, "p_value": p_value,
        "ci_95_lower": ci_lower, "ci_95_upper": ci_upper,
        "n_control": n_c, "n_treatment": n_t,
        "significant": p_value < 0.05,
    }

    logger.info(f"  Control: {p_c:.4f}, Treatment: {p_t:.4f}")
    logger.info(f"  Lift: {lift:.2%}, p={p_value:.4f}")
    logger.info(f"  95% CI for diff: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return result


def revenue_test(df):
    logger.info("Revenue test (Mann-Whitney U)...")

    ctrl_rev = df.loc[df["treatment"] == 0, "revenue"]
    treat_rev = df.loc[df["treatment"] == 1, "revenue"]

    t_stat, t_pval = stats.ttest_ind(ctrl_rev, treat_rev, equal_var=False)
    u_stat, u_pval = mannwhitneyu(ctrl_rev, treat_rev, alternative="two-sided")

    ctrl_mean = ctrl_rev.mean()
    treat_mean = treat_rev.mean()
    lift = (treat_mean - ctrl_mean) / max(ctrl_mean, 1e-10)

    n_c, n_t = len(ctrl_rev), len(treat_rev)
    se = np.sqrt(ctrl_rev.var() / n_c + treat_rev.var() / n_t)
    ci_lower = (treat_mean - ctrl_mean) - 1.96 * se
    ci_upper = (treat_mean - ctrl_mean) + 1.96 * se

    result = {
        "metric": "revenue",
        "control_mean": ctrl_mean, "treatment_mean": treat_mean,
        "absolute_diff": treat_mean - ctrl_mean, "relative_lift": lift,
        "t_stat": t_stat, "t_pval": t_pval,
        "u_stat": u_stat, "u_pval": u_pval,
        "ci_95_lower": ci_lower, "ci_95_upper": ci_upper,
        "control_median": ctrl_rev.median(), "treatment_median": treat_rev.median(),
    }

    logger.info(f"  Control: ${ctrl_mean:.2f}, Treatment: ${treat_mean:.2f}")
    logger.info(f"  Lift: {lift:.2%}")
    logger.info(f"  Welch's t: p={t_pval:.4f}, Mann-Whitney: p={u_pval:.4f}")

    return result


def bootstrap_test(df, metric="converted", n_boot=10000, seed=RANDOM_STATE):
    logger.info(f"Bootstrap test on '{metric}' ({n_boot} iterations)...")
    rng = np.random.RandomState(seed)

    ctrl = df.loc[df["treatment"] == 0, metric].values
    treat = df.loc[df["treatment"] == 1, metric].values
    observed_diff = treat.mean() - ctrl.mean()

    boot_diffs = np.zeros(n_boot)
    for i in range(n_boot):
        boot_ctrl = rng.choice(ctrl, size=len(ctrl), replace=True)
        boot_treat = rng.choice(treat, size=len(treat), replace=True)
        boot_diffs[i] = boot_treat.mean() - boot_ctrl.mean()

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_value = np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)
    p_value = min(2 * p_value, 1.0)

    result = {
        "observed_diff": observed_diff,
        "bootstrap_mean": boot_diffs.mean(),
        "bootstrap_std": boot_diffs.std(),
        "ci_95_lower": ci_lower, "ci_95_upper": ci_upper,
        "p_value": p_value,
    }

    logger.info(f"  Observed diff: {observed_diff:.4f}")
    logger.info(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return result, boot_diffs


def bayesian_test(df, metric="converted", n_samples=50000, seed=RANDOM_STATE):
    logger.info(f"Bayesian A/B test on '{metric}'...")
    rng = np.random.RandomState(seed)

    ctrl = df.loc[df["treatment"] == 0, metric]
    treat = df.loc[df["treatment"] == 1, metric]

    alpha_c = 1 + ctrl.sum()
    beta_c = 1 + len(ctrl) - ctrl.sum()
    alpha_t = 1 + treat.sum()
    beta_t = 1 + len(treat) - treat.sum()

    samples_c = rng.beta(alpha_c, beta_c, n_samples)
    samples_t = rng.beta(alpha_t, beta_t, n_samples)
    diff_samples = samples_t - samples_c

    prob_t_better = (diff_samples > 0).mean()
    expected_loss_c = np.maximum(diff_samples, 0).mean()
    expected_loss_t = np.maximum(-diff_samples, 0).mean()

    hdi_lower = np.percentile(diff_samples, 2.5)
    hdi_upper = np.percentile(diff_samples, 97.5)

    result = {
        "prob_treatment_better": prob_t_better,
        "expected_loss_control": expected_loss_c,
        "expected_loss_treatment": expected_loss_t,
        "posterior_mean_diff": diff_samples.mean(),
        "hdi_95_lower": hdi_lower, "hdi_95_upper": hdi_upper,
    }

    logger.info(f"  P(treatment > control): {prob_t_better:.4f}")
    logger.info(f"  Expected loss if choosing control: {expected_loss_c:.5f}")
    logger.info(f"  95% HDI: [{hdi_lower:.4f}, {hdi_upper:.4f}]")

    return result, diff_samples


def power_analysis(baseline_rate=0.05, mde=0.01, alpha=0.05, power=0.8):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + mde
    n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1))
          + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / mde) ** 2
    return int(np.ceil(n))


def power_curve(baseline_rate=0.05, alpha=0.05, power=0.8):
    mdes = np.arange(0.005, 0.05, 0.001)
    sample_sizes = [power_analysis(baseline_rate, mde, alpha, power) for mde in mdes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mdes * 100, np.array(sample_sizes) / 1000, color="#2c3e50", linewidth=2)
    ax.set_xlabel("Minimum Detectable Effect (percentage points)")
    ax.set_ylabel("Required Sample Size per Group (×1000)")
    ax.set_title(f"Power Analysis (baseline={baseline_rate:.1%}, α={alpha}, power={power})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "power_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved power_curve.png")


def segmented_effects(df, segments=None):
    logger.info("Computing segmented treatment effects...")

    if segments is None:
        segments = {
            "age_group": pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 100],
                                labels=["18-25", "26-35", "36-45", "46-55", "55+"]),
            "tenure": pd.cut(df["days_since_signup"], bins=[0, 30, 90, 180, 365, 2000],
                             labels=["<30d", "30-90d", "90-180d", "180-365d", ">1y"]),
            "purchase_history": pd.cut(df["past_purchases"], bins=[-1, 0, 2, 5, 100],
                                       labels=["none", "1-2", "3-5", "5+"]),
            "device": df["device_mobile"].map({0: "desktop", 1: "mobile"}),
            "region": df["region"],
            "channel": df["channel"],
            "subscriber": df["is_subscriber"].map({0: "no", 1: "yes"}),
        }

    all_results = []
    for seg_name, seg_values in segments.items():
        for val in seg_values.dropna().unique():
            mask = seg_values == val
            subset = df[mask]
            ctrl = subset[subset["treatment"] == 0]
            treat = subset[subset["treatment"] == 1]

            if len(ctrl) < 50 or len(treat) < 50:
                continue

            p_c = ctrl["converted"].mean()
            p_t = treat["converted"].mean()
            diff = p_t - p_c
            se = np.sqrt(p_c * (1 - p_c) / len(ctrl) + p_t * (1 - p_t) / len(treat))

            z = diff / max(se, 1e-10)
            p_val = 2 * (1 - norm.cdf(abs(z)))

            all_results.append({
                "segment": seg_name, "value": val,
                "n_control": len(ctrl), "n_treatment": len(treat),
                "control_rate": p_c, "treatment_rate": p_t,
                "lift": diff, "se": se,
                "ci_lower": diff - 1.96 * se, "ci_upper": diff + 1.96 * se,
                "p_value": p_val,
                "significant": p_val < 0.05,
            })

    return pd.DataFrame(all_results)


def plot_segmented_effects(seg_df, top_segments=None):
    if top_segments is None:
        top_segments = ["purchase_history", "channel", "age_group", "subscriber"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, seg_name in zip(axes.flat, top_segments):
        seg_data = seg_df[seg_df["segment"] == seg_name].sort_values("lift")
        y_pos = range(len(seg_data))
        ax.barh(y_pos, seg_data["lift"], color="#2c3e50", alpha=0.7)
        ax.errorbar(seg_data["lift"], y_pos,
                     xerr=1.96 * seg_data["se"], fmt="none", color="#e74c3c", capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(seg_data["value"])
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Treatment Effect by {seg_name}")
        ax.set_xlabel("Conversion Lift (pp)")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "segmented_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved segmented_effects.png")


def multiple_testing_correction(seg_df, method="bonferroni"):
    logger.info(f"Applying {method} correction...")
    n_tests = len(seg_df)

    if method == "bonferroni":
        seg_df["p_corrected"] = (seg_df["p_value"] * n_tests).clip(upper=1.0)
    elif method == "bh":
        ranked = seg_df["p_value"].rank()
        seg_df["p_corrected"] = (seg_df["p_value"] * n_tests / ranked).clip(upper=1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    seg_df["significant_corrected"] = seg_df["p_corrected"] < 0.05

    n_before = seg_df["significant"].sum()
    n_after = seg_df["significant_corrected"].sum()
    logger.info(f"  Significant before correction: {n_before}")
    logger.info(f"  Significant after correction: {n_after}")

    return seg_df


def plot_results_summary(freq_result, boot_result, bayes_result, boot_diffs, bayes_diffs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    metrics = ["converted", "revenue"]
    lifts = [freq_result["relative_lift"]]
    ci_lower = [freq_result["ci_95_lower"]]
    ci_upper = [freq_result["ci_95_upper"]]
    ax.barh([0], [freq_result["absolute_diff"]], color="#2c3e50", alpha=0.7)
    ax.errorbar([freq_result["absolute_diff"]], [0],
                xerr=[[freq_result["absolute_diff"] - freq_result["ci_95_lower"]],
                       [freq_result["ci_95_upper"] - freq_result["absolute_diff"]]],
                fmt="o", color="#e74c3c", capsize=5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(["conversion"])
    ax.set_title(f"Treatment Effect (p={freq_result['p_value']:.4f})")
    ax.set_xlabel("Absolute Difference")

    ax = axes[0, 1]
    ax.hist(boot_diffs, bins=80, color="#2c3e50", alpha=0.7, density=True, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(boot_result["ci_95_lower"], color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.axvline(boot_result["ci_95_upper"], color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.axvline(boot_result["observed_diff"], color="#2c3e50", linewidth=2)
    ax.set_title("Bootstrap Distribution of Treatment Effect")
    ax.set_xlabel("Difference in Conversion Rate")

    ax = axes[1, 0]
    ax.hist(bayes_diffs, bins=80, color="#2c3e50", alpha=0.7, density=True, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(bayes_result["hdi_95_lower"], color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.axvline(bayes_result["hdi_95_upper"], color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.set_title(f"Bayesian Posterior (P(treat>ctrl) = {bayes_result['prob_treatment_better']:.3f})")
    ax.set_xlabel("Posterior Difference")

    ax = axes[1, 1]
    labels = ["Frequentist\n(z-test)", "Bootstrap", "Bayesian"]
    diffs = [freq_result["absolute_diff"], boot_result["observed_diff"], bayes_result["posterior_mean_diff"]]
    ci_lo = [freq_result["ci_95_lower"], boot_result["ci_95_lower"], bayes_result["hdi_95_lower"]]
    ci_hi = [freq_result["ci_95_upper"], boot_result["ci_95_upper"], bayes_result["hdi_95_upper"]]
    colors = ["#2c3e50", "#e74c3c", "#27ae60"]

    for i, (label, d, lo, hi, color) in enumerate(zip(labels, diffs, ci_lo, ci_hi, colors)):
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                     capsize=5, markersize=8, linewidth=2)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Comparison of Methods")
    ax.set_xlabel("Treatment Effect (conversion rate diff)")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "test_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved test_results.png")


def plot_sequential_peeking(seq_df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(seq_df["day"], seq_df["control_rate"] * 100, label="Control",
            color="#2c3e50", linewidth=2)
    ax.plot(seq_df["day"], seq_df["treatment_rate"] * 100, label="Treatment",
            color="#e74c3c", linewidth=2)
    ax.set_ylabel("Cumulative Conversion Rate (%)")
    ax.set_title("Sequential Results — Peeking Problem")
    ax.legend()

    ax = axes[1]
    ax.plot(seq_df["day"], seq_df["p_value"], color="#2c3e50", linewidth=2, marker="o", markersize=3)
    ax.axhline(0.05, color="#e74c3c", linestyle="--", label="α = 0.05")

    alpha_spending = 0.05 * np.sqrt(seq_df["day"] / seq_df["day"].max())
    ax.plot(seq_df["day"], alpha_spending, color="#27ae60", linestyle="--",
            label="O'Brien-Fleming spending", linewidth=2)

    ax.set_xlabel("Day")
    ax.set_ylabel("p-value")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "sequential_peeking.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sequential_peeking.png")


def simpsons_paradox_check(df):
    logger.info("Checking for Simpson's paradox...")

    overall_ctrl = df[df["treatment"] == 0]["converted"].mean()
    overall_treat = df[df["treatment"] == 1]["converted"].mean()
    overall_direction = "treatment" if overall_treat > overall_ctrl else "control"

    reversed_segments = []
    for seg_name, seg_col in [("channel", "channel"), ("region", "region")]:
        for val in df[seg_col].unique():
            subset = df[df[seg_col] == val]
            ctrl = subset[subset["treatment"] == 0]["converted"].mean()
            treat = subset[subset["treatment"] == 1]["converted"].mean()
            direction = "treatment" if treat > ctrl else "control"
            if direction != overall_direction:
                reversed_segments.append({
                    "segment": seg_name, "value": val,
                    "control_rate": ctrl, "treatment_rate": treat,
                    "direction": direction,
                })

    if reversed_segments:
        logger.info(f"  Found {len(reversed_segments)} reversed segments (potential Simpson's paradox)")
        return pd.DataFrame(reversed_segments)
    else:
        logger.info("  No Simpson's paradox detected")
        return pd.DataFrame()


def sample_ratio_mismatch_check(df, expected_ratio=0.5):
    logger.info("Checking for Sample Ratio Mismatch...")
    n_treat = (df["treatment"] == 1).sum()
    n_ctrl = (df["treatment"] == 0).sum()
    n_total = n_treat + n_ctrl

    observed_ratio = n_treat / n_total
    expected_treat = n_total * expected_ratio
    expected_ctrl = n_total * (1 - expected_ratio)
    chi2 = (n_treat - expected_treat) ** 2 / expected_treat + (n_ctrl - expected_ctrl) ** 2 / expected_ctrl
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    result = {
        "n_treatment": n_treat, "n_control": n_ctrl,
        "observed_ratio": observed_ratio, "expected_ratio": expected_ratio,
        "chi2": chi2, "p_value": p_value,
        "srm_detected": p_value < 0.001,
    }

    if p_value < 0.001:
        logger.info(f"  SRM DETECTED: observed ratio = {observed_ratio:.4f}, p = {p_value:.6f}")
    else:
        logger.info(f"  No SRM: observed ratio = {observed_ratio:.4f}, p = {p_value:.4f}")

    return result


def cuped_analysis(df, metric="converted", covariate="sessions_last_week"):
    logger.info(f"CUPED variance reduction using '{covariate}'...")

    ctrl = df[df["treatment"] == 0]
    treat = df[df["treatment"] == 1]

    raw_diff = treat[metric].mean() - ctrl[metric].mean()
    raw_se = np.sqrt(ctrl[metric].var() / len(ctrl) + treat[metric].var() / len(treat))

    theta = np.cov(df[metric], df[covariate])[0, 1] / max(np.var(df[covariate]), 1e-10)
    df = df.copy()
    df["adjusted_metric"] = df[metric] - theta * (df[covariate] - df[covariate].mean())

    ctrl_adj = df[df["treatment"] == 0]["adjusted_metric"]
    treat_adj = df[df["treatment"] == 1]["adjusted_metric"]

    cuped_diff = treat_adj.mean() - ctrl_adj.mean()
    cuped_se = np.sqrt(ctrl_adj.var() / len(ctrl_adj) + treat_adj.var() / len(treat_adj))

    variance_reduction = 1 - (cuped_se ** 2) / (raw_se ** 2)

    result = {
        "raw_diff": raw_diff, "raw_se": raw_se,
        "raw_ci_lower": raw_diff - 1.96 * raw_se,
        "raw_ci_upper": raw_diff + 1.96 * raw_se,
        "cuped_diff": cuped_diff, "cuped_se": cuped_se,
        "cuped_ci_lower": cuped_diff - 1.96 * cuped_se,
        "cuped_ci_upper": cuped_diff + 1.96 * cuped_se,
        "theta": theta,
        "variance_reduction": variance_reduction,
    }

    logger.info(f"  Raw: {raw_diff:.5f} +/- {1.96 * raw_se:.5f}")
    logger.info(f"  CUPED: {cuped_diff:.5f} +/- {1.96 * cuped_se:.5f}")
    logger.info(f"  Variance reduction: {variance_reduction:.1%}")

    return result


def cuped_multi_covariate(df, metric="converted", covariates=None):
    if covariates is None:
        covariates = ["sessions_last_week", "past_purchases", "total_spent", "days_since_signup"]

    logger.info(f"CUPED with {len(covariates)} covariates...")

    valid_covariates = [c for c in covariates if c in df.columns and df[c].std() > 0]
    if not valid_covariates:
        return None

    Y = df[metric].values
    X_cov = df[valid_covariates].values
    X_cov_centered = X_cov - X_cov.mean(axis=0)

    cov_xy = np.array([np.cov(Y, X_cov[:, i])[0, 1] for i in range(X_cov.shape[1])])
    cov_xx = np.cov(X_cov.T)
    if cov_xx.ndim == 0:
        cov_xx = np.array([[cov_xx]])
    try:
        theta = np.linalg.solve(cov_xx, cov_xy)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(cov_xx, cov_xy, rcond=None)[0]

    adjusted = Y - X_cov_centered @ theta
    df = df.copy()
    df["adjusted_metric_multi"] = adjusted

    ctrl = df[df["treatment"] == 0]
    treat = df[df["treatment"] == 1]

    raw_se = np.sqrt(df.loc[ctrl.index, metric].var() / len(ctrl) + df.loc[treat.index, metric].var() / len(treat))
    cuped_se = np.sqrt(ctrl["adjusted_metric_multi"].var() / len(ctrl) + treat["adjusted_metric_multi"].var() / len(treat))
    variance_reduction = 1 - (cuped_se ** 2) / (raw_se ** 2)

    diff = treat["adjusted_metric_multi"].mean() - ctrl["adjusted_metric_multi"].mean()

    result = {
        "diff": diff, "se": cuped_se,
        "ci_lower": diff - 1.96 * cuped_se,
        "ci_upper": diff + 1.96 * cuped_se,
        "variance_reduction": variance_reduction,
        "covariates_used": valid_covariates,
        "theta": dict(zip(valid_covariates, theta)),
    }

    logger.info(f"  Multi-CUPED diff: {diff:.5f} +/- {1.96 * cuped_se:.5f}")
    logger.info(f"  Variance reduction: {variance_reduction:.1%}")

    return result


def delta_method_ratio(df, numerator="revenue", denominator="pages_viewed"):
    logger.info(f"Delta method for ratio metric ({numerator}/{denominator})...")

    ctrl = df[df["treatment"] == 0]
    treat = df[df["treatment"] == 1]

    def ratio_stats(group):
        n = len(group)
        y = group[numerator].values
        x = group[denominator].values.astype(float)
        x = np.where(x == 0, 1, x)
        mu_y = y.mean()
        mu_x = x.mean()
        var_y = y.var()
        var_x = x.var()
        cov_xy = np.cov(y, x)[0, 1]
        ratio = mu_y / max(mu_x, 1e-10)
        var_ratio = (1 / (mu_x ** 2)) * (var_y - 2 * ratio * cov_xy + ratio ** 2 * var_x)
        se = np.sqrt(var_ratio / n)
        return ratio, se

    r_ctrl, se_ctrl = ratio_stats(ctrl)
    r_treat, se_treat = ratio_stats(treat)
    diff = r_treat - r_ctrl
    se_diff = np.sqrt(se_ctrl ** 2 + se_treat ** 2)
    z = diff / max(se_diff, 1e-10)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    result = {
        "control_ratio": r_ctrl, "treatment_ratio": r_treat,
        "diff": diff, "se": se_diff,
        "ci_lower": diff - 1.96 * se_diff,
        "ci_upper": diff + 1.96 * se_diff,
        "z": z, "p_value": p_value,
    }

    logger.info(f"  Control: {r_ctrl:.4f}, Treatment: {r_treat:.4f}")
    logger.info(f"  Diff: {diff:.4f} +/- {1.96 * se_diff:.4f}, p={p_value:.4f}")

    return result


def stratified_analysis(df, strata_col="region", metric="converted"):
    logger.info(f"Stratified analysis by '{strata_col}'...")

    strata = df[strata_col].unique()
    weights = {}
    stratum_effects = {}

    for s in strata:
        subset = df[df[strata_col] == s]
        ctrl = subset[subset["treatment"] == 0]
        treat = subset[subset["treatment"] == 1]
        if len(ctrl) < 10 or len(treat) < 10:
            continue
        diff = treat[metric].mean() - ctrl[metric].mean()
        n_s = len(subset)
        se = np.sqrt(ctrl[metric].var() / len(ctrl) + treat[metric].var() / len(treat))
        weights[s] = n_s
        stratum_effects[s] = {"diff": diff, "se": se, "n": n_s}

    total_n = sum(weights.values())
    weighted_diff = sum(stratum_effects[s]["diff"] * weights[s] / total_n for s in stratum_effects)
    weighted_se = np.sqrt(sum(
        (weights[s] / total_n) ** 2 * stratum_effects[s]["se"] ** 2
        for s in stratum_effects
    ))

    naive_ctrl = df[df["treatment"] == 0][metric].mean()
    naive_treat = df[df["treatment"] == 1][metric].mean()
    naive_diff = naive_treat - naive_ctrl

    result = {
        "strata_col": strata_col,
        "n_strata": len(stratum_effects),
        "weighted_diff": weighted_diff,
        "weighted_se": weighted_se,
        "weighted_ci_lower": weighted_diff - 1.96 * weighted_se,
        "weighted_ci_upper": weighted_diff + 1.96 * weighted_se,
        "naive_diff": naive_diff,
        "stratum_details": stratum_effects,
    }

    logger.info(f"  Naive diff: {naive_diff:.5f}")
    logger.info(f"  Stratified diff: {weighted_diff:.5f} +/- {1.96 * weighted_se:.5f}")

    return result


def plot_cuped_comparison(raw_result, cuped_result, multi_cuped_result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["Raw", "CUPED\n(single)", "CUPED\n(multi)"]
    diffs = [raw_result["absolute_diff"],
             cuped_result["cuped_diff"],
             multi_cuped_result["diff"]]
    ci_widths = [
        1.96 * np.sqrt(
            raw_result["control_rate"] * (1 - raw_result["control_rate"]) / raw_result["n_control"]
            + raw_result["treatment_rate"] * (1 - raw_result["treatment_rate"]) / raw_result["n_treatment"]
        ),
        1.96 * cuped_result["cuped_se"],
        1.96 * multi_cuped_result["se"],
    ]
    colors = ["#95a5a6", "#2c3e50", "#e74c3c"]

    for i, (m, d, w, c) in enumerate(zip(methods, diffs, ci_widths, colors)):
        axes[0].errorbar(d, i, xerr=w, fmt="o", color=c, capsize=5, markersize=8, linewidth=2)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Treatment Effect Estimate Comparison")
    axes[0].set_xlabel("Conversion Rate Difference")

    var_reductions = [0, cuped_result["variance_reduction"], multi_cuped_result["variance_reduction"]]
    axes[1].bar(methods, [v * 100 for v in var_reductions], color=colors)
    axes[1].set_ylabel("Variance Reduction (%)")
    axes[1].set_title("CUPED Variance Reduction")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "cuped_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cuped_comparison.png")
