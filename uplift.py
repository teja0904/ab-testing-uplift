import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from config import ASSETS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

FEATURE_COLS = [
    "age", "is_male", "days_since_signup", "past_purchases", "total_spent",
    "device_mobile", "sessions_last_week", "is_subscriber",
]
CATEGORICAL_COLS = ["region", "channel"]


def prepare_features(df):
    X = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        dummies = pd.get_dummies(df[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
    return X


class SLearner:
    def __init__(self):
        self.model = None

    def fit(self, X, treatment, y):
        X_aug = X.copy()
        X_aug["treatment"] = treatment.values
        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=32, min_child_samples=50,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        )
        self.model.fit(X_aug, y)

    def predict_uplift(self, X):
        X_treat = X.copy()
        X_treat["treatment"] = 1
        X_ctrl = X.copy()
        X_ctrl["treatment"] = 0
        return self.model.predict_proba(X_treat)[:, 1] - self.model.predict_proba(X_ctrl)[:, 1]


class TLearner:
    def __init__(self):
        self.model_ctrl = None
        self.model_treat = None

    def fit(self, X, treatment, y):
        ctrl_mask = treatment == 0
        treat_mask = treatment == 1

        params = dict(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=32, min_child_samples=50,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        )
        self.model_ctrl = lgb.LGBMClassifier(**params)
        self.model_ctrl.fit(X[ctrl_mask], y[ctrl_mask])

        self.model_treat = lgb.LGBMClassifier(**params)
        self.model_treat.fit(X[treat_mask], y[treat_mask])

    def predict_uplift(self, X):
        return self.model_treat.predict_proba(X)[:, 1] - self.model_ctrl.predict_proba(X)[:, 1]


class XLearner:
    def __init__(self):
        self.model_ctrl = None
        self.model_treat = None
        self.tau_ctrl = None
        self.tau_treat = None
        self.propensity_model = None

    def fit(self, X, treatment, y):
        ctrl_mask = treatment == 0
        treat_mask = treatment == 1

        params = dict(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=32, min_child_samples=50,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        )

        self.model_ctrl = lgb.LGBMClassifier(**params)
        self.model_ctrl.fit(X[ctrl_mask], y[ctrl_mask])

        self.model_treat = lgb.LGBMClassifier(**params)
        self.model_treat.fit(X[treat_mask], y[treat_mask])

        d_treat = y[treat_mask].values - self.model_ctrl.predict_proba(X[treat_mask])[:, 1]
        d_ctrl = self.model_treat.predict_proba(X[ctrl_mask])[:, 1] - y[ctrl_mask].values

        reg_params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=24, min_child_samples=50,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        )
        self.tau_treat = lgb.LGBMRegressor(**reg_params)
        self.tau_treat.fit(X[treat_mask], d_treat)

        self.tau_ctrl = lgb.LGBMRegressor(**reg_params)
        self.tau_ctrl.fit(X[ctrl_mask], d_ctrl)

        self.propensity_model = lgb.LGBMClassifier(**params)
        self.propensity_model.fit(X, treatment)

    def predict_uplift(self, X):
        tau_t = self.tau_treat.predict(X)
        tau_c = self.tau_ctrl.predict(X)
        g = self.propensity_model.predict_proba(X)[:, 1]
        return g * tau_c + (1 - g) * tau_t


def evaluate_uplift(df, uplift_scores, name, n_bins=10):
    df = df.copy()
    df["uplift_score"] = uplift_scores
    df["uplift_bin"] = pd.qcut(df["uplift_score"], n_bins, labels=False, duplicates="drop")

    bin_results = []
    for b in sorted(df["uplift_bin"].unique()):
        subset = df[df["uplift_bin"] == b]
        ctrl = subset[subset["treatment"] == 0]
        treat = subset[subset["treatment"] == 1]

        if len(ctrl) < 10 or len(treat) < 10:
            continue

        observed_uplift = treat["converted"].mean() - ctrl["converted"].mean()
        bin_results.append({
            "bin": b,
            "n": len(subset),
            "mean_score": subset["uplift_score"].mean(),
            "observed_uplift": observed_uplift,
            "control_rate": ctrl["converted"].mean(),
            "treatment_rate": treat["converted"].mean(),
        })

    return pd.DataFrame(bin_results)


def qini_curve(df, uplift_scores):
    df = df.copy()
    df["uplift_score"] = uplift_scores
    df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)

    n = len(df)
    incremental_gains = []
    random_gains = []

    overall_treat_rate = df["treatment"].mean()
    overall_ctrl_rate = 1 - overall_treat_rate

    for k in range(100, n, max(n // 100, 1)):
        top_k = df.iloc[:k]
        treat_k = top_k[top_k["treatment"] == 1]
        ctrl_k = top_k[top_k["treatment"] == 0]

        if len(treat_k) < 5 or len(ctrl_k) < 5:
            continue

        n_t = len(treat_k)
        n_c = len(ctrl_k)
        gain = treat_k["converted"].sum() - ctrl_k["converted"].sum() * (n_t / max(n_c, 1))
        incremental_gains.append((k / n, gain))

        total_treat = df[df["treatment"] == 1]["converted"].sum()
        total_ctrl = df[df["treatment"] == 0]["converted"].sum()
        n_t_all = (df["treatment"] == 1).sum()
        n_c_all = (df["treatment"] == 0).sum()
        random_gain = (k / n) * (total_treat - total_ctrl * (n_t_all / max(n_c_all, 1)))
        random_gains.append((k / n, random_gain))

    return incremental_gains, random_gains


def run_uplift_models(df):
    logger.info("Running uplift models...")

    X = prepare_features(df)
    treatment = df["treatment"]
    y = df["converted"]

    models = {
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "X-Learner": XLearner(),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"  Fitting {name}...")
        model.fit(X, treatment, y)
        scores = model.predict_uplift(X)
        results[name] = scores

        bin_eval = evaluate_uplift(df, scores, name)
        correlation = bin_eval["mean_score"].corr(bin_eval["observed_uplift"])
        logger.info(f"    Score-uplift correlation: {correlation:.3f}")

    if "true_treatment_effect" in df.columns:
        logger.info("  Correlation with true treatment effect:")
        for name, scores in results.items():
            corr = np.corrcoef(scores, df["true_treatment_effect"])[0, 1]
            logger.info(f"    {name}: {corr:.3f}")

    return results


def plot_uplift_results(df, results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for name, scores in results.items():
        bin_eval = evaluate_uplift(df, scores, name)
        ax.plot(bin_eval["bin"], bin_eval["observed_uplift"], marker="o", label=name, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Uplift Score Decile")
    ax.set_ylabel("Observed Uplift")
    ax.set_title("Uplift by Score Decile")
    ax.legend()

    ax = axes[0, 1]
    for name, scores in results.items():
        gains, random = qini_curve(df, scores)
        if gains:
            pcts, vals = zip(*gains)
            ax.plot(pcts, vals, label=name, linewidth=2)
    if random:
        pcts_r, vals_r = zip(*random)
        ax.plot(pcts_r, vals_r, "k--", alpha=0.4, label="Random")
    ax.set_xlabel("Fraction of Population Targeted")
    ax.set_ylabel("Incremental Conversions")
    ax.set_title("Qini Curves")
    ax.legend()

    ax = axes[1, 0]
    if "true_treatment_effect" in df.columns:
        for name, scores in results.items():
            ax.scatter(df["true_treatment_effect"], scores, alpha=0.02, s=3, label=name)
        ax.plot([df["true_treatment_effect"].min(), df["true_treatment_effect"].max()],
                [df["true_treatment_effect"].min(), df["true_treatment_effect"].max()],
                "k--", alpha=0.3)
        ax.set_xlabel("True Treatment Effect")
        ax.set_ylabel("Predicted Uplift")
        ax.set_title("Predicted vs True Uplift")
        ax.legend(markerscale=10)

    ax = axes[1, 1]
    best_model = list(results.keys())[-1]
    scores = results[best_model]
    df_temp = df.copy()
    df_temp["uplift_bin"] = pd.qcut(scores, 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"],
                                      duplicates="drop")
    bin_summary = df_temp.groupby("uplift_bin").agg(
        n=("user_id", "count"),
        conversion_ctrl=("converted", lambda x: x[df_temp.loc[x.index, "treatment"] == 0].mean()),
        conversion_treat=("converted", lambda x: x[df_temp.loc[x.index, "treatment"] == 1].mean()),
    )
    x = range(len(bin_summary))
    w = 0.35
    ax.bar([i - w/2 for i in x], bin_summary["conversion_ctrl"], w,
           label="Control", color="#2c3e50", alpha=0.7)
    ax.bar([i + w/2 for i in x], bin_summary["conversion_treat"], w,
           label="Treatment", color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_summary.index, rotation=15)
    ax.set_ylabel("Conversion Rate")
    ax.set_title(f"Conversion by Uplift Quintile ({best_model})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "uplift_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved uplift_results.png")


def cross_validated_uplift(df, n_folds=5):
    logger.info(f"Running {n_folds}-fold cross-validated uplift evaluation...")
    from sklearn.model_selection import KFold

    X = prepare_features(df)
    treatment = df["treatment"]
    y = df["converted"]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    models = {"S-Learner": SLearner, "T-Learner": TLearner, "X-Learner": XLearner}
    cv_scores = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        t_train, t_val = treatment.iloc[train_idx], treatment.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, cls in models.items():
            model = cls()
            model.fit(X_train, t_train, y_train)
            scores = model.predict_uplift(X_val)

            val_df = df.iloc[val_idx].copy()
            val_df["uplift_score"] = scores
            bin_eval = evaluate_uplift(val_df, scores, name)

            if len(bin_eval) >= 2:
                corr = bin_eval["mean_score"].corr(bin_eval["observed_uplift"])
                cv_scores[name].append(corr if not np.isnan(corr) else 0)

    for name in cv_scores:
        scores = cv_scores[name]
        if scores:
            logger.info(f"  {name}: mean corr = {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    return cv_scores


def propensity_score_analysis(df):
    logger.info("Estimating propensity scores...")
    X = prepare_features(df)
    treatment = df["treatment"].values

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        num_leaves=16, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, treatment)
    propensity = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(treatment, propensity)
    logger.info(f"  Propensity model AUC: {auc:.4f} (should be ~0.5 for RCT)")

    df = df.copy()
    df["propensity"] = propensity

    trimmed = df[(df["propensity"] > 0.1) & (df["propensity"] < 0.9)].copy()
    logger.info(f"  After trimming: {len(trimmed)} / {len(df)} ({len(trimmed)/len(df):.1%})")

    ctrl = trimmed[trimmed["treatment"] == 0]
    treat = trimmed[trimmed["treatment"] == 1]

    w_ctrl = 1 / (1 - ctrl["propensity"])
    w_treat = 1 / treat["propensity"]

    ate_ipw = (
        (treat["converted"] * w_treat).sum() / w_treat.sum()
        - (ctrl["converted"] * w_ctrl).sum() / w_ctrl.sum()
    )

    naive_ate = treat["converted"].mean() - ctrl["converted"].mean()

    logger.info(f"  Naive ATE: {naive_ate:.4f}")
    logger.info(f"  IPW ATE: {ate_ipw:.4f}")

    return {
        "propensity_auc": auc,
        "naive_ate": naive_ate,
        "ipw_ate": ate_ipw,
        "n_trimmed": len(trimmed),
        "propensity_mean": propensity.mean(),
        "propensity_std": propensity.std(),
    }


def plot_propensity_diagnostics(df, prop_result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    propensity = df.get("propensity")
    if propensity is None:
        X = prepare_features(df)
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            num_leaves=16, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X, df["treatment"].values)
        propensity = model.predict_proba(X)[:, 1]

    axes[0].hist(propensity[df["treatment"] == 0], bins=50, alpha=0.6,
                 label="Control", color="#2c3e50", density=True)
    axes[0].hist(propensity[df["treatment"] == 1], bins=50, alpha=0.6,
                 label="Treatment", color="#e74c3c", density=True)
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_title(f"Propensity Score Distribution (AUC={prop_result['propensity_auc']:.3f})")
    axes[0].legend()

    methods = ["Naive ATE", "IPW ATE"]
    values = [prop_result["naive_ate"], prop_result["ipw_ate"]]
    colors = ["#2c3e50", "#e74c3c"]
    axes[1].bar(methods, values, color=colors, alpha=0.7)
    axes[1].set_ylabel("Treatment Effect")
    axes[1].set_title("ATE Estimates")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "propensity_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved propensity_diagnostics.png")


def targeting_analysis(df, results):
    logger.info("Targeting analysis...")
    best_model = "X-Learner"
    scores = results.get(best_model, list(results.values())[-1])

    df = df.copy()
    df["uplift_score"] = scores

    percentiles = np.arange(0.1, 1.0, 0.05)
    targeting_results = []

    for pct in percentiles:
        threshold = df["uplift_score"].quantile(1 - pct)
        targeted = df[df["uplift_score"] >= threshold]
        not_targeted = df[df["uplift_score"] < threshold]

        t_ctrl = targeted[targeted["treatment"] == 0]["converted"].mean()
        t_treat = targeted[targeted["treatment"] == 1]["converted"].mean()
        t_uplift = t_treat - t_ctrl

        if len(not_targeted) > 100:
            nt_ctrl = not_targeted[not_targeted["treatment"] == 0]["converted"].mean()
            nt_treat = not_targeted[not_targeted["treatment"] == 1]["converted"].mean()
            nt_uplift = nt_treat - nt_ctrl
        else:
            nt_uplift = 0

        targeting_results.append({
            "pct_targeted": pct, "n_targeted": len(targeted),
            "targeted_uplift": t_uplift, "not_targeted_uplift": nt_uplift,
            "incremental_conversions": t_uplift * len(targeted[targeted["treatment"] == 1]),
        })

    result_df = pd.DataFrame(targeting_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result_df["pct_targeted"] * 100, result_df["targeted_uplift"] * 100,
            color="#2c3e50", linewidth=2, label="Targeted group uplift")
    ax.plot(result_df["pct_targeted"] * 100, result_df["not_targeted_uplift"] * 100,
            color="#e74c3c", linewidth=2, linestyle="--", label="Non-targeted group uplift")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("% of Population Targeted")
    ax.set_ylabel("Observed Uplift (pp)")
    ax.set_title("Targeting Strategy — Who Should Receive the Treatment?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "targeting_strategy.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved targeting_strategy.png")

    return result_df
