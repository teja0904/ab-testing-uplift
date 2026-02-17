import logging
from config import ASSETS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run():
    from simulator import generate_experiment, generate_sequential_data
    from analysis import (
        check_balance, plot_balance, frequentist_test, revenue_test,
        bootstrap_test, bayesian_test, power_curve, segmented_effects,
        plot_segmented_effects, multiple_testing_correction,
        plot_results_summary, plot_sequential_peeking, simpsons_paradox_check,
        sample_ratio_mismatch_check, cuped_analysis, cuped_multi_covariate,
        delta_method_ratio, stratified_analysis, plot_cuped_comparison,
    )
    from uplift import (
        run_uplift_models, plot_uplift_results, targeting_analysis,
        cross_validated_uplift, propensity_score_analysis, plot_propensity_diagnostics,
    )

    ASSETS_DIR.mkdir(exist_ok=True)

    df = generate_experiment(n=50000, assignment="random")

    srm = sample_ratio_mismatch_check(df)

    balance = check_balance(df)
    plot_balance(balance)
    logger.info(f"\n{balance.to_string(index=False)}")

    power_curve()

    freq_result = frequentist_test(df, metric="converted")
    rev_result = revenue_test(df)
    boot_result, boot_diffs = bootstrap_test(df, metric="converted")
    bayes_result, bayes_diffs = bayesian_test(df, metric="converted")

    plot_results_summary(freq_result, boot_result, bayes_result, boot_diffs, bayes_diffs)

    cuped_result = cuped_analysis(df, metric="converted", covariate="sessions_last_week")
    multi_cuped = cuped_multi_covariate(df, metric="converted")
    if multi_cuped:
        plot_cuped_comparison(freq_result, cuped_result, multi_cuped)

    delta_result = delta_method_ratio(df, numerator="revenue", denominator="pages_viewed")

    strat_result = stratified_analysis(df, strata_col="region", metric="converted")

    seg_df = segmented_effects(df)
    plot_segmented_effects(seg_df)
    seg_df = multiple_testing_correction(seg_df, method="bonferroni")
    logger.info(f"  Segments significant after correction:\n"
                f"{seg_df[seg_df['significant_corrected']].to_string(index=False)}")

    simpsons = simpsons_paradox_check(df)
    if len(simpsons) > 0:
        logger.info(f"  Simpson's paradox found:\n{simpsons.to_string(index=False)}")

    seq_df = generate_sequential_data()
    plot_sequential_peeking(seq_df)

    prop_result = propensity_score_analysis(df)
    plot_propensity_diagnostics(df, prop_result)

    uplift_results = run_uplift_models(df)
    plot_uplift_results(df, uplift_results)
    targeting_analysis(df, uplift_results)

    cv_scores = cross_validated_uplift(df)

    logger.info("Done.")


if __name__ == "__main__":
    run()
