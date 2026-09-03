//! Deterministic Bayesian bookkeeping for `/investigate`. The model
//! estimates likelihoods in natural language / structured JSON; everything
//! in this module is the arithmetic Junebug performs instead of trusting a
//! model to do it. Pure functions, no I/O, no `serde_json::Value` — testable
//! without a provider.

const EPS: f64 = 1e-6;

/// Natural-log odds of a probability, clamped away from 0/1 so `ln` never
/// sees zero or a negative denominator.
fn logit(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

fn sigmoid(log_odds: f64) -> f64 {
    1.0 / (1.0 + (-log_odds).exp())
}

/// One piece of evidence's effect on one hypothesis: `P(E|H) / P(E|¬H)`.
/// Greater than 1 supports the hypothesis, less than 1 contradicts it,
/// close to 1 is neutral.
#[must_use]
pub fn likelihood_ratio(given_h: f64, given_not_h: f64) -> f64 {
    given_h.max(EPS) / given_not_h.max(EPS)
}

/// Sequential Bayesian update: `Odds(H|E1..En) = Odds(H) * product(LR_i)`,
/// converted back to a probability. This is per-hypothesis — it does not
/// know about sibling hypotheses, matching how each evaluator call is blind
/// to them. Call `normalize` afterward to make a hypothesis set's posteriors
/// comparable as a ranked belief distribution.
#[must_use]
pub fn posterior(prior: f64, likelihood_ratios: &[f64]) -> f64 {
    let log_odds = likelihood_ratios
        .iter()
        .fold(logit(prior), |acc, lr| acc + lr.max(EPS).ln());
    sigmoid(log_odds)
}

/// Rescales a set of independently-computed posteriors so they sum to 1, for
/// display as a ranked belief distribution. An all-zero input is left
/// untouched rather than divide-by-zero.
pub fn normalize(posteriors: &mut [f64]) {
    let sum: f64 = posteriors.iter().sum();
    if sum > EPS {
        for p in posteriors {
            *p /= sum;
        }
    }
}

/// Shannon entropy in nats: `H(Hypotheses)`, the input an
/// expected-information-gain discriminator would subtract from (not yet
/// wired into an active planning loop — see `INVESTIGATE_MODE_PLAN.md` §8).
#[must_use]
pub fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > EPS)
        .map(|&p| -p * p.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{EPS, entropy, likelihood_ratio, normalize, posterior};

    #[test]
    fn strong_supporting_evidence_raises_posterior_above_prior() {
        let lr = likelihood_ratio(0.8, 0.2); // 4.0
        assert!(posterior(0.3, &[lr]) > 0.3);
    }

    #[test]
    fn contradicting_evidence_lowers_posterior_below_prior() {
        let lr = likelihood_ratio(0.1, 0.6);
        assert!(posterior(0.5, &[lr]) < 0.5);
    }

    #[test]
    fn neutral_evidence_leaves_posterior_unchanged() {
        let lr = likelihood_ratio(0.5, 0.5);
        assert!((posterior(0.4, &[lr]) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn multiple_supporting_observations_compound() {
        let lrs = [likelihood_ratio(0.7, 0.3), likelihood_ratio(0.8, 0.2)];
        let one = posterior(0.3, &lrs[..1]);
        let two = posterior(0.3, &lrs);
        assert!(two > one);
    }

    #[test]
    fn posterior_stays_within_unit_interval_for_extreme_inputs() {
        let value = posterior(0.99, &[likelihood_ratio(1.0, 1e-9); 20]);
        assert!((0.0..=1.0).contains(&value) && value > 0.0);
        let value = posterior(0.01, &[likelihood_ratio(1e-9, 1.0); 20]);
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn normalize_scales_to_sum_one() {
        let mut values = [0.6, 0.3, 0.1];
        normalize(&mut values);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_is_a_no_op_on_all_zero_input() {
        let mut values = [0.0, 0.0];
        normalize(&mut values);
        assert!(values.iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn entropy_is_zero_for_a_certain_outcome() {
        assert!(entropy(&[1.0, 0.0, 0.0]) < 1e-9);
    }

    #[test]
    fn entropy_is_maximal_for_a_uniform_distribution() {
        let uniform = entropy(&[0.25, 0.25, 0.25, 0.25]);
        let skewed = entropy(&[0.7, 0.1, 0.1, 0.1]);
        assert!(uniform > skewed);
    }

    #[test]
    fn a_scenario_with_lopsided_evidence_converges_high() {
        // One hypothesis with three strongly-supporting observations should
        // land clearly above 0.8, not hover near its 0.4 prior.
        let lrs = [
            likelihood_ratio(0.9, 0.1),
            likelihood_ratio(0.85, 0.15),
            likelihood_ratio(0.8, 0.2),
        ];
        assert!(posterior(0.4, &lrs) > 0.8);
    }

    #[test]
    fn balanced_evidence_leaves_no_hypothesis_dominant() {
        let a = posterior(0.33, &[likelihood_ratio(0.55, 0.45)]);
        let b = posterior(0.33, &[likelihood_ratio(0.5, 0.5)]);
        let c = posterior(0.34, &[likelihood_ratio(0.45, 0.55)]);
        let mut values = [a, b, c];
        normalize(&mut values);
        assert!(values.iter().all(|&p| p > 0.2 && p < 0.45));
    }
}
