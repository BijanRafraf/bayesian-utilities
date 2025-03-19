
fn beta_function(alpha: f64, beta: f64) -> f64 {
    libm::exp(libm::lgamma(alpha) + libm::lgamma(beta) - libm::lgamma(alpha + beta))
}

fn beta_pdf_point(alpha: f64, beta: f64, x: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        return 0.0;
    }
    let denominator = beta_function(alpha, beta);
    x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0) / denominator
}

fn beta_pdf(alpha: f64, beta: f64, x: Vec<f64>) -> Vec<f64> {
    let denominator = beta_function(alpha, beta);
    x.iter().map(|&x| x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0) / denominator).collect()
}

fn mean_var_from_beta(alpha: f64, beta: f64) -> (f64, f64) {
    let mean = alpha / (alpha + beta);
    let var = alpha * beta / ((alpha + beta).powf(2.0) * (alpha + beta + 1.0));
    (mean, var)
}

pub fn binomial_mean_posterior_pdf(n: u64, k: u64, x: Vec<f64>) -> (Vec<f64>, f64, f64, f64) {
    let alpha = k as f64 + 1.0;
    let beta = n as f64 - k as f64 + 1.0;
    let pdf = beta_pdf(alpha, beta, x);
    let (mu, var) = mean_var_from_beta(alpha, beta);
    let val_at_mu = beta_pdf_point(alpha, beta, mu);
    return (pdf, mu, var, val_at_mu);
}
