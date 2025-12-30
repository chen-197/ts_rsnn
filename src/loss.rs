use ndarray::prelude::*;
use rayon::prelude::*;

pub trait Loss: Send + Sync {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64;
    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64>;
}

pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let sum_sq_error: f64 = predicted.as_slice().unwrap().par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        sum_sq_error / predicted.len() as f64
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        let n = predicted.len() as f64;
        let mut grad = Array2::zeros(predicted.raw_dim());
        grad.as_slice_mut().unwrap().par_iter_mut()
            .zip(predicted.as_slice().unwrap().par_iter())
            .zip(actual.as_slice().unwrap().par_iter())
            .for_each(|((g, p), a)| { *g = 2.0 * (p - a) / n; });
        grad
    }
}

pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        let sum_error: f64 = predicted.as_slice().unwrap().par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| {
                let eps = 1e-15;
                let p_safe = p.clamp(eps, 1.0 - eps);
                -a * p_safe.ln() - (1.0 - a) * (1.0 - p_safe).ln()
            })
            .sum();
        sum_error / predicted.len() as f64
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        let mut grad = Array2::zeros(predicted.raw_dim());
        grad.as_slice_mut().unwrap().par_iter_mut()
            .zip(predicted.as_slice().unwrap().par_iter())
            .zip(actual.as_slice().unwrap().par_iter())
            .for_each(|((g, p), a)| {
                let eps = 1e-15;
                let denominator = (p * (1.0 - p)).max(eps);
                *g = (p - a) / denominator;
            });
        grad
    }
}