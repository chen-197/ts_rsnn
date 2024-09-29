use ndarray::prelude::*;

use rayon::prelude::*;

pub trait Loss: Send + Sync {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64;
    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64>;
}

pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        predicted
            .as_slice()
            .unwrap()
            .par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predicted.len() as f64
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        let grad: Vec<f64> = predicted
            .as_slice()
            .unwrap()
            .par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| 2.0 * (p - a) / predicted.len() as f64)
            .collect();

        Array2::from_shape_vec(predicted.raw_dim(), grad).unwrap()
    }
}

pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn compute(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> f64 {
        predicted
            .as_slice()
            .unwrap()
            .par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| {
                -a * p.ln() - (1.0 - a) * (1.0 - p).ln()
            })
            .sum::<f64>()
            / predicted.len() as f64
    }

    fn gradient(&self, predicted: &Array2<f64>, actual: &Array2<f64>) -> Array2<f64> {
        let grad: Vec<f64> = predicted
            .as_slice()
            .unwrap()
            .par_iter()
            .zip(actual.as_slice().unwrap().par_iter())
            .map(|(p, a)| {
                (p - a) / (p * (1.0 - p)).max(1e-7)
            })
            .collect();

        Array2::from_shape_vec(predicted.raw_dim(), grad).unwrap()
    }
}
