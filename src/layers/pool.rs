use ndarray::{Array4, Axis, s};
use rayon::prelude::*;
use crate::layers::Layer;
use crate::LayerType;
use crate::Activation;
pub struct MaxPool2D {
    pool_size: usize,
    stride: usize,
}

impl MaxPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        MaxPool2D { pool_size, stride }
    }

    pub fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        let (batch_size, channels, height, width) = input.dim();
        let output_height = (height - self.pool_size) / self.stride + 1;
        let output_width = (width - self.pool_size) / self.stride + 1;

        let mut output = Array4::zeros((batch_size, channels, output_height, output_width));

        output.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut batch| {
            for d in 0..channels {
                for ((_, i, j), o) in batch.indexed_iter_mut() {
                    let slice = input.slice(s![
                        ..,
                        d,
                        i * self.stride..i * self.stride + self.pool_size,
                        j * self.stride..j * self.stride + self.pool_size
                    ]);

                    let mut max_value = f64::MIN;
                    for value in slice.iter() {
                        if *value > max_value {
                            max_value = *value;
                        }
                    }
                    *o = max_value;
                }
            }
        });

        output
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        self.forward(input)
    }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        unimplemented!()
    }

    fn update_weights(&mut self, _learning_rate: f64) {
    }

    fn layer_type(&self) -> LayerType {
        LayerType::MaxPool2D
    }
}
