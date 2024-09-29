use ndarray::{Array4, Array1, Axis, s};
use rayon::prelude::*;
use crate::Layer;
use crate::LayerType;
use rand::Rng;
use crate::Activation;
pub enum Initializer {
    He,
    Xavier,
    Zero,
    Uniform(f64, f64),
}

pub struct Conv2D {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weights: Array4<f64>,
    biases: Array1<f64>,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, initializer: Initializer) -> Self {
        let weights = Self::initialize_weights(in_channels, out_channels, kernel_size, initializer);
        let biases = Array1::zeros(out_channels);
        Conv2D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            biases,
        }
    }

    fn initialize_weights(in_channels: usize, out_channels: usize, kernel_size: usize, initializer: Initializer) -> Array4<f64> {
        let fan_in = in_channels * kernel_size * kernel_size;
        let _fan_out = out_channels * kernel_size * kernel_size;
        let mut rng = rand::thread_rng();

        match initializer {
            Initializer::He => {
                let scale = (2.0 / fan_in as f64).sqrt();
                Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| {
                    rng.gen_range(-scale..scale)
                })
            },
            Initializer::Xavier => {
                let scale = (1.0 / fan_in as f64).sqrt();
                Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| {
                    rng.gen_range(-scale..scale)
                })
            },
            Initializer::Zero => {
                Array4::zeros((out_channels, in_channels, kernel_size, kernel_size))
            },
            Initializer::Uniform(low, high) => {
                Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| {
                    rng.gen_range(low..high)
                })
            },
        }
    }

    pub fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        let (batch_size, in_channels, height, width) = input.dim();
        assert_eq!(in_channels, self.in_channels, "Input channels must match Conv2D in_channels");

        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        assert!(
            padded_height >= self.kernel_size && padded_width >= self.kernel_size,
            "Input size is too small after padding for the given kernel size"
        );

        let output_height = (padded_height - self.kernel_size) / self.stride + 1;
        let output_width = (padded_width - self.kernel_size) / self.stride + 1;

        let mut output = Array4::zeros((batch_size, self.out_channels, output_height, output_width));
        let input = input.pad(self.padding);

        output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(b, mut output_batch)| {
            for c in 0..self.out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let h_start = h * self.stride;
                        let w_start = w * self.stride;
                        let h_end = h_start + self.kernel_size;
                        let w_end = w_start + self.kernel_size;

                        let input_slice = input.slice(s![b, .., h_start..h_end, w_start..w_end]);
                        let weight = self.weights.slice(s![c, .., .., ..]);
                        let result: f64 = input_slice.iter().zip(weight.iter()).map(|(a, b)| a * b).sum();
                        output_batch[[c, h, w]] = result + self.biases[c];
                    }
                }
            }
        });
        output
    }
}

pub trait Pad {
    fn pad(&self, padding: usize) -> Self;
}

impl Pad for Array4<f64> {
    fn pad(&self, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let (batch_size, channels, height, width) = self.dim();
        let padded_height = height + 2 * padding;
        let padded_width = width + 2 * padding;

        let mut padded = Array4::zeros((batch_size, channels, padded_height, padded_width));

        padded
            .slice_mut(s![
                ..,
                ..,
                padding..padding + height,
                padding..padding + width
            ])
            .assign(self);

        padded
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        self.forward(input)
    }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        unimplemented!()
    }

    fn update_weights(&mut self, _learning_rate: f64) {
        unimplemented!()
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Conv2D
    }
}
