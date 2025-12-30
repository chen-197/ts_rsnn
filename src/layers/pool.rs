use ndarray::{Array4, s};
use crate::layers::Layer;
use crate::optimizer::Optimizer;
use crate::LayerType;
use std::any::Any;

pub struct MaxPool2D {
    pub pool_size: usize,
    pub stride: usize,
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

        ndarray::Zip::indexed(output.outer_iter_mut())
            .par_for_each(|b, mut out_batch| {
                ndarray::Zip::indexed(out_batch.outer_iter_mut())
                    .par_for_each(|c, mut out_channel| {
                         for i in 0..output_height {
                            for j in 0..output_width {
                                let h_start = i * self.stride;
                                let w_start = j * self.stride;
                                let slice = input.slice(s![b, c, h_start..h_start + self.pool_size, w_start..w_start + self.pool_size]);
                                let mut max_value = f64::MIN;
                                for v in slice.iter() {
                                    if *v > max_value { max_value = *v; }
                                }
                                out_channel[[i, j]] = max_value;
                            }
                        }
                    });
            });
        output
    }
    
    pub fn backward_impl(&self, input: Array4<f64>, grad_output: Array4<f64>) -> Array4<f64> {
         //let (batch_size, channels, _height, _width) = input.dim(); 
         let mut grad_input = Array4::zeros(input.dim());
         let (_, _, out_h, out_w) = grad_output.dim();

         ndarray::Zip::indexed(grad_input.outer_iter_mut())
            .par_for_each(|b, mut grad_in_batch| {
                 ndarray::Zip::indexed(grad_in_batch.outer_iter_mut())
                    .par_for_each(|c, mut grad_in_channel| {
                        for i in 0..out_h {
                            for j in 0..out_w {
                                let h_start = i * self.stride;
                                let w_start = j * self.stride;
                                
                                let slice = input.slice(s![b, c, h_start..h_start+self.pool_size, w_start..w_start+self.pool_size]);
                                let mut max_val = f64::MIN;
                                let mut max_idx = (0, 0);
                                
                                for (y, row) in slice.outer_iter().enumerate() {
                                    for (x, &val) in row.iter().enumerate() {
                                        if val > max_val {
                                            max_val = val;
                                            max_idx = (y, x);
                                        }
                                    }
                                }
                                let grad_val = grad_output[[b, c, i, j]];
                                grad_in_channel[[h_start + max_idx.0, w_start + max_idx.1]] += grad_val;
                            }
                        }
                    });
            });
         grad_input
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> { self.forward(input) }
    
    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, _activation_derivative: Array4<f64>) -> Array4<f64> {
        // Correctly calls the inherent implementation
        self.backward_impl(input, grad_output)
    }

    fn update_weights(&mut self, _optimizer: &mut dyn Optimizer, _layer_id: usize) {}
    fn layer_type(&self) -> LayerType { LayerType::MaxPool2D }
    fn as_any(&self) -> &dyn Any { self }
}