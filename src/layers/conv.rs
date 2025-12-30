use ndarray::{Array1, Array4, Array2, Axis, s};
use rayon::prelude::*;
use crate::Layer;
use crate::LayerType;
use crate::optimizer::Optimizer;
use rand::Rng;
use std::any::Any;

pub enum Initializer {
    He,
    Xavier,
    Zero,
    Uniform(f64, f64),
}

pub struct Conv2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Array4<f64>,
    pub biases: Array1<f64>,
    grad_weights: Array4<f64>,
    grad_biases: Array1<f64>,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, initializer: Initializer) -> Self {
        let weights = Self::initialize_weights(in_channels, out_channels, kernel_size, initializer);
        let biases = Array1::zeros(out_channels);
        let grad_weights = Array4::zeros((out_channels, in_channels, kernel_size, kernel_size));
        let grad_biases = Array1::zeros(out_channels);

        Conv2D {
            in_channels, out_channels, kernel_size, stride, padding, weights, biases, grad_weights, grad_biases,
        }
    }

    // Getters
    pub fn out_channels(&self) -> usize { self.out_channels }
    pub fn in_channels(&self) -> usize { self.in_channels }
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    pub fn stride(&self) -> usize { self.stride }
    pub fn padding(&self) -> usize { self.padding }
    pub fn set_weights(&mut self, weights: Array4<f64>) { self.weights = weights; }
    pub fn set_biases(&mut self, biases: Array1<f64>) { self.biases = biases; }

    fn initialize_weights(in_channels: usize, out_channels: usize, kernel_size: usize, initializer: Initializer) -> Array4<f64> {
        let fan_in = in_channels * kernel_size * kernel_size;
        let mut rng = rand::thread_rng();
        match initializer {
            Initializer::He => {
                let scale = (2.0 / fan_in as f64).sqrt();
                Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| rng.gen_range(-scale..scale))
            },
            Initializer::Xavier => {
                let scale = (1.0 / fan_in as f64).sqrt();
                Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| rng.gen_range(-scale..scale))
            },
            Initializer::Zero => Array4::zeros((out_channels, in_channels, kernel_size, kernel_size)),
            Initializer::Uniform(low, high) => Array4::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| rng.gen_range(low..high)),
        }
    }

    pub fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        let (batch_size, in_channels, height, width) = input.dim();
        assert_eq!(in_channels, self.in_channels);

        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        let output_height = (padded_height - self.kernel_size) / self.stride + 1;
        let output_width = (padded_width - self.kernel_size) / self.stride + 1;

        let input_padded = input.pad(self.padding);
        let mut output = Array4::zeros((batch_size, self.out_channels, output_height, output_width));

        // Zip 并行遍历 Output 的 (Batch, Channel) 轴
        ndarray::Zip::indexed(output.outer_iter_mut())
            .par_for_each(|b, mut out_batch| {
                ndarray::Zip::indexed(out_batch.outer_iter_mut())
                    .par_for_each(|c_out, mut out_channel| {
                         let weight = self.weights.slice(s![c_out, .., .., ..]);
                         let bias = self.biases[c_out];

                         for h in 0..output_height {
                            for w in 0..output_width {
                                let h_start = h * self.stride;
                                let w_start = w * self.stride;
                                
                                let input_slice = input_padded.slice(s![b, .., h_start..h_start+self.kernel_size, w_start..w_start+self.kernel_size]);
                                // 展平 slice 进行点积
                                let val = input_slice.iter().zip(weight.iter()).fold(0.0, |acc, (&i, &w)| acc + i * w);
                                out_channel[[h, w]] = val + bias;
                            }
                        }
                    });
            });
        output
    }
}

pub trait Pad {
    fn pad(&self, padding: usize) -> Self;
}

impl Pad for Array4<f64> {
    fn pad(&self, padding: usize) -> Self {
        if padding == 0 { return self.clone(); }
        let (b, c, h, w) = self.dim();
        let mut padded = Array4::zeros((b, c, h + 2 * padding, w + 2 * padding));
        padded.slice_mut(s![.., .., padding..padding+h, padding..padding+w]).assign(self);
        padded
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> { self.forward(input) }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        let (batch_size, _in_c, in_h, in_w) = input.dim();
        let (_bs, _out_c, out_h, out_w) = grad_output.dim();

        // 1. Delta 计算 (In-place parallel)
        let mut delta = grad_output;
        ndarray::Zip::from(&mut delta).and(&activation_derivative_output).par_for_each(|d, &adv| *d *= adv);

        // 2. Bias 梯度
        let grad_biases_local = delta.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        self.grad_biases += &grad_biases_local;

        // 3. Weight 梯度 (Map-Reduce Optimization)
        let input_padded = input.pad(self.padding);
        let stride = self.stride;
        let kernel_size = self.kernel_size;
        
        let num_tasks = self.out_channels * self.in_channels;
        
        // 每个线程计算一部分 Kernel 的梯度，最后合并
        let computed_grads: Vec<(usize, usize, Array2<f64>)> = (0..num_tasks).into_par_iter()
            .map(|idx| {
                let c_out = idx / self.in_channels;
                let c_in = idx % self.in_channels;
                let mut local_kernel_grad = Array2::<f64>::zeros((kernel_size, kernel_size));
                
                for b in 0..batch_size {
                    for h in 0..out_h {
                        for w in 0..out_w {
                            let d_val = delta[[b, c_out, h, w]];
                            if d_val == 0.0 { continue; } // 稀疏性优化

                            let h_start = h * stride;
                            let w_start = w * stride;

                            for ky in 0..kernel_size {
                                for kx in 0..kernel_size {
                                    local_kernel_grad[[ky, kx]] += input_padded[[b, c_in, h_start + ky, w_start + kx]] * d_val;
                                }
                            }
                        }
                    }
                }
                (c_out, c_in, local_kernel_grad)
            })
            .collect();

        for (c_out, c_in, g) in computed_grads {
            let mut w_slice = self.grad_weights.slice_mut(s![c_out, c_in, .., ..]);
            w_slice += &g;
        }

        // 4. Input 梯度 (Full Parallel Write)
        let (padded_batch, padded_channels, padded_h, padded_w) = input_padded.dim();
        let mut grad_input_padded = Array4::<f64>::zeros((padded_batch, padded_channels, padded_h, padded_w));

        // 按 Batch 并行，确保无锁写入
        ndarray::Zip::indexed(grad_input_padded.outer_iter_mut())
            .par_for_each(|b, mut grad_in_batch| {
                for c_out in 0..self.out_channels {
                     let weight_kernel = self.weights.slice(s![c_out, .., .., ..]);
                     for h in 0..out_h {
                         for w in 0..out_w {
                             let d_val = delta[[b, c_out, h, w]];
                             if d_val == 0.0 { continue; } 

                             let h_start = h * stride;
                             let w_start = w * stride;

                             for c_in in 0..self.in_channels {
                                 for ky in 0..kernel_size {
                                     for kx in 0..kernel_size {
                                         grad_in_batch[[c_in, h_start + ky, w_start + kx]] += d_val * weight_kernel[[c_in, ky, kx]];
                                     }
                                 }
                             }
                         }
                     }
                }
            });

        let grad_input = grad_input_padded.slice(s![.., .., self.padding..self.padding+in_h, self.padding..self.padding+in_w]).to_owned();
        grad_input
    }

    fn update_weights(&mut self, optimizer: &mut dyn Optimizer, layer_id: usize) {
        {
            let mut w_view = self.weights.view_mut().into_dyn();
            let g_view = self.grad_weights.view().into_dyn();
            optimizer.update(layer_id, "weights", &mut w_view, &g_view);
        }
        {
            let mut b_view = self.biases.view_mut().into_dyn();
            let g_view = self.grad_biases.view().into_dyn();
            optimizer.update(layer_id, "biases", &mut b_view, &g_view);
        }
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }
    
    fn layer_type(&self) -> LayerType { LayerType::Conv2D }
    fn as_any(&self) -> &dyn Any { self }
}