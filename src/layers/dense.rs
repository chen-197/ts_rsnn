use ndarray::prelude::*;
use crate::optimizer::Optimizer;
use crate::Layer;
use crate::LayerType;
use rand::Rng;
use std::any::Any;
use ndarray::{Array2, Array4};

pub struct Dense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    grad_weights: Array2<f64>,
    grad_biases: Array1<f64>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-scale..scale));
        let biases = Array1::zeros(output_size);
        Dense {
            weights, biases,
            grad_weights: Array2::zeros((input_size, output_size)),
            grad_biases: Array1::zeros(output_size),
        }
    }

    pub fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights;
    }

    pub fn set_biases(&mut self, biases: Array1<f64>) {
        self.biases = biases;
    }

    pub fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        let (batch_size, ..) = input.dim();
        let input_use = input.into_shape((batch_size, self.weights.nrows())).unwrap();

        // 1. Matrix Multiplication
        let mut output_use = parallel_matrix_multiplication(&input_use, &self.weights);
        
        // 2. Add Biases
        ndarray::Zip::from(output_use.axis_iter_mut(Axis(0)))
            .par_for_each(|mut row| {
                row += &self.biases;
            });

        output_use.into_shape((batch_size, 1, 1, self.weights.ncols())).unwrap()
    }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        let (batch_size, ..) = input.dim();
        let input_use = input.clone().into_shape((batch_size, self.weights.nrows())).unwrap();
        let grad_output_use = grad_output.into_shape((batch_size, self.weights.ncols())).unwrap();
        let activation_derivative_output_use = activation_derivative_output.into_shape((batch_size, self.weights.ncols())).unwrap();

        // 1. Delta
        let mut delta = activation_derivative_output_use;
        ndarray::Zip::from(&mut delta).and(&grad_output_use).par_for_each(|d, g| *d *= *g);

        // 2. Grad Weights
        let input_t = input_use.t().to_owned(); 
        let grad_w = parallel_matrix_multiplication(&input_t, &delta);
        self.grad_weights += &grad_w;

        // 3. Grad Biases
        let grad_b = delta.sum_axis(Axis(0));
        self.grad_biases += &grad_b;

        // 4. Grad Input
        let weights_t = self.weights.t().to_owned();
        let grad_input_use = parallel_matrix_multiplication(&delta, &weights_t);

        // 将梯度 Reshape 回原始输入的形状
        grad_input_use.into_shape(input.dim()).unwrap()
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
}

fn parallel_matrix_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (n, _m) = (a.nrows(), a.ncols());
    let (_, p) = (b.nrows(), b.ncols());
    let mut c = Array2::<f64>::zeros((n, p));

    ndarray::Zip::indexed(c.outer_iter_mut())
        .par_for_each(|i, mut row| {
            let a_row = a.row(i);
            for j in 0..p {
                row[j] = a_row.dot(&b.column(j));
            }
        });
    c
}

impl Layer for Dense {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> { self.forward(input) }
    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        self.backward(input, grad_output, activation_derivative_output)
    }
    fn update_weights(&mut self, optimizer: &mut dyn Optimizer, layer_id: usize) {
        self.update_weights(optimizer, layer_id)
    }
    fn layer_type(&self) -> LayerType { LayerType::Dense }
    fn as_any(&self) -> &dyn Any { self }
}