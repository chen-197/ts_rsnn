use ndarray::prelude::*;
use crate::activation::Activation;
use crate::Layer;
use crate::LayerType;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    grad_weights: Array2<f64>,
    grad_biases: Array1<f64>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // 初始化权重
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::ones(output_size);
        let grad_weights = Array2::zeros((input_size, output_size));
        let grad_biases = Array1::zeros(output_size);
        Dense {
            weights,
            biases,
            grad_weights,
            grad_biases,
        }
    }

    pub fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        // 将输入展平成2D数组
        let (batch_size, _channels, _height, _width) = input.dim();
        let input_use = input.into_shape((batch_size, self.weights.nrows())).unwrap();

        // 矩阵乘法
        let mut output_use = parallel_matrix_multiplication(&input_use, &self.weights) + &self.biases;

        // 将输出重新转换为4D数组
        output_use.into_shape((batch_size, 1, 1, self.weights.ncols())).unwrap()
    }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64> {
        // 将输入和输出梯度展平成2D数组
        let (batch_size, _channels, _height, _width) = input.dim();
        let input_use = input.into_shape((batch_size, self.weights.nrows())).unwrap();
        let mut grad_output_use = grad_output.into_shape((batch_size, self.weights.ncols())).unwrap();
        let mut activation_derivative_output_use = activation_derivative_output.into_shape((batch_size, self.weights.ncols())).unwrap();

        // 计算激活梯度与误差梯度之乘积，并行化
        let mut a_g_m_e_g = activation_derivative_output_use;
        //a_g_m_e_g.par_mapv_inplace(|x| activation.derivative(x));
        a_g_m_e_g.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(grad_output_use.axis_iter_mut(Axis(0)))
            .for_each(|(mut a,b)| {
                a *= &b;
            });

        // 本地变量保存权重和偏置梯度
        let grad_weights = Arc::new(Mutex::new(Array2::zeros(self.grad_weights.dim())));
        let grad_biases = Arc::new(Mutex::new(Array1::zeros(self.grad_biases.dim())));
        //println!("输入：{:?}",input_use);
        //println!("激活：{:?}",a_g_m_e_g);
        // 并行计算权重和偏置梯度
        input_use.axis_iter(Axis(0))
            .into_par_iter()
            .zip(a_g_m_e_g.axis_iter(Axis(0)))
            .for_each(|(input_row, grad_output_row)| {
                //println!("左：{:?}",input_row.insert_axis(Axis(1)));
                //println!("右：{:?}",&grad_output_row.insert_axis(Axis(0)));
                //println!("外积：{:?}",input_row.insert_axis(Axis(1)).dot(&grad_output_row.insert_axis(Axis(0))));
                //let grad_w = input_row.insert_axis(Axis(1)).dot(&grad_output_row.insert_axis(Axis(0)));
                let grad_w = parallel_matrix_multiplication(&input_row.insert_axis(Axis(1)).to_owned(),&grad_output_row.insert_axis(Axis(0)).to_owned());
                {
                    let mut grad_weights_lock = grad_weights.lock().unwrap();
                    *grad_weights_lock += &grad_w;
                }
                {
                    let mut grad_biases_lock = grad_biases.lock().unwrap();
                    *grad_biases_lock += &grad_output_row;
                }
            });

        // 将并行计算的结果累加到类的成员变量
        self.grad_weights += &*grad_weights.lock().unwrap();
        self.grad_biases += &*grad_biases.lock().unwrap();

        // 计算输入梯度
        //let grad_input_use = a_g_m_e_g.dot(&self.weights.t());
        let grad_input_use = parallel_matrix_multiplication(&a_g_m_e_g,&self.weights.t().to_owned());

        // 将输入梯度重新转换为4D数组
        grad_input_use.into_shape((batch_size, 1, 1, self.weights.nrows())).unwrap()
    }

    fn update_weights(&mut self, learning_rate: f64) {
        // 并行更新权重
        self.weights
            .as_slice_mut()
            .expect("Failed to get mutable slice of weights")
            .par_iter_mut()
            .zip(self.grad_weights.as_slice().expect("Failed to get slice of grad_weights").par_iter())
            .for_each(|(w, &grad_w)| {
                *w -= learning_rate * grad_w;
            });

        // 并行更新偏置
        self.biases
            .as_slice_mut()
            .expect("Failed to get mutable slice of biases")
            .par_iter_mut()
            .zip(self.grad_biases.as_slice().expect("Failed to get slice of grad_biases").par_iter())
            .for_each(|(b, &grad_b)| {
                *b -= learning_rate * grad_b;
            });

        // 清空梯度
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }
}

impl Layer for Dense {
    fn forward(&self, input: Array4<f64>) -> Array4<f64> {
        self.forward(input)
    }

    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: ndarray::Array4<f64>) -> Array4<f64> {
        self.backward(input, grad_output, activation_derivative_output)
    }

    fn update_weights(&mut self, learning_rate: f64) {
        self.update_weights(learning_rate)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Dense
    }
}

fn parallel_matrix_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (n, m) = (a.nrows(), a.ncols());
    let (_, p) = (b.nrows(), b.ncols());

    assert!(m == b.nrows(), "Number of columns in A must equal number of rows in B");

    let mut c = Array2::<f64>::zeros((n, p));

    c.axis_chunks_iter_mut(Axis(0), 1)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let mut row = row.index_axis_mut(Axis(0), 0);
            for j in 0..p {
                row[j] = a.row(i).dot(&b.column(j));
            }
        });

    c
}