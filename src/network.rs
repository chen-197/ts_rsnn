use crate::layers::*;
use crate::activation::Activation;
use ndarray::prelude::*;
use ndarray::ArrayD;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn Activation>>,
    inputs: Vec<Array4<f64>>,
    origin_outputs: Vec<Array4<f64>>,
    has_fc_layer: bool,
    output_shapes: Vec<[usize; 3]>, // 存储每一层的输出形状（不包括批次大小）
}

impl Network {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            activations: Vec::new(),
            inputs: Vec::new(),
            origin_outputs: Vec::new(),
            has_fc_layer: false,
            output_shapes: Vec::new(), // 初始为空
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>, activation: Option<Box<dyn Activation>>, output_shape: Option<[usize; 3]>) {
        if self.has_fc_layer {
            match layer.layer_type() {
                LayerType::Conv2D | LayerType::MaxPool2D => {
                    panic!("Cannot add Conv2D or MaxPool2D layer after a fully connected layer");
                }
                _ => {}
            }
        }

        if let LayerType::Dense = layer.layer_type() {
            self.has_fc_layer = true;
        }

        self.layers.push(layer);
        self.activations.push(activation.unwrap_or_else(|| Box::new(crate::activation::ReLU)));

        if let Some(shape) = output_shape {
            self.output_shapes.push(shape);
        }
    }

    fn reshape_output(&self, output: Array4<f64>, target_shape: [usize; 3]) -> Array4<f64> {
        let batch_size = output.shape()[0]; // 自动获取批次大小
        let mut reshaped_target_shape = target_shape.to_vec();
        reshaped_target_shape.insert(0, batch_size); // 插入批次大小

        assert_eq!(
            output.len(),
            reshaped_target_shape.iter().product::<usize>(),
            "Output length and target shape length do not match"
        );

        let reshaped_output: ArrayD<f64> = output.into_shape(IxDyn(&reshaped_target_shape)).unwrap();
        reshaped_output.into_dimensionality::<Ix4>().unwrap()
    }

    pub fn forward(&mut self, input: Array4<f64>) -> Array4<f64> {
        self.inputs.clear();
        self.origin_outputs.clear();
        let mut output = input.clone();
        self.inputs.push(input);

        for (layer, (activation, target_shape)) in self.layers.iter().zip(self.activations.iter().zip(&self.output_shapes)) {
            output = layer.forward(output);
            self.origin_outputs.push(output.clone());
            output.mapv_inplace(|x| activation.activate(x));
            self.inputs.push(output.clone());

            output = self.reshape_output(output, *target_shape);
        }
        output
    }

    pub fn backward(&mut self, loss_grad: Array4<f64>) {
        let mut grad = loss_grad;
        for (layer, (input, (origin_output, activation))) in self.layers.iter_mut().rev().zip(self.inputs.iter().rev().skip(1).zip(self.origin_outputs.iter().rev().zip(self.activations.iter().rev()))) {
            if let LayerType::Dense = layer.layer_type() {
                let mut activation_derivative_output = origin_output.clone();
                activation_derivative_output.par_mapv_inplace(|x| activation.derivative(x));
                grad = layer.backward(input.clone(), grad.clone(), activation_derivative_output.clone());
            } else {
                // 遇到非全连接层，停止反向传播
                break;
            }
        }
    }

    pub fn update_weights(&mut self, learning_rate: f64) {
        for layer in self.layers.iter_mut() {
            if let LayerType::Dense = layer.layer_type() {
                layer.update_weights(learning_rate);
            }
        }
    }
}
