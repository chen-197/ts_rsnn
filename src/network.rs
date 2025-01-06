use crate::activation::Activation;
use crate::activation_from_name;
use crate::layers::conv::Initializer;
use crate::layers::*;
use crate::{Conv2D, Dense, LayerType, MaxPool2D};
use ndarray::prelude::*;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{self, File};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
struct LayerData {
    layer_type: String,
    weights: Option<Vec<f64>>,
    biases: Option<Vec<f64>>,
    input_shape: Option<Vec<usize>>,
    output_shape: Option<Vec<usize>>,
    hyperparameters: Option<String>, // e.g., "kernel_size: 3, stride: 1, padding: 1"
    activation: Option<String>,      // 激活函数的名称
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn Activation>>,
    inputs: Vec<Array4<f64>>,
    origin_outputs: Vec<Array4<f64>>,
    has_fc_layer: bool,
    output_shapes: Vec<[usize; 3]>, // 存储每一层的输出形状（不包括批次大小）
}

impl Network {
    pub fn get_layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }

    pub fn get_activations(&self) -> &Vec<Box<dyn Activation>> {
        &self.activations // 添加 Getter 方法
    }

    /// 保存模型到文件
    pub fn save_model(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut layers_data = vec![];

        for ((layer, output_shape), activation) in self
            .layers
            .iter()
            .zip(&self.output_shapes)
            .zip(&self.activations)
        {
            let layer_data = match layer.layer_type() {
                LayerType::Dense => {
                    let dense = layer.as_any().downcast_ref::<Dense>().unwrap();
                    LayerData {
                        layer_type: "Dense".to_string(),
                        weights: Some(dense.weights.iter().cloned().collect()),
                        biases: Some(dense.biases.iter().cloned().collect()),
                        input_shape: None,
                        output_shape: Some(output_shape.clone().to_vec()),
                        hyperparameters: None,
                        activation: Some(activation.name().to_string()), // 保存激活函数的名称
                    }
                }
                LayerType::Conv2D => {
                    let conv = layer.as_any().downcast_ref::<Conv2D>().unwrap();
                    LayerData {
                        layer_type: "Conv2D".to_string(),
                        weights: Some(conv.weights.iter().cloned().collect()),
                        biases: Some(conv.biases.iter().cloned().collect()),
                        input_shape: None,
                        output_shape: Some(output_shape.clone().to_vec()),
                        hyperparameters: Some(format!(
                            "kernel_size: {}, stride: {}, padding: {}",
                            conv.kernel_size, conv.stride, conv.padding
                        )),
                        activation: Some(activation.name().to_string()),
                    }
                }
                LayerType::MaxPool2D => {
                    let pool = layer.as_any().downcast_ref::<MaxPool2D>().unwrap();
                    LayerData {
                        layer_type: "MaxPool2D".to_string(),
                        weights: None,
                        biases: None,
                        input_shape: None,
                        output_shape: Some(output_shape.clone().to_vec()),
                        hyperparameters: Some(format!(
                            "pool_size: {}, stride: {}",
                            pool.pool_size, pool.stride
                        )),
                        activation: Some(activation.name().to_string()),
                    }
                }
                _ => unimplemented!("Unsupported layer type"),
            };
            layers_data.push(layer_data);
        }

        // 将层数据保存为 JSON 格式
        let json = serde_json::to_string_pretty(&layers_data)?;
        let mut file = File::create(file_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// 从文件加载模型
    pub fn load_model(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(file_path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let layers_data: Vec<LayerData> = serde_json::from_str(&json)?;

        self.layers.clear();
        self.activations.clear();
        self.output_shapes.clear();

        for layer_data in layers_data {
            let activation = if let Some(activation_name) = layer_data.activation {
                activation_from_name(&activation_name) // 使用现有的 activation_from_name 函数
            } else {
                panic!("Missing activation function in saved model!");
            };

            match layer_data.layer_type.as_str() {
                "Dense" => {
                    let input_size = layer_data.weights.as_ref().unwrap().len()
                        / layer_data.output_shape.as_ref().unwrap()[2];
                    let output_size = layer_data.output_shape.as_ref().unwrap()[2];
                    let mut dense = Dense::new(input_size, output_size);
                    dense.set_weights(
                        Array2::from_shape_vec(
                            (input_size, output_size),
                            layer_data.weights.unwrap(),
                        )
                        .unwrap(),
                    );
                    dense.set_biases(Array1::from(layer_data.biases.unwrap()));
                    self.add_layer(
                        Box::new(dense),
                        Some(activation), // 恢复激活函数
                        layer_data
                            .output_shape
                            .map(|shape| [shape[0], shape[1], shape[2]]),
                    );
                }
                "Conv2D" => {
                    // 解析超参数
                    let params = if let Some(hyperparams) = &layer_data.hyperparameters {
                        parse_hyperparameters(hyperparams, &["kernel_size", "stride", "padding"])
                    } else {
                        panic!("Missing hyperparameters for Conv2D layer");
                    };
                    let (kernel_size, stride, padding) = (params[0], params[1], params[2]);

                    // 从 JSON 数据推导出通道数
                    let out_channels = layer_data.output_shape.as_ref().unwrap()[0];
                    let in_channels = layer_data.weights.as_ref().unwrap().len()
                        / (out_channels * kernel_size * kernel_size);

                    // 初始化卷积层
                    let mut conv = Conv2D::new(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        crate::layers::conv::Initializer::Zero,
                    );

                    // 检查权重大小
                    let weights = layer_data.weights.unwrap();
                    let expected_size =
                        conv.out_channels * conv.in_channels * conv.kernel_size * conv.kernel_size;
                    assert_eq!(
                        weights.len(),
                        expected_size,
                        "Expected weights size {}, but got {}",
                        expected_size,
                        weights.len()
                    );

                    // 设置权重
                    conv.set_weights(
                        Array4::from_shape_vec(
                            (
                                conv.out_channels,
                                conv.in_channels,
                                conv.kernel_size,
                                conv.kernel_size,
                            ),
                            weights,
                        )
                        .unwrap(),
                    );

                    // 设置偏置
                    conv.set_biases(Array1::from(layer_data.biases.unwrap()));

                    // 添加到网络
                    self.add_layer(
                        Box::new(conv),
                        Some(activation),
                        layer_data
                            .output_shape
                            .map(|shape| [shape[0], shape[1], shape[2]]),
                    );
                }

                "MaxPool2D" => {
                    // 使用 parse_hyperparameters 提取参数
                    let params = if let Some(hyperparams) = &layer_data.hyperparameters {
                        parse_hyperparameters(hyperparams, &["pool_size", "stride"])
                    } else {
                        panic!("Missing hyperparameters for MaxPool2D layer");
                    };
                    let (pool_size, stride) = (params[0], params[1]);

                    let pool = MaxPool2D::new(pool_size, stride);

                    self.add_layer(
                        Box::new(pool),
                        Some(activation),
                        layer_data
                            .output_shape
                            .map(|shape| [shape[0], shape[1], shape[2]]),
                    );
                }

                _ => unimplemented!("Unsupported layer type"),
            }
        }

        Ok(())
    }

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

    pub fn add_layer(
        &mut self,
        layer: Box<dyn Layer>,
        activation: Option<Box<dyn Activation>>,
        output_shape: Option<[usize; 3]>,
    ) {
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
        self.activations
            .push(activation.unwrap_or_else(|| Box::new(crate::activation::ReLU)));

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

        let reshaped_output: ArrayD<f64> =
            output.into_shape(IxDyn(&reshaped_target_shape)).unwrap();
        reshaped_output.into_dimensionality::<Ix4>().unwrap()
    }

    pub fn forward(&mut self, input: Array4<f64>) -> Array4<f64> {
        self.inputs.clear();
        self.origin_outputs.clear();
        let mut output = input.clone();
        self.inputs.push(input);

        for (layer, (activation, target_shape)) in self
            .layers
            .iter()
            .zip(self.activations.iter().zip(&self.output_shapes))
        {
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
        for (layer, (input, (origin_output, activation))) in self.layers.iter_mut().rev().zip(
            self.inputs.iter().rev().skip(1).zip(
                self.origin_outputs
                    .iter()
                    .rev()
                    .zip(self.activations.iter().rev()),
            ),
        ) {
            if let LayerType::Dense = layer.layer_type() {
                let mut activation_derivative_output = origin_output.clone();
                activation_derivative_output.par_mapv_inplace(|x| activation.derivative(x));
                grad = layer.backward(
                    input.clone(),
                    grad.clone(),
                    activation_derivative_output.clone(),
                );
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

fn parse_hyperparameters(hyperparams: &str, keys: &[&str]) -> Vec<usize> {
    keys.iter()
        .map(|key| {
            hyperparams
                .split(", ")
                .find(|s| s.starts_with(&format!("{}:", key)))
                .unwrap_or_else(|| panic!("Missing {} in hyperparameters: {}", key, hyperparams))
                .split(": ")
                .last()
                .unwrap()
                .parse::<usize>()
                .unwrap_or_else(|_| {
                    panic!(
                        "Invalid value for {} in hyperparameters: {}",
                        key, hyperparams
                    )
                })
        })
        .collect()
}
