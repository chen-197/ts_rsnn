use crate::activation::Activation;
use crate::activation_from_name;
use crate::layers::conv::Initializer;
use crate::layers::*;
use crate::optimizer::Optimizer;
use crate::{Conv2D, Dense, LayerType, MaxPool2D};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{File};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
struct LayerData {
    layer_type: String,
    weights: Option<Vec<f64>>,
    biases: Option<Vec<f64>>,
    input_shape: Option<Vec<usize>>,
    output_shape: Option<Vec<usize>>,
    hyperparameters: Option<Value>,
    activation: Option<String>,
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn Activation>>,
    inputs: Vec<Array4<f64>>,
    origin_outputs: Vec<Array4<f64>>,
    has_fc_layer: bool,
    output_shapes: Vec<[usize; 3]>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl Network {
    pub fn get_layers(&self) -> &Vec<Box<dyn Layer>> { &self.layers }
    pub fn get_activations(&self) -> &Vec<Box<dyn Activation>> { &self.activations }

    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
        self.optimizer = Some(optimizer);
    }

    pub fn save_model(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut layers_data = vec![];

        for ((layer, output_shape), activation) in self.layers.iter().zip(&self.output_shapes).zip(&self.activations) {
            let layer_data = match layer.layer_type() {
                LayerType::Dense => {
                    let dense = layer.as_any().downcast_ref::<Dense>().unwrap();
                    let input_size = dense.weights.nrows();
                    let output_size = dense.weights.ncols();
                    LayerData {
                        layer_type: "Dense".to_string(),
                        weights: Some(dense.weights.iter().cloned().collect()),
                        biases: Some(dense.biases.iter().cloned().collect()),
                        input_shape: Some(vec![input_size]),
                        output_shape: Some(output_shape.to_vec()),
                        hyperparameters: Some(json!({ "input_size": input_size, "output_size": output_size })),
                        activation: Some(activation.name().to_string()),
                    }
                }
                LayerType::Conv2D => {
                    let conv = layer.as_any().downcast_ref::<Conv2D>().unwrap();
                    LayerData {
                        layer_type: "Conv2D".to_string(),
                        weights: Some(conv.weights.iter().cloned().collect()),
                        biases: Some(conv.biases.iter().cloned().collect()),
                        input_shape: Some(vec![conv.in_channels]),
                        output_shape: Some(output_shape.to_vec()),
                        hyperparameters: Some(json!({ "in_channels": conv.in_channels, "out_channels": conv.out_channels, "kernel_size": conv.kernel_size, "stride": conv.stride, "padding": conv.padding })),
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
                        output_shape: Some(output_shape.to_vec()),
                        hyperparameters: Some(json!({ "pool_size": pool.pool_size, "stride": pool.stride })),
                        activation: Some(activation.name().to_string()),
                    }
                }
                _ => return Err(format!("Unsupported layer type: {:?}", layer.layer_type()).into()),
            };
            layers_data.push(layer_data);
        }

        let json = serde_json::to_string_pretty(&layers_data)?;
        let mut file = File::create(file_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load_model(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(file_path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let layers_data: Vec<LayerData> = serde_json::from_str(&json)?;

        self.layers.clear();
        self.activations.clear();
        self.output_shapes.clear();
        self.has_fc_layer = false;

        for (_i, layer_data) in layers_data.into_iter().enumerate() {
            let activation = activation_from_name(layer_data.activation.as_ref().ok_or("Missing activation")?);
            let output_shape_vec = layer_data.output_shape.as_ref().ok_or("Missing output shape")?;
            let output_shape = [output_shape_vec[0], output_shape_vec[1], output_shape_vec[2]];
            let params = layer_data.hyperparameters.as_ref().ok_or("Missing hyperparameters")?;

            match layer_data.layer_type.as_str() {
                "Dense" => {
                    let input_size = params["input_size"].as_u64().unwrap() as usize;
                    let output_size = params["output_size"].as_u64().unwrap() as usize;
                    let mut dense = Dense::new(input_size, output_size);
                    dense.set_weights(Array2::from_shape_vec((input_size, output_size), layer_data.weights.unwrap())?);
                    dense.set_biases(Array1::from(layer_data.biases.unwrap()));
                    self.add_layer(Box::new(dense), Some(activation), Some(output_shape));
                }
                "Conv2D" => {
                    let in_channels = params["in_channels"].as_u64().unwrap() as usize;
                    let out_channels = params["out_channels"].as_u64().unwrap() as usize;
                    let kernel_size = params["kernel_size"].as_u64().unwrap() as usize;
                    let stride = params["stride"].as_u64().unwrap() as usize;
                    let padding = params["padding"].as_u64().unwrap() as usize;
                    let mut conv = Conv2D::new(in_channels, out_channels, kernel_size, stride, padding, Initializer::Zero);
                    conv.set_weights(Array4::from_shape_vec((out_channels, in_channels, kernel_size, kernel_size), layer_data.weights.unwrap())?);
                    conv.set_biases(Array1::from(layer_data.biases.unwrap()));
                    self.add_layer(Box::new(conv), Some(activation), Some(output_shape));
                }
                "MaxPool2D" => {
                    let pool_size = params["pool_size"].as_u64().unwrap() as usize;
                    let stride = params["stride"].as_u64().unwrap() as usize;
                    self.add_layer(Box::new(MaxPool2D::new(pool_size, stride)), Some(activation), Some(output_shape));
                }
                _ => return Err("Unsupported layer type".into()),
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
            output_shapes: Vec::new(),
            optimizer: None,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>, activation: Option<Box<dyn Activation>>, output_shape: Option<[usize; 3]>) {
        if self.has_fc_layer {
            match layer.layer_type() {
                LayerType::Conv2D | LayerType::MaxPool2D => panic!("Cannot add Conv/Pool after Dense"),
                _ => {}
            }
        }
        if let LayerType::Dense = layer.layer_type() { self.has_fc_layer = true; }
        self.layers.push(layer);
        self.activations.push(activation.unwrap_or_else(|| Box::new(crate::activation::ReLU)));
        if let Some(shape) = output_shape { self.output_shapes.push(shape); }
    }

    fn reshape_output(&self, output: Array4<f64>, target_shape: [usize; 3]) -> Array4<f64> {
        let batch_size = output.shape()[0];
        let mut reshaped_target = target_shape.to_vec();
        reshaped_target.insert(0, batch_size);
        // Fixed: Use IxDyn then Dimensionality cast
        let reshaped_dyn = output.into_shape(IxDyn(&reshaped_target)).unwrap();
        reshaped_dyn.into_dimensionality::<Ix4>().unwrap()
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
            let mut activation_derivative_output = origin_output.clone();
            activation_derivative_output.par_mapv_inplace(|x| activation.derivative(x));
            grad = layer.backward(input.clone(), grad.clone(), activation_derivative_output.clone());
        }
    }

    pub fn update_weights(&mut self) {
        if let Some(ref mut opt) = self.optimizer {
            opt.step();
            for (i, layer) in self.layers.iter_mut().enumerate() {
                layer.update_weights(opt.as_mut(), i);
            }
        } else {
            println!("Warning: No optimizer set.");
        }
    }
}