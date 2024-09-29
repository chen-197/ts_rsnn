pub mod dense;
pub mod conv;
pub mod pool;
use crate::Activation;
pub use dense::Dense;
pub use conv::Conv2D;
pub use pool::MaxPool2D;
use ndarray::Array4;
pub trait Layer {
    fn forward(&self, input: Array4<f64>) -> Array4<f64>;
    fn backward(&mut self, input: Array4<f64>, grad_output: Array4<f64>, activation_derivative_output: Array4<f64>) -> Array4<f64>;
    fn update_weights(&mut self, learning_rate: f64);
    fn layer_type(&self) -> LayerType;
}

#[derive(PartialEq)]
pub enum LayerType {
    Conv2D,
    MaxPool2D,
    Dense,
    Other,
}
