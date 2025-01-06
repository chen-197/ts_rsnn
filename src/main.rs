use rust_nn::network::Network;
use rust_nn::layers::{Dense, Conv2D, MaxPool2D};
use rust_nn::activation::{ReLU, Sigmoid, None};
use rust_nn::loss::{MeanSquaredError, Loss};
use rust_nn::layers::conv::Initializer;
use ndarray::{s, Array4, Ix4};
use rust_nn::image_utils::{load_image_as_tensor, load_image_dataset};
use image::GrayImage;
use image::Luma;
fn main() {
    let mut network = Network::new();

    // 添加卷积层和池化层
    network.add_layer(
        Box::new(Conv2D::new(1, 8, 3, 1, 1, Initializer::He)),
        Some(Box::new(ReLU)),
        Some([8, 28, 28]),
    );
    network.add_layer(
        Box::new(MaxPool2D::new(2, 2)),
        Some(Box::new(ReLU)),
        Some([8, 14, 14]),
    );
    network.add_layer(
        Box::new(Dense::new(8*8, 8*8)), // [batch_size, 1, 1, 10]
        Some(Box::new(Sigmoid)),
        Some([1, 1, 8*8]), // 只指定通道数和尺寸
    );

    // 保存模型
    network.save_model("test_model.json").unwrap();

    // 加载模型
    let mut loaded_network = Network::new();
    loaded_network.load_model("test_model.json").unwrap();

    // 使用 getter 方法访问 activations
    for (layer, activation) in loaded_network.get_layers().iter().zip(loaded_network.get_activations().iter()) {
        println!("Layer: {:?}, Activation: {}", layer.layer_type(), activation.name());
    }
}
