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

    // 设置初始输入形状
    let input_channels = 1;
    let input_height = 125;
    let input_width = 105;

    // 添加卷积层、池化层和全连接层，并指定输出形状（不包括批次大小）
    /*
    network.add_layer(
        Box::new(Conv2D::new(input_channels, 8, 3, 1, 1, Initializer::He)), // [batch_size, 8, 28, 28]
        Some(Box::new(ReLU)),
        Some([8, 125, 105]), // 只指定通道数和尺寸
    );
    network.add_layer(
        Box::new(MaxPool2D::new(2, 2)), // [batch_size, 8, 14, 14]
        Some(Box::new(None)),
        Some([8, 62, 52]), // 只指定通道数和尺寸
    );
    
    network.add_layer(
        Box::new(Dense::new(125*105, 125*105)), // [batch_size, 1, 1, 128]
        Some(Box::new(ReLU)),
        Some([1, 1, 125*105]), // 只指定通道数和尺寸
    );*/
    network.add_layer(
        Box::new(Dense::new(125*105, 105*125)), // [batch_size, 1, 1, 10]
        Some(Box::new(Sigmoid)),
        Some([1, 1, 105*125]), // 只指定通道数和尺寸
    );

    // 加载图像数据集并调整图像尺寸
    let original_dir = r"C:\Users\chen-\Downloads\BaiduNetDiskDownloads\cwhzld\test_one"; // 替换为实际原始图像路径
    let processed_dir = r"C:\Users\chen-\Downloads\BaiduNetDiskDownloads\cwhzld\test_two"; // 替换为实际处理后图像路径
    let image_size = Some((105,125)); // 图像大小
    let x = load_image_dataset(original_dir, image_size);
    let y = load_image_dataset(processed_dir, image_size);
    println!("1");
    let num_samples = x.shape()[0];

    // 选择损失函数
    let loss_fn: Box<dyn Loss> = Box::new(MeanSquaredError);

    // 训练轮数
    let epochs = 355;
    let learning_rate = 0.5;
    let batch_size = 5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        // 分批次训练
        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let x_batch = x.slice(s![batch_start..batch_end, .., .., ..]).to_owned();
            let y_batch = y.slice(s![batch_start..batch_end, .., .., ..]).to_owned();

            // 进行前向传播
            let output = network.forward(x_batch.clone());

            // 将输出和标签展平为2D张量
            let output_shape = output.shape().to_vec();
            let output_2d = output.into_shape((batch_end - batch_start, output_shape[1] * output_shape[2] * output_shape[3])).unwrap();
            let y_batch_shape = y_batch.shape().to_vec();
            let y_batch_2d = y_batch.into_shape((batch_end - batch_start, y_batch_shape[1] * y_batch_shape[2] * y_batch_shape[3])).unwrap();

            // 计算损失
            let loss = loss_fn.compute(&output_2d, &y_batch_2d);
            total_loss += loss;

            // 计算损失的梯度
            let loss_grad = loss_fn.gradient(&output_2d, &y_batch_2d);

            // 将梯度恢复为原来的4D张量形状
            let loss_grad_4d = loss_grad.into_shape(Ix4(output_shape[0], output_shape[1], output_shape[2], output_shape[3])).unwrap();

            network.backward(loss_grad_4d);

            // 更新权重
            network.update_weights(learning_rate*(1.0-(epoch as f64)/epochs as f64+0.01));
        }

        // 打印每轮的平均损失
        if epoch % 1 == 0 {
            println!("Epoch {}: Average Loss: {}", epoch + 1, total_loss / num_samples as f64);
        }
    }

    // 使用训练好的网络进行预测
    let test_image_path = r"C:\Users\chen-\Downloads\BaiduNetDiskDownloads\cwhzld\test_one\1.png"; // 替换为实际图像路径
    let test_input = load_image_as_tensor(test_image_path, image_size);
    let prediction = network.forward(test_input.clone());

    // 假设预测结果是[1, 1, 28, 28]形状
    let prediction = prediction.into_shape((1, 1, 125, 105)).unwrap();
    let prediction = prediction.index_axis(ndarray::Axis(0), 0).to_owned(); // 去掉批次维度
    let prediction = prediction.index_axis(ndarray::Axis(0), 0).to_owned(); // 去掉通道维度

    // 创建图像并保存
    let mut img = GrayImage::new(105, 125);
    for (y, row) in prediction.outer_iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let pixel = (value * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Luma([pixel]));
        }
    }

    img.save("output.png").expect("Failed to save image");
    println!("Prediction saved as output.png");
}