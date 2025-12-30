//测试用例
mod mnist_loader; // 引入加载器

use rust_nn::network::Network;
use rust_nn::layers::{Dense, Conv2D, MaxPool2D};
use rust_nn::activation::{ReLU, Sigmoid};
use rust_nn::layers::conv::Initializer;
use rust_nn::optimizer::Adam;
use rust_nn::loss::{CrossEntropy, Loss};
use ndarray::{Array4, s};
use std::time::Instant;

fn main() {
    println!("Loading MNIST dataset...");
    let data = mnist_loader::load_data();
    println!("Data loaded. Train size: {}, Test size: {}", data.train_images.dim().0, data.test_images.dim().0);

    // 2. 构建网络
    let mut network = Network::new();
    
    network.set_optimizer(Box::new(Adam::new(0.001)));

    // Layer 1: Conv2D (1 input channel, 8 filters, 3x3 kernel)
    // Input: [Batch, 1, 28, 28] -> Output: [Batch, 8, 28, 28] (Padding=1)
    network.add_layer(
        Box::new(Conv2D::new(1, 8, 3, 1, 1, Initializer::He)),
        Some(Box::new(ReLU)),
        Some([8, 28, 28]),
    );

    // Layer 2: MaxPool (2x2)
    // Input: [Batch, 8, 28, 28] -> Output: [Batch, 8, 14, 14]
    network.add_layer(
        Box::new(MaxPool2D::new(2, 2)),
        Some(Box::new(ReLU)), // 池化层后通常不需要激活，但加了也无害，这里主要起传递作用
        Some([8, 14, 14]),
    );

    // Layer 3: Dense (Fully Connected)
    // Input size: 8 * 14 * 14 = 1568
    // Output size: 10 
    network.add_layer(
        Box::new(Dense::new(8 * 14 * 14, 10)),
        Some(Box::new(Sigmoid)), 
        Some([1, 1, 10]),
    );

    let loss_func = CrossEntropy;
    let batch_size = 32;
    let epochs = 3; 
    let train_len = data.train_images.dim().0;

    // 3. 训练循环
    for epoch in 0..epochs {
        let start_time = Instant::now();
        let mut total_loss = 0.0;
        let mut batches = 0;

        for i in (0..train_len).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, train_len);
            
            // 获取当前 Batch
            let input_batch = data.train_images.slice(s![i..end, .., .., ..]).to_owned();
            let label_batch = data.train_labels.slice(s![i..end, .., .., ..]).to_owned();
            
            // Forward
            let output = network.forward(input_batch);

            // 准备计算 Loss (转换为 2D: Batch x 10)
            let output_2d = output.clone().into_shape((output.dim().0, 10)).unwrap();
            let label_2d = label_batch.clone().into_shape((label_batch.dim().0, 10)).unwrap();

            // Compute Loss & Gradient
            let loss = loss_func.compute(&output_2d, &label_2d);
            let loss_grad_2d = loss_func.gradient(&output_2d, &label_2d);

            // 将梯度转回 4D 传给 Network Backward: [Batch, 1, 1, 10]
            let loss_grad_4d = loss_grad_2d.into_shape((output.dim().0, 1, 1, 10)).unwrap();

            // Backward & Update
            network.backward(loss_grad_4d);
            network.update_weights();

            total_loss += loss;
            batches += 1;

            if batches % 100 == 0 {
                print!("\rEpoch {} | Batch {}/{} | Loss: {:.4}", epoch+1, batches, train_len/batch_size, loss);
            }
        }
        
        let duration = start_time.elapsed();
        println!("\nEpoch {} finished in {:.2?} | Avg Loss: {:.4}", epoch+1, duration, total_loss / batches as f64);
        
        // 4. 每个 Epoch 结束后进行评估
        evaluate(&mut network, &data.test_images, &data.test_labels);
    }
    // 1. 保存模型
    let save_path = "mnist_model.json";
    println!("Saving trained model to '{}'...", save_path);
    if let Err(e) = network.save_model(save_path) {
        eprintln!("Failed to save model: {}", e);
        return;
    }
    println!("Model saved successfully.");

    println!("Loading model into a fresh Network instance...");
    let mut loaded_network = Network::new();
    
    if let Err(e) = loaded_network.load_model(save_path) {
        eprintln!("Failed to load model: {}", e);
        return;
    }
    println!("Model loaded successfully.");

    // 3. 验证加载后的模型
    // 理论上，loaded_network 的准确率应该和训练结束时的 network 完全一致
    println!("Evaluating loaded model on test set...");
    evaluate(&mut loaded_network, &data.test_images, &data.test_labels);

    println!("---------------------------------");
    println!("Verification Complete!");
}

fn evaluate(network: &mut Network, images: &Array4<f64>, labels: &Array4<f64>) {
    let output = network.forward(images.clone());
    let batch_size = output.dim().0;
    let mut correct = 0;

    for i in 0..batch_size {
        // 找到预测概率最大的索引
        let pred_idx = argmax(output.slice(s![i, 0, 0, ..]));
        // 找到真实标签的索引
        let true_idx = argmax(labels.slice(s![i, 0, 0, ..]));

        if pred_idx == true_idx {
            correct += 1;
        }
    }

    println!("Test Accuracy: {:.2}% ({}/{})", (correct as f64 / batch_size as f64) * 100.0, correct, batch_size);
}

fn argmax(view: ndarray::ArrayView1<f64>) -> usize {
    let mut max_idx = 0;
    let mut max_val = f64::MIN;
    for (i, &val) in view.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}