**rust_nn: 高性能纯 Rust 深度学习框架**

rust_nn 是一个从零开始编写的、轻量级且高性能的深度学习库。它完全使用 Rust 语言实现，不依赖任何底层的 C++ 库（如 OpenBLAS 或 CUDA），却能通过优秀的并行化策略在 CPU 上实现惊人的训练速度。

该项目旨在为初学者熟悉Rust本身以及神经网络相关原理提供的Demo，支持卷积神经网络（CNN）和全连接网络（MLP）的构建、训练与推理。

✨ 项目特点
 * 纯 Rust 实现：无任何复杂的 C/C++ 依赖链。
 * 极致并行性能：基于 Rayon 和 ndarray 实现了细粒度的并行计算（Batch × Channel 级并行）。
 * 功能完备：
   * 支持 Conv2D（卷积层）、MaxPool2D（池化层）、Dense（全连接层）。
   * 实现 反向传播（Backpropagation） 与自动梯度推导。
   * 内置 Adam 和 SGD 优化器。
   * 支持 ReLU、Sigmoid 等激活函数。
   * 支持 CrossEntropy（交叉熵）和 MSE（均方误差）损失函数。
 * 模型持久化：支持将训练好的模型保存为 JSON 文件，并随时加载进行推理。
 * 零依赖推理：推理阶段仅需标准库支持，极易集成到其他 Rust 应用中。
📦 如何在其他项目中引入
你可以通过 Cargo 直接引入此项目作为依赖。
1. 添加依赖
在你的新项目 Cargo.toml 中添加：
```toml
[dependencies]
ndarray = "0.15"  # 必须引入 ndarray 用于数据构造
# 方式一：通过 Git 引入 (推荐)
rust_nn = { git = "https://github.com/你的用户名/你的仓库名.git", branch = "main" }

# 方式二：本地路径引入 (开发调试用)
# rust_nn = { path = "../rust_nn" }
```

2. ⚠️ 关键性能配置 (必读)
为了获得预期的性能，考虑在你的项目 Cargo.toml 中添加以下编译优化配置：
```toml
[profile.release]
opt-level = 3       # 最高优化等级
lto = true          # 开启链接时优化
codegen-units = 1   # 牺牲编译速度换取运行速度
strip = true        # 减小二进制体积
panic = "abort"     # 发生错误直接退出
```

🚀 快速开始
准备工作：MNIST 数据集
本项目包含一个内置的 MNIST 训练示例。在运行之前，请确保项目根目录下存在 data 文件夹，并包含以下 未压缩 的数据集文件（注意文件名必须完全一致，无 .gz 后缀）：
/data
  ├── train-images-idx3-ubyte
  ├── train-labels-idx1-ubyte
  ├── t10k-images-idx3-ubyte
  └── t10k-labels-idx1-ubyte

运行训练示例
我们提供了一个完整的训练脚本 src/main.rs，它会训练一个 CNN 网络，并展示结果。
```
cargo run --release
```

📚 API 使用指南
1. 构建网络
```rust
use rust_nn::network::Network;
use rust_nn::layers::{Dense, Conv2D, MaxPool2D};
use rust_nn::activation::{ReLU, Sigmoid};
use rust_nn::layers::conv::Initializer;
use rust_nn::optimizer::Adam;

let mut network = Network::new();

// 设置优化器 (Adam, 学习率 0.001)
network.set_optimizer(Box::new(Adam::new(0.001)));

// 添加卷积层: Input [1, 28, 28] -> Output [8, 28, 28]
network.add_layer(
    Box::new(Conv2D::new(1, 8, 3, 1, 1, Initializer::He)),
    Some(Box::new(ReLU)),
    Some([8, 28, 28]),
);

// 添加池化层: Output [8, 14, 14]
network.add_layer(
    Box::new(MaxPool2D::new(2, 2)),
    Some(Box::new(ReLU)), 
    Some([8, 14, 14]),
);

// 添加全连接层: 8*14*14 -> 10
network.add_layer(
    Box::new(Dense::new(1568, 10)),
    Some(Box::new(Sigmoid)),
    Some([1, 1, 10]),
);
```

2. 前向传播与训练
```rust
use ndarray::Array4;

// 构造输入 [Batch, Channel, Height, Width]
let input = Array4::<f64>::zeros((1, 1, 28, 28));

// 前向传播
let output = network.forward(input);

// 反向传播 (假设 loss_grad 是计算好的损失梯度)
let loss_grad = Array4::<f64>::zeros((1, 1, 1, 10)); 
network.backward(loss_grad);

// 更新权重
network.update_weights();
```

3. 保存与加载
```rust
// 保存
network.save_model("my_model.json").unwrap();

// 加载
let mut new_network = Network::new();
new_network.load_model("my_model.json").unwrap();
```

📄 许可证
本项目采用 GPL v3 许可证。这意味着你可以自由使用、修改和分发本项目，但如果你在本项目基础上开发了新的软件并发布，该衍生软件也必须开源并使用 GPL v3 许可。
详情请参阅 LICENSE 文件。
