use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use std::collections::HashMap;

/// 优化器特征
pub trait Optimizer: Send + Sync {
    /// 在每一轮 batch 更新前调用（可选，用于增加时间步 t 等）
    fn step(&mut self);

    /// 更新参数
    /// layer_id: 层的索引，用于区分不同层的参数
    /// param_id: 参数名称（如 "weights", "biases"）
    /// param: 可变的参数数据（视图）
    /// grad: 梯度数据（视图）
    fn update(&mut self, layer_id: usize, param_id: &str, param: &mut ArrayViewMutD<f64>, grad: &ArrayViewD<f64>);
}

/// 基础的随机梯度下降 (SGD)
pub struct SGD {
    pub lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {}

    fn update(&mut self, _layer_id: usize, _param_id: &str, param: &mut ArrayViewMutD<f64>, grad: &ArrayViewD<f64>) {
        // param = param - lr * grad
        param.zip_mut_with(grad, |p, g| *p -= self.lr * *g);
    }
}

/// Adam 优化器
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    t: i32, // 时间步
    // 状态存储: Key 是 (layer_id, param_id), Value 是 (m, v)
    state: HashMap<(usize, String), (ArrayD<f64>, ArrayD<f64>)>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            state: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
    }

    fn update(&mut self, layer_id: usize, param_id: &str, param: &mut ArrayViewMutD<f64>, grad: &ArrayViewD<f64>) {
        let key = (layer_id, param_id.to_string());

        // 如果状态不存在，初始化为 0
        let (m, v) = self.state.entry(key).or_insert_with(|| {
            (
                ArrayD::zeros(grad.shape()), // m
                ArrayD::zeros(grad.shape()), // v
            )
        });

        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let t = self.t as f64;

        // 计算偏差修正项
        let correction1 = 1.0 - beta1.powf(t);
        let correction2 = 1.0 - beta2.powf(t);

        ndarray::azip!((p in param, g in grad, m in m, v in v) {
            // 更新一阶矩
            *m = beta1 * (*m) + (1.0 - beta1) * *g;
            // 更新二阶矩
            *v = beta2 * (*v) + (1.0 - beta2) * *g * *g;

            // 计算修正后的估计量
            let m_hat = *m / correction1;
            let v_hat = *v / correction2;

            // 更新参数
            *p -= lr * m_hat / (v_hat.sqrt() + epsilon);
        });
    }
}