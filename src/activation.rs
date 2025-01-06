pub trait Activation: Send + Sync {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
    fn name(&self) -> &'static str; // 激活函数名称，用于序列化与反序列化
}

pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        let sig = self.activate(x);
        sig * (1.0 - sig)
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

pub struct None;

impl Activation for None {
    fn activate(&self, x: f64) -> f64 {
        x
    }
    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
    fn name(&self) -> &'static str {
        "None"
    }
}

// 用于反序列化激活函数
pub fn activation_from_name(name: &str) -> Box<dyn Activation> {
    match name {
        "ReLU" => Box::new(ReLU),
        "Sigmoid" => Box::new(Sigmoid),
        "None" => Box::new(None),
        _ => panic!("Unknown activation function: {}", name),
    }
}
