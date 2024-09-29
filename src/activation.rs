pub trait Activation: Send + Sync {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn derivative(&self, _x: f64) -> f64 {
        if _x > 0.0 { 1.0 } else { 0.0 }
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
}

pub struct None;

impl Activation for None {
    fn activate(&self, x: f64) -> f64 { x }
    fn derivative(&self, _x: f64) -> f64 { 1.0 }
}
