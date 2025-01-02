use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    activation: Relu,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    output: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig;

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            hidden1: LinearConfig::new(768, 16).init(device),
            hidden2: LinearConfig::new(16, 10).init(device),
            output: LinearConfig::new(10, 10).init(device),
            activation: Relu::new()
        }
    }
}
