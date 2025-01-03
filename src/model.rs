use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    activation: Relu,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, pixel_count] = images.dims();
    
        let x = images;
    
        let x = self.hidden1.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden2.forward(x);
        let x = self.activation.forward(x);
        let x = self.output.forward(x);
    
        softmax(x, 1)
    }
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
