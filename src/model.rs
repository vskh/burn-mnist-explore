use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use crate::data::MnistBatch;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    activation: Relu,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> {
        // let [batch_size, pixel_count] = images.dims();

        let x = images;

        let x = self.hidden1.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden2.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
    pub fn forward_classification(
        &self,
        images: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
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
            activation: Relu::new(),
        }
    }
}
