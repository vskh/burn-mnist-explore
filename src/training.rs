use crate::data::MnistBatcher;
use crate::model::ModelConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::LearnerBuilder;
use std::path::Path;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir<P: AsRef<Path>>(artifact_dir: P) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(&artifact_dir).ok();
    std::fs::create_dir_all(&artifact_dir).ok();
}

pub fn train<B: AutodiffBackend, P: AsRef<Path>>(
    artifact_dir: P,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(&artifact_dir);
    config
        .save(format!("{}/config.json", artifact_dir.as_ref().display()))
        .expect("Failed to save training config.");

    B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_test = MnistBatcher::<B::InnerBackend>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let dataloader_validate = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_validate);

    model_trained
        .save_file(
            format!("{}/model", artifact_dir.as_ref().display()),
            &CompactRecorder::new(),
        )
        .expect("Failed to save trained model.");
}
