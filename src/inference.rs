use std::path::Path;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use crate::data::MnistBatcher;
use crate::training::TrainingConfig;

pub fn infer<B: Backend, P: AsRef<Path>>(artifact_dir: P, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{}/config.json", artifact_dir.as_ref().display()))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{}/model", artifact_dir.as_ref().display()).into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}