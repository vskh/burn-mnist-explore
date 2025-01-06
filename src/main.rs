use crate::args::{Args, Mode};
use crate::model::ModelConfig;
use crate::training::{train, TrainingConfig};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use clap::Parser;
use std::path::PathBuf;
use burn::data::dataset::Dataset;
use burn::data::dataset::vision::MnistDataset;
use crate::inference::infer;

mod args;
mod data;
mod inference;
mod model;
mod training;

type BE = Wgpu;
type DiffBE = Autodiff<BE>;

fn main() {
    let args = Args::parse();

    let device = WgpuDevice::default();
    let artifacts_dir = PathBuf::from("artifacts");

    match args.mode {
        Mode::Train => train::<DiffBE, _>(
            artifacts_dir,
            TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
            device,
        ),
        Mode::Infer => {
            infer::<BE, _>(artifacts_dir, device, MnistDataset::test().get(args.id).unwrap())
        }
    }
}
