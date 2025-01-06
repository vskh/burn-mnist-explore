use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Mode {
    Train,
    Infer
}

#[derive(Parser)]
#[command(name = "burn-mnist-explore")]
#[command(version = "0.1")]
#[command(about = "Train model/classify using data from MNIST handwritten digits.", long_about = None)]
pub struct Args {
    #[arg(value_enum)]
    pub mode: Mode
}