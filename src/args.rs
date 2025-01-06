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
    #[arg(long, value_enum)]
    pub mode: Mode,

    #[arg(
        long,
        requires_if("infer", "mode"),
        value_name = "MNIST-ID",
        long_help = "Image ID to give to the model from MNIST dataset. Dataset can be explored at https://observablehq.com/@davidalber/mnist-viewer"
    )]
    pub id: usize
}