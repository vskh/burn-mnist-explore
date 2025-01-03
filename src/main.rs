use burn::backend::Wgpu;
use crate::model::ModelConfig;

mod model;
mod data;

type BE = Wgpu;

fn main() {
    let device = Default::default();
    let model = ModelConfig::new().init::<BE>(&device);

    println!("Model: {}", model);

    
}
