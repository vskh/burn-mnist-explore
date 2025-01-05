use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::module::{Content, DisplaySettings, ModuleDisplayDefault};
use burn::tensor::Tensor;

fn main() {
    println!("Hello, Tensors!");

    let device: WgpuDevice = Default::default();

    let t1: Tensor<Wgpu, 1> = Tensor::ones([5], &device);
    println!("t1 = {}", t1);

    let t2: Tensor<Wgpu, 2> = Tensor::ones([5, 5], &device);
    println!("t2 = {}", t2);

    let ct1: Tensor<Wgpu, 2> = Tensor::ones([1, 5], &device);
    let ct2: Tensor<Wgpu, 2> = Tensor::ones([1, 5], &device);
    println!("ct1 = {}\nct2 = {}", ct1, ct2);

    let ct = Tensor::cat(vec![ct1, ct2], 0);
    println!("[Concat along dim test] ct = {:?}", ct);
}