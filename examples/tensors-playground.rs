use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Distribution, Tensor};

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

    let tf: Tensor<Wgpu, 2> = Tensor::random([2, 5], Distribution::Uniform(0., 1.), &device);
    println!("[ARGMAX along dim test] tf = {}", tf);
    println!("[ARGMAX along dim test] argmax(tf, 0) = {}", tf.clone().argmax(0)); // finds which row# has the biggest number (result is 1r x 5c)
    println!("[ARGMAX along dim test] argmax(tf, 1) = {}", tf.argmax(1)); // finds which column# has the biggest number (result is 2r x 1c)

}