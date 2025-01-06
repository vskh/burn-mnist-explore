# Burn MNIST exploration
Use [Burn](https://burn.dev/) to train a simplistic 768-16-16-10 DNN
mentioned in [3blue1brown](https://www.youtube.com/@3blue1brown)
[Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series
on [MNIST dataset](https://yann.lecun.com/exdb/mnist/index.html) and see what happens.

(Just following the Burn book really)

## Results
Reached 91.9% accuracy :)

### ======================== Learner Summary ========================

Model:
>  ```
>  Model {
>    activation: Relu
>    hidden1: Linear {d_input: 768, d_output: 16, bias: true, params: 12304}
>    hidden2: Linear {d_input: 16, d_output: 10, bias: true, params: 170}
>    output: Linear {d_input: 10, d_output: 10, bias: true, params: 110}
>    params: 12584
>  }
>  ```
Total Epochs: 10


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 53.035   | 1        | 91.817   | 10       |
| Train | Loss     | 0.285    | 10       | 1.536    | 1        |
| Valid | Accuracy | 78.310   | 1        | 91.940   | 10       |
| Valid | Loss     | 0.282    | 10       | 0.897    | 1        |
