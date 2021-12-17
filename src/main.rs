//use convnet::layers::neuron::output;
//use convnet::nodes::neuron;
//use convnet::weights::Weight1d;
//use convnet::functions::get_function;
//use convnet::dataloader::mnist::Mnist2d;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{StandardNormal};
use ndarray::{Array1, Array2};
extern crate blas_src;
fn main() {
    /*
    let mut layer = output::L1d::new(2, 3, "linear");
    let input = vec![0.3, 0.4, 0.2];
    let output = layer.forward1d(input);
    println!("Output 1: {}", output[0]);
    println!("Output 2: {}", output[1]);
    let path_train_images = String::from("/home/ugot/Documents/mnist/train-images.idx3-ubyte");
    let path_test_images = String::from("/home/ugot/Documents/mnist/t10k-images.idx3-ubyte");
    let path_train_labels = String::from("/home/ugot/Documents/mnist/train-labels.idx1-ubyte");
    let path_test_labels = String::from("/home/ugot/Documents/mnist/t10k-labels.idx1-ubyte");
    Mnist2d::init(path_train_images, path_train_labels,
                path_test_images, path_test_labels, 33.0, 78.0);
    print!("Done");
     */
    //let a = Array::random((2, 5), Normal::new(2., 3.));
    let w = Array2::<f64>::random((128, 800), StandardNormal);
    let inp = Array1::<f64>::random((800), StandardNormal);
    let b = Array1::<f64>::random((128), StandardNormal);
    let out = w.dot(&inp) + b;
    assert_eq!(out.shape(), &[128]);
}
