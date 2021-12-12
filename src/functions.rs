pub trait Function{
    fn output(&mut self, input:f64) -> f64;
    fn derivative(&mut self, input:f64) -> f64;
}
pub mod non_linear_old;
mod non_linear;

pub struct Linear {
    pub output: f64,
    pub derivative: f64,
}
impl Function for Linear {
    fn output(&mut self, input:f64) -> f64{
        input
    }
    fn derivative(&mut self, input:f64) -> f64{
        input
    }
}

//Register function here
use crate::functions::non_linear_old::{ReLu, Sigmoid, Tanh};

pub fn get_function(func: &str) -> Box<dyn Function> {
    match func {
        "relu" => {Box::new(ReLu{output:0.0, derivative:0.0})},
        "sigmoid" => {Box::new(Sigmoid{output:0.0, derivative:0.0})},
        "tanh" => {Box::new(Tanh{output:0.0, derivative:0.0})},
        "linear" => {Box::new(Linear{output:0.0, derivative:0.0})},
        _ => {panic!("Function name incorrect, must be one of the following [relu, sigmoid, tanh, linear]")}
    }
}