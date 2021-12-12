use crate::functions::{Function, get_function};
use crate::weights::Weight1d;

pub struct Output{
    pub output: f64,
    pub gradient: f64,
    pub synapse: Vec<Weight1d>,
    pub act_func: Box<dyn Function>,
}

impl Output{
    #![allow(unused)]
    pub fn init(output:f64, gradient:f64, length:i64, func:&str) -> Output{
        let mut synapse = Vec::with_capacity(length as usize);
        for _ in 0..length{
            synapse.push(Weight1d::new());
        }
        let act_func = get_function(func);
        Output{output, gradient,synapse,act_func}
    }
    pub fn forward(&mut self, input: &Vec<f64>){
        assert_eq!(self.synapse.len(), input.len(), "input to weight mismatch, expected {} found {}", self.synapse.len(), input.len());
        let length = input.len();
        let mut sum = 0.0;
        for i in 0..length {
            sum += input[i] * self.synapse[i].get_weight();
        }
        self.output = self.act_func.output(sum);
    }
    pub fn backward(&mut self){}

    pub fn output(&mut self)-> f64{
        let output = self.output;
        return output
    }
}
