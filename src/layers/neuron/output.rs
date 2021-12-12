use crate::layers::{Layer, Layer1d, Layer2d};
use crate::nodes::neuron::output::Output;

#[cfg(test)]
mod tests {
    use crate::layers::neuron::output;
    use crate::nodes::neuron;
    use crate::weights::Weight1d;
    use crate::functions::get_function;

    #[test]
    fn output_forward_new() {
        /*
        let mut layer = output::L1d::new(2, 3, "linear");
        let input = vec![0.3, 0.4, 0.2];
        let _ = layer.forward1d(input);
        //println!("Output 1: {}", output[0]);
        //println!("Output 2: {}", output[1]);
        //assert_eq!();

         */
    }
    #[test]
    fn output_forward_init() {
        /*
        let input = vec![0.99, 0.13];
        let synapse1 = vec![Weight1d::init(0.16), Weight1d::init(0.22)];
        let neuron1 = neuron::output::Output{output:0.0, gradient:0.0, synapse:synapse1, act_func:get_function("linear")};
        let synapse2 = vec![Weight1d::init(0.11), Weight1d::init(0.31)];
        let neuron2 = neuron::output::Output{output:0.0, gradient:0.0, synapse:synapse2, act_func:get_function("linear")};
        let mut layer = output::L1d::init(vec![neuron1, neuron2], 2);
        let output = layer.forward1d(input);
        assert_eq!(0.187, (output[0]* 1000.0).round() / 1000.0);
        assert_eq!(0.149, (output[1]* 1000.0).round() / 1000.0);

         */
    }
}
pub struct L1d {}

pub struct Data{
    pub nodes: Vec<Output>,
    pub incoming_length: usize,
}

impl  L1d{
    #![allow(unused)]
    pub fn new (length:i64, incoming_length: i64, activation_function: &str) -> Box<Data> {
        let mut vec_nodes = Vec::with_capacity(length as usize);
        for _ in 0..length{
            let neu = Output::init(0.0, 0.0, incoming_length, activation_function);
            vec_nodes.push(neu);
        }
        Box::new(Data{ nodes: vec_nodes, incoming_length:incoming_length as usize})
    }

    pub fn init(nodes: Vec<Output>, incoming_length:i64) -> Box<Data> {
        Box::new(Data{nodes, incoming_length:incoming_length as usize})
    }
}
/*
impl Layer for Data{}
impl Layer1d for Data{
    fn forward1d(&mut self, input: Vec<f64>)->Vec<f64>{
        assert_eq!(self.incoming_length, input.len(), "input length mismatch, expected {} found {}", self.incoming_length, input.len());
        let mut output = Vec::new();
        for i in 0..self.nodes.len(){
            self.nodes[i].forward(&input);
            output.push(self.nodes[i].output());
        }
        output
    }
    fn backward1d(&mut self, _: Vec<f64>)->Vec<f64>{
        unimplemented!()
    }
    fn length1d(&self) -> usize {
        self.nodes.len()
    }
}
impl  Layer2d for Data {
    fn forward2d(&self, _: Vec<Vec<Vec<f64>>>){unimplemented!()}
    fn backward2d(&self, _:Vec<Vec<Vec<f64>>>){unimplemented!()}
    fn length2d(&self) -> usize{unimplemented!()}
}

 */