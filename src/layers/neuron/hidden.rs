use crate::layers::{Layer, Layer1d, Layer2d};
use crate::nodes::neuron::hidden::Hidden;
use crate::weights::Weight;
use ndarray::{ArrayBase, OwnedRepr, Array1, Ix, Ix1, Ix2, Ix3, Dim, Array};
use crate::functions::{Function, get_function};

#[cfg(test)]
mod tests {
    use crate::layers::neuron::hidden;
    use crate::nodes::neuron;
    use crate::weights::Weight1d;
    use crate::functions::get_function;

    #[test]
    fn hidden_forward_new() {
        /*
        let mut layer = hidden::L1d::new(2, 3, "relu");
        let input = vec![0.3, 0.4, 0.2];
        layer.forward1d(input);
        //assert_eq!();

         */
    }
    #[test]
    fn hidden_forward_init() {
        /*
        let input = vec![0.99, 0.13];
        let synapse1 = vec![Weight1d::init(0.16), Weight1d::init(0.22)];
        let neuron1 = neuron::hidden::Hidden{output:0.0, gradient:0.0, synapse:synapse1, act_func:get_function("linear")};
        let synapse2 = vec![Weight1d::init(0.11), Weight1d::init(0.31)];
        let neuron2 = neuron::hidden::Hidden{output:0.0, gradient:0.0, synapse:synapse2, act_func:get_function("linear")};
        let mut layer = hidden::L1d::init(vec![neuron1, neuron2], 2);
        let output = layer.forward1d(input);
        assert_eq!(0.187, (output[0]* 1000.0).round() / 1000.0);
        assert_eq!(0.149, (output[1]* 1000.0).round() / 1000.0);
         */
    }
}
struct L1d{}

struct Data{
    pub input: ArrayBase<OwnedRepr<f64>, Ix1>,
    pub weight: Weight,
    pub incoming_length: usize,
    pub out_length: usize,
    pub act_func: Box<dyn Function>,
}

impl  L1d{
    #![allow(unused)]
    fn new (incoming_length:usize, out_length: usize, activation_function: &str) -> Box<dyn Layer1d> {
        let act_func = get_function(activation_function);
        let mut input = Array1::zeros((incoming_length));
        let mut weight = Weight::new(out_length , incoming_length);
        Box::new(Data{ input, weight, incoming_length, out_length, act_func})
    }

    fn init(weight:Weight, incoming_length:usize, out_length:usize, activation_function:&str) -> Box<dyn Layer1d> {
        let act_func = get_function(activation_function);
        let mut input = Array1::zeros((incoming_length));
        Box::new(Data{input, weight, incoming_length,out_length, act_func})
    }
}

impl Layer for Data{}
impl Layer1d for Data{
    fn forward1d(&mut self, input: ArrayBase<OwnedRepr<f64>, Ix1>)->ArrayBase<OwnedRepr<f64>, Ix1>{
        assert_eq!(self.incoming_length, input.len(), "input length mismatch, expected {} found {}", self.incoming_length, input.len());
        let output: ArrayBase<OwnedRepr<f64>, Ix1> = self.weight.value.dot(&input);

        self.input = input;
        output
    }
    fn backward1d(&mut self, _: ArrayBase<OwnedRepr<f64>, Ix1>)->ArrayBase<OwnedRepr<f64>, Ix1>{
        unimplemented!()
    }
    fn length1d(&self) -> usize {
        self.out_length
    }
}
impl  Layer2d for Data {
    fn forward2d(&self, _: ArrayBase<OwnedRepr<f64>, Ix3>){unimplemented!()}
    fn backward2d(&self, _:ArrayBase<OwnedRepr<f64>, Ix3>){unimplemented!()}
    fn length2d(&self) -> usize{unimplemented!()}
}