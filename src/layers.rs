use ndarray::{ArrayBase, OwnedRepr,Ix1, Ix3};

#[cfg(test)]
mod tests {
    use crate::weights::Weight1d;
    use crate::nodes::neuron;
    use crate::functions::get_function;
    use crate::layers::neuron::output;

    #[test]
    fn layer_forward() {
        /*
        let input = vec![0.99, 0.13];
        let hidden_synapse1 = vec![Weight1d::init(0.16), Weight1d::init(0.22)];
        let hidden_neuron1 = neuron::hidden::Hidden{
            output:0.0,
            gradient:0.0,
            synapse: hidden_synapse1,
            act_func:get_function("linear")
        };
        let hidden_synapse2 = vec![Weight1d::init(0.11), Weight1d::init(0.31)];
        let hidden_neuron2 = neuron::hidden::Hidden{
            output:0.0,
            gradient:0.0,
            synapse: hidden_synapse2,
            act_func:get_function("linear")
        };
        let mut hidden_layer = hidden::L1d::init(vec![hidden_neuron1, hidden_neuron2], 2);
        let hidden = hidden_layer.forward1d(input);

        let output_synapse = vec![Weight1d::init(0.16), Weight1d::init(0.22)];
        let output_neuron1 = neuron::output::Output{output:0.0, gradient:0.0, synapse: output_synapse, act_func:get_function("linear")};
        let output_synapse2 = vec![Weight1d::init(0.11), Weight1d::init(0.31)];
        let output_neuron2 = neuron::output::Output{output:0.0, gradient:0.0, synapse: output_synapse2, act_func:get_function("linear")};
        let mut layer = output::L1d::init(vec![output_neuron1, output_neuron2], 2);
        let output = output_layer.forward1d(hidden);
        //assert_eq!();

         */
    }
}
pub trait Layer: Layer1d + Layer2d {
}

pub trait Layer1d {
    fn forward1d(&mut self, input: ArrayBase<OwnedRepr<f64>, Ix1>)->ArrayBase<OwnedRepr<f64>, Ix1>;
    fn backward1d(&mut self, gradient:ArrayBase<OwnedRepr<f64>, Ix1>)->ArrayBase<OwnedRepr<f64>, Ix1>;
    fn length1d(&self) -> usize;
}
pub trait Layer2d {
    fn forward2d(&self, input: ArrayBase<OwnedRepr<f64>, Ix3>);
    fn backward2d(&self, gradient:ArrayBase<OwnedRepr<f64>, Ix3>);
    fn length2d(&self) -> usize;
}
pub mod neuron{
    pub mod output;
    pub mod hidden;
}