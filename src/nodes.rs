#[cfg(test)]
mod tests {
    use crate::nodes::neuron::output::Output;
    use crate::functions::get_function;
    use crate::weights::Weight1d;


    #[test]
    fn neuron_output() {
        let input = vec![0.45];
        let weight = Weight1d::init(0.16);
        let vec_weight = vec![weight];
        let func = get_function("linear");
        let mut neu = Output{
            output: 0.0,
            gradient: 0.0,
            synapse: vec_weight,
            act_func: func
        };
        neu.forward(&input);
        assert_eq!(0.072, (neu.output()* 1000.0).round() / 1000.0);
    }

}

pub mod neuron {

    pub mod hidden;

    pub mod output;
}