use crate::functions::Function;

pub struct ReLu {
    pub output: f64,
    pub derivative: f64,
}

pub struct Sigmoid {
    pub output: f64,
    pub derivative: f64,
}

pub struct Tanh {
    pub output: f64,
    pub derivative: f64,
}

impl Function for ReLu {
    fn output(&mut self, input:f64) -> f64{
        if input < 0.0 {
            self.output = 0.0;
            0.0
        } else {
            self.output = 1.0;
            1.0
        }
    }
    fn derivative(&mut self, input:f64) -> f64{
        input
    }
}

impl Function for Sigmoid {
    fn output(&mut self, input:f64) -> f64{
        if input < 0.0 {
            self.output = 0.0;
            0.0
        } else {
            self.output = 1.0;
            1.0
        }
    }
    fn derivative(&mut self, input:f64) -> f64{
        input
    }
}

impl Function for Tanh {
    fn output(&mut self, input:f64) -> f64{
        if input < 0.0 {
            self.output = 0.0;
            0.0
        } else {
            self.output = 1.0;
            1.0
        }
    }
    fn derivative(&mut self, input:f64) -> f64{
        input
    }
}
