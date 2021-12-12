pub mod gradient;
pub mod layers;
pub mod functions;
pub mod nodes;
pub mod weights;
pub mod model;
pub mod dataloader;

pub struct Data {
    x1: f64,
    x2: f64,
    z1: f64,
    z2: f64,
    b1: f64,
    b2: f64,
}
impl Data {
    pub fn new(data: Vec<f64>) -> Data{
        Data{
            x1: data[0],
            x2: data[1],
            z1: data[2],
            z2: data[3],
            b1: data[4],
            b2: data[5],
        }
    }
}

pub struct Weights {
    w1: f64,
    w2: f64,
    w3: f64,
    w4: f64,
    w5: f64,
    w6: f64,
    w7: f64,
    w8: f64,
}

impl Weights {
    pub fn new(weights: Vec<f64>) -> Weights{
        Weights{
            w1: weights[0],
            w2: weights[1],
            w3: weights[2],
            w4: weights[3],
            w5: weights[4],
            w6: weights[5],
            w7: weights[6],
            w8: weights[7],
        }
    }
    pub fn weight_adjustment(&mut self, backward: &gradient::backward::Backprop, lr: &f64) {
        self.w1 = self.w1 - backward.dw1*lr;
        self.w2 = self.w2 - backward.dw2*lr;
        self.w3 = self.w3 - backward.dw3*lr;
        self.w4 = self.w4 - backward.dw4*lr;

        self.w5 = self.w5 - backward.dw5*lr;
        self.w6 = self.w6 - backward.dw6*lr;
        self.w7 = self.w7 - backward.dw7*lr;
        self.w8 = self.w8 - backward.dw8*lr;
    }
}
pub mod forward{
    use crate::{Weights, Data};
    const E:f64 = 2.72;

    pub struct Forwardprop {
        pub nety1: f64,
        pub nety2: f64,
        pub outy1: f64,
        pub outy2: f64,
        pub netz1: f64,
        pub netz2: f64,
        pub outz1: f64,
        pub outz2: f64,
        pub ez1: f64,
        pub ez2: f64,
        pub etotal: f64,
    }

    pub fn sigmoid(input: f64) -> f64{
        1.0/(1.0+(E.powf(-input)))
    }

    impl Forwardprop {
        pub fn new() -> Forwardprop{
            Forwardprop{
                nety1: 0.0,
                nety2: 0.0,
                outy1: 0.0,
                outy2: 0.0,
                netz1: 0.0,
                netz2: 0.0,
                outz1: 0.0,
                outz2: 0.0,
                ez1: 0.0,
                ez2: 0.0,
                etotal: 0.0,
            }
        }
        pub fn forward(&mut self, weights: &Weights, data: &Data) {
            self.nety1 = weights.w1* data.x1 + weights.w2*data.x2 + data.b1*1.0;
            self.nety2 = weights.w3*data.x1 + weights.w4*data.x2 + data.b1*1.0;
            self.outy1 = sigmoid(self.nety1);
            self.outy2 = sigmoid(self.nety2);
            //output layer
            self.netz1 = weights.w5*self.outy1 + weights.w6*self.outy2 + data.b2*1.0;
            self.netz2 = weights.w7*self.outy1 + weights.w8*self.outy2 + data.b2*1.0;
            self.outz1 = sigmoid(self.netz1);
            self.outz2 = sigmoid(self.netz2);
            //Calculate total error
            self.ez1 = 0.5*(data.z1-self.outz1).powf(2.0);
            self.ez2 = 0.5*(data.z2-self.outz2).powf(2.0);
            self.etotal = self.ez1 + self.ez2;
        }
    }
}
