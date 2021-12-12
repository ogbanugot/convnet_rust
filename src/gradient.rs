pub mod backward{
    use crate::{Weights, Data};
    use crate::forward::Forwardprop;
    pub struct Backprop {
        pub dw1: f64,
        pub dw2: f64,
        pub dw3: f64,
        pub dw4: f64,
        pub dw5: f64,
        pub dw6: f64,
        pub dw7: f64,
        pub dw8: f64,
    }

    impl Backprop {
        pub fn new() -> Backprop{
            Backprop{
                dw1: 0.0,
                dw2: 0.0,
                dw3: 0.0,
                dw4: 0.0,
                dw5: 0.0,
                dw6: 0.0,
                dw7: 0.0,
                dw8: 0.0,
            }
        }
        pub fn backward(&mut self, weights: &Weights, data: &Data, forward: &Forwardprop) {
            let doutz1 = forward.outz1-data.z1;
            let doutz2 = forward.outz2-data.z2;
            let dnetz1 = forward.outz1*(1.0-forward.outz1);
            let dnetz2 = forward.outz2*(1.0-forward.outz2);

            //gradient with respect to output layer weights
            self.dw5 = doutz1*(forward.outy1*dnetz1);
            self.dw6 = doutz1*(forward.outy2*dnetz1);
            self.dw7 = doutz2*(forward.outy1*dnetz2);
            self.dw8 = doutz2*(forward.outy2*dnetz2);

            //gradient with respect to hidden layer outputs
            let d_ez1_outy1 = weights.w5*(doutz1*dnetz1);
            let d_ez2_outy1 = weights.w7*(doutz2*dnetz2);
            let douty1 = d_ez1_outy1 + d_ez2_outy1;
            let d_ez1_outy2 = weights.w6*(doutz1*dnetz1);
            let d_ez2_outy2 = weights.w8*(doutz2*dnetz2);
            let douty2 = d_ez1_outy2 + d_ez2_outy2;
            let dnety1 = forward.outy1*(1.0-forward.outy1);
            let dnety2 = forward.outy2*(1.0-forward.outy2);

            //gradient with respect to hidden layer weights
            self.dw1 = douty1*(data.x1*dnety1);
            self.dw2 = douty1*(data.x2*dnety1);
            self.dw3 = douty2*(data.x1*dnety2);
            self.dw4 = douty2*(data.x2*dnety2);
        }
    }
}