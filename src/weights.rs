use rand::{thread_rng, Rng};
use ndarray::{ArrayBase, OwnedRepr, Array2, Ix, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{StandardNormal, Normal, Uniform};

pub struct Weight1d {
    pub value: f64,
    pub gradient: f64,
}

impl Weight1d{
    #![allow(unused)]
    pub fn new() -> Weight1d{
        let mut rng = thread_rng();
        let w = rng.gen_range(0.0..1.0);
        Weight1d{value:w, gradient:0.0}
    }
    pub fn init(value:f64) ->Weight1d{
        Weight1d{value, gradient:0.0}
    }
    pub fn get_weight(&self) -> f64 {
        self.value
    }
    pub fn get_gradient(&self)-> f64 {
        self.gradient
    }
}

pub struct Weight2d {
    pub value: f64,
    pub gradient: f64,
}
impl Weight2d{
    #![allow(unused)]
    pub fn init(_:i64, _:i64) -> Weight2d{
        unimplemented!()
    }
    pub fn get_weight(&self, _:i64, _:i64)-> f64 {
        self.value
    }
    pub fn get_gradient(&self, _:i64, _:i64)-> f64 {
        self.gradient
    }
    pub fn unroll(&self) -> Vec<f64>{
        unimplemented!()
    }
}

pub struct Weight {
    pub value: ArrayBase<OwnedRepr<f64>, Ix2>,
    pub gradient: ArrayBase<OwnedRepr<f64>, Ix2>,
}

impl Weight{
    #![allow(unused)]
    pub fn new(shape_x:usize, shape_y:usize) -> Weight{
        let w = Array2::<f64>::random((shape_x, shape_y), StandardNormal);
        let g = Array2::<f64>::zeros((shape_x, shape_y));

        Weight{value:w, gradient:g}
    }
    pub fn init(value:ArrayBase<OwnedRepr<f64>, Ix2>) ->Weight{
        let shape = value.shape();
        let g = Array2::<f64>::zeros((shape[0], shape[1]));
        Weight{value, gradient:g}
    }
    // pub fn get_weight(&self) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    //     self.value
    // }
    // pub fn get_gradient(&self)-> ArrayBase<OwnedRepr<f64>, Ix2> {
    //     self.gradient
    // }
}

pub struct Filter {
    pub value: f64,
    pub gradient: f64,
}
impl Filter{
    #![allow(unused)]
    pub fn init(_:i64, _:i64) -> Weight2d{
        unimplemented!()
    }
    pub fn get_weight(&self, _:i64, _:i64)-> f64 {
        self.value
    }
    pub fn get_gradient(&self, _:i64, _:i64)-> f64 {
        self.gradient
    }
    pub fn unroll(&self) -> Vec<f64>{
        unimplemented!()
    }
}