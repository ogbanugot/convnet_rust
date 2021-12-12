use std::fs::File;
use std::io::Read;
use std::fs;
use crate::Data;
pub mod mnist;

pub trait DataBatch1d:Iterator{}
pub trait DataBatch2d:Iterator{}
pub trait LabelsBatch:Iterator{}

pub enum InputType{
    One,
    Two,
}
pub struct Labels{
    pub batch:Vec<Vec<f64>>,
}
impl LabelsBatch for Labels{}
impl Iterator for Labels{
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item>{
        unimplemented!()
    }
}

pub struct Data1d{
    pub batch:Vec<Vec<f64>>,
}
impl DataBatch1d for Data1d{}
impl Iterator for Data1d{
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}
pub struct Data2d{
    pub batch:Vec<Vec<Vec<f64>>>,
}
impl DataBatch2d for Data2d{}
impl Iterator for Data2d{
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item>{
        unimplemented!()
    }
}

pub struct Load1d {
}
pub struct Batches1D{
    pub data: Vec<Data1d>,
    pub labels: Vec<Labels>,
}
impl  Iterator for Batches1D{
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

pub struct Load2d {
}
pub struct Batches2D{
    pub data:Vec<Data2d>,
    pub labels:Vec<Labels>,
}
impl  Iterator for Batches2D{
    type Item = Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

pub trait Input: Input1d + Input2d{}
pub trait Input1d{
    fn values(&self) -> Vec<f64>;
    fn shape(&self) -> InputType;
}
pub trait Input2d{
    fn values(&self) -> Vec<Vec<Vec<f64>>>;
    fn shape(&self) -> InputType;
}
pub struct  InputData1d{
    pub values: Vec<f64>,
    pub shape_2d_x:Option<i64>,
    pub shape_2d_y:Option<i64>,
}
pub struct  InputData2d{
    pub values: Vec<Vec<f64>>,
}


