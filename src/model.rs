use crate::layers::Layer;
use crate::dataloader::Input;

pub struct Model{
    layers:Vec<Box<dyn Layer>>,
}

impl Model{
    #![allow(unused)]
    fn new()->Model{
        let layers = Vec::new();
        Model{layers }
    }
    fn add<T: 'static + Layer>(&mut self, layer:T){
        let layer= Box::new(layer);
        self.layers.push(layer);
    }
    fn forward(&mut self, input:Box<dyn Input>){
        //first layer
        //assert_eq!(input.len(), )
        //for i in 1..self.layers.len(){

        //}
    }
}