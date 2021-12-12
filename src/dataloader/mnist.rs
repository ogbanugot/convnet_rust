use crate::dataloader::{Input2d, Batches2D, DataBatch2d, LabelsBatch, InputData2d, Data2d};
use std::fs::File;
use std::fs;
use std::io::Read;

pub struct Mnist2d{
    data: MnistData2d,
    batches: MnistBatches2d,
}
impl Mnist2d{
    pub fn init(path_to_train_images_binary:String, path_to_train_labels_binary:String,
                path_to_test_images_binary:String, path_to_test_labels_binary:String,
                mean:f64, std:f64) -> MnistData2d{
        let train_images = Mnist2d::load_images(path_to_train_images_binary, mean,std);
        let test_images = Mnist2d::load_images(path_to_test_images_binary, mean,std);
        let train_labels = Mnist2d::load_labels(path_to_train_labels_binary);
        let test_labels = Mnist2d::load_labels(path_to_test_labels_binary);
        MnistData2d {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }
    fn load_images(path_to_images_binary:String, mean:f64, std:f64) -> Vec<Vec<Vec<f64>>> {
        let images_buffer = get_file_as_byte_vec(&path_to_images_binary);
        //remove magic number
        let image_bytes = &images_buffer[16..images_buffer.len()];
        let number_of_images = image_bytes.len()/784;
        let mut images = Vec::with_capacity(number_of_images);
        // get each image
        let mut counter = 0;
        for _ in 0..number_of_images {
            let mut pixels = vec![vec![0.0; 28]; 28];
            for i in 0..28{
                for j in 0..28{
                    //Normalize
                    let b = image_bytes[counter];
                    let c = (b as f64) - mean;
                    let d = c / std;
                    pixels[i][j] = d;
                    counter += 1;
                }
            }
            images.push(pixels);
        } // each image
        images
    }
    fn load_labels(path_to_labels_binary:String) -> Vec<Vec<f64>> {
        let labels_buffer = get_file_as_byte_vec(&path_to_labels_binary);
        //remove magic number
        let number_of_images = labels_buffer.len() - 8.0 as usize;
        let label_bytes = &labels_buffer[8..labels_buffer.len()];
        let mut labels = Vec::with_capacity(number_of_images as usize);
        // get each image
        let mut counter = 0;
        for c in 0..number_of_images {
            let lbl = label_bytes[c];
            let lbl_encoded = label_encode_mnist(lbl as i64);
            labels.push(lbl_encoded);
        } // each image
        labels
    }
    fn batches(data:Vec<Vec<Vec<f64>>>, labels:Vec<Vec<f64>>){
        let v: Vec<i32> = vec![1, 1, 1, 2, 2, 2, 3, 3, 3];
        let v_chunked: Vec<Vec<i32>> = v.chunks(3).map(|x| x.to_vec()).collect();
    }
}

pub struct MnistData2d {
    pub train_images: Vec<Vec<Vec<f64>>>,
    pub train_labels: Vec<Vec<f64>>,
    pub test_images: Vec<Vec<Vec<f64>>>,
    pub test_labels: Vec<Vec<f64>>,
}

pub struct MnistBatches2d {
    train_batch: Batches2D,
    test_batch: Batches2D,
}

pub fn get_file_as_byte_vec(filename: &String) -> Vec<u8> {
    let mut f = File::open(&filename).expect("no file found");
    let metadata = fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    buffer
}

pub fn label_encode_mnist(lbl:i64) ->Vec<f64>{
    let zero = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let one = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let two = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let three = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let four = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let five = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let six = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let seven = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let eight = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let nine =  vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    match lbl{
        0 => zero,
        1 => one,
        2 => two,
        3 => three,
        4 => four,
        5 => five,
        6 => six,
        7 => seven,
        8 => eight,
        9 => nine,
        _ => {panic!("Number must be between 0 and 9")}
    }
}

