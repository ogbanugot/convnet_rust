//use convnet::Data;
//use convnet::forward::Forwardprop;
//use convnet::gradient::backward::Backprop;
//use convnet::Weights;

fn main() {
    /*
    let data_vec = vec![0.05, 0.07, 0.01, 0.99, 0.45, 0.55];
    let weight_vec = vec![0.14, 0.15, 0.18, 0.24, 0.35, 0.40, 0.45, 0.48];
    let epoch = 300;
    let lr = 0.5;
    let mut weights = Weights::new(weight_vec);
    let data = Data::new(data_vec);
    let mut forwardprop = Forwardprop::new();
    let mut backwardprop = Backprop::new();
    for _number in 1..epoch{
        forwardprop.forward(&weights,&data);
        println!("Total error {}", forwardprop.etotal);
        backwardprop.backward(&weights,&data,&forwardprop);
        weights.weight_adjustment(&backwardprop,&lr);
    }
     */
    /*
    let v1 = vec![1, 2, 3, 4];
    let v2 = vec![5, 6, 7, 8];
    let v3 = vec![v1, v2];
    let v3_iter = v3.iter();
    for v in v3_iter{
        let v_iter = v.iter();
        for val in v_iter{
            println!("{}", val);
        }
    }
     */
    let v1 = vec![1, 2, 3, 4];
    let v2 = &v1[0..2];
    let v3 = v2.iter();
    for val in v3{
        println!("{}", val);
    }
    //let v2 = vec![5, 6, 7, 8];
    //let v3 = vec![v1, v2];
    //println!("Len: {}", v2[0].len());
}