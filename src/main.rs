use std::{
    error::Error, 
    fs::File, 
    io::{BufRead, BufReader}
};
use opencv::{
    core, 
    dnn, 
    highgui, 
    imgproc, 
    prelude::{self, MatTrait, MatTraitManual, NetTrait}, 
    types, 
    videoio::{self, VideoCaptureTrait}
};

mod args;

fn main() {
    // Do not return Result from main, it prints the Debug
    // representation of the error. 
    // Executing it in another function prints the Display version 
    // of the error.
    if let Err(err) = try_main() {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), Box<dyn Error>> {
    // argument parsing
    let args = args::Args::parse()?;

    // initialize video capture 
    let mut video_capture = videoio::VideoCapture::from_file(&args.file, videoio::CAP_ANY)?;

    // initialize neural network
    let mut net = dnn::read_net_from_darknet(&args.config, &args.weights)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;

    // get coco class names
    let classes = read_file(&args.coco)?;

    // run application
    run(&mut video_capture, &mut net, &classes)?;
    
    Ok(())
}

fn run(video_capture: &mut videoio::VideoCapture, net: &mut dnn::Net, classes: &types::VectorOfString) -> Result<(), Box<dyn Error>> {
    // preallocate images
    let mut img = prelude::Mat::default()?;
    let mut blob = prelude::Mat::default()?;

    // define parameters for the network
    let conf_threshold= 0.5_f32;
    let nms_threshold = 0.4_f32;
    let inp_width = 416;
    let inp_height = 416;

    // start video processing
    while highgui::wait_key(1)? < 0 {
        // extract each frame from the video along with the frame size
        video_capture.read(&mut img)?;
        let img_width = img.cols();
        let img_height = img.rows();

        // break loop if video capture has no more frames
        if !video_capture.grab()? {
            println!("Video processing finished");
            break
        }

        // create a 4D blob from image
        dnn::blob_from_image_to(&img, 
                                &mut blob, 
                                1./255., 
                                core::Size::new(inp_width, inp_height), 
                                core::Scalar::new(0.,0.,0., 0.), 
                                false, 
                                false, 
                                core::CV_32F)?;

        // get the names of output layer for bbox naming
        let names = get_output_names(&net)?;

        // forward propagation through the network
        let mut net_output = types::VectorOfMat::new();
        net.set_input(&blob, "", 1.0, core::Scalar::new(0.,0.,0., 0.))?;
        net.forward(&mut net_output, &names)?;

        // scan through all bounding boxes and keep only the ones with high confidence
        let mut class_ids = types::VectorOfi32::new();
        let mut confidences = types::VectorOff32::new();
        let mut boxes = types::VectorOfRect::new();

        // remove the bounding boxes with low confidence using non-maxima suppression
        for (i, matrix) in net_output.iter().enumerate() {          
            for j in 0..matrix.rows() {
                let data = matrix.at_row::<f32>(j as i32)?; 
                let scores = net_output.get(i)?.row(j)?.col_range(&core::Range::new(5, net_output.get(i)?.cols())?)?;
                let mut class_id_point = core::Point::default();
                let mut confidence = 0_f64;

                core::min_max_loc(&scores, 
                               &mut 0., 
                               &mut confidence, 
                               &mut core::Point::new(0,0), 
                               &mut class_id_point, 
                               &core::no_array()?)?;
                if confidence > conf_threshold as f64 {
                    let center_x = (data[0] *  img_width as f32) as i32;
                    let center_y = (data[1] * img_height as f32) as i32;
                    let width = (data[2] * img_width as f32) as i32;                 // w
                    let height = (data[3] * img_height as f32) as i32;               // h
                    let left = center_x - (width / 2);                               // x
                    let top = center_y - (height / 2);                               // y

                    class_ids.push(class_id_point.x);
                    confidences.push(confidence as f32);
                    boxes.push(core::Rect::new(left, top, width, height));
                }
            }
        }

        // perform non maximum suppression to remove redundant overlapping boxes with lower confidences
        let mut indices = types::VectorOfi32::new();
        dnn::nms_boxes(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1., 0)?;

        for num in indices.iter() {
            // get bounding box and associated class
            let bbox = boxes.get(num as usize)?;     
            let label = classes.get(class_ids.get(num as usize)? as usize)?;
            
            // draw predicted bounding box with associated class
            imgproc::rectangle(&mut img, bbox, core::Scalar::new(255., 18., 50., 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(&mut img, 
                              &label, 
                              core::Point::new(bbox.x, bbox.y), 
                              imgproc::FONT_HERSHEY_SIMPLEX, 
                              0.75, 
                              core::Scalar::new(0., 0., 0., 0.), 
                              1, 
                              imgproc::LINE_8, 
                              false)?;
        }

        // show frame 
        highgui::imshow("image", &img)?;
    }

    Ok(())
}

fn get_output_names(net: &dnn::Net) -> Result<types::VectorOfString, Box<dyn Error>> {
    let layers = net.get_unconnected_out_layers()?;
    let layer_names = net.get_layer_names()?;
    
    let mut names = types::VectorOfString::with_capacity(layers.len());

    for (i, _num) in layers.iter().enumerate() {
        let value = layer_names.get((layers.get(i).unwrap() - 1) as usize)?;
        names.insert(i, &value)?;
    }

    Ok(names)
}

fn read_file(file_name: &str) -> Result<types::VectorOfString, Box<dyn Error>> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut vec = types::VectorOfString::new();

    for line in reader.lines() {
        vec.push(line.unwrap().as_str());
    }

    Ok(vec)
}
