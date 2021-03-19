use std::error::Error;
use clap::{App, Arg};


pub struct Args {
    pub file: String,
    pub weights: String,
    pub config: String,
    pub coco: String,
}

impl Args {
    pub fn parse() -> Result<Self, Box<dyn Error>> {
        // parse arguments
        // file: video file input
        // weights: network weights
        // config: network config file
        // coco: coco names file
        let matches = App::new("Knowledge Component Extraction")
        .version("0.1.0")
        .about("Analyzes programming videos and extracts knowledge components")
        .arg(Arg::with_name("file")
                .short("f")
                .long("file")
                .takes_value(true)
                .help("Video or image file as input"))
        .arg(Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Yolo weights"))
        .arg(Arg::with_name("config")
                .short("cfg")
                .long("config")
                .takes_value(true)
                .help("Yolo config file"))
        .arg(Arg::with_name("coco")
                .short("n")
                .long("coco")
                .takes_value(true)
                .help("Coco names"))
        .get_matches();

        // read arguments for network configuration/weights and video file processing
        let video = matches.value_of("file").expect("provide video file").to_owned();
        let model_configuration = matches.value_of("config").expect("provide config file").to_owned();
        let model_weights = matches.value_of("weights").expect("provide weights file").to_owned();
        let coco_names = matches.value_of("coco").expect("provide coco names files").to_owned();

        Ok(Self {
            file: video,
            config: model_configuration,
            weights: model_weights,
            coco: coco_names
        })
    }
}