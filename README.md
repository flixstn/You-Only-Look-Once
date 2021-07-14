# You Only Look Once: Unified, Real-Time Object Detection
A Rust implementation of [Yolo](https://arxiv.org/abs/1506.02640) for object detection and tracking.

#### Run
* Install [OpenCV](https://opencv.org/)
* Run `get_model.sh` to get the config files for the neural network or download it from [here](https://pjreddie.com/darknet/yolo/).
* `cargo run -- --file path/to/video.mp4 --config path/to/yolov3.cfg --weights path/to/yolov3.weights --coco path/to/coco.names`
