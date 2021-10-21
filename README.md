# You Only Look Once: Unified, Real-Time Object Detection
A Rust implementation of [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) for object detection and tracking.

#### Run
* Install [OpenCV](https://opencv.org/)
* Run `get_model.sh` to get the config files for the neural network or download it from [here](https://pjreddie.com/darknet/yolo/).
* `cargo run -- --file path/to/video.mp4 --config path/to/yolo.cfg --weights path/to/yolo.weights --coco path/to/coco.names`
