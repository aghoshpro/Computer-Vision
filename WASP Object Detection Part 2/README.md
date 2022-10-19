## WASP Object Detectoion Part 2 ([FAQ on Object Detection Using SSD mobilenet](https://madhumitamenon.medium.com/faq-on-object-detection-using-ssd-mobilenet-b8bf31924601))
This particular project detects objects using the mobileNet SSD method. Therefore before proceeding, three files are a pre-requisite — ‘coco.names’, ‘ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt’ and ‘frozen_inference_graph.pb’.

[CoCo Data Set](https://cocodataset.org/#home): The offcial COCO Webpage

### Dataset
[Coco Names](https://github.com/nightrome/cocostuff/blob/master/labels.md): Contains lables of 182 objects that can be detected on the video stream.

### Model Used
[MobileNet-SSD v3](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API): OpenCV needs an extra configuration file to import object detection models from TensorFlow. It's based on a text version of the same serialized graph in protocol buffers format (protobuf).

### Output
https://user-images.githubusercontent.com/71174892/196668206-004059e0-b1c9-4d19-823d-237aa0a582b3.mp4

