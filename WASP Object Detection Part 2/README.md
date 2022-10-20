## WASP Object Detection Part 2 ([FAQ on Object Detection Using SSD mobilenet](https://madhumitamenon.medium.com/faq-on-object-detection-using-ssd-mobilenet-b8bf31924601))
**WASP_AS_M1_Umeå1**: Arka Ghosh,  Divya Baura,  Joannes Vermant, Julian Mendez and Sabine Houy

This particular project detects objects using the mobileNet SSD method. Therefore before proceeding, three files are a pre-requisite —---------
* ‘coco.names’
* ‘ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt’ 
* ‘frozen_inference_graph.pb’.

### **Reading Materials**

[Object Detection using SSD Mobilenet and Tensorflow Object Detection](https://medium.com/@techmayank2000/object-detection-using-ssd-mobilenetv2-using-tensorflow-api-can-detect-any-single-class-from-31a31bbd0691)

************************************************************************************************************************************************************************
[COCO (Common Objects In Context)](https://cocodataset.org/#home): is a set of challenging, high quality datasets for computer vision, mostly state-of-the-art neural networks. This name is also used to name a format used by those datasets.

### Dataset
[Coco Names](https://github.com/nightrome/cocostuff/blob/master/labels.md): Contains lables of 182 objects that can be detected on the video stream.

### Model Used
[MobileNet-SSD v3](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API): OpenCV needs an extra configuration file to import object detection models from TensorFlow. It's based on a text version of the same serialized graph in protocol buffers format (protobuf).

### Output

#### Output 01 (at 55% detection accuracy)
https://user-images.githubusercontent.com/71174892/196672496-30e54277-dfee-425e-add1-73aedeaefccd.mp4

#### Output 02 (at 60% detection accuracy)
https://user-images.githubusercontent.com/71174892/196671666-5cba1fa3-e1b7-4f75-83d9-a55d4e4aabe0.mp4
