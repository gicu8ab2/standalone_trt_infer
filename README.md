## Standalone TensorRT inferencing with Yolov5s (tag3.0) on COCO-pretrained models

This code is forked from [TensorRTX](https://github.com/wang-xinyu/tensorrtx) based on the 
Yolov5s tag3.0 Pytorch model at [Ultralytics-Yolov5](https://github.com/ultralytics/yolov5) and 
applies the following modifications

* abstracts away the engine build process (done to make engine file compatible also with Nvidia Deepstream)
so no requirement to install TensorRT
* writes metadata to txt files containing bounding box, objectness confidence, and object class
* annotates output files to include objectness confidence and object centroid
* supply yolov5s.engine file designed to work with Nvidia Deepstream

### Dependencies

* OpenCV 4.X (might work with older OpenCV)
* CUDA >=10.1
* (Only tested on x86_64 architecture with RTX2080Ti GPU)

### Build yolov5s TensorRT inference app (COCO-pretrained)

	mkdir -p build && cd build
	mv ../yolov5s.engine.bz2 .
	bzip2 -d yolov5s.engine.bz2
	cmake ..
	make

	
### Run yolov5s TRT inference on test image directory

	./yolov5s_TRT_infer ../test_dir
	
	
### Sample output	

Sample annotated output image

![Sample Annotated Image](_sample_1080p_056.png "Results")

Sample metadata output: 
(left, up, width, height, confidence, object_class)

	382 457 116 263 0.895267 0
	211 429 126 360 0.835960 0
	75 451 130 332 0.801693 0
	738 508 480 354 0.935859 2
	1417 498 493 215 0.916631 2
	1073 489 174 79 0.864593 2
	622 470 113 88 0.757637 2
	555 470 76 60 0.663252 2
	
	
## References

* [Ultralytics-Yolov5](https://github.com/ultralytics/yolov5)
* [TensorRTX](https://github.com/wang-xinyu/tensorrtx)
* [Nvidia TensorRT Installation Documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)



	
	

