# Export Yolov7-mask to ONNX and tensorRT

This implimentation is based on [yolov7](https://github.com/WongKinYiu/yolov7/tree/mask).

## Install

- [TensorRT OSS Plugin](https://github.com/hiennguyen9874/TensorRT)

- [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

## Export

### Export to onnx

- `python3 export.py --weights weights/yolov7-mask.pt --img-size 640 640 --batch-size 1 --end2end --max-wh 640 --simplify --cleanup --topk-all 100 --iou-thres 0.65 --conf-thres 0.35`

### Export to tensorRT

- `python3 export.py --weights weights/yolov7-mask.pt --img-size 640 640 --batch-size 1 --end2end --max-wh 640 --simplify --cleanup --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --trt --dynamic-batch`

- `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-mask.onnx --saveEngine=./weights/yolov7-mask-nms.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640 --shapes=images:1x3x640x640`

## Test

### ONNX

[[scripts]](./tools/Yolov7onnx_mask.ipynb)

### TensorRT

[[scripts]](./tools/YOLOv7trt_mask.ipynb)

## Deepstream

[github.com/hiennguyen9874/deepstream-yolov7-mask](https://github.com/hiennguyen9874/deepstream-yolov7-mask)

## TODO

- ONNX::RoiAlign coordinate_transformation_mode