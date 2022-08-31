# Export Yolov7-mask to ONNX and tensorRT

## Install

- TensorRT OSS Plugin: [github.com/hiennguyen9874/TensorRT](https://github.com/hiennguyen9874/TensorRT)

## Export to onnx

- `python3 export.py --weights weights/yolov7-mask.pt --img-size 640 640 --batch-size 1 --grid --end2end --max-wh 640 --simplify --mask --cleanup --topk-all 100 --iou-thres 0.65 --conf-thres 0.35`

## Export to tensorRT

- `python3 export.py --weights weights/yolov7-mask.pt --img-size 640 640 --batch-size 1 --grid --end2end --max-wh 640 --simplify --mask --cleanup --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --trt`

- `CUDA_VISIBLE_DEVICES=1 /usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-mask.onnx --saveEngine=./weights/yolov7-mask-nms.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --shapes=images:1x3x640x640`
