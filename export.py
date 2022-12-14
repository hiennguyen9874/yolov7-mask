import argparse
import sys
import time
import warnings

sys.path.append("./")  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import End2End, End2EndMask, attempt_load
from utils.activations import Hardswish, SiLU
from utils.add_nms import RegisterNMS
from utils.general import check_img_size, set_logging
from utils.torch_utils import select_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./yolor-csp-c.pt", help="weights path")
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[640, 640], help="image size"
    )  # height, width
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="dynamic batch onnx for tensorrt and onnx-runtime",
    )
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument(
        "--max-wh",
        type=int,
        default=None,
        help="max width,height using for nms",
    )
    parser.add_argument("--topk-all", type=int, default=100, help="topk objects for every images")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="iou threshold for NMS")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="conf threshold for NMS")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--simplify", action="store_true", help="simplify onnx model")
    parser.add_argument("--include-nms", action="store_true", help="export end2end onnx")
    parser.add_argument("--fp16", action="store_true", help="CoreML FP16 half-precision export")
    parser.add_argument("--int8", action="store_true", help="CoreML INT8 quantization")
    parser.add_argument(
        "--trt", action="store_true", help="True for tensorrt, false for onnx-runtime"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="True for using onnx_graphsurgeon to sort and remove unused",
    )
    parser.add_argument("--attn-resolution", type=int, default=14, help="attn-resolution")
    parser.add_argument("--mask-resolution", type=int, default=56, help="mask-resolution")
    parser.add_argument("--num-base", type=int, default=5, help="num-base")
    parser.add_argument("--pooler-scale", type=float, default=0.25, help="RoiAlign: scale")
    parser.add_argument("--sampling-ratio", type=int, default=1, help="RoiAlign: sampling-ratio")

    opt = parser.parse_args()

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(
        device
    )  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = False  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    import onnx

    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    f = opt.weights.replace(".pt", ".onnx")  # filename
    model.eval()
    output_names = ["output"]

    if opt.end2end and opt.trt:
        output_names = [
            "num_dets",
            "det_boxes",
            "det_scores",
            "det_classes",
            "det_mask",
        ]

    dynamic_axes = None

    if opt.dynamic_batch:
        opt.batch_size = "batch"
        dynamic_axes = {"images": {0: "batch"}}
        if opt.end2end:
            if opt.trt:
                output_axes = {
                    "num_dets": {0: "batch"},
                    "det_boxes": {0: "batch"},
                    "det_scores": {0: "batch"},
                    "det_classes": {0: "batch"},
                    "det_mask": {0: "batch"},
                }
            else:
                output_axes = {"output": {0: "num_dets"}}
        else:
            output_axes = {"output": {0: "batch"}}
        dynamic_axes.update(output_axes)

    if opt.end2end:
        if opt.trt:
            shapes = [
                opt.batch_size,
                1,
                opt.batch_size,
                opt.topk_all,
                4,
                opt.batch_size,
                opt.topk_all,
                1,
                opt.batch_size,
                opt.topk_all,
                1,
                opt.batch_size,
                opt.topk_all,
                opt.mask_resolution * opt.mask_resolution,
            ]
        else:
            shapes = ["num_dets", 1 + 4 + 1 + 1 + opt.mask_resolution * opt.mask_resolution]

    if opt.end2end:
        print(
            "\nStarting export end2end-mask onnx model for %s..." % "TensorRT"
            if opt.max_wh is None
            else "onnxruntime"
        )
        model = End2EndMask(
            model,
            opt.topk_all,
            opt.iou_thres,
            opt.conf_thres,
            opt.max_wh,
            device,
            trt=opt.trt,
            attn_resolution=opt.attn_resolution,
            mask_resolution=opt.mask_resolution,
            num_base=opt.num_base,
            pooler_scale=opt.pooler_scale,
        )
    else:
        model.model[-1].concat = True

    torch.onnx.export(
        model,
        img,
        f,
        verbose=False,
        opset_version=14,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
    )

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if opt.end2end:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    # Metadata
    d = {
        "stride": int(max(model.model.stride if opt.end2end else model.stride)),
        "names": model.model.names if opt.end2end else model.names,
    }
    for k, v in d.items():
        meta = onnx_model.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(onnx_model, f)

    if opt.simplify:
        try:
            import onnxsim

            print("\nStarting to simplify ONNX...")
            onnx_model, check = onnxsim.simplify(
                onnx_model,
                include_subgraph=True,
                # check_n=10,
                test_input_shapes={"images": list(img.shape)},
            )
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")

    if opt.cleanup:
        try:
            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(onnx_model)
            graph = graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model, f)
    print("ONNX export success, saved as %s" % f)

    if opt.include_nms:
        print("Registering NMS plugin for ONNX...")
        mo = RegisterNMS(f)
        mo.register_nms()
        mo.save(f)

    # except Exception as e:
    #     print("ONNX export failure: %s" % e)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )
