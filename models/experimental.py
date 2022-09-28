import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.google_utils import attempt_download

from models.common import Conv, DWConv


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        device = boxes.device
        dtype = boxes.dtype
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), device=device, dtype=torch.int32
        )
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, device=device, dtype=dtype)
        det_scores = torch.randn(batch_size, max_output_boxes, device=device, dtype=dtype)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), device=device, dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        out = g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_wh = max_wh  # if max_wh != 0 : non-agnostic else : agnostic

        self.register_buffer("max_obj", torch.tensor([max_obj]))
        self.register_buffer("iou_threshold", torch.tensor([iou_thres]))
        self.register_buffer("score_threshold", torch.tensor([score_thres]))
        self.register_buffer(
            "convert_matrix",
            torch.tensor(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-0.5, 0, 0.5, 0],
                    [0, -0.5, 0, 0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(
            nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold
        )
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y]
        selected_categories = category_id[X, Y].float()
        selected_scores = max_score[X, Y]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device("cpu")
        self.background_class = (-1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(
            boxes,
            scores,
            self.background_class,
            self.iou_threshold,
            self.max_obj,
            self.score_activation,
            self.score_threshold,
        )
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None
    ):
        super().__init__()
        device = device if device else torch.device("cpu")
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


class ORT_RoiAlign(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        output_height=56,
        output_width=56,
        sampling_ratio=1,
        spatial_scale=0.25,
        # coordinate_transformation_mode="output_half_pixel",
        mode="avg",
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale,
        # coordinate_transformation_mode="output_half_pixel",
        mode="avg",
    ):
        return g.op(
            "RoiAlign",
            X,
            rois,
            batch_indices,
            # coordinate_transformation_mode=coordinate_transformation_mode,
            mode_s=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )


class ONNX_ORT_MASK(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=640,
        attn_resolution=14,
        mask_resolution=56,
        num_base=5,
        pooler_scale=0.25,
        sampling_ratio=0,
        device=None,
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_wh = max_wh  # if max_wh != 0 : non-agnostic else : agnostic
        self.attn_resolution = attn_resolution
        self.mask_resolution = mask_resolution
        self.num_base = num_base
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

        self.register_buffer("max_obj", torch.tensor([max_obj]))
        self.register_buffer("iou_threshold", torch.tensor([iou_thres]))
        self.register_buffer("score_threshold", torch.tensor([score_thres]))
        self.register_buffer(
            "convert_matrix",
            torch.tensor(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-0.5, 0, 0.5, 0],
                    [0, -0.5, 0, 0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5:]
        attn = x[1]
        bases = x[2]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        )
        total_object = selected_indices.shape[0]
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y]
        selected_categories = category_id[X, Y].float()
        selected_scores = max_score[X, Y]
        selected_attn = attn[X, Y]

        pooled_bases = ORT_RoiAlign.apply(
            bases,
            selected_boxes,
            X,
            self.mask_resolution,
            self.mask_resolution,
            self.sampling_ratio,
            self.pooler_scale,
        )

        X = X.unsqueeze(1).float()

        selected_attn = selected_attn.view(
            total_object, self.num_base, self.attn_resolution, self.attn_resolution
        )
        selected_attn = F.interpolate(
            selected_attn, (self.mask_resolution, self.mask_resolution), mode="bilinear"
        ).softmax(dim=1)
        masks_preds = (
            (pooled_bases * selected_attn)
            .sum(dim=1)
            .view(total_object, self.mask_resolution * self.mask_resolution)
            .sigmoid()
        )
        return torch.cat([X, selected_boxes, selected_categories, selected_scores, masks_preds], 1)


class TRT_NMS2(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes_per_class=100,
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return_selected_indices = torch.zeros(
            (batch * max_output_boxes_per_class, 3), dtype=torch.int64, device=device
        )
        return_selected_indices[: selected_indices.shape[0]] = selected_indices
        return return_selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes_per_class=100,
    ):
        return g.op(
            "TRT::EfficientNMS_ONNX_TRT",
            boxes,
            scores,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_per_class_i=max_output_boxes_per_class,
            center_point_box_i=0,
            outputs=1,
        )


class TRT_NMS3(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return_selected_indices = torch.zeros(
            (batch * max_output_boxes_per_class, 3), dtype=torch.int64, device=device
        )
        return_selected_indices[: selected_indices.shape[0]] = selected_indices
        return return_selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class TRT_RoiAlign(torch.autograd.Function):
    """TensorRT RoiAlign operation"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        output_height,
        output_width,
        spatial_scale,
        sampling_ratio,
        aligned=1,
        mode="avg",
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        output_height,
        output_width,
        spatial_scale,
        sampling_ratio,
        aligned=1,
        mode="avg",
    ):
        return g.op(
            "TRT::RoIAlignDynamic_TRT",
            X,
            rois,
            output_height_i=output_height,
            output_width_i=output_width,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            mode_s=mode,
            aligned_i=aligned,
            outputs=1,
        )


class ONNX_TRT_MASK(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=640,
        attn_resolution=14,
        mask_resolution=56,
        num_base=5,
        pooler_scale=0.25,
        sampling_ratio=0,
        device=None,
    ):
        super().__init__()
        self.type_nms_padding = 1  # or 1, or 2

        self.device = device if device else torch.device("cpu")
        self.max_wh = max_wh
        self.max_obj_i = max_obj
        self.attn_resolution = attn_resolution
        self.mask_resolution = mask_resolution
        self.num_base = num_base
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

        # For EfficientNMS_ONNX_TRT
        # self.iou_threshold = iou_thres
        # self.score_threshold = score_thres

        # For NonMaxSuppression
        self.register_buffer("max_obj", torch.tensor([max_obj]))
        self.register_buffer("iou_threshold", torch.tensor([iou_thres]))
        self.register_buffer("score_threshold", torch.tensor([score_thres]))

        self.register_buffer(
            "convert_matrix",
            torch.tensor(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-0.5, 0, 0.5, 0],
                    [0, -0.5, 0, 0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5:]
        attn = x[1]
        bases = x[2]

        batch_size = boxes.shape[0]

        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()

        selected_indices = TRT_NMS3.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        ).to(torch.long)

        # # TODO: EfficientNMS_ONNX_TRT current not working
        # selected_indices = TRT_NMS2.apply(
        #     nmsbox,
        #     max_score_tp,
        #     self.score_threshold,
        #     self.iou_threshold,
        #     self.max_obj,
        # ).to(torch.long)

        total_object = selected_indices.shape[0]

        # TODO: split dynamic size not work in tensorRT
        # selected_indices = torch.split(selected_indices, num_object)[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]

        selected_boxes = boxes[X, Y]
        selected_categories = category_id[X, Y].float()
        selected_scores = max_score[X, Y]
        selected_attn = attn[X, Y]
        # X = X.unsqueeze(1).float()

        pooled_bases = TRT_RoiAlign.apply(
            bases,
            # torch.cat((X, selected_boxes), dim=1),
            torch.cat((X.unsqueeze(1).float(), selected_boxes), dim=1),
            self.mask_resolution,
            self.mask_resolution,
            self.pooler_scale,
            self.sampling_ratio,
        )

        selected_attn = selected_attn.view(
            total_object, self.num_base, self.attn_resolution, self.attn_resolution
        )
        selected_attn = F.interpolate(
            selected_attn, (self.mask_resolution, self.mask_resolution), mode="bilinear"
        ).softmax(dim=1)
        masks_preds = (
            (pooled_bases * selected_attn)
            .sum(dim=1)
            .view(total_object, self.mask_resolution * self.mask_resolution)
            .sigmoid()
        )

        if self.type_nms_padding == 0 or self.type_nms_padding == 2:
            # If sum(axis=1) is zero
            num_object1 = (
                torch.topk(
                    torch.where(
                        selected_indices.sum(dim=1) > 0,
                        torch.arange(0, total_object, 1, device=self.device, dtype=torch.int32),
                        torch.zeros(total_object, device=self.device, dtype=torch.int32),
                    ).to(torch.float),
                    k=1,
                    largest=True,
                )[1]
                + 1
            ).reshape((1,))
            num_object = num_object1
        elif self.type_nms_padding == 1 or self.type_nms_padding == 2:
            # Check lag not change
            selected_indices_lag = (selected_indices[1:] - selected_indices[:-1]).sum(dim=1)
            num_object2 = (
                torch.topk(
                    torch.where(
                        selected_indices_lag != 0,
                        torch.arange(0, total_object - 1, device=self.device, dtype=torch.int32),
                        torch.zeros((1,), device=self.device, dtype=torch.int32),
                    ).to(torch.float),
                    k=1,
                    largest=True,
                )[1]
                + 2
            ).reshape((1,))
            num_object = num_object2

        if self.type_nms_padding == 2:
            num_object = (selected_indices_lag.sum() != 0).to(torch.float32) * torch.min(
                num_object1, num_object2
            )

        batch_indices_per_batch = torch.where(
            (
                X.unsqueeze(dim=1)
                == torch.arange(0, batch_size, dtype=X.dtype, device=self.device).unsqueeze(dim=0)
            )
            & torch.where(
                torch.arange(0, total_object, device=self.device, dtype=torch.int32) < num_object,
                torch.ones((1,), device=self.device, dtype=torch.int32),
                torch.zeros((1,), device=self.device, dtype=torch.int32),
            )
            .to(torch.bool)
            .unsqueeze(dim=1),
            torch.ones((1,), device=self.device, dtype=torch.int32),
            torch.zeros((1,), device=self.device, dtype=torch.int32),
        )

        num_det = batch_indices_per_batch.sum(dim=0).view(batch_size, 1).to(torch.int32)

        # TODO: Not working in deepstream
        # obj_idxs = (
        #     torch.cumsum((batch_indices_per_batch).float(), axis=0) * batch_indices_per_batch
        # ).sum(dim=1).to(torch.long) - 1

        # det_boxes = torch.zeros(
        #     (batch_size, self.max_obj_i, 4), device=self.device, dtype=torch.float32
        # )
        # det_scores = torch.zeros(
        #     (batch_size, self.max_obj_i, 1), device=self.device, dtype=torch.float32
        # )
        # det_classes = torch.zeros(
        #     (batch_size, self.max_obj_i, 1), device=self.device, dtype=torch.float32
        # )
        # det_mask = torch.zeros(
        #     (batch_size, self.max_obj_i, self.mask_resolution * self.mask_resolution),
        #     device=self.device,
        #     dtype=torch.float32,
        # )
        # det_attn = torch.zeros(
        #     (batch_size, self.max_obj_i, self.num_base * self.attn_resolution * self.attn_resolution), device=self.device, dtype=torch.float32
        # )

        # det_boxes[X, obj_idxs] = selected_boxes.to(torch.float32)
        # det_scores[X, obj_idxs] = selected_scores.to(torch.float32)
        # det_classes[X, obj_idxs] = selected_categories.to(torch.float32)
        # det_mask[X, obj_idxs] = masks_preds.to(torch.float32)
        # det_attn[X, obj_idxs] = selected_attn
        # return num_det.to(torch.int32), det_boxes, det_scores, det_classes, det_attn, bases

        idxs = (
            torch.topk(
                batch_indices_per_batch.to(torch.float32)
                * torch.arange(0, total_object, dtype=torch.int32, device=self.device).unsqueeze(
                    dim=1
                ),
                k=self.max_obj_i,
                dim=0,
                largest=True,
                sorted=True,
            )[0]
            .t()
            .contiguous()
            .view(-1)
            .to(torch.long)
        )

        det_boxes = selected_boxes[idxs].view(batch_size, self.max_obj_i, 4).to(torch.float32)
        det_scores = selected_scores[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        det_classes = (
            selected_categories[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        )
        det_mask = (
            masks_preds[idxs]
            .view(batch_size, self.max_obj_i, self.mask_resolution * self.mask_resolution)
            .to(torch.float32)
        )
        return num_det, det_boxes, det_scores, det_classes, det_mask


class TRT_NMS4(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        device = boxes.device
        dtype = boxes.dtype

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), device=device, dtype=torch.int32
        )
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, device=device, dtype=dtype)
        det_scores = torch.randn(batch_size, max_output_boxes, device=device, dtype=dtype)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), device=device, dtype=torch.int32
        )
        det_indices = torch.randint(
            0,
            num_boxes,
            (batch_size, max_output_boxes),
            device=device,
            dtype=torch.int32,
        )
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        out = g.op(
            "TRT::EfficientNMSCustom_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        num_det, det_boxes, det_scores, det_classes, det_indices = out
        return num_det, det_boxes, det_scores, det_classes, det_indices


class TRT_ROIAlign2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_map,
        roi,
        pooled_size=56,
        image_size=640,
        sampling_ratio=1,
        roi_coords_absolute=1,
        roi_coords_swap=0,
        roi_coords_transform=2,
        legacy=0,
    ):
        device = roi.device
        dtype = roi.dtype
        ROI_N, ROI_R, ROI_D = roi.shape
        F_N, F_C, F_H, F_W = feature_map.shape
        assert ROI_N == F_N
        return torch.randn(
            (ROI_N, ROI_R, F_C, pooled_size, pooled_size), device=device, dtype=dtype
        )

    @staticmethod
    def symbolic(
        g,
        feature_map,
        roi,
        pooled_size=56,
        image_size=640,
        sampling_ratio=1,
        roi_coords_absolute=1,
        roi_coords_swap=0,
        roi_coords_transform=2,
        legacy=0,
    ):
        return g.op(
            "TRT::RoIAlign2Dynamic_TRT",
            feature_map,
            roi,
            pooled_size_i=pooled_size,
            sampling_ratio_i=sampling_ratio,
            roi_coords_absolute_i=roi_coords_absolute,
            roi_coords_swap_i=roi_coords_swap,
            roi_coords_transform_i=roi_coords_transform,
            image_size_i=image_size,
            legacy_i=legacy,
            outputs=1,
        )


class ONNX_TRT_MASK2(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        image_size=640,
        attn_resolution=14,
        mask_resolution=56,
        num_base=5,
        pooler_scale=0.25,
        sampling_ratio=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.background_class = (-1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.score_activation = 0
        self.score_threshold = score_thres
        self.image_size = image_size
        self.attn_resolution = attn_resolution
        self.mask_resolution = mask_resolution
        self.num_base = num_base
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio
        self.roi_align_type = 2  # 1, or 2

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5:]
        attn = x[1]
        bases = x[2]

        batch_size = boxes.shape[0]
        bases_dim = bases.shape[1]

        scores *= conf
        num_det, det_boxes, det_scores, det_classes, det_indices = TRT_NMS4.apply(
            boxes,
            scores,
            self.background_class,
            self.iou_threshold,
            self.max_obj,
            self.score_activation,
            self.score_threshold,
        )

        # return num_det, det_boxes, det_scores, det_classes, det_indices, attn, bases

        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)

        det_attn = attn[
            batch_indices.view(batch_size * self.max_obj).to(torch.long),
            det_indices.view(batch_size * self.max_obj).to(torch.long),
        ].view(
            batch_size, self.max_obj, self.num_base * self.attn_resolution * self.attn_resolution
        )

        if self.roi_align_type == 1:
            pooled_bases = TRT_RoiAlign.apply(
                bases,
                torch.cat((batch_indices.unsqueeze(2).float(), det_boxes), dim=2).view(
                    batch_size * self.max_obj, 5
                ),
                self.mask_resolution,
                self.mask_resolution,
                self.pooler_scale,
                self.sampling_ratio,
            )
        else:
            pooled_bases = TRT_ROIAlign2.apply(
                bases,
                det_boxes.view(batch_size, self.max_obj, 4),
                self.mask_resolution,
                self.image_size,
            )

        pooled_bases = pooled_bases.view(
            (batch_size * self.max_obj, bases_dim, self.mask_resolution, self.mask_resolution)
        )

        det_attn = det_attn.view(
            batch_size * self.max_obj, self.num_base, self.attn_resolution, self.attn_resolution
        )

        det_attn = F.interpolate(
            det_attn, (self.mask_resolution, self.mask_resolution), mode="bilinear"
        ).softmax(dim=1)

        masks_preds = (
            (pooled_bases * det_attn)
            .sum(dim=1)
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
            .sigmoid()
        )

        return num_det, det_boxes, det_scores, det_classes, masks_preds


class End2EndMask(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        trt=True,
        attn_resolution=14,
        mask_resolution=56,
        num_base=5,
        pooler_scale=0.25,
        sampling_ratio=0,
    ):
        super().__init__()
        device = device if device else torch.device("cpu")
        assert isinstance(max_wh, (int))

        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT_MASK2 if trt else ONNX_ORT_MASK
        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            max_wh=max_wh,
            device=device,
            attn_resolution=attn_resolution,
            mask_resolution=mask_resolution,
            num_base=num_base,
            pooler_scale=pooler_scale,
            sampling_ratio=sampling_ratio,
        )
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        )  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
