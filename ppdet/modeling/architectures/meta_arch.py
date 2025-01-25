from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import typing

from ppdet.core.workspace import register
from ppdet.modeling.post_process import nms

__all__ = ['BaseArch']


@register
class BaseArch(nn.Layer):
    def __init__(self, data_format='NCHW', ):
        super(BaseArch, self).__init__()
        self.data_format = data_format
        self.inputs = {}

        self.fuse_norm = False

    def load_meanstd(self, cfg_transform):
        scale = 1.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for item in cfg_transform:
            if 'NormalizeImage' in item:
                mean = np.array(
                    item['NormalizeImage']['mean'], dtype=np.float32)
                std = np.array(item['NormalizeImage']['std'], dtype=np.float32)
                if item['NormalizeImage'].get('is_scale', True):
                    scale = 1. / 255.
                break
        if self.data_format == 'NHWC':
            self.scale = paddle.to_tensor(scale / std).reshape((1, 1, 1, 3))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 1, 1, 3))
        else:
            self.scale = paddle.to_tensor(scale / std).reshape((1, 3, 1, 1))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 3, 1, 1))

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = image * self.scale + self.bias
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
            self.inputs = inputs

        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            inputs_list = []
            # multi-scale input
            if not isinstance(inputs, typing.Sequence):
                inputs_list.append(inputs)
            else:
                inputs_list.extend(inputs)
            outs = []
            for inp in inputs_list:
                if self.fuse_norm:
                    self.inputs['image'] = inp['image'] * self.scale + self.bias
                    self.inputs['im_shape'] = inp['im_shape']
                    self.inputs['scale_factor'] = inp['scale_factor']
                else:
                    self.inputs = inp
                outs.append(self.get_pred())
            if self.exclude_nms:
                #out = [self.merge_multi_scale_predictions(outs),]
                out = outs
            elif len(outs) == 0:
                out = outs[0]
            else:
                out = self.merge_multi_scale_predictions(outs)
            # out = outs[0]
        return out

    def merge_multi_scale_predictions(self, outs, weights=None):
        from ppdet.modeling.WBF import weighted_boxes_fusion, nms
        # default values for architectures not included in following list
        # pred_bboxes = paddle.concat([o['bbox'] for o in outs], 1)
        # pred_scores = paddle.concat([o['bbox_num'] for o in outs], 2)
        # pred_bboxes = outs[0]['bbox']
        # pred_scores = outs[0]['bbox_num']

        label = [o['bbox'][:, 0] for o in outs]
        score = [o['bbox'][:, 1] for o in outs]
        bbox = [o['bbox'][:, 2:] for o in outs]
        # pred_scores = [o['bbox_num'] for o in outs]
        bbox_pred = weighted_boxes_fusion(bbox, score, label, weights=weights,
                                          iou_thr=0.65,
                                          skip_box_thr=0.0,
                                          conf_type='avg',
                                          allows_overflow=False)
        # bbox_pred = nms(bbox, score, label,iou_thr=0.55,)
        output = {'bbox': bbox_pred, 'bbox_num': [bbox_pred.shape[0], ]}
        # print(output)
        return output

    def build_inputs(self, data, input_def):
        inputs = {}
        for i, k in enumerate(input_def):
            inputs[k] = data[i]
        return inputs

    def model_arch(self, ):
        pass

    def get_loss(self, ):
        raise NotImplementedError("Should implement get_loss method!")

    def get_pred(self, ):
        raise NotImplementedError("Should implement get_pred method!")
