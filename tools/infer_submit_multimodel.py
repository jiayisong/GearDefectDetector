# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import sys

# add python path of PadleDetection to sys.path
import time

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import warnings

warnings.filterwarnings('ignore')

import glob
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.core.workspace import create
from PIL import Image, ImageOps, ImageFile
import paddle
import typing
from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results, KeyPointTopDownCOCOEval, \
    KeyPointTopDownMPIIEval

from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from argparse import ArgumentParser
from ppdet.utils.logger import setup_logger

logger = setup_logger('eval')


infer_dir = '/mnt/DataSets/齿轮检测数据集/testB/'

def parse_args():
    parser = ArgumentParser()
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
        "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
        "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def main(configs, weights, output_dir):
    draw_threshold = 0.5
    visualize = True
    fuse_weights = [1 for _ in range(len(weights))]
    # fuse_weights = [7, 4, 5, 3, 6, 2, ]
    FLAGS = parse_args()
    cfg = load_config(configs.pop(0))
    merge_args(cfg, FLAGS)
    cfg.weights = weights.pop(0)
    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_version()

    init_parallel_env()
    # build trainer
    cfg[cfg.architecture]['exclude_nms'] = True
    trainer = Trainer(cfg, mode='test')
    # load weights
    trainer.load_weights(cfg.weights)

    images = get_test_images(infer_dir, None)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer.dataset.set_images(images)
    loader = create('TestReader')(trainer.dataset, 0)

    imid2path = trainer.dataset.get_imid2path()

    def setup_metrics_for_loader():
        # mem
        metrics = copy.deepcopy(trainer._metrics)
        mode = trainer.mode
        save_prediction_only = trainer.cfg[
            'save_prediction_only'] if 'save_prediction_only' in trainer.cfg else None
        output_eval = trainer.cfg[
            'output_eval'] if 'output_eval' in trainer.cfg else None

        # modify
        trainer.mode = '_test'
        trainer.cfg['save_prediction_only'] = True
        trainer.cfg['output_eval'] = output_dir
        trainer.cfg['imid2path'] = imid2path
        trainer._init_metrics()

        # restore
        trainer.mode = mode
        trainer.cfg.pop('save_prediction_only')
        if save_prediction_only is not None:
            trainer.cfg['save_prediction_only'] = save_prediction_only

        trainer.cfg.pop('output_eval')
        if output_eval is not None:
            trainer.cfg['output_eval'] = output_eval

        trainer.cfg.pop('imid2path')

        _metrics = copy.deepcopy(trainer._metrics)
        trainer._metrics = metrics

        return _metrics

    metrics = setup_metrics_for_loader()
    clsid2catid, catid2name = _coco17_category()

    # Run Infer
    trainer.status['mode'] = 'test'
    trainer.model.eval()
    if trainer.cfg.get('print_flops', False):
        flops_loader = create('TestReader')(trainer.dataset, 0)
        trainer._flops(flops_loader)

    models = []
    loaders = []
    for c, w in zip(configs, weights):
        m, l = build_model_loader(load_config(c), w, images)
        models.append(m)
        loaders.append(l)
    results = []
    for step_id, data in enumerate(tzip(loader, *loaders)):
        trainer.status['step_id'] = step_id
        # forward
        with paddle.no_grad():
            outs = trainer.model(data[0])
            fuse_weight = [fuse_weights[0] for _ in range(len(outs))]
            for m, d, fw in zip(models, data[1:], fuse_weights[1:]):
                oo = m(d)
                outs.extend(oo)
                fuse_weight.extend([fw for _ in range(len(oo))])
            outs = trainer.model.merge_multi_scale_predictions(outs, weights=fuse_weight)
        for _m in metrics:
            _m.update(data[0], outs)
        for key in ['im_shape', 'scale_factor', 'im_id']:
            if isinstance(data[0], typing.Sequence):
                outs[key] = data[0][0][key]
            else:
                outs[key] = data[0][key]
        for key, value in outs.items():
            if hasattr(value, 'numpy'):
                outs[key] = value.numpy()
        results.append(outs)

    for _m in metrics:
        _m.accumulate()
        _m.reset()
    if visualize:
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']
            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                trainer.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                    if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] \
                    if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] \
                    if 'segm' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] \
                    if 'keypoint' in batch_res else None
                image = visualize_results(
                    image, bbox_res, mask_res, segm_res, keypoint_res,
                    int(im_id), catid2name, draw_threshold)
                trainer.status['result_image'] = np.array(image.copy())
                if trainer._compose_callback:
                    trainer._compose_callback.on_step_end(trainer.status)
                # save image with detection
                save_name = trainer._get_save_image_name(output_dir,
                                                         image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)
                start = end

    with open(os.path.join(output_dir, 'id2path.json'), 'w') as f:
        json.dump(trainer.dataset.get_imid2path(), f)

    with open(os.path.join(output_dir, 'bbox.json'), 'r') as f1:
        results = json.load(f1)
    with open(os.path.join(output_dir, 'id2path.json'), 'r') as f2:
        id2path = json.load(f2)
    upload_json = []
    for i in range(len(results)):
        dt = {}
        dt['name'] = os.path.basename(id2path[str(results[i]['image_id'])])
        dt['category_id'] = results[i]['category_id']
        xmin, ymin, w, h = results[i]['bbox']
        dt['bbox'] = [xmin, ymin, xmin + w, ymin + h]
        dt['score'] = results[i]['score']
        upload_json.append(dt)

    # 生成上传文件
    with open(os.path.join(output_dir, 'upload.json'), 'w') as f:
        json.dump(upload_json, f)


def build_model_loader(cfg, weights, images):
    cfg[cfg.architecture]['exclude_nms'] = True
    model = create(cfg.architecture)
    model.load_meanstd(cfg['TestReader']['sample_transforms'])
    load_pretrain_weight(model, weights)
    logger.debug("Load weights {} to start training".format(weights))
    model.eval()

    dataset = cfg['TestDataset'] = create('TestDataset')()
    dataset.set_images(images)
    loader = create('TestReader')(dataset, 0)
    return model, loader


def _coco17_category():
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
    catid2name.pop(0)

    return clsid2catid, catid2name


if __name__ == '__main__':
    # 推理L
    output_dir = 'output/L_556575/'
    configs = [
        'configs/testB/ppyoloe_adamw_flip3_clip2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip2_75.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip2_55.yml',
    ]
    weights = [
        'output/ppyoloe_adamw_flip3_clip2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip2_75/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip2_55/best_model.pdparams',
    ]
    main(configs, weights, output_dir)
    # 推理M
    output_dir = 'output/M_556575/'
    configs = [
        'configs/testB/ppyoloe_adamw_flip3_clip_m2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_m2_75.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_m2_55.yml',
    ]
    weights = [
        'output/ppyoloe_adamw_flip3_clip_m2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_m2_75/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_m2_55/best_model.pdparams',
    ]
    main(configs, weights, output_dir)
    # 推理X
    output_dir = 'output/X_556575/'
    configs = [
        'configs/testB/ppyoloe_adamw_flip3_clip_x2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_x2_75.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_x2_55.yml',
    ]
    weights = [
        'output/ppyoloe_adamw_flip3_clip_x2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_x2_75/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_x2_55/best_model.pdparams',
    ]
    main(configs, weights, output_dir)
    # 推理S
    output_dir = 'output/S_556575/'
    configs = [
        'configs/testB/ppyoloe_adamw_flip3_clip_s2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_s2_75.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_s2_55.yml',
    ]
    weights = [
        'output/ppyoloe_adamw_flip3_clip_s2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_s2_75/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_s2_55/best_model.pdparams',
    ]
    main(configs, weights, output_dir)
