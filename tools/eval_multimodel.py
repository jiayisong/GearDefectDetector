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

import os
import sys

# add python path of PadleDetection to sys.path
import time

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
from ppdet.core.workspace import create
import paddle
import typing
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.slim import build_slim_model
from argparse import ArgumentParser
from ppdet.utils.logger import setup_logger
logger = setup_logger('eval')


def parse_args():
    parser = ArgumentParser()
    args = parser.parse_args()
    return args




def main():
    exclude_nms = True
    configs = [
         'configs/testB/ppyoloe_adamw_flip3_clip2.yml',
         'configs/testB/ppyoloe_adamw_flip3_clip2_75.yml',
         'configs/testB/ppyoloe_adamw_flip3_clip2_55.yml',
       # 'configs/testB/ppyoloe_adamw_flip3_clip_flip2.yml',
       # 'configs/testB/ppyoloe_adamw_flip3_clip_rot2.yml',
      #  'configs/testB/ppyoloe_adamw_flip3_clip_fliprot2.yml',
     #   'configs/testB/ppyoloe_adamw_flip3_clip_s2.yml',
      #  'configs/testB/ppyoloe_adamw_flip3_clip_x2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_m2.yml',
       'configs/testB/ppyoloe_adamw_flip3_clip_m2_55.yml',
       'configs/testB/ppyoloe_adamw_flip3_clip_m2_75.yml',
         'configs/testB/ppyoloe_adamw_flip3_clip_s2.yml',
         'configs/testB/ppyoloe_adamw_flip3_clip_s2_55.yml',
         'configs/testB/ppyoloe_adamw_flip3_clip_s2_75.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_x2.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_x2_55.yml',
        'configs/testB/ppyoloe_adamw_flip3_clip_x2_75.yml',
      #  'configs/sniper/hyd_cls1.yml',
      #  'configs/sniper/hyd_cls2.yml',
     #   'configs/sniper/hyd_cls3.yml',
      #  'configs/testB/ppyoloe_adamw_flip3_clip_s2.yml',
      #  'configs/sniper/hyd.yml',
       # 'configs/testB/ppyoloe_cascade_rcnn_rpn_nms2.yml'
       # 'configs/sniper/ppyoloe_adamw_flip3_mr2.yml',
    ]
    weights = [
         'output/ppyoloe_adamw_flip3_clip2/best_model.pdparams',
         'output/ppyoloe_adamw_flip3_clip2_75/best_model.pdparams',
         'output/ppyoloe_adamw_flip3_clip2_55/best_model.pdparams',
      #  'output/ppyoloe_adamw_flip3_clip_flip2/best_model.pdparams',
      #  'output/ppyoloe_adamw_flip3_clip_rot2/best_model.pdparams',
       # 'output/ppyoloe_adamw_flip3_clip_fliprot2/best_model.pdparams',
     #   'output/ppyoloe_adamw_flip3_clip_s2/best_model.pdparams',
     #   'output/ppyoloe_adamw_flip3_clip_x2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_m2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_m2_55/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_m2_75/best_model.pdparams',
         'output/ppyoloe_adamw_flip3_clip_s2/best_model.pdparams',
         'output/ppyoloe_adamw_flip3_clip_s2_55/best_model.pdparams',
         'output/ppyoloe_adamw_flip3_clip_s2_75/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_x2/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_x2_55/best_model.pdparams',
        'output/ppyoloe_adamw_flip3_clip_x2_75/best_model.pdparams',
      #  'output/hyd/best_model.pdparams',
      #  'output/hyd/best_model.pdparams',
      #  'output/hyd/best_model.pdparams',
     # 'output/ppyoloe_cascade_rcnn_rpn_nms2/best_model.pdparams'

    ]
    fuse_weights = [1 for _ in range(len(weights))]
    #fuse_weights = [7, 4, 5, 3, 6, 2, ]
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
    cfg[cfg.architecture]['exclude_nms'] = exclude_nms
    trainer = Trainer(cfg, mode='eval')
    # load weights
    trainer.load_weights(cfg.weights)
    models = []
    loaders = []
    for c, w in zip(configs, weights):
        m, l = build_model_loader(load_config(c), w, exclude_nms)
        models.append(m)
        loaders.append(l)
    # training
    with paddle.no_grad():
        sample_num = 0
        tic = time.time()
        trainer._compose_callback.on_epoch_begin(trainer.status)
        trainer.status['mode'] = 'eval'
        trainer.model.eval()
        if trainer.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(trainer.mode.capitalize()))(
                trainer.dataset, trainer.cfg.worker_num, trainer._eval_batch_sampler)
            trainer._flops(flops_loader)
        for step_id, data in enumerate(zip(trainer.loader, *loaders)):
            trainer.status['step_id'] = step_id
            trainer._compose_callback.on_step_begin(trainer.status)
            # forward
            if trainer.use_amp:
                with paddle.amp.auto_cast(
                        enable=trainer.cfg.use_gpu,
                        custom_white_list=trainer.custom_white_list,
                        custom_black_list=trainer.custom_black_list,
                        level=trainer.amp_level):
                    outs = trainer.model(data[0])
            else:
                outs = trainer.model(data[0])
            fuse_weight = [fuse_weights[0] for _ in range(len(outs))]
            for m, d, fw in zip(models, data[1:], fuse_weights[1:]):
                if trainer.use_amp:
                    with paddle.amp.auto_cast(
                            enable=trainer.cfg.use_gpu,
                            custom_white_list=trainer.custom_white_list,
                            custom_black_list=trainer.custom_black_list,
                            level=trainer.amp_level):
                        oo = m(d)
                else:
                    oo = m(d)
                outs.extend(oo)
                    #outs = m(d)
                fuse_weight.extend([fw for _ in range(len(oo))])
            outs = trainer.model.merge_multi_scale_predictions(outs, weights=fuse_weight)
            # update metrics
            for metric in trainer._metrics:
                metric.update(data[0], outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data[0], typing.Sequence):
                sample_num += data[0][0]['im_id'].numpy().shape[0]
            else:
                sample_num += data[0]['im_id'].numpy().shape[0]
            trainer._compose_callback.on_step_end(trainer.status)

        trainer.status['sample_num'] = sample_num
        trainer.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in trainer._metrics:
            metric.accumulate()
            metric.log()
        trainer._compose_callback.on_epoch_end(trainer.status)
        # reset metric states for metric may performed multiple times
        trainer._reset_metrics()


def build_model_loader(cfg, weights, exclude_nms):
    cfg[cfg.architecture]['exclude_nms'] = exclude_nms
    model = create(cfg.architecture)
    model.load_meanstd(cfg['TestReader']['sample_transforms'])
    load_pretrain_weight(model, weights)
    logger.debug("Load weights {} to start training".format(weights))
    model.eval()

    dataset = cfg['EvalDataset'] = create('EvalDataset')()
    eval_batch_sampler = paddle.io.BatchSampler(dataset, batch_size=cfg.EvalReader['batch_size'])
    reader_name = 'EvalReader'

    loader = create(reader_name)(dataset, cfg.worker_num, eval_batch_sampler)
    return model, loader


if __name__ == '__main__':
    main()
