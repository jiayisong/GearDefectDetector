from ppdet.modeling.WBF import weighted_boxes_fusion, nms
import json
import os
from tqdm import tqdm

weights = [1, 1, 1, 1]
test_img_dir = '/mnt/DataSets/齿轮检测数据集/testB/'
files = [
    'output/S_556575/upload.json',
    'output/M_556575/upload.json',
    'output/L_556575/upload.json',
    'output/X_556575/upload.json',
]
out_files = 'output/SMLX_556575/upload.json'
datas = []
imgs = os.listdir(test_img_dir)
for i in files:
    with open(i, 'r', encoding='utf-8') as file:
        datas.append(json.load(file))
results = []
for img in tqdm(imgs):
    labels = []
    bboxs = []
    scores = []
    for data in datas:
        label = []
        bbox = []
        score = []
        for d in data:
            if d['name'] == img:
                bbox.append(d['bbox'])
                label.append(d['category_id'])
                score.append(d['score'])
        labels.append(label)
        scores.append(score)
        bboxs.append(bbox)

    overall_boxes = weighted_boxes_fusion(bboxs, scores, labels, weights=weights,
                                          iou_thr=0.65,
                                          skip_box_thr=0.001,
                                          conf_type='avg',
                                          allows_overflow=False)
    bboxs_new = overall_boxes[:, 2:]
    scores_new = overall_boxes[:, 1]
    labels_new = overall_boxes[:, 0]

    for b, s, l in zip(bboxs_new, scores_new, labels_new):
        results.append({
            'name': img,
            'category_id': int(l),
            'score': s,
            'bbox': b.tolist(),
        })
with open(out_files, 'w', encoding='utf-8') as f:
    json.dump(results, f)
