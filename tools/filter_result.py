from ppdet.modeling.WBF import weighted_boxes_fusion, nms
import json
import os
from tqdm import tqdm


files = '/mnt/jys/YYDS/output/B_SML_556575/upload.json'
out_files = '/mnt/jys/YYDS/output/B_SML_556575/upload_filter.json'
with open(files, 'r', encoding='utf-8') as file:
    datas=json.load(file)

results = []
for d in tqdm(datas):
    if d['score'] >= 0.001:
        results.append(d)

with open(out_files, 'w', encoding='utf-8') as f:
    json.dump(results, f)
