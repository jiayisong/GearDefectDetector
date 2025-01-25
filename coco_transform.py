import os
import json
import math
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.use('TkAgg')
'''

def plot_dist(df_train):
    ps = np.zeros(len(df_train))
    for i in range(len(df_train)):
        ps[i] = math.log(df_train['area'][i])
    plt.title('area dist', )
    sns.distplot(ps, bins=10000)


input_dir = 'E:/Dataset/chilun/val.json'
output_dir = 'E:/Dataset/chilun/val_resize.json'

with open(input_dir, 'r') as f1:
    results = json.load(f1)
    new = []
    for i in tqdm(results['annotations']):
        for j in results['images']:
            if j['id'] == i['image_id']:
                w = j['width']
                h = j['height']
        if h == 1500:
            if w == 1000:
                target_size = [1902, 1076]
            elif w == 1400:
                target_size = [1500, 841]
            else:
                raise RuntimeError('img shape error')
        elif h == 2000:
            target_size = [3354, 1262]
        else:
            raise RuntimeError('img shape error')
        i['bbox'][0] *= target_size[1] / w
        i['bbox'][1] *= target_size[0] / h
        i['bbox'][2] *= target_size[1] / w
        i['bbox'][3] *= target_size[0] / h
        i['area'] = i['bbox'][2] * i['bbox'][3]
        if i['area'] < 0:
            print(i['bbox'])
        new.append(i)
    results['annotations'] = new
    new = []
    for i in tqdm(results['images']):
        name = i['file_name']
        im = cv2.imread('E:/Dataset/chilun/train/' + name)
        h, w, _ = im.shape
        if h == 1500:
            if w == 1000:
                target_size = [1902, 1076]
            elif w == 1400:
                target_size = [1500, 841]
            else:
                raise RuntimeError('img shape error')
        elif h == 2000:
            target_size = [3354, 1262]
        else:
            raise RuntimeError('img shape error')
        im = cv2.resize(im, target_size[::-1], interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('E:/Dataset/chilun/trainval_resize/' + name, im)
        i['width'] = target_size[1]
        i['height'] = target_size[0]
        new.append(i)
    results['images'] = new
    with open(output_dir, 'w') as f:
        json.dump(results, f)

'''


with open('E:/Dataset/chilun/train_coco_resize.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
train_anno = pd.DataFrame(train_data['annotations'])
train_fig = pd.DataFrame(train_data['images'])
df_train = pd.merge(left=train_fig, right=train_anno, how='inner', left_on='id', right_on='image_id')
df_train['bbox_xmin'] = df_train['bbox'].apply(lambda x: x[0])
df_train['bbox_ymin'] = df_train['bbox'].apply(lambda x: x[1])
df_train['bbox_w'] = df_train['bbox'].apply(lambda x: x[2])
df_train['bbox_h'] = df_train['bbox'].apply(lambda x: x[3])
df_train['bbox_xcenter'] = df_train['bbox'].apply(lambda x: (x[0] + 0.5 * x[2]))
df_train['bbox_ycenter'] = df_train['bbox'].apply(lambda x: (x[1] + 0.5 * x[3]))
print(df_train.area.describe([(i + 1) * 0.02 for i in range(50)]))
#df_train.boxplot(column="area", by= "category_id",)
#df_train.area.plot(kind="density",)


print('类别1')
print(df_train[df_train['category_id'] == 1].area.describe([(i + 1) * 0.02 for i in range(50)]))
#df_train[df_train['category_id'] == 1].area.plot(kind="density",)
print('类别2')
print(df_train[df_train['category_id'] == 2].area.describe([(i + 1) * 0.02 for i in range(50)]))
#df_train[df_train['category_id'] == 2].area.plot(kind="density",)
print('类别3')
print(df_train[df_train['category_id'] == 3].area.describe([(i + 1) * 0.02 for i in range(50)]))
#df_train[df_train['category_id'] == 3].area.plot(kind="density",)
plt.grid()
plt.show()


