import os
from PIL import Image, ImageDraw, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
import shutil
import json
import os
import cv2
import time
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from IPython.display import display, HTML

from pathlib import Path
import glob


# myfont = FontProperties(fname=r"resources/NotoSansCJKsc-Medium.otf", size=12)
# plt.rcParams['figure.figsize'] = (12, 12)
# plt.rcParams['font.family']= myfont.get_family()
# plt.rcParams['font.sans-serif'] = myfont.get_name()
# plt.rcParams['axes.unicode_minus'] = False

def draw_bbox_in_image(image, bbox, edgecolor, linewidth=1):
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], fill=edgecolor)
    return image


def draw_line_in_image(image, points, edgecolor, linewidth=1):
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=edgecolor, width=linewidth)
    return image


def cv2PIL(im):
    try:
        return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
    except:
        return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def PIL2cv(im):
    try:
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)
    except:
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def rgb2gray(rgb):
    rgb_ = rgb.copy()
    bw_threshold = 128  # 黑白化
    rgb_[rgb_[:, :, 0] < bw_threshold] = 0
    rgb_[rgb_[:, :, 0] >= bw_threshold] = 255
    return rgb_.max(axis=2)  # 黑白化后 取最大值


# 变换图像亮度值
def image_illum_transform(img, mean_illum=0.5):
    img = img.astype(np.uint8)
    # print(f'Lighting_4_', img)

    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    mean = np.mean(illum)
    # std = np.std(illum)
    gamma = np.log(mean_illum) / np.log(mean)
    illum = np.power(illum, gamma)

    # print(f'Lighting_3_{gamma}/{illum}')

    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # print(f'Lighting_5_', img)
    return img.astype(np.uint8)


# 获取齿轮图像的两个左右切边
def get_gear_edge(img, threshold=0.30):
    leftLoc = -1
    rightLoc = -1
    mmLeft = 0
    mmRight = 0
    imgGray = rgb2gray(img)
    #     print(imgGray.shape)
    #     print(f'liuyz_1__', imgGray[:,0].shape)
    imgMean = np.mean(imgGray, axis=0)
    imgMean_len = len(imgMean)
    mid = round(imgMean_len / 2)
    #     print(imgMean[mid-10:mid+10])
    for i in range(mid):
        #         left_ratio = abs(imgMean[i]-imgMean[1])/(imgMean[i+1]+0.0001)
        #         print(f'left_1__{i}/{mid}_', imgMean[i], imgMean[i+1], left_ratio)
        if leftLoc == -1:
            left_ratio = abs(imgMean[i] - imgMean[1]) / (imgMean[i + 1] + 0.0001)
            if left_ratio >= threshold:
                #                 print(f'left_2__{i}/{mid}_', imgMean[i], imgMean[i+1], left_ratio)
                #                 print(imgMean[mid-i-20:mid-i+20])
                leftLoc = i
                leftLoc -= 10
                if leftLoc < 0: leftLoc = 0

        #         right_ratio = (abs(imgMean[imgMean_len-1-i]-imgMean[imgMean_len-1-i-1])/(imgMean[imgMean_len-1-i-1]+0.0001))
        #         print(f'right_1_{i}/{mid}_', imgMean[imgMean_len-1-i], imgMean[imgMean_len-1-i-1], right_ratio)
        if rightLoc == -1:
            right_ratio = (abs(imgMean[imgMean_len - 1 - i] - imgMean[imgMean_len - 1 - i - 1]) / (
                        imgMean[imgMean_len - 1 - i - 1] + 0.0001))
            if right_ratio >= threshold:
                #                 print(f'right_2_{i}/{mid}_', imgMean[imgMean_len-1-i], imgMean[imgMean_len-1-i-1], right_ratio)
                #                 print(imgMean[imgMean_len-1-i-20:imgMean_len-1-i+20])
                rightLoc = imgMean_len - 1 - i
                rightLoc += 10
                if rightLoc >= imgMean_len: rightLoc = imgMean_len - 1
        if leftLoc >= 0 and rightLoc >= 0:
            break
    return leftLoc, rightLoc


# 根据齿轮图像获取斜率（可用于数据斜向增强）
def get_gear_slope(image, leftLoc=None, rightLoc=None, show=False, debug=False):
    #     image = Image.open(img_file)
    def get_best_distribution(slopes, bins=1000):
        import pandas as pd
        s = pd.Series(slopes)
        ss = s.value_counts(bins=bins)
        if len(ss) < 0:
            return 0, [None, None]
        count = ss.iloc[0]
        index = str(ss.__dict__['_index'][0]).replace('(', '').replace(']', '')
        splits = index.split(',')
        return count, [float(splits[0]), float(splits[1])]

    def max_length_num(str):
        max_list = []
        locs = []
        for i in range(len(str)):
            new_list = []
            for j in range(i + 1, len(str)):
                if str[j].isdigit():
                    new_list.append(str[j])
                else:
                    break
                if len(max_list) < len(new_list):
                    max_list = new_list
                    locs.append(i)
        return "".join(max_list), locs

    def find_best_markLen(image_whiteblack, markLoc):
        sss = ''
        for i in range(image_whiteblack.shape[0]):
            sss += '1' if image_whiteblack[:, markLoc][i] else 'T'

        maxLengthNumStr, loc = max_length_num(sss)
        maxLen = len(maxLengthNumStr)
        if debug:
            print(f'maxLengthNumStr=', maxLen, loc)
        locs = []
        markLen = 0
        for i in range(int(maxLen / 2)):
            markLen = maxLen - i
            loop = True
            start = 0
            locs = []
            while (loop):
                loc = sss.find('1' * (markLen), start)
                if loc >= 0:
                    last_point = start - markLen
                    start = loc + markLen
                    if (loc - last_point) < 100:
                        continue
                    locs.append(loc)
                else:
                    loop = False
            if debug and 0 == 1:
                print(f'____find[{i}]_{markLen}__locs={len(locs)}/{locs}')
            if len(locs) >= 8:
                return markLen, locs
        return markLen, locs

    i_image = image_illum_transform(PIL2cv(image), mean_illum=0.2)
    image = cv2PIL(i_image)
    image_whiteblack = image.convert('1')
    image_whiteblack = np.array(image_whiteblack)
    img_height = image_whiteblack.shape[0]
    img_width = image_whiteblack.shape[1]

    if debug:
        print(f'image_whiteblack.shape={image_whiteblack.shape}')
        print(f'leftLoc,rightLoc={leftLoc}, {rightLoc}')

    imgMidX = int(img_width / 2)
    markLeftX = imgMidX - 400
    markRightX = imgMidX + 400
    step = 30
    markLocXs = []
    for i in range(math.ceil((markRightX - markLeftX) / step)):
        markX = markLeftX + i * step
        markLocXs.append(markX)
    markLens = []
    markLocs = []
    for i in range(len(markLocXs)):
        markLen, locs = find_best_markLen(image_whiteblack, markLocXs[i])
        markLens.append(markLen)
        markLocs.append(locs)

    if debug:
        print(f'markLocXs = {len(markLocXs)} / {markLocXs}')
        print(f'markLens = {markLens}')
        print(f'markLocs = {markLocs}')

    best_slope = None

    slopes = []
    max_slope, min_slope = -9999999, 9999999
    line_count = 0
    slop_scope = [-100, 0]
    min_h = 50
    draw_candidate_line_max = 0  # 100
    slope_lines = []
    for i in range(len(markLocXs)):
        for j in range(len(markLocXs) - i - 1):
            locs1 = markLocs[i]
            locs2 = markLocs[i + j + 1]
            w = markLocXs[i + j + 1] - markLocXs[i]
            for k in range(len(locs1)):
                for kk in range(len(locs2)):
                    h = locs1[k] + int(markLens[i] / 2) - (locs2[kk] + int(markLens[i + j + 1] / 2))
                    if h < 0 and abs(h) > min_h:
                        slope = h / w
                        if len(slop_scope) == 2:
                            if slope < slop_scope[0] or slope > slop_scope[1]: continue
                        slopes.append(slope)
                        if slope > max_slope: max_slope = slope
                        if slope < min_slope: min_slope = slope
                        draw_line = [(markLocXs[i], locs1[k] + int(markLens[i] / 2)),
                                     (markLocXs[i + j + 1], locs2[kk] + int(markLens[i + j + 1] / 2))]
                        slope_lines.append([slope, draw_line])
                        if show and draw_candidate_line_max > 0 and line_count < draw_candidate_line_max:
                            line_count += 1
                            image = draw_line_in_image(image, draw_line, "#ff00ff", linewidth=2)
    if debug:
        print(f'slopes: ', len(slopes), (min_slope, max_slope), (max_slope - min_slope), line_count,
              draw_candidate_line_max)
    if len(slopes) == 0:
        return None, None, None, None
    slope_max_count, [best_slope1, best_slope2] = get_best_distribution(slopes, bins=len(slopes))
    best_slope = (best_slope1 + best_slope2) / 2
    if debug:
        print(f'best_slope:({best_slope:5.02f},{best_slope1:5.02f},{best_slope2:5.02f}) / {slope_max_count}')
    best_dist = None
    distances = []
    for i in range(len(markLocs)):
        locs = markLocs[i]
        for j in range(len(locs) - 1):
            distances.append(locs[j + 1] - locs[j])
    if len(distances) == 0:
        return None, None, None, None
    if debug:
        print(f'distances={distances}')
    dist_max_count, [best_dist1, best_dist2] = get_best_distribution(distances, bins=len(distances) * 3)
    best_dist = (best_dist1 + best_dist2) / 2
    if debug:
        print(f'best_dist:({best_dist:5.02f},{best_dist1:5.02f},{best_dist2:5.02f}) / {dist_max_count}')

    x0 = markLocXs[0]
    y0 = img_height
    draw_best_lines = []
    draw_best_line_count = 0
    max_line_length = 0
    max_line_box = None
    midYs = []
    for slope_line in slope_lines:
        if slope_line[0] >= best_slope1 and slope_line[0] <= best_slope2:
            draw_best_line_count += 1
            draw_best_lines.append(slope_line)
            if show:
                image = draw_line_in_image(image, slope_line[1], "#ff00ff", linewidth=2)
            if slope_line[1][0][1] < y0:
                y0 = slope_line[1][0][1]
                x0 = slope_line[1][0][0]
            if slope_line[1][1][0] - slope_line[1][0][0] > max_line_length:
                max_line_length = slope_line[1][1][0] - slope_line[1][0][0]
                max_line_box = slope_line[1]
            midY0 = slope_line[1][1][1]
            midY0 += slope_line[0] * (slope_line[1][1][0] - imgMidX)
            midYs.append(midY0)

    if debug:
        print(f'draw_best_lines={draw_best_lines}')
        print(f'midYs={midYs}')
    midY_max_count, [best_midY1, best_midY2] = get_best_distribution(midYs, bins=len(midYs) * 5)
    best_midY = (best_midY1 + best_midY2) / 2
    if debug:
        print(f'best_midY:({best_midY:5.02f},{best_midY1:5.02f},{best_midY2:5.02f}) / {midY_max_count}')
    best_midY_up = divmod(best_midY, best_dist)[1]
    best_midY_center = int(img_height / best_dist / 2) * best_dist + best_midY_up
    if debug:
        print(f'best_midY/best_dist: {best_midY_up} = {best_midY}/{best_dist}')

    if show:
        dpi = 200
        plt_width = img_width / dpi
        plt_height = img_height / dpi
        if leftLoc is not None:
            image = draw_bbox_in_image(image,
                                       [leftLoc, 0, 2, img_height - 1],
                                       '#00ff00', linewidth=2)
        if rightLoc is not None:
            image = draw_bbox_in_image(image,
                                       [rightLoc, 0, 2, img_height - 1],
                                       '#00ff00', linewidth=2)
        for i in range(len(markLocXs)):
            image = draw_bbox_in_image(image,
                                       [markLocXs[i] - 2, 0, 2, img_height - 1],
                                       '#ffaa00', linewidth=2)
            for loc in markLocs[i]:
                image = draw_bbox_in_image(image,
                                           [markLocXs[i], loc, 8, markLens[i]],
                                           "#ff0000", linewidth=2)
        if max_line_box is not None:
            image = draw_line_in_image(image, max_line_box,
                                       "#00ff00", linewidth=5)
        if debug:
            print(f'slope_max_count={slope_max_count}, draw_best_line_count={draw_best_line_count}')

        if y0 == img_height:
            y0 = markLocs[0][0] + int(markLens[0] / 2)
            x0 = markLocXs[0]
        h = abs(best_slope * (markLocXs[-1] - x0))
        image = draw_line_in_image(image,
                                   [(x0, y0),
                                    (markLocXs[-1], y0 + h)],
                                   "#ff0000", linewidth=6)
        image = draw_line_in_image(image,
                                   [(x0, y0 + best_dist),
                                    (markLocXs[-1], y0 + h + best_dist)],
                                   "#ff0000", linewidth=6)

        # image = draw_line_in_image(image,
        #         [(imgMidX, best_midY),
        #          (imgMidX, best_midY + 10)],
        #         "#00ffaa", linewidth=5)
        # image = draw_line_in_image(image,
        #         [(imgMidX, best_midY_up),
        #          (imgMidX, best_midY_up + 10)],
        #         "#00ffaa", linewidth=10)
        image = draw_line_in_image(image,
                                   [(imgMidX, best_midY_center),
                                    (imgMidX, best_midY_center + 10)],
                                   "#00ffaa", linewidth=15)

        plt.rcParams['figure.figsize'] = (plt_width, plt_height)
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        plt.imshow(image)  # , cmap=plt.get_cmap('gray'))  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    return best_slope, best_dist, imgMidX, best_midY_center


def get_gear_features(img_file):
    if not os.path.exists(img_file):
        print(f'file [{img_file}] not found. ')
        return
    img_plt = mpimg.imread(img_file)
    img_height = img_plt.shape[0]
    img_width = img_plt.shape[1]
    leftLoc, rightLoc = get_gear_edge(img_plt, threshold=0.30)
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)

    best_slope, best_dist, imgMidX, best_midY_center = get_gear_slope(image, leftLoc=leftLoc, rightLoc=rightLoc)
    print((best_slope, best_dist, imgMidX, best_midY_center))

    if best_slope is not None:
        h = abs(best_slope * (rightLoc - leftLoc))
        y = best_midY_center + int((imgMidX - leftLoc) * best_slope)
        image = draw_line_in_image(image, [(leftLoc, y), (rightLoc, y + h)], "#ff0000", linewidth=6)
        image = draw_line_in_image(image, [(leftLoc, y + best_dist), (rightLoc, y + h + best_dist)], "#ff0000",
                                   linewidth=6)
        image = draw_line_in_image(image, [(leftLoc, 0), (leftLoc, img_height - 1)], '#00ff00', linewidth=2)
        image = draw_line_in_image(image, [(rightLoc, 0), (rightLoc, img_height - 1)], '#00ff00', linewidth=2)
        image.save(f'./images/result.jpg')
        image = image.resize((int(img_width / 4), int(img_height / 4)))
        image.show()


img_file = './demo/3_chimian_23671226_20210513_105659644_5.jpg'
# img_file = './demo/1_10__H2_817171_IO-NIO198M_210303A0125-2-1.jpg'
# img_file = './demo/1_11__H2_817171_IO-NIO198M_210415A0002-2-2.jpg'
# img_file = './demo/1_5__H2_817171_IO-NIO198M_210415A0002-2-2.jpg'
# img_file = './demo/1_6__H2_817171_IO-NIO198M_210415A0002-2-1.jpg'

get_gear_features(img_file)
