from PIL import Image
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

def readImg(im_fn):
    im = Image.open(im_fn)
    return im

def split_result(result,count=4):
    res = []
    h,w,c = result.shape
    w = w//4 
    for i in range(count):
        img = result[:,w*i:w*(i+1)]
        res.append(img) # 变三通道
    return res

def crop_and_resize(img,center,crop_length,target_shape,inter = None):
    crop_img = img[center[1]-crop_length//2:center[1]+crop_length//2,
                   center[0]-crop_length//2:center[0]+crop_length//2]
    if inter == 1:#最近邻插值
        return cv2.resize(crop_img, target_shape, interpolation=cv2.INTER_NEAREST)
    elif inter == 2:#双线性插值
        return cv2.resize(crop_img, target_shape, interpolation=cv2.INTER_LINEAR)
    elif inter == 3:#双三次插值
        return cv2.resize(crop_img, target_shape, interpolation=cv2.INTER_CUBIC)
    else:
        raise TypeError("please choose current interpolation!!!")

if __name__ == "__main__":
    result = readImg('/ssd/lzq/sf3/output/s_3/result_img/Original_GroundTruth_Prediction2.png')
    img_path = '/ssd/lzq/sf3/data/STARE/images/im0081.ppm'
    save_path = '/ssd/lzq/sf3/visualization/result_detail_visualization'
    w0 = 590
    h0 = 360
    block = 110

    ori_img = readImg(img_path)
    idx = img_path.split('/')[-1].split('.')[0]

    result_list = split_result(result)
    prob_block = crop_and_resize(result_list[1],(w0,h0),block,(ori_img.shape[0],ori_img.shape[0]),inter=2)
    bin_block = crop_and_resize(result_list[2],(w0,h0),block,(ori_img.shape[0],ori_img.shape[0]),inter=1)
    gt_block = crop_and_resize(result_list[3],(w0,h0),block,(ori_img.shape[0],ori_img.shape[0]),inter=1)

    cv2.imwrite(save_path+'/{}_prob_block.png'.format(idx), prob_block)
    cv2.imwrite(save_path+'/{}_bin_block.png'.format(idx), bin_block)
    cv2.imwrite(save_path+'/{}_gt_block.png'.format(idx), gt_block)

    ori_block = crop_and_resize(ori_img,(w0,h0),block,(ori_img.shape[0],ori_img.shape[0]),inter=2)
    cv2.imwrite(save_path+'/{}_ori__block.png'.format(idx), ori_block)
    # 拼接
    # res = np.concatenate((ori_img,ori_block,prob_block,bin_block,gt_block),axis=1)

    cv2.rectangle(ori_img, (w0-block//2,h0-block//2),(w0+block//2,h0+block//2),255,5)
    cv2.line(ori_img, (w0-block//2,h0-block//2),(ori_img.shape[1],0), 255, 5)
    cv2.line(ori_img, (w0-block//2,h0+block//2),ori_img.shape[::-1][1:], 255, 5)
    cv2.imwrite(save_path+'/{}_ori_img.png'.format(idx), ori_img)

