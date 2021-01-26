from PIL import Image
import os,sys
sys.path.append('../')
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

def crop_and_resize(img,center,crop_length,target_shape):
    crop_img = img[center[1]-crop_length//2:center[1]+crop_length//2,
                   center[0]-crop_length//2:center[0]+crop_length//2]
    return cv2.resize(crop_img, target_shape, interpolation=cv2.INTER_CUBIC)

if __name__ == "__main__":
    result = readImg('/ssd/lzq/sf3/output/s_3/result_img/Original_GroundTruth_Prediction2.png')
    ori_img = readImg('/ssd/lzq/sf3/data/STARE/images/im0081.ppm')
    save_path = '/ssd/lzq/sf3/result_detail_visualization'
    w0 = 580
    h0 = 360
    block = 90

    result_list = split_result(result)
    _, prob_block, bin_block, gt_block= [crop_and_resize(img,(w0,h0),block,ori_img.shape[::-1][1:]) for img in result_list]

    ori_block = crop_and_resize(ori_img,(w0,h0),block,ori_img.shape[::-1][1:])

    # 拼接
    res = np.concatenate((ori_img,ori_block,prob_block,bin_block,gt_block),axis=1)

    cv2.rectangle(res, (w0-block//2,h0-block//2),(w0+block//2,h0+block//2),255,2)
    cv2.line(res, (w0-block//2,h0-block//2),(ori_img.shape[1],0), 255, 2)
    cv2.line(res, (w0-block//2,h0+block//2),ori_img.shape[::-1][1:], 255, 2)
    cv2.imwrite(save_path+'/detail_comp_result_0081.png', res)

