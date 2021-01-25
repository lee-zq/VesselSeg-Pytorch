import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def concat_result(ori_img,pred_res,gt):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))
    gt = data = np.transpose(gt,(1,2,0))
    # 预测二值图
    binary = deepcopy(pred_res)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0  
    # 拼接
    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
        gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
    total_img = np.concatenate((ori_img,pred_res,binary,gt),axis=1)
    return total_img

#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  #the image is between 0-1
    img.save(filename)
    return img

# def pred_to_imgs(pred, patch_height, patch_width):
#     assert (len(pred.shape)==3)  #3D array: (N_patches,height*width,2)
#     assert (pred.shape[2]==2 )  #check the classes are 2

#     pred_images = pred[:, :, 1]
#     pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
#     return pred_images