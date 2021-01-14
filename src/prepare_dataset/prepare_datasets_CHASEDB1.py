#==========================================================
#
# 将训练集按照图片数量比例分配至训练集和验证集
#
#============================================================

import os
import h5py,imageio
import cv2
import numpy as np
from PIL import Image

def readImg(im_fn):
    im = cv2.imread(im_fn)
    if im is None :
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:,:] #mask和gt是灰度图，取单通道
            #  _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY) #单通道不需要二值化处理，只有test中2nd_manual是四通道，暂时不考虑
    # print('loading image ：', im_fn)
    return im

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

channels = 3
height = 960
width = 999

def get_train_val(imgs_dir,groundTruth_dir,borderMasks_dir,file_list):
    length = len(file_list)

    imgs = np.empty((length,height,width,channels))
    groundTruth = np.empty((length,height,width))
    border_masks = np.empty((length,height,width))
    for i,files in enumerate(file_list): #list all files, directories in the path
            #original
            print("original image: " +files)
            img = readImg(imgs_dir+files)
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[0:9] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            g_truth = readImg(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth[:,:,0])
            #corresponding border masks
            border_masks_name = files[0:9] + ".png"
            print("border masks name: " + border_masks_name)
            b_mask = readImg(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask[:,:,0])

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (length,channels,height,width))
    groundTruth = np.reshape(groundTruth,(length,1,height,width))
    border_masks = np.reshape(border_masks,(length,1,height,width))
    assert(groundTruth.shape == (length,1,height,width))
    assert(border_masks.shape == (length,1,height,width))
    print('data shape:',imgs.shape,groundTruth.shape,border_masks.shape)
    return imgs, groundTruth, border_masks

if __name__ == '__main__':
    #------------Path of the images --------------------------------------------------------------
    original_imgs_train = "./data/CHASEDB1/images/"
    groundTruth_imgs_train = "./data/CHASEDB1/1st_label/"
    borderMasks_imgs_train = "./data/CHASEDB1/mask/"
    #---------------------------------------------------------------------------------------------
    dataset_path = "./data/CHASEDB4_data/"
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    #getting the training and valid datasets
    val_rate = 0.25
    file_list = sorted(os.listdir(original_imgs_train))
    split_pos = int(len(file_list) * (1 - val_rate)) # 28
    train_list = file_list[:21]#+file_list[21:]
    val_list =  file_list[21:]
    print('train_list:',train_list)
    print('test_list:', val_list)
    
    imgs_train, groundTruth_train, border_masks_train = get_train_val(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,train_list)
    print("saving train datasets")
    write_hdf5(imgs_train, dataset_path + "imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_path + "groundTruth_train.hdf5")
    write_hdf5(border_masks_train,dataset_path + "borderMasks_train.hdf5")

    imgs_val, groundTruth_val, border_masks_val = get_train_val(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,val_list)
    print("saving train datasets")
    write_hdf5(imgs_val, dataset_path + "imgs_test.hdf5")
    write_hdf5(groundTruth_val, dataset_path + "groundTruth_test.hdf5")
    write_hdf5(border_masks_val,dataset_path + "borderMasks_test.hdf5")
