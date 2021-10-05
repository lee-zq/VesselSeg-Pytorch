import random
from os.path import join
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from lib.metrics import Evaluate
from lib.visualize import group_images, save_img
from lib.extract_patches import get_data_train
from lib.datasetV2 import data_preprocess,create_patch_idx,TrainDatasetV2
from tqdm import tqdm

# ========================get dataloader==============================
def get_dataloader(args):
    """
    该函数将数据集加载并直接提取所有训练样本图像块到内存，所以内存占用率较高，容易导致内存溢出
    """
    patches_imgs_train, patches_masks_train = get_data_train(
        data_path_list = args.train_data_path_list,
        patch_height = args.train_patch_height,
        patch_width = args.train_patch_width,
        N_patches = args.N_patches,
        inside_FOV = args.inside_FOV #select the patches only inside the FOV  (default == False)
    )
    val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(args.val_ratio*patches_masks_train.shape[0])))
    train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...],mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=6)

    val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...],mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=6)
    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        N_sample = min(patches_imgs_train.shape[0], 50)
        save_img(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader


def get_dataloaderV2(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)

    patches_idx = create_patch_idx(fovs_train, args)

    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
        visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
            visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
            if i>=N_sample-1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader

# =======================train======================== 
def train(train_loader,net,criterion,optimizer,device):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    return log

# ========================val=============================== 
def val(val_loader,net,criterion,device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets,outputs[:,1])
    log = OrderedDict([('val_loss', val_loss.avg), 
                       ('val_acc', evaluater.confusion_matrix()[1]), 
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc())])
    return log