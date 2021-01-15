import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from collections import OrderedDict
from lib.help_functions import *
import os
import argparse
from lib.logger import Logger, Print_Logger
# extract_patches.py
from lib.extract_patches import recompone_overlap, kill_border, pred_only_FOV, get_data_test_overlap
# pre_processing.py
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc

setpu_seed(2020)
class Test_on_testSet():
    #====================extract test parameters===================
    def __init__(self,args):
        self.args = args
        assert (args.stride_height <= args.patch_height and args.stride_width <= args.patch_width)
        # save path
        self.path_experiment = args.outf + args.save +'/'
        #============ Load the data and divide in patches=================================
        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list = args.test_data_path_list,
            patch_height = args.patch_height,
            patch_width = args.patch_width,
            stride_height = args.stride_height,
            stride_width = args.stride_width
            )

        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size,shuffle=False, num_workers=3)

    def inference(self,net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = torch.nn.functional.softmax(outputs,dim=1)
                outputs = outputs.permute(0,2,3,1)
                outputs = outputs.view(-1,outputs.shape[1]*outputs.shape[2],2)
                outputs = outputs.data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds,axis=0)
        #===== Convert the prediction arrays in corresponding images
        self.pred_patches = pred_to_imgs(predictions, self.args.patch_height, self.args.patch_width, "prob")

    def evaluate(self):
        #========== Elaborate and visualize the predicted images ====================
        pred_imgs = recompone_overlap(self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)# predictions
        orig_imgs = my_PreProc(self.test_imgs)     #originals
        gtruth_masks = self.test_masks

        kill_border(pred_imgs, self.test_FOVs)  #DRIVE MASK  #only for visualization
        ## back to original dimensions
        pred_imgs = pred_imgs[:,:,0:self.img_height,0:self.img_width]
        ######################保存结果图########
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)

        N_predicted = orig_imgs.shape[0]
        group = 1
        assert (N_predicted%group==0)
        for i in range(int(N_predicted/group)):
            orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)  # 原图
            masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group) # GT
            pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)  # 预测概率图
            binary = copy.deepcopy(pred_stripe)
            binary[binary>=0.5]=1
            binary[binary<0.5]=0  # 预测二值图
            total_img = np.concatenate((orig_stripe,pred_stripe,binary,masks_stripe),axis=1)   # 拼接
            visualize(total_img,self.save_img_path +"/Original_GroundTruth_Prediction"+str(i))#.show()
        ######################
        #predictions only inside the FOV
        y_scores, y_true = pred_only_FOV(pred_imgs,self.test_masks, self.test_FOVs)  #returns data only inside the FOV
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true,y_scores)
        # log = OrderedDict([('val_auc_roc', eval.auc_roc()), ('val_f1', eval.f1_score())])
        log = eval.save_all_result(plot_curve=True)

        # save labels and probs
        np.save('{}result.npy'.format(self.path_experiment),np.asarray([y_true,y_scores]))

        return dict_round(log,6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='../experiments/', help='trained model will be saved at here')
    parser.add_argument('--test_data_path_list', default='/ssd/lzq/projects/vesselseg/src/prepare_dataset/data_path_list/DRIVE/train.txt')

    parser.add_argument('--patch_height', default=96)
    parser.add_argument('--patch_width', default=96)
    parser.add_argument('--batch_size', default=32, type=int,help='batch size')
    # testing
    parser.add_argument('--average_mode', default=True)
    parser.add_argument('--stride_height', default=8)
    parser.add_argument('--stride_width', default=8)

    parser.add_argument('--save',default='test', help='save path name')
    # hardware setting
    args = parser.parse_args()
    sys.stdout = Print_Logger(os.path.join('../experiments/',args.save,'test_log.txt'))
    # net = models.denseunet.Dense_Unet(1,2,filters=64)
    net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16)
    net.cuda()

    cudnn.benchmark = True
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(join('../experiments/', args.save, 'best_model.pth'))
    # checkpoint = torch.load(join('./output/', args.save, 'latest_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test_on_testSet(args)
    eval.inference(net)
    print(eval.evaluate())