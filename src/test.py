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
from lib.extract_patches import *
# pre_processing.py
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc

setpu_seed(2020)


class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <=
                args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )

        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = outputs.permute(0, 2, 3, 1)
                outputs = outputs.view(-1, outputs.shape[1]*outputs.shape[2], 2)
                outputs = outputs.data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        #===== Convert the prediction arrays in corresponding images
        self.pred_patches = pred_to_imgs(predictions, self.args.test_patch_height, self.args.test_patch_width)

    def evaluate(self):
        #========== Elaborate and visualize the predicted images ====================
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## back to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True)
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment),
                np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])
            visualize(total_img,join(self.save_img_path, "img_prob_bin_gt_"+img_name_list[i]+'.png'))


if __name__ == '__main__':
    args = parse_args()
    # args.save = 'test16'
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # net = models.denseunet.Dense_Unet(1,2,filters=64)
    net = models.UNetFamily.U_Net(1,2).to(device)
    # net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
