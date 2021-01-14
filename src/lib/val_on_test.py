import joblib
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
# help_functions.py
from collections import OrderedDict
from lib.help_functions import *
# extract_patches.py
from lib.extract_patches import recompone
from lib.extract_patches import recompone_overlap
from lib.extract_patches import kill_border
from lib.extract_patches import pred_only_FOV
from lib.extract_patches import get_data_testing
from lib.extract_patches import get_data_testing_overlap
# pre_processing.py
from lib.pre_processing import my_PreProc
from os.path import join
from lib.dataset.dataset import TestDataset
import matplotlib.pylab as pylab
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed

setpu_seed(2020)
class Val_on_testSet():
    #====================extract test parameters===================
    def __init__(self,args):
        self.args = args
        # original test images
        DRIVE_test_imgs_original = args.path_data + args.test_img
        self.test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
        self.full_img_height = self.test_imgs_orig.shape[2]
        self.full_img_width = self.test_imgs_orig.shape[3]
        # the border masks provided by the DRIVE (for FOV selection)
        DRIVE_test_border_masks = args.path_data + args.test_mask
        self.test_border_masks = load_hdf5(DRIVE_test_border_masks)
        assert (args.stride_height < args.test_patch_height and args.stride_width < args.test_patch_width)
        # save path
        self.path_experiment = args.outf + args.save +'/'
        #============ Load the data and divide in patches=================================
        if args.average_mode == True:  # 该模式下，会重叠取测试patch，最终结果由平均得到
            self.patches_imgs_test, self.new_height, self.new_width, self.masks_test = get_data_testing_overlap(
                DRIVE_test_imgs_original = DRIVE_test_imgs_original,  # original
                DRIVE_test_groudTruth = args.path_data + args.test_gt,
                Imgs_to_test = args.full_images_to_test,
                patch_height = args.test_patch_height,
                patch_width = args.test_patch_width,
                stride_height = args.stride_height,
                stride_width = args.stride_width
            )
            gtruth_masks = self.masks_test  # ground truth masks

        else:  # 该模式下直接评估不重叠的所有测试patch，性能略差
            self.patches_imgs_test, self.patches_masks_test = get_data_testing(
                DRIVE_test_imgs_original = DRIVE_test_imgs_original,  # original
                DRIVE_test_groudTruth = args.path_data + args.test_gt,  # masks
                Imgs_to_test = args.full_images_to_test,
                patch_height = args.test_patch_height,
                patch_width = args.test_patch_width,
            )
            gtruth_masks = recompone(self.patches_masks_test,13,12)
        self.gtruth_masks = gtruth_masks[:,:,0:self.full_img_height,0:self.full_img_width]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                  shuffle=False, num_workers=6)
    def inference(self,net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = torch.nn.functional.softmax(outputs,dim=1)
                outputs = outputs.permute(0,2,3,1)
                shape = list(outputs.shape)
                outputs = outputs.view(-1,shape[1]*shape[2],2)
                outputs = outputs.data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds,axis=0)
        #===== Convert the prediction arrays in corresponding images
        self.pred_patches = pred_to_imgs(predictions, self.args.test_patch_height, self.args.test_patch_width, "prob")

    def evaluate(self):
        #========== Elaborate and visualize the predicted images ====================
        if self.args.average_mode == True:
            pred_imgs = recompone_overlap(self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)# predictions
        else:
            pred_imgs = recompone(self.pred_patches,13,12)       # predictions
        kill_border(pred_imgs, self.test_border_masks)  #DRIVE MASK  #only for visualization
        ## back to original dimensions
        pred_imgs = pred_imgs[:,:,0:self.full_img_height,0:self.full_img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_FOV(pred_imgs,self.gtruth_masks, self.test_border_masks)  #returns data only inside the FOV
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true,y_scores)
        log = OrderedDict([('val_auc_roc', eval.auc_roc()), ('val_acc',eval.confusion_matrix()[1]),('val_f1', eval.f1_score())])
        return log

if __name__ == '__main__':
    trained_model = 'test_new_val'

    # net = models.denseunet.Dense_Unet(1,2,filters=64)
    net = models.LadderNetv6(inplanes=1, num_classes=2, layers=4, filters=10)
    net.cuda()
    args = joblib.load('./output/%s/args.pkl' % trained_model)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, './output/%s/args.pkl' % args.save)
    cudnn.benchmark = True
    # Load checkpoint.
    path_experiment = args.outf + args.save + '/'
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(join(path_experiment, 'best_model.pt7'))
    net.load_state_dict(checkpoint['net'])

    eval = Val_on_testSet(args)
    eval.inference(net)
    print(eval.evaluate())