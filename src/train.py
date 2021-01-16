import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random,sys
from os.path import join
import torch
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.losses.lovasz_loss import lovasz_with_softmax
from lib.help_functions import *
from lib.common import *
from lib.dataset import TrainDataset,TestDataset
from config import parse_args
from lib.logger import Logger, Print_Logger
from collections import OrderedDict
from lib.metrics import Evaluate
import models
from test import Test_on_testSet


#  Load the data and divided in patches
def get_dataloader(args):
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
    # Save some samples of  feeding to the neural network
    N_sample = min(patches_imgs_train.shape[0], 100)
    visualize(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
              join(args.outf, args.save, "sample_input_imgs.png"))
    visualize(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
              join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader

# train 
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

# val 
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

            outputs = torch.nn.functional.softmax(outputs,dim=1)
            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets,outputs[:,1])
    log = OrderedDict([('val_loss', val_loss.avg), ('val_acc', evaluater.confusion_matrix()[1]), 
                        ('val_f1', evaluater.f1_score()),('val_auc_roc', evaluater.auc_roc())])
    return log

def main():
    setpu_seed(2021)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    save_path = join(args.outf, args.save)
    save_args(args,save_path)
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))

    # net = models.UNetFamily.R2AttU_Net(1,2).to(device)
    net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,48,48)).to(device).to(device=device))  # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    # criterion = CrossEntropy2d()

    # create a list of learning rate with epochs
    # lr_epoch = np.array([50, args.N_epochs])
    # lr_value = np.array([0.001, 0.0001])
    # lr_schedule = make_lr_schedule(lr_epoch,lr_value)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    # optimizer = optim.SGD(net.parameters(),lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)

    train_loader, val_loader = get_dataloader(args) 
    # eval_tool = Test_on_testSet(args) 
    best = {'epoch':0,'AUC_roc':0.5} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # early stop 计数器
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f)' % ((epoch), args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr']))

        train_log = train(train_loader,net,criterion, optimizer,device)
        val_log = val(val_loader,net,criterion,device)
        # eval_tool.inference(net)
        # val_log = eval_tool.evaluate()
        log.update(epoch,train_log,val_log)
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('Saving best model')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
