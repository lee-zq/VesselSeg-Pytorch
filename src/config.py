
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='../experiments/', help='trained model will be saved at here')
    parser.add_argument('--save',default='test', help='save path name')
    # parser.add_argument('--model', '-m', default='untitled')
    # parser.add_argument('--dataset', default="DRIVE", help='dataset name')
    # parser.add_argument('--aug', default=True, type=bool,help='augment dataset(已实现,内置) ')

    # data
    parser.add_argument('--train_data_path_list',default='/ssd/lzq/projects/vesselseg/src/prepare_dataset/data_path_list/DRIVE/train.txt')

    # parser.add_argument('--val_img', default='imgs_val.hdf5')
    # parser.add_argument('--val_gt', default='groundTruth_val.hdf5')
    # parser.add_argument('--val_mask', default='borderMasks_val.hdf5')

    parser.add_argument('--test_data_path_list', default='/ssd/lzq/projects/vesselseg/src/prepare_dataset/data_path_list/DRIVE/train.txt')

    parser.add_argument('--patch_height', default=48)
    parser.add_argument('--patch_width', default=48)
    parser.add_argument('--N_subimgs', default=100000)
    parser.add_argument('--inside_FOV', default=False)
    parser.add_argument('--val_portion',default=0.1)
    # model parameters
    parser.add_argument('--in_channels', default=1, type=int,help='input channels')
    parser.add_argument('--classes', default=2, type=int,help='output channels')

    # training
    parser.add_argument('--N_epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int,help='batch size')
    parser.add_argument('--early-stop', default=None, type=int,help='early stopping (default: 20)')
    # parser.add_argument('--loss', default='BCEWithLogitsLoss',help='loss function')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='optimizer')
    parser.add_argument('--lr', default=0.0001, type=float,
                         help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')

    parser.add_argument('--pre_trained',default=None,help='(path of trained _model)load trained model to continue train')

    # pre_trained checkpoint
    parser.add_argument('--start_epoch',default=1)

    # testing
    parser.add_argument('--full_images_to_test', default=20,help='N full images to be predicted')
    parser.add_argument('--N_group_visual', default=1,help='Grouping of the predicted images')
    parser.add_argument('--average_mode', default=True)
    parser.add_argument('--test_patch_height', default=96)
    parser.add_argument('--test_patch_width', default=96)
    parser.add_argument('--stride_height', default=8)
    parser.add_argument('--stride_width', default=8)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool, help='use cuda compute')

    args = parser.parse_args()

    return args