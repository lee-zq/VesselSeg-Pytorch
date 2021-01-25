
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='UNet_vessel_seg',
                        help='save name of experiment in args.outf directory')

    # data
    parser.add_argument('--train_data_path_list',
                        default='./prepare_dataset/data_path_list/DRIVE/train.txt')
    parser.add_argument('--test_data_path_list',
                        default='./prepare_dataset/data_path_list/DRIVE/test.txt')

    parser.add_argument('--train_patch_height', default=64)
    parser.add_argument('--train_patch_width', default=64)
    parser.add_argument('--N_patches', default=100000)
    parser.add_argument('--inside_FOV', default=True)
    parser.add_argument('--val_ratio', default=0.1)
    # model parameters
    parser.add_argument('--in_channels', default=1,
                        type=int, help='input channels')
    parser.add_argument('--classes', default=2,
                        type=int, help='output channels')

    # training
    parser.add_argument('--N_epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=6, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=True, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1)
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=96)
    parser.add_argument('--test_patch_width', default=96)
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda compute')

    args = parser.parse_args()

    return args
