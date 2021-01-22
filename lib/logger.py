import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
from .common import dict_round
class Logger():
    def __init__(self,save_name):
        self.log = None
        self.summary = None
        self.name = save_name
        self.time_now = time.strftime('_%Y-%m-%d-%H-%M', time.localtime())

    def update(self,epoch,train_log,val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        item = dict_round(item,6) # 保留小数点后6位有效数字
        print(item)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/log%s.csv' %(self.name,self.time_now), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.name)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)
    def save_graph(self,model,input):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.name)
        self.summary.add_graph(model, (input,))
        print("Architecture of Model have saved in Tensorboard!")

# def save_sample_res(args,input,output,target):
#     cnt = 1
#     for in1, out1, tar1 in zip(input.detach().cpu().numpy(),
#                                np.argmax(output.detach().cpu().numpy(), axis=1),
#                                np.argmax(target.detach().cpu().numpy(), axis=1)):
#         plt.subplot(1, 3, 1), plt.imshow(in1[0]), plt.title('input_image')
#         plt.subplot(1, 3, 2), plt.imshow(out1), plt.title('prediction')
#         plt.subplot(1, 3, 3), plt.imshow(tar1), plt.title('groundtruth')
#         plt.savefig('outputs/{}/result_{}.png'.format(args.name,cnt))
#         cnt+=1

import sys
import os

class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# sys.stdout = Logger(os.path.join(save_path,'test_log.txt'))