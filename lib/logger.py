
import sys
import os
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
from .common import dict_round

# Record data in tensorboard and .csv files during training stage
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

# Record the information printed in the terminal
class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
# call by
# sys.stdout = Logger(os.path.join(save_path,'test_log.txt'))

