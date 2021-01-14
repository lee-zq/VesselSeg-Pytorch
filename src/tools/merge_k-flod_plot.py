"""
合并k折交叉验证在一个数据集上的测试结果，给出合并后的performance
"""
import sys
sys.path.append('/ssd/lzq/sf3')
from lib.metrics import Evaluate
import numpy as np

# second = ['stare1','stare2','stare3','stare4']
# second = ['chase1','chase2','chase3','chase4']
second = ['s_1','s_2','s_3','s_4']
# second = ['c_1','c_2','c_3','c_4']
save_path = './output/STARE'
agent = Evaluate(save_path=save_path)
for i in second:
    data = np.load('./output/{}/result.npy'.format(i))
    y_true, y_prob = data[0],data[1]
    agent.add_batch(y_true,y_prob)
np.save('{}/result.npy'.format(save_path),np.asarray([agent.target,agent.output]))
agent.save_all_result(plot_curve=True)