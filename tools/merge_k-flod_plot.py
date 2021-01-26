"""
Because the STARE and CHASE_DB1 dataset are not divided into trainset 
and testset, cross-validation should be used.The script combines the 
test results of k-fold cross-validation on a dataset to give the final performance.

Attention: Run from the root directory of the project, otherwise the package may not be found
"""
import sys
sys.path.append('./') # The root directory of the project
from lib.metrics import Evaluate
import numpy as np

# second = ['stare1','stare2','stare3','stare4']
# second = ['chase1','chase2','chase3','chase4']
second = ['s_1','s_2','s_3','s_4']
# second = ['c_1','c_2','c_3','c_4']
save_path = './experiments/STARE'
agent = Evaluate(save_path=save_path)
for i in second:
    data = np.load('./experiments/{}/result.npy'.format(i))
    y_true, y_prob = data[0],data[1]
    agent.add_batch(y_true,y_prob)
np.save('{}/result.npy'.format(save_path),np.asarray([agent.target,agent.output]))
agent.save_all_result(plot_curve=True)