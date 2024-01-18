from baselines.load_data import load_mnist_1d, load_movielen, load_yelp

from EESEG import EE_SEG
import numpy as np
import os
import torch



from env import *
# from fake_net import *



import time
def printtime():
    print("%d:%d:%d"%(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec))


if __name__ == '__main__':
    dataset = 'iris'
   
    runing_times = 1
    regrets_all = []
    b = load_yelp()


    for i in range(runing_times):  

        lr_1 = 0.01 #learning rate for exploitation network
        lr_2 = 0.001 #learning rate for exploration network
        lr_3 = 0.001 #learning rate for decision maker


        regrets_seg = []

        sum_regret_seg = 0
        ee_seg = EE_SEG(b.dim, b.n_arm, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3, f1_hidden=100, f2_hidden=128, neural_decision_maker = False)

        for t in range(2000):
            context, rwd = b.step()
            
            arm_select_seg, seg_scores = ee_seg.predict(context, t)

            reward_seg = rwd[arm_select_seg]

            regret_seg = np.max(rwd) - reward_seg
            #print(t, ee_scores)

            #tensor = torch.tensor(context, dtype=torch.float32).to("cuda")
            ee_seg.update(context, reward_seg, t)
            

            sum_regret_seg += regret_seg

            if t<1000:
                if t%10 == 0:
                    loss_1_seg, loss_2_seg, loss_3_seg = ee_seg.train(t)
            else:
                if t%100 == 0:
                    loss_1_seg, loss_2_seg, loss_3_seg = ee_seg.train(t)

            regrets_seg.append(sum_regret_seg)

            if t % 5 == 0:
                print("round:"+str(t)+" eeseg:regret:"+str(sum_regret_seg)+" average_regret:"+str(sum_regret_seg/(t+1))+" loss1:"+str(loss_1_seg)+" loss2:"+str(loss_2_seg)+" loss3:"+str(loss_3_seg))
                printtime()
                # wandb.log({"seg_sum":df})
    regrets_all.append(regrets_seg)
    path = os.getcwd()
    np.save('{}/results/seg_yelp.npy'.format(path), regrets_all)
