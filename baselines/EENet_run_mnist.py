from baselines.load_data import load_mnist_1d
from EENet import EE_Net
from EEHes import EE_Hes
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
    b = load_mnist_1d()


    for i in range(runing_times):  

        lr_1 = 0.01 #learning rate for exploitation network
        lr_2 = 0.001 #learning rate for exploration network
        lr_3 = 0.001 #learning rate for decision maker

        regrets = []
        regrets_hes = []
        sum_regret = 0
        sum_regret_hes = 0
        ee_net = EE_Net(b.dim, b.n_arm, pool_step_size = 50, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3,  hidden=100,_kernel_size=100, _stride=50, neural_decision_maker = False)
        ee_hes = EE_Hes(b.dim, b.n_arm, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3, f1_hidden=100, f2_hidden=128, neural_decision_maker = False)

        for t in range(2000):
            context, rwd = b.step()
            
            arm_select, ee_scores = ee_net.predict(context, t)
            arm_select_hes, hes_scores = ee_hes.predict(context, t)

            reward = rwd[arm_select]
            reward_hes = rwd[arm_select_hes]

            regret = np.max(rwd) - reward
            regret_hes = np.max(rwd) - reward_hes
            #print(t, ee_scores)

            #tensor = torch.tensor(context, dtype=torch.float32).to("cuda")
            ee_net.update(context, reward, t)
            ee_hes.update(context, reward_hes, t)
            

            sum_regret += regret
            sum_regret_hes += regret_hes

            if t<1000:
                if t%10 == 0:
                    loss_1, loss_2, loss_3  = ee_net.train(t)
                    loss_1_hes, loss_2_hes, loss_3_hes = ee_hes.train(t)
            else:
                if t%100 == 0:
                    loss_1, loss_2, loss_3  = ee_net.train(t)
                    loss_1_hes, loss_2_hes, loss_3_hes = ee_hes.train(t)

            regrets.append(sum_regret)
            regrets_hes.append(sum_regret_hes)

            if t % 50 == 0:
                print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}, loss_3:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1, loss_2, loss_3))
                print("hes:regret:"+str(sum_regret_hes)+" average_regret:"+str(sum_regret_hes/(t+1))+" loss1:"+str(loss_1_hes)+" loss2:"+str(loss_2_hes)+" loss3:"+str(loss_3_hes))
                printtime()
                # wandb.log({"hes_sum":df})
            
        print(' regret: {:},  average_regret: {:.2f}'.format(sum_regret, sum_regret/(t+1)))
        regrets_all.append(regrets)

    path = os.getcwd()
    np.save('{}/results/eenet_results_iris.npy'.format(path), regrets_all)