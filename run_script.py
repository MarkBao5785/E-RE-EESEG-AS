from NeuralTS import NeuralTS
from NeuralUCB import NeuralUCBDiag
from KernelUCB import KernelUCB
from LinUCB import Linearucb
from EENet import EE_Net
from EESEG import EE_SEG

import numpy as np
import os
import torch
import argparse
import random
from baselines.load_data import load_mnist_1d, load_movielen, load_yelp
from dataset_gen import *
#IRIS and experimental_contextual_bandit
import time



def load_dataset(data, exp_setting=None):
    
    def exp_pass():
        return
    
    dic = {"movielens":load_movielen, "iris":IRIS, "yelp":load_yelp}
    b = dic.get(data, exp_pass)()
    if(b != None):
        return b
    else:
        T = exp_setting.T
        n_arms = exp_setting.n_arms
        reward_func = exp_setting.reward_func
        context = exp_setting.context
        noise_std = exp_setting.noise_std
        seed = exp_setting.seed
        bandit = Experimental_ContextualBandit(T, n_arms, n_features, reward_func, context, noise_std=noise_std, seed=seed)
        bandit.reset_rewards()
        b = bandit
    return b

def model_prediction(model, context, t, model_name):
    if ((model_name == "EESEG") | (model_name == "EENet")):
        arm_select, _ = model.predict(context, t)
    else:
        arm_select = model.select(context)
    return arm_select

def model_update(model_name, model, context, reward, t, arm_select):
    loss = 0
    if((model_name == "LinUCB") | (model_name == "KernelUCB")):
        model.train(context[arm_select],reward)
        return "None"
    elif((model_name == "EESEG") | (model_name == "EENet")):
        model.update(context, reward, t)
    else:
        model.update(context[arm_select], reward)
    if t<1000:
        if t%10 == 0:
            loss = model.train(t)
    else:
        if t%100 == 0:
            loss = model.train(t)
    return loss


def load_model(model_name, dataset, lr_1, lr_2, lr_3, default_lambda, default_nu):
    if(model_name == "EENet"):
        if(b.dim == 7840):
            model = EE_Net(dataset.dim, dataset.n_arms, pool_step_size = 50, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3,  hidden=100,_kernel_size=100, _stride=50, neural_decision_maker = False)
        else:
            model = EE_Net(dataset.dim, dataset.n_arms, pool_step_size = 20, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3,  hidden=100,_kernel_size=10, _stride=5, neural_decision_maker = False)
    elif(model_name == "EESEG"):
        model = EE_SEG(dataset.dim, dataset.n_arms, lr_1 = lr_1, lr_2 = lr_2, lr_3 = lr_3, f1_hidden=100, f2_hidden=128, neural_decision_maker = False)
    elif(model_name == "NeuralUCB"):
        model = NeuralUCBDiag(dataset.dim, lamdba=default_lambda, nu=default_nu, hidden=100)
    elif(model_name == "NeuralTS"):
        model = NeuralTS(dataset.dim, dataset.n_arms, m=100, sigma=default_lambda, nu=default_nu)
    elif(model_name == "KernelUCB"):
        model = KernelUCB(dataset.dim, default_lambda, default_nu)
    elif(model_name == "LinUCB"):
        model = Linearucb(dataset.dim, default_lambda, default_nu)
    return model

def printtime():
    print("%d:%d:%d"%(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run methods')
    parser.add_argument('--dataset', default='yelp', type=str, help='yelp, movielens, iris, linear, quadratic, cosine')
    parser.add_argument("--method", default="EENet", help="EENet, EESEG")

    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    arg_lambda = 0.1
    arg_nu = 0.001



    runing_times = 3
    regrets_all = []

    # Setup for experimental setting
    T = int(2000)
    n_arms = 3
    n_features = 6
    noise_std = 0.1

    SEED = 42

    arm_feature = np.random.randn(n_features)
    arm_feature /= np.linalg.norm(arm_feature, ord=2)
    
    reward_func = 0
    if(dataset == "linear"):
        reward_func = lambda x: 10*np.dot(arm_feature, x)
    elif(dataset == "quadratic"):
        reward_func = lambda x: 100*np.dot(arm_feature, x)**2
    elif(dataset == "cosine"):
        reward_func = lambda x: np.cos(2*np.pi*np.dot(x, arm_feature))
    setting = Experimental_Setting(T, n_arms, n_features, reward_func, arm_feature, noise_std, SEED)
    
    #load dataset    
    b = load_dataset(dataset, setting)

    for i in range(runing_times):  

        lr_1 = 0.01 #learning rate for exploitation network
        lr_2 = 0.001 #learning rate for exploration network
        lr_3 = 0.001 #learning rate for decision maker


        regrets = []

        sum_regret = 0
        model = load_model(method ,b, lr_1, lr_2, lr_3, arg_lambda, arg_nu)
        

        for t in range(T):
            context, rwd = b.step()
            
            arm_select = model_prediction(model, context, t, method)

            reward = rwd[arm_select]

            loss = model_update(method, model, context, reward, t, arm_select)
            regret = np.max(rwd) - reward

            sum_regret += regret
            regrets.append(sum_regret)

            if t % 50 == 0:
                print("method:"+method+" round:"+str(t)+" regret:"+str(sum_regret)+" average_regret:"+str(sum_regret/(t+1))+" loss:"+str(loss))
                printtime()
        regrets_all.append(regrets)
    path = os.getcwd()
    np.save('results_multi/'+method+'_'+dataset+'.npy'.format(path), regrets_all)
