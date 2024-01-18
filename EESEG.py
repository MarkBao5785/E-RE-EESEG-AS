import numpy as np

from EESEGClass import Exploitation, Exploration, Decision_maker

class EE_SEG:
    def __init__(self, dim, n_arm, lr_1 = 0.01, lr_2 = 0.01, lr_3 = 0.01, f1_hidden=100, f2_hidden=128, neural_decision_maker = False):
        #Network 1
        self.f_1 = Exploitation(dim, n_arm, lr_1, f1_hidden)
        self.n_arm = n_arm
        # number of dimensions of aggregated for f_2  
        f_2_input_dim = f1_hidden*9
        #Network 2
        self.f_2 = Exploration(f_2_input_dim, lr_2, f2_hidden)
        
        #Network 3
        self.f_3 = Decision_maker(2, 20, lr_3) 
        
        self.arm_select = 0
        
        self.exploit_scores = []
        self.explore_scores = []
        self.ee_scores = []
        self.hessian = []
        
        self.contexts = []
        self.rewards = []
        self.decision_maker = neural_decision_maker
        
    def predict(self, context, t):  # np.array 
        self.exploit_scores, self.hessian = self.f_1.output_and_hessian(context)
        self.exploit_scores = self.exploit_scores.detach().cpu().reshape([self.n_arm,1]).numpy()
        self.explore_scores = self.f_2.output(self.hessian).reshape([self.n_arm,1])
        self.ee_scores = np.concatenate((self.exploit_scores, self.explore_scores), axis=1)
        
        if self.decision_maker and t > 500:  
            # neural decision maker
            self.arm_select = self.f_3.select(self.ee_scores)
        else:
            # linear decision maker
            f_2_weight = 1.0
            if t > 1000: f_2_weight = 0.1
            suml = self.exploit_scores + f_2_weight * (self.explore_scores-1.0)
            self.arm_select = np.argmax(suml)
        return self.arm_select, self.ee_scores
    
    def update(self, context, r_1, t):
        # update exploitation network
        self.f_1.update(context[self.arm_select], r_1)
        
        self.contexts.append(context[self.arm_select])
        self.rewards.append(r_1)
        
        # update exploration network
        f_1_predict = self.exploit_scores[self.arm_select][0]
        r_2 = (r_1 - f_1_predict) + 1.0
        self.f_2.update(self.hessian[self.arm_select], r_2)
        
        # add additional samples to exploration net when the selected arm is not optimal
        if t < 1000:
            if r_1 == 0:
                index = 0
                for i in self.hessian:
                    c = 1.2
                    if index != self.arm_select:
                        self.f_2.update(i, c)
                    index += 1
        
        # update decision maker
        self.f_3.update(self.ee_scores[self.arm_select], float(r_1))

    def train(self, t):
        #train networks
        loss_1 = self.f_1.train()        
        loss_2 = self.f_2.train()
        if self.decision_maker:
            loss_3 = self.f_3.train()
        else:
            loss_3 = 0.0
        return loss_1, loss_2, loss_3
    