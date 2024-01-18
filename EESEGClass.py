
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from skimage.measure import block_reduce




if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu" 
device = torch.device(dev)

# aquire weight and bias from a fc1-relu-fc2 model
def get_wandb(net):
    w0 = net.fc1.weight
    b0 = net.fc1.bias
    w1 = net.fc2.weight
    b1 = net.fc2.bias
    return w0,b0,w1,b1

# mimic a fc1-relu-fc2 model, for calculating the hessian(Output->Weight/Bias)
def mimic_net(context, w0, b0, w1, b1):
    relu = torch.nn.ReLU()
    hidden = relu(context.mm(w0.T)+b0)
    output = hidden.mm(w1.T)+b1
    return output

#abbr for hessian
h = torch.autograd.functional.hessian

#construct the function for calculating the hessian(Output->Weight/Bias)
def return_hessian_func(context):
    def hes_func(W0,B0,W1,B1):
        output = mimic_net(context, W0,B0,W1,B1)
        output = output**2
        return output
    return hes_func

def hessian_process(hessian):
    h00 = hessian[0][0].sum(dim = (1,2,3)).detach().cpu().numpy()
    h01 = hessian[0][1].sum(dim = (1,2)).detach().cpu().numpy()
    h02 = hessian[0][2].sum(dim = (1,2,3)).detach().cpu().numpy()
    h03 = hessian[0][3].sum(dim = (1,2)).detach().cpu().numpy()
    h11 = hessian[1][1].sum(dim = 1).detach().cpu().numpy()
    h12 = hessian[1][2].sum(dim = (1,2)).detach().cpu().numpy()
    h13 = hessian[1][3].sum(dim = 1).detach().cpu().numpy()
    h22 = hessian[2][2].sum(dim = (0,2,3)).detach().cpu().numpy()
    h23 = hessian[2][3].sum(dim = (0,2)).detach().cpu().numpy()
    # h33 is a constant scalar; others are ignored for the symmetry of hessian.
    return np.concatenate((h00,h01,h02,h03,h11,h12,h13,h22,h23))
'''Network Structure'''

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Network_decision_maker(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_decision_maker, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    

'''Network functions'''

class Exploitation:
    def __init__(self, input_dim, num_arm, lr = 0.01, hidden=100):
        '''input_dim: number of dimensions of input'''    
        '''num_arm: number of arms'''
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''
        
        self.func = Network_exploitation(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []        
        self.lr = lr
        self.n_arm = num_arm
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
    def output_and_hessian(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        results = self.func(tensor)
        hes_list = []
        para = get_wandb(self.func)
        for arm in range(self.n_arm):
            context_current = tensor[arm].unsqueeze(0)
            output_func = return_hessian_func(context_current)
            hessian = h(output_func, para)

            processed_hessian = hessian_process(hessian)
            hes_list.append(processed_hessian)
        return results, hes_list
    
    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                loss = (self.func(c.to(device)) - r)**2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

            

    
class Exploration:
    def __init__(self, input_dim, lr, hidden):
        self.func = Network_exploration(input_dim=input_dim, hidden=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr

    
    def update(self, context, reward):
        tensor = torch.from_numpy(context).float().to(device)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.unsqueeze(tensor, 0)
        self.context_list.append(tensor)
        self.reward.append(reward)
        
    def output(self, context):
        tensor = torch.from_numpy(np.array(context)).float().to(device)
        tensor = torch.unsqueeze(tensor, 1)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return res
    

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=0.0001)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                output = self.func(c.to(device))
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 2000:
                    return tot_loss / cnt
            if batch_loss / length <= 2e-3:
                #print("batched loss",  batch_loss / length)
                return batch_loss / length     
            
            


class Decision_maker:
    def __init__(self, input_dim, hidden=20, lr = 0.01):
        self.func = Network_decision_maker(input_dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        print("f3_lr", self.lr)

    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        
        
    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        ress = self.func(tensor).cpu()
        res = ress.detach().numpy()
        return np.argmax(res)

    def train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                target = torch.tensor([r]).unsqueeze(1).to(device)
                output = self.func(c.to(device))
                loss = (output - r)**2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length                   
    
