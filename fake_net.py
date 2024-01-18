import torch

def Tensor(anything):
    return torch.tensor(anything, dtype = torch.float32).to("cuda")



#context should be matrix, cuda, float32
def mimic_net(context, w0, b0, w1, b1):
    relu = torch.nn.ReLU()
    hidden = relu(context.mm(w0.T)+b0)
    output = hidden.mm(w1.T)+b1
    return output

h = torch.autograd.functional.hessian

def return_hes_func(context,target):
    context = context.unsqueeze(0)
    def hes_func(W0,B0,W1,B1):
        output = net1(context, W0, B0, W1, B1)
        loss = (output-target)**2
        return loss
    return hes_func

def get_wandb(net):
    w0 = net.fc1.weight
    b0 = net.fc1.bias
    w1 = net.fc2.weight
    b1 = net.fc2.bias
    return w0,b0,w1,b1
    
def calculate_hessian(context, net):
    w0 = net.fc1.weight
    b0 = net.fc1.bias
    w1 = net.fc2.weight
    b1 = net.fc2.bias
    hes = h(net1, (context, w0, b0, w1, b1))
    return hes

