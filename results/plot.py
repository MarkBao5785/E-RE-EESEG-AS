import numpy as np
import matplotlib.pyplot as plt
import sys

dataset = sys.argv[1]
print(dataset)

def load(filename):
    data = np.load(filename)
    data = data.squeeze()
    return data

def data_plot(p, x, data, colour, lineStyle, Label):
    p.plot(x, data, color = colour, linestyle = lineStyle, label = Label)

def output(p, dataset):
    p.xlabel('Rounds')
    p.ylabel('Regret')
    p.legend()
    p.title(dataset)
    p.savefig(dataset+"_edit.png",dpi=500)

def extractname(filename:str):
    filename = filename.split(".")[0]
    filename = filename.split("_")
    method = filename[0]
    dataset = filename[1]
    return method,dataset

def create_data(filename):
    data = load(filename)
    method, dataset = extractname(filename)
    return method,dataset,data

def plot_output(filename):
    x = range(2000)
    plt.figure(figsize=(10,6))
    method,dataset = extractname(filename)

    data = load(filename)
    data_plot(plt, x, data, 'red', '-', method+"_"+dataset)
    
    output(plt, method+"_"+dataset)


 
x = range(2000)
plt.figure(figsize=(10, 6))

method = ["E-RE-EESEG-AS","EENet","LinUCB","NeuralTS","NeuralUCB"]

color = ["red","black","yellow","blue","green","purple"]
style = ["-",":","--","-.","-","-."]

for i in range(5):
    data = load(method[i]+"_"+dataset+".npy")
    data_plot(plt, x, data, color[i], style[i], method[i])

output(plt, dataset)
