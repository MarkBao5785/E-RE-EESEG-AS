import pandas as pd
import numpy as np

def extractname(filename:str):
    filename = filename.split(".")[0]
    filename = filename.split("_")
    method = filename[0]
    dataset = filename[1]
    return method,dataset

def file2csv(filenames):
    frame = {}
    for filename in filenames:
        data = np.load(filename).squeeze()
        method,dataset = extractname(filename)
        frame["%s_%s"%(method,dataset)] = data
    df = pd.DataFrame(frame)
    df.to_csv("%s.csv"%dataset)
    return df
