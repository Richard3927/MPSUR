import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

def random_mask_feature(data, ratio):

    a = random.randint(1,10)
    if a <= int(10*ratio):
        mask_data = int(len(data)*ratio)
        random_numbers = [random.randint(0, len(data)-1) for _ in range(mask_data)]
        for i in random_numbers:
            data[i] = 0
        return data
    else:
        return data      

def normalization(data):   ########def定义归一化函数，主要的目的：使模型更快收敛
    #_range = np.max(data) - np.min(data)
    #return (data - np.min(data)) / _range

    _range = np.max(abs(data))      ######找到图像中绝对值的最大值
    return data / _range    #######数据除以最大值，使数据在[-1,1]之间，让模型收敛更快，使训练效果更好

def data_read_CB(data, label):
    data = data.reshape((-1,5396))
    series = random_mask_feature(data[:,:20],0.2)
    text = random_mask_feature(data[:,20:].reshape((-1,5376)),0.5)
    # series = normalization(series)
    # text = normalization(text)

    return np.array(series), np.array(text), np.array(label).astype(np.int64)
    

def data_read_CB_test(data, label):
    data = data.reshape((-1,5396))
    series = data[:,:20]
    text = data[:,20:].reshape((-1,5376))
    # series = normalization(series)
    # text = normalization(text)

    return np.array(series), np.array(text), np.array(label).astype(np.int64)
    
    
def data_read_CB_mask(data, label, index):
    data = data.reshape((-1,5396))
    series = random_mask_feature(data[:,:20],0.2)
    data_text = np.delete(data[:,20:].reshape((-1,768)),index,0)
    #print(data_text.shape)
    text = random_mask_feature(data_text.reshape((-1,6*768)),0.5)
    #print(text.shape)
    # series = normalization(series)
    # text = normalization(text)

    return np.array(series), np.array(text), np.array(label).astype(np.int64)   


def data_read_CB_test_mask(data, label, index):
    data = data.reshape((-1,5396))
    series =  data[:,:20]
    data_text = np.delete(data[:,20:].reshape((-1,768)),index,0)
    #print(data_text.shape)
    text = data_text.reshape((-1,6*768))
    #print(text.shape)
    # series = normalization(series)
    # text = normalization(text)
    graph = np.loadtxt("/disk1/xly/un-planned_reoperation/data/adj_template_1.txt")
    return np.array(series), np.array(text), np.array(graph), np.array(label).astype(np.int64) 

  
# data_test_ex_V1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_07_10/5gezhongxin_test_3.txt")
# data_test_ex1_V1 = np.loadtxt("/disk1/xly/un-planned_reoperation/data/2025_07_10/5gezhongxin_test_text_CB.txt")

# data_test_ex_V1 = np.concatenate((data_test_ex_V1, data_test_ex1_V1), axis=1)
# X_ex_V1 = data_test_ex_V1[:, 2:]
# y_ex_V1 = data_test_ex_V1[:, 1]
# a,b,c = data_read_CB_mask(X_ex_V1[1],y_ex_V1[1],1)
    
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, label = data_read_CB_test(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, label
        


class DatasetSplit_test(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, label = data_read_CB_test(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, label



class DatasetSplit_test_mask(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label, index):
        self.all_data = all_data
        self.all_label = all_label
        self.index = index
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, graph, label = data_read_CB_test_mask(self.all_data[item],self.all_label[item], self.index)

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, graph, label
        
 
        
# a = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] 
# print(random_mask_feature(a,1))

def data_read_LLama2_CM(data, label):
    data = data.reshape((-1,35860))
    series = data[:,:20]
    text = data[:,20:].reshape((-1,35840))
    # series = normalization(series)
    # text = normalization(text)

    return np.array(series), np.array(text), np.array(label).astype(np.int64)
    
    



def data_read_llama2(data, label):
    data = data.reshape((-1,35860))#35860
    series = data[:,:20]
    text = data[:,20:].reshape((-1,35840))
    # series = normalization(series)
    # text = normalization(text)
    graph = np.loadtxt("/disk1/xly/un-planned_reoperation/data/adj_template_1.txt")
    return np.array(series), np.array(text), np.array(graph), np.array(label).astype(np.int64)
 
class DatasetSplit_LLama2(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self): 
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, graph, label = data_read_llama2(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, graph, label
        
        
        
def data_read_llama3(data, label):
    data = data.reshape((-1,28692))
    series = data[:,:20]
    text = data[:,20:].reshape((-1,28672))
    # series = normalization(series)
    # text = normalization(text)
    graph = np.loadtxt("/disk1/xly/un-planned_reoperation/data/adj_template_1.txt")
    return np.array(series), np.array(text), np.array(graph), np.array(label).astype(np.int64)
 
    # return np.array(series), np.array(text), np.array(label).astype(np.int64)
 
class DatasetSplit_LLama3(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, graph, label = data_read_llama3(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, graph, label
        


def data_read_CB_test_g(data, label):
    data = data.reshape((-1,5396))#5396
    series = data[:,:20]
    text = data[:,20:].reshape((-1,5376))#5376
    # series = normalization(series)
    # text = normalization(text)
    graph = np.loadtxt("/disk1/xly/un-planned_reoperation/data/adj_template_1.txt")
    return np.array(series), np.array(text), np.array(graph), np.array(label).astype(np.int64)

        
        
class DatasetSplit_g(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, graph, label = data_read_CB_test_g(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, graph, label
        


class DatasetSplit_test_g(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, all_data, all_label):
        self.all_data = all_data
        self.all_label = all_label
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item): 
        series, text, graph, label = data_read_CB_test_g(self.all_data[item],self.all_label[item])

        #print(image.dtype)
        #print(torch.tensor(image))
        return series, text, graph, label