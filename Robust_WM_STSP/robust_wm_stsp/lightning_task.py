import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import os

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

class DMTS_Dataset(torch.utils.data.Dataset):
    'Characterizes a delay-match to sample task'

    def __init__(self, inps,out_des,list_IDs, labels,test_ons, distracted_bools):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.inps = inps
        self.out_des = out_des
        self.test_ons = test_ons
        self.distracted_bools = distracted_bools

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        inp = self.inps[ID]
        y = self.labels[ID].long()
        out = self.out_des[ID]
        test_on = self.test_ons[ID]
        distracted_bool = self.distracted_bools[ID]
        
        return inp,out,y,test_on,distracted_bool

def generate_one_DMTS_IO(sample_mat, samp, noise_level, dt,alpha, time_limits = [-1,5.5], possible_delays = [1.  , 1.41, 2.  , 2.83, 4.  ], use_distractor = 1):

    
    num_samples = len(sample_mat)
    
    distractor1 = np.random.choice(np.delete(np.arange(num_samples),samp))
    
    #by convention the 8th and 9th samples are possible mid-delay distractors
    mid_delay_distractor_ind = np.random.choice([7,8])
    delay_length = 1000*np.random.choice(possible_delays)
    
    tvec = np.arange(time_limits[0],time_limits[1],1/1000)
    TIME_STEPS = len(tvec)
    
    
    inp = torch.zeros((TIME_STEPS,num_samples+1))   
    out_des = torch.zeros((TIME_STEPS,num_samples+1))     

    samp_on = np.argmin(np.abs(tvec-0))
    samp_off = samp_on + 500
    
    dis_on = int(delay_length/2) + samp_off
    dis_off  = dis_on + 250

    test_on = samp_off + int(delay_length)
    test_off = test_on + 500  
    
    #present sample
    inp[samp_on:samp_off,:-1] = sample_mat[samp]
    
    
    
    if np.heaviside(np.random.rand()-0.5,0):
        #present distractor on 50% of trials
        inp[dis_on:dis_off,:-1] = sample_mat[mid_delay_distractor_ind]
        distracted_bool = 1
    else:
        distracted_bool = 0
    
    #present test and sample
    inp[test_on:test_off,:-1] = sample_mat[samp] + sample_mat[distractor1]
    
    #fixate signal
    inp[0:test_on,-1] = 1 # fixation signal, answer when it goes off    
    
    #desired output
    #out_des[int(3000/dt):int(3500/dt),samp] = 1
    out_des[test_on:,samp] = 1
    out_des[0:test_on,-1] = 0
    
    inp += np.sqrt(2/alpha)*noise_level*torch.randn(inp.shape)
    
    
    return inp[::dt] , out_des[::dt], int(test_on/dt), distracted_bool



def generate_DMTS(dt = 100, tau = 100, time_limits = [-1,5.5], num_samples = 8+2, variable_delay = True, mid_delay_distractor = True):
    
    """"
   Generates one delayed-match to sample dataset. 
    
    ARGS:
        -dt: timestep to use
        -time_limits: beginning and end of trial. sample On = 0s. Units of s. 
        -num_samples: size of sample pool
        
    
    
    RETURNS:
        -inps: inputs into the rnn, size batch by time by num_samples + 1
        -out_des: desired outputs from the rnn, size batch by time by num_samples + 1
        -partition: training and testing IDs
        -labels: sample label for each element in dataset
    """   

    #use binary encoding of sample images
    sample_mat = torch.eye( num_samples )

    
    TIME_STEPS = len(np.arange(time_limits[0],time_limits[1], dt/1000))
        

    noise_level = 0.01
    
    num_train = int(2**14)#int(0.6*0.5*(2**13))
    num_test = int(2**12)#int(0.4*0.5*(2**13))
    num_val = int(2**12)

    num_examples = num_train + num_test + num_val #int(0.5*(2**10))

    inps = torch.empty((num_examples,TIME_STEPS, num_samples+1), requires_grad = False)
    out_des = torch.empty((num_examples,TIME_STEPS, num_samples+1),requires_grad = False)
    test_ons = torch.empty((num_examples),requires_grad = False)
    distracted_bools = torch.empty((num_examples),requires_grad = False)
    
    labels = torch.empty(num_examples)
    list_IDs = []

    for i in range(num_examples):
        samp = np.random.choice(np.arange(num_samples-2))   
        inps[i],out_des[i],test_ons[i],distracted_bools[i] = generate_one_DMTS_IO(sample_mat = sample_mat, samp = samp, noise_level = noise_level, dt = dt, alpha = dt/tau,time_limits = time_limits)
        labels[i] = samp
        list_IDs.append(i)

    partition = {'train': list_IDs[:num_train], 'test': list_IDs[num_train: num_train+ num_test],'val':list_IDs[num_train + num_test: num_train+ num_test+ num_val]}
    
    return inps,out_des,partition,labels,test_ons,distracted_bools


def get_DMTS_training_test_val_sets(dt_ann,train_batch = 256,test_batch = 256,val_batch = 2**12):
    
    inps,out_des,partition,labels,test_ons,distracted_bools = generate_DMTS(dt = dt_ann)

    training_set = DMTS_Dataset(inps,out_des,partition['train'],labels,test_ons,distracted_bools)


    test_set = DMTS_Dataset(inps,out_des,partition['test'],labels,test_ons, distracted_bools)


    validation_set = DMTS_Dataset(inps,out_des,partition['val'],labels,test_ons, distracted_bools)

    
    
    return training_set,test_set,validation_set


class dDMTSDataModule(pl.LightningDataModule):
    def __init__(self, dt_ann: int = 15, batch_size: int = 256):
        super().__init__()
        self.dt_ann = dt_ann  
        self.batch_size = batch_size
            

    def setup(self,stage = None):
        self.training_data,self.test_data,self.validation_data = get_DMTS_training_test_val_sets(self.dt_ann)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS,drop_last = True,pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=2**11,num_workers = NUM_WORKERS, drop_last = True, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.validation_data, batch_size=1024,num_workers = NUM_WORKERS, drop_last = True, pin_memory=False)