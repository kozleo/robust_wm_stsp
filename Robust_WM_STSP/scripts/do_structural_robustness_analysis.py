from robust_wm_stsp.lightning_networks import dDMTSNet
from robust_wm_stsp.lightning_task import dDMTSDataModule
from robust_wm_stsp.lightning_helper import process_noise_robustness, structural_robustness
import numpy as np
import os
from functools import partial
from multiprocessing import Pool
#import xarray as xr









      







def parse_network_filename(filename): 
    #split hyperparam delims
    x = filename.split("--")   

    rnn_type = x[0].split("=")[-1]
    nl = x[1].split("=")[-1]
    hs = int(x[2].split("=")[-1])
    act_reg = float(x[3].split("=")[-1])
    gamma = float(x[4].split("=")[-1])
    param_reg = float(x[5].split("=")[-1])
    epoch = int(x[6].split("=")[-1])
    val_acc = float(x[7].split("=")[-1][0:3])
    
    
    return rnn_type, nl, hs, act_reg,gamma, param_reg, epoch, val_acc

'''
attributes = ["rnn_type", "nl","hs", "act_reg","gamma", "param_reg", "epoch", "val_acc","robustness"]  
model_IDs = np.arange(num_models)      

data_dict = {"rnn_type":[],
 "nl": [],
 "hs": [], 
 "act_reg": [],
 "gamma": [], 
 "param_reg": [], 
 "epoch": [], 
 "val_acc": [],
 "robustness": []}
'''


#xr.DataArray(np.random.randn(num_models,9),dims = ["model","attribute"],coords = [model_IDs, attributes ] )



# iterate over files in
# that directory


    #model = dDMTSNet.load_from_checkpoint(f)


    # checking if it is a file
    #if os.path.isfile(f):
        #print(f)


print('Loading tester...')
#import datamodule and tester
dt_ann = 15
#load data generator
dDMTS = dDMTSDataModule(dt_ann = dt_ann)  
dDMTS.setup()
tester = dDMTS.test_dataloader()





#create partial function with fixed tester
#do_struct_analysis = partial(structural_robustness,tester = tester, num_noise_levels = 20)
#do_process_noise_analysis = partial(structural_robustness,tester = tester, num_noise_levels = 20)

def load_model_and_do_robustness(filename, tester,num_noise_levels = 20):
    #load model
    model = dDMTSNet.load_from_checkpoint(filename)
   
    #do structural robustness
    struct_accs,struct_robust_vals = structural_robustness(model,tester,min_noise = 0,max_noise = 0.95,num_noise_levels = num_noise_levels)
    #do process noise robustness
    proc_accs,proc_robust_vals = process_noise_robustness(model,tester,min_noise = 0,max_noise = 0.95,num_noise_levels = num_noise_levels)
    
    return np.mean(struct_accs*struct_robust_vals), np.mean(proc_accs*proc_robust_vals)










# assign directory
directory = '/home/leo/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/scripts/_lightning_sandbox/checkpoints'

num_models = len([name for name in os.listdir('.') if os.path.isfile(name)])
path, dirs, files = next(os.walk(directory))
num_models = len(files)


data_dict = {"rnn_type":[],
"nl": [],
"hs": [], 
"act_reg": [],
"gamma": [], 
"param_reg": [], 
"epoch": [], 
"val_acc": [],
"struct_robustness": [],
"proc_robustness": []}


fs = []
for i, filename in enumerate(os.listdir(directory)):
    print(f'Analayzing {i} out of {num_models}')
    f = os.path.join(directory, filename)
    fs.append(f)

    rnn_type, nl, hs, act_reg,gamma, param_reg, epoch, val_acc = parse_network_filename(f)
    data_dict['rnn_type'].append(rnn_type)
    data_dict['nl'].append(nl)
    data_dict['hs'].append(hs)
    data_dict['act_reg'].append(act_reg)
    data_dict['gamma'].append(gamma)
    data_dict['param_reg'].append(param_reg)
    data_dict['epoch'].append(epoch)
    data_dict['val_acc'].append(val_acc)

    struct_robustness, proc_robustness = load_model_and_do_robustness(f,tester)

    data_dict['struct_robustness'].append(struct_robustness)
    data_dict['proc_robustness'].append(proc_robustness)



import pickle

with open('saved_robustness.pkl', 'wb') as f:
    pickle.dump(data_dict, f)




    




    


'''
print('loading model')
#load trained models
modeldir = '/om2/user/leokoz8/code/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/results/trained_networks/'
model = dDMTSNet.load_from_checkpoint(modeldir + 'rnn-sample-dDMTS-epoch=99-val_acc=0.29--stsp--relu-v1.ckpt')

print('Doing struct analysis')
#do info analysis 
accs,struct_robust_vals = structural_robustness(model,tester)
robustness = np.mean(accs*struct_robust_vals)
print(f'Finished struct analysis! Robust = {robustness}')
'''
