import torch
from copy import deepcopy
import numpy as np
import os
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.linalg import orthogonal_procrustes
from robust_wm_stsp.lightning_task import dDMTSDataModule

def distance_between_rnn_distracted_and_undistracted(rnn,generator):
    
    num_delays = 4
    num_samps = 8
    
    
    inp, out_des,y,test_on,dis_bool = next(iter(generator))

    out_readout, out_hidden,w_hidden,_= rnn(inp)
        
    
    unique_delay_times = torch.unique(test_on)
    
    if rnn.fixed_syn == False:    
    
        dists_neur = []
        dists_syn = []

        for i in unique_delay_times:    
            distance_vs_time_neur = 0
            distance_vs_time_syn = 0
            for j in range(num_samps):                
                dis_inds = torch.where((dis_bool == 1) & (y == j) & (test_on == i))[0]
                non_dis_inds = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                
                if len(dis_inds)==0 or len(non_dis_inds) == 0:
                    print('Bad! Batch size too small for undistracted vs distracted.')


                mean_dis_traj_neur = out_hidden[dis_inds].mean(0)
                mean_non_dis_traj_neur = out_hidden[non_dis_inds].mean(0)  

                mean_dis_traj_syn = w_hidden[dis_inds].mean(0)
                mean_non_dis_traj_syn = w_hidden[non_dis_inds].mean(0)  


                distance_vs_time_neur += ((mean_non_dis_traj_neur - mean_dis_traj_neur)**2).mean(1)
                distance_vs_time_syn += ((mean_non_dis_traj_syn - mean_dis_traj_syn)**2).mean(1)

            dists_neur.append(distance_vs_time_neur.cpu().detach())
            dists_syn.append(distance_vs_time_syn.cpu().detach())

        return dists_neur,dists_syn
    else:
        dists_neur = []       
        for i in unique_delay_times:    
            distance_vs_time_neur = 0
            distance_vs_time_syn = 0
            for j in range(num_samps):                
                dis_inds = torch.where((dis_bool == 1) & (y == j) & (test_on == i))[0]
                non_dis_inds = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                
                if len(dis_inds)==0 or len(non_dis_inds) == 0:
                    print('Bad! Batch size too small.')
          

                mean_dis_traj_neur = out_hidden[dis_inds].mean(0)
                mean_non_dis_traj_neur = out_hidden[non_dis_inds].mean(0)  

                distance_vs_time_neur += ((mean_non_dis_traj_neur - mean_dis_traj_neur)**2).mean(1)

            dists_neur.append(distance_vs_time_neur.cpu().detach())

        return dists_neur,_

def distance_between_rnn_samples(rnn,generator):
    
    num_delays = 4
    num_samps = 8
    
    inp, out_des,y,test_on,dis_bool = next(iter(generator))
    
    out_readout, out_hidden,w_hidden,_= rnn(inp)
        
        
        
    unique_delay_times = torch.unique(test_on)
    
    if rnn.fixed_syn == False:
    
        dists_neur = []
        dists_syn = []


        for i in unique_delay_times:    
            distance_vs_time_neur = 0
            distance_vs_time_syn = 0        
            for j in range(num_samps): 
                for k in range(num_samps):   
                    non_dis_inds_j = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                    non_dis_inds_k = torch.where((dis_bool == 0) & (y == k) & (test_on == i))[0]
                    
                    
                    if len(non_dis_inds_j)==0 or len(non_dis_inds_k) == 0:
                        print('Bad! Batch size too small. Distance between samples.')
                    
                    else:

                        mean_traj_j_neur = out_hidden[non_dis_inds_j].mean(0)
                        mean_traj_k_neur = out_hidden[non_dis_inds_k].mean(0)   

                        mean_traj_j_syn = w_hidden[non_dis_inds_j].mean(0)
                        mean_traj_k_syn = w_hidden[non_dis_inds_k].mean(0)                


                        distance_vs_time_neur += ((mean_traj_j_neur - mean_traj_k_neur)**2).mean(1)

                        distance_vs_time_syn += ((mean_traj_j_syn - mean_traj_k_syn)**2).mean(1)

            dists_neur.append(distance_vs_time_neur.cpu().detach())
            dists_syn.append(distance_vs_time_syn.cpu().detach())
            

        return dists_neur,dists_syn
    else:
        dists_neur = []       


        for i in unique_delay_times:    
            distance_vs_time_neur = 0
            distance_vs_time_syn = 0        
            for j in range(num_samps): 
                for k in range(num_samps):   
                    non_dis_inds_j = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                    non_dis_inds_k = torch.where((dis_bool == 0) & (y == k) & (test_on == i))[0]
                    
                    
                    if len(non_dis_inds_j)==0 or len(non_dis_inds_k) == 0:
                        print('Bad! Batch size too small. Distance between samples.')
                 

                    mean_traj_j_neur = out_hidden[non_dis_inds_j].mean(0)
                    mean_traj_k_neur = out_hidden[non_dis_inds_k].mean(0)            

                    distance_vs_time_neur += ((mean_traj_j_neur - mean_traj_k_neur)**2).mean(1)                    

            dists_neur.append(distance_vs_time_neur.cpu().detach())
            

        return dists_neur,_    

    
    
    
    

def get_validation_accuracy(rnn,generator):
    inp, out_des,y,test_on,dis_bool = next(iter(generator))
           
    out_readout,_,_,_ = rnn(inp)       

    accs = np.zeros(out_readout.shape[0])
    #test model performance                    
    for i in range(out_readout.shape[0]):
        curr_max = out_readout[i,int(test_on[i])+int(500/rnn.dt_ann):int(test_on[i])+2*int(500/rnn.dt_ann),:-1].argmax(dim = 1).cpu().detach().numpy()
        accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)
        
    return accs.mean()


def decoder(rnn,generator,include_distractor = True):
    
    num_delays = 4
    num_samps = 8
    
    inp, out_des,y,test_on,dis_bool = next(iter(generator))
    
    out_readout, out_hidden,w_hidden,_= rnn(inp)

    unique_delay_times = torch.unique(test_on)
    
    if rnn.fixed_syn == False:
    
        info_neur = []
        info_syn = []
    
        for i in unique_delay_times:   
           
            if include_distractor == False:
                inds = torch.where((dis_bool == 0) & (test_on == i))[0]
            else:
                inds = torch.where(test_on == i)[0]
                
            
            neur_scores = do_info_over_trials(out_hidden[inds].detach().numpy(),y[inds])
            syn_scores = do_info_over_trials(w_hidden[inds].detach().numpy(),y[inds])
            
            info_neur.append(neur_scores)
            info_syn.append(syn_scores)

        return info_neur,info_syn
    else:
        info_neur = []
        for i in unique_delay_times:    

            if include_distractor == False:
                inds = torch.where((dis_bool == 0) & (test_on == i))[0]
            else:
                inds = torch.where(test_on == i)[0]
                
            neur_scores = do_info_over_trials(out_hidden[inds].detach().numpy(),y[inds])  
            info_neur.append(neur_scores)
            
        return info_neur,_    

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def do_info_over_trials(X,y,rbf_kernel = True):
    #import data here
    T = X.shape[1]
    #split data into training and test trials
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    #piece together steps (scale and regress)
    if rbf_kernel == False:
        pipe = make_pipeline(StandardScaler(), svm.LinearSVC(max_iter = 1000))
    else:
        pipe = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf',max_iter = 1000))
    #for each time point, train a classifier and then get its performance on test set
    scores = np.zeros(T)
    for time in range(T):
        pipe.fit(X_train[:,time], y_train)    
        scores[time] = pipe.score(X_test[:,time], y_test)    
    return scores

def rnn_norm_of_response(rnn,generator,dt_ann = 15):
    
    num_delays = 4
    num_samps = 8
    
    
    inp, out_des,y,test_on,dis_bool = next(iter(generator))

    out_readout, out_hidden,w_hidden,_= rnn(inp)
        
    
    unique_delay_times = torch.unique(test_on)
    
    if rnn.fixed_syn == False:    
    
        norms_neur = []
        norms_syn = []
        
        norms_neur_distracted = []
        norms_syn_distracted = []

        for i in unique_delay_times:    
            
            norms_vs_time_neur = []
            norms_vs_time_neur_distracted = []           
            
            norms_vs_time_syn = []
            norms_vs_time_syn_distracted = []           
            
            
            for j in range(num_samps):                
                dis_inds = torch.where((dis_bool == 1) & (y == j) & (test_on == i))[0]
                non_dis_inds = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                
                if len(dis_inds)==0 or len(non_dis_inds) == 0:
                    print('Bad! Batch size too small for undistracted vs distracted.')


                mean_dis_traj_neur = out_hidden[dis_inds].norm(p = 2, dim = 2).mean(0)
                mean_non_dis_traj_neur = out_hidden[non_dis_inds].norm(p = 2, dim = 2).mean(0)

                mean_dis_traj_syn = w_hidden[dis_inds].norm(p = 2, dim = 2).mean(0)
                mean_non_dis_traj_syn = w_hidden[non_dis_inds].norm(p = 2, dim = 2).mean(0)
                
                norms_vs_time_neur.append(mean_non_dis_traj_neur- mean_non_dis_traj_neur[int(1000/dt_ann)])
                norms_vs_time_neur_distracted.append(mean_dis_traj_neur - mean_dis_traj_neur[int(1000/dt_ann)])          
                
                
                norms_vs_time_syn.append(mean_non_dis_traj_syn - mean_non_dis_traj_syn[int(1000/dt_ann)])
                norms_vs_time_syn_distracted.append(mean_dis_traj_syn- mean_dis_traj_syn[int(1000/dt_ann)])
                
            norms_neur.append(norms_vs_time_neur)
            norms_neur_distracted.append(norms_vs_time_neur_distracted)
            
            norms_syn.append(norms_vs_time_syn)
            norms_syn_distracted.append(norms_vs_time_syn_distracted)

        return norms_neur,norms_neur_distracted,norms_syn,norms_syn_distracted
    else:
        norms_neur = []     
        norms_neur_distracted = []   
        for i in unique_delay_times:    
            norms_vs_time_neur = []
            norms_vs_time_neur_distracted = []           

            for j in range(num_samps):                
                dis_inds = torch.where((dis_bool == 1) & (y == j) & (test_on == i))[0]
                non_dis_inds = torch.where((dis_bool == 0) & (y == j) & (test_on == i))[0]
                
                if len(dis_inds)==0 or len(non_dis_inds) == 0:
                    print('Bad! Batch size too small.')          

                mean_dis_traj_neur = out_hidden[dis_inds].norm(p = 2, dim = 2).mean(0)
                mean_non_dis_traj_neur = out_hidden[non_dis_inds].norm(p = 2, dim = 2).mean(0)
                
                norms_vs_time_neur.append(mean_non_dis_traj_neur - mean_non_dis_traj_neur[int(1000/dt_ann)])
                norms_vs_time_neur_distracted.append(mean_dis_traj_neur - mean_dis_traj_neur[int(1000/dt_ann)])            
                
            norms_neur.append(norms_vs_time_neur)
            norms_neur_distracted.append(norms_vs_time_neur_distracted)

        return norms_neur, norms_neur_distracted,_,_


def structural_robustness(model,tester,min_noise = 0,max_noise = 0.95,num_noise_levels = 20):

    #perturbation values to loop through
    structural_noise_levels = np.linspace(min_noise,max_noise,num_noise_levels)
    
    #for storing network accuracy
    accs = []

    #loop over structural noise levels    
    for p in structural_noise_levels:    
        
        #import fresh copy
        pert_model = deepcopy(model)

        #add perturbation mask
        with torch.no_grad():

            pert_model.rnn.struc_perturb_mask = torch.FloatTensor(model.rnn.hidden_size, model.rnn.hidden_size).uniform_() > p            

            acc = get_validation_accuracy(pert_model,tester)

        accs.append(acc)
    
    return np.asarray(accs), structural_noise_levels


def process_noise_robustness(model,tester,min_noise = 0,max_noise = 0.95,num_noise_levels = 20):

    #perturbation values to loop through
    process_noise_levels = np.linspace(min_noise,max_noise,num_noise_levels)
    
    #for storing network accuracy
    accs = []

    #loop over structural noise levels    
    for p in process_noise_levels:    
        
        #import fresh copy
        pert_model = deepcopy(model)

        #add perturbation mask
        with torch.no_grad():

            pert_model.rnn.process_noise = p           

            acc = get_validation_accuracy(pert_model,tester)

        accs.append(acc)
    
    return np.asarray(accs), process_noise_levels



def get_max_min_similar(corrs, inds):
    
    inds_corrs = np.stack(corrs[inds])
    
    most_brain_similar = np.nanargmax(inds_corrs[:,-1])
    least_brain_similar = np.nanargmin(inds_corrs[:,-1])

    return inds[most_brain_similar], inds[least_brain_similar]




def do_LDA_neural_data(neural_data):

    clf_sample = LinearDiscriminantAnalysis()
    clf_pretest = LinearDiscriminantAnalysis()

    sample = np.concatenate([neural_data[x]['sample'] for x in neural_data.keys()],axis = 0) 
    pre_test = np.concatenate([neural_data[x]['pre_test'] for x in neural_data.keys()],axis = 0)
    y = np.concatenate([neural_data[x]['labels'] for x in neural_data.keys()])

    sample_lda = clf_sample.fit_transform(sample,y)
    pretest_lda = clf_pretest.fit_transform(pre_test,y)

    #find orthogonal transformation that aligns decoder readouts
    R_samp2pretest, _ = orthogonal_procrustes(sample_lda,pretest_lda)
    samp_new = sample_lda @ R_samp2pretest


    return_dict = {"samp": samp_new,
    "pre_test":pretest_lda,
    "y":y}  

    return return_dict

from robust_wm_stsp.lightning_networks import dDMTSNet
def get_model_from_index(df,idx):

    '''
    Hacky way to load the selected model...
    '''
    directory = '/home/leo/ExpStableDynamics/plos_comp_bio_rebuttal/Robust_WM_STSP/scripts/_lightning_sandbox/checkpoints'


    for i, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        rnn_type_str = 'rnn='+df.loc[idx]['rnn_type']
        nl_str = 'nl='+df.loc[idx]['nl']
        hs_str = 'hs='+str(df.loc[idx]['hs'])
        act_reg_str= 'act_reg='+str(df.loc[idx]['act_reg'])
        gamma_str='gamma='+ str(df.loc[idx]['gamma'])
        param_reg='param_reg='+ str(df.loc[idx]['param_reg'])

        if (rnn_type_str in f) and (nl_str in f) and (hs_str in f) and (act_reg_str in f) and (gamma_str in f) and (param_reg in f):
            print(f)
            return dDMTSNet.load_from_checkpoint(f)


def get_info_curve_from_model(model,tester,distractor = True):

    inp, out_des,y,test_on,dis_bool = next(iter(tester))

    out_readout, out_hidden,w_hidden,_= model(inp)


    unique_delay_times = torch.unique(test_on)

    with torch.no_grad():

        if distractor == True:
             inds = torch.where((test_on == unique_delay_times[-1]) & (dis_bool == 1))[0]
        else:
            inds = torch.where((test_on == unique_delay_times[-1]) & (dis_bool == 0))[0]

        neur_scores = do_info_over_trials(out_hidden[inds].detach().numpy(),y[inds])

    return neur_scores



def get_pca_from_model(model,tester,neurs = True,model_name = None):
    
    inp, out_des,y,test_on,dis_bool = next(iter(tester))
    out_readout, out_hidden,w_hidden,_= model(inp) 

    unique_delay_times = torch.unique(test_on)
    
    longest_delay = np.max(unique_delay_times.detach().numpy())
    long_delay_inds = (test_on == longest_delay)

    frac_sample_neurons = 1
    times = np.linspace(-1,5.5,out_hidden.shape[1]) 

    if model_name == 'ah':
        sample_times = ((times >= 0.25) & (times < 0.75))
        #sample_times = ((times >= .75) & (times < 1.25))
    else:
        sample_times = ((times >= .5) & (times < 1))
        
    delay_times = ((times >= 0) & (times < 1))
    pretest_times = ((times >= 4) & (times < 4.5))

    out_hidden = out_hidden[long_delay_inds,:,:].detach().numpy()
    w_hidden = w_hidden[long_delay_inds,:,:].detach().numpy()
    
    #pca = PCA(n_components=3)
    clf_sample = LinearDiscriminantAnalysis()
    clf_delay = LinearDiscriminantAnalysis()
    clf_pretest = LinearDiscriminantAnalysis()
    #clf = TruncatedSVD(n_components=3, n_iter=10, random_state=42)

    #perform LDA on the sample times    
    if neurs == True:
        out_2_dim_red_sample = out_hidden[:,sample_times,:].mean(axis = 1).reshape((-1,out_hidden.shape[-1]),order = 'C')
        out_2_dim_red_delay = out_hidden[:,delay_times,:].mean(axis = 1).reshape((-1,out_hidden.shape[-1]),order = 'C')
        out_2_dim_red_pretest= out_hidden[:,pretest_times,:].mean(axis = 1).reshape((-1,out_hidden.shape[-1]),order = 'C')
    else:
        out_2_dim_red_sample = w_hidden[:,sample_times,:].mean(axis = 1).reshape((-1,w_hidden.shape[-1]),order = 'C')
        out_2_dim_red_delay = w_hidden[:,delay_times,:].mean(axis = 1).reshape((-1,w_hidden.shape[-1]),order = 'C')
        out_2_dim_red_pretest = w_hidden[:,pretest_times,:].mean(axis = 1).reshape((-1,w_hidden.shape[-1]),order = 'C')

    with torch.no_grad():

        times_repeat = np.tile(times,out_hidden.shape[0])       
        
        y_repeat_sample = y[long_delay_inds].detach().numpy()
        y_repeat_delay = y[long_delay_inds].detach().numpy()
        y_repeat_pretest = y[long_delay_inds].detach().numpy()


        out_2_dim_red_sample_train, out_2_dim_red_sample_test, y_repeat_sample_train, y_repeat_sample_test = train_test_split(out_2_dim_red_sample, y_repeat_sample, test_size=0.8,random_state = 42)
        out_2_dim_red_delay_train, out_2_dim_red_delay_test, y_repeat_delay_train, y_repeat_delay_test = train_test_split(out_2_dim_red_delay, y_repeat_delay, test_size=0.8,random_state = 42)
        out_2_dim_red_pretest_train, out_2_dim_red_pretest_test, y_repeat_pretest_train, y_repeat_pretest_test = train_test_split(out_2_dim_red_pretest, y_repeat_pretest, test_size=0.8,random_state = 42)

        clf_sample.fit(out_2_dim_red_sample_train,y_repeat_sample_train)
        clf_delay.fit(out_2_dim_red_delay_train,y_repeat_delay_train)
        clf_pretest.fit(out_2_dim_red_pretest_train,y_repeat_pretest_train)

        dim_red_out_sample = clf_sample.transform(out_2_dim_red_sample_test)#,y_repeat_sample_test)        
        dim_red_out_delay = clf_delay.transform(out_2_dim_red_delay_test)#,y_repeat_delay_test) 
        dim_red_out_pretest = clf_pretest.transform(out_2_dim_red_pretest_test)#,y_repeat_pretest_test) 

        #find orthogonal transformation that aligns decoder readouts
        R_samp2pretest, _ = orthogonal_procrustes(dim_red_out_sample,dim_red_out_pretest)
        samp_new = dim_red_out_sample @ R_samp2pretest

        #R_pretest2delay, _ = orthogonal_procrustes(dim_red_out_pretest,dim_red_out_delay)
        #pretest_new = dim_red_out_pretest @ R_pretest2delay


    return_dict = {"dim_red_out_sample": samp_new,
    "dim_red_out_delay":dim_red_out_delay,
    "dim_red_out_pretest": dim_red_out_pretest,
    "y_repeat_sample": y_repeat_sample_test,
    "y_repeat_delay": y_repeat_delay_test,
    "y_repeat_pretest": y_repeat_pretest_test,
    "times_repeat": times_repeat}  

    return return_dict


def get_most_similar_LDA(corrs,inds,tester,df,neurs = True,model_name = None):
    
    most_brain_similar, least_brain_similar = get_max_min_similar(corrs, inds)
    model = get_model_from_index(df,most_brain_similar)
    return_dict = get_pca_from_model(model,tester,neurs,model_name)
    
    return return_dict


def get_time_segs(times):
    times_three_segment = []
    for t in times:
        if t <= 0:
            times_three_segment.append('w')
        if t > 0 and t <= 0.5:
            times_three_segment.append('w')
        if t > 0.5 and t <= 2.5:
            times_three_segment.append('r')
        if t > 2.5 and t <= 4:
            times_three_segment.append('b')
        if t > 4 and t <= 4.5:
            times_three_segment.append('w')
        if t > 4.5:
            times_three_segment.append('g')        

    return times_three_segment


def get_info_curves_from_inds(inds):
    most_brain_similar, least_brain_similar = get_max_min_similar(corrs, inds)
    
    model_least = get_model_from_index(least_brain_similar)
    info_curve_least = get_info_curve_from_model(model_least,tester)

    model_most = get_model_from_index(most_brain_similar)
    info_curve_most = get_info_curve_from_model(model_most,tester)

    return info_curve_most,info_curve_least


def load_val_tester():
    dt_ann = 15
    #load data generator
    dDMTS = dDMTSDataModule(dt_ann = dt_ann)  
    dDMTS.setup()
    tester = dDMTS.val_dataloader()

    return tester