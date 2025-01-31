import numpy as np
import seaborn as sns
from sklearn.preprocessing import normalize
from binning import bin_data
import matplotlib.pyplot as plt

def plot_ct_vs_isodepth(cell_type_df,expert_ind,gastonmix_labels,isodepth,num_bins=20,num_cts=5,
                       figsize=(7,6),colors=None):
    
    ct_df_ind=cell_type_df.iloc[gastonmix_labels == expert_ind]
    ct_prop_mat=normalize(ct_df_ind,axis=0,norm='l1')
    
    N=ct_df_ind.shape[0]
    binning_output=bin_data(np.ones((N,10)), np.zeros(np.sum(gastonmix_labels==expert_ind)), isodepth, 
                             ct_df_ind, np.array(['test' for i in range(10)]), num_bins=num_bins)
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    ct_count_mat=binning_output['binned_cell_type_mat'].T # len(unique_cell_types) x binned_labels
    cell_type_names=binning_output['cell_type_names']
    
    L=len(np.unique(binned_labels))

    # get top cts
    top_cts=np.argsort(np.sum(ct_count_mat,1))[-num_cts:]
    
    fig,axs=plt.subplots(figsize=figsize)
    
    if colors is None:
        colors=[None for _ in range(num_cts)]
    
    for _,ct in enumerate(top_cts):
        plt.plot(unique_binned_isodepths,ct_count_mat[ct,:] / np.sum(ct_count_mat[top_cts,:],0),label=cell_type_names[ct],lw=5,c=colors[_])
    plt.legend(frameon=False,fontsize=15)
    sns.despine()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()