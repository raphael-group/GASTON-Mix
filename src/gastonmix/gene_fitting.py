import numpy as np
import segmented_fit
import binning
import seaborn as sns
from sklearn.preprocessing import normalize

from importlib import reload
reload(binning)

def perform_regressions_and_binning(counts_mat, expert_ind,gastonmix_labels, isodepth, gene_labels,
                                    cell_type_df=None,num_cts=5,t=0.1,q=0.2,umi_threshold=None,
                                    zero_fit_threshold=None,num_bins=7):

    # Piecewise linear fit parameters
    t=0.1 # set slope=0 if LLR p-value > 0.1

    if umi_threshold is None:
        umi_threshold=np.quantile( np.count_nonzero( counts_mat[gastonmix_labels==expert_ind,:], 0 ), q )

    if zero_fit_threshold is None:
        zero_fit_threshold=np.quantile( np.sum( counts_mat[gastonmix_labels==expert_ind,:], 0 ), q )

    # get cell types to plot
    if cell_type_df is not None:
        ct_df_ind=cell_type_df.iloc[gastonmix_labels == expert_ind]
        ct_prop_mat=normalize(ct_df_ind,axis=0,norm='l1')
        
        binning_output=binning.bin_data(counts_mat[gastonmix_labels==expert_ind,:], np.zeros(np.sum(gastonmix_labels==expert_ind)), 
                                        isodepth, cell_type_df.iloc[gastonmix_labels==expert_ind], gene_labels, 
                                        num_bins=num_bins, umi_threshold=umi_threshold)
        
        unique_binned_isodepths=binning_output['unique_binned_isodepths']
        binned_labels=binning_output['binned_labels']
        ct_count_mat=binning_output['binned_cell_type_mat'].T # len(unique_cell_types) x binned_labels
        cell_type_names=binning_output['cell_type_names']
    
        # get top CTs
        num_cts=5
        top_cts=np.argsort(np.sum(ct_count_mat,1))[-num_cts:]
        ct_list=cell_type_names[top_cts]
    else:
        binning_output=binning.bin_data(counts_mat[gastonmix_labels==expert_ind,:], np.zeros(np.sum(gastonmix_labels==expert_ind)), 
                                        isodepth, None, gene_labels, num_bins=num_bins, umi_threshold=umi_threshold)
        ct_list=[]
        ct_df_ind=None
    
    ####################################
    

    
    ####################################
    # compute piecewise linear fits
    pw_fit_dict=segmented_fit.pw_linear_fit(counts_mat[gastonmix_labels==expert_ind,:], np.zeros(np.sum(gastonmix_labels==expert_ind)), 
                                                isodepth, ct_df_ind, ct_list, 
                                                zero_fit_threshold=zero_fit_threshold,t=t,umi_threshold=umi_threshold,isodepth_mult_factor=1)

    return pw_fit_dict, binning_output,ct_list

###################################################################
# get gradient genes; see https://github.com/raphael-group/GASTON/blob/main/src/gaston/spatial_gene_classification.py
###################################################################
from collections import defaultdict

def get_cont_genes(pw_fit_dict, binning_output, q=0.95, ct_attributable=False, domain_cts=None, ct_perc=0.6):
    cont_genes=defaultdict(list) # dict of gene -> [list of domains]
    gene_labels_idx=binning_output['gene_labels_idx']
    

    slope_mat_all,_,_,_=pw_fit_dict['all_cell_types']
    slope_q=np.quantile(np.abs(slope_mat_all), q,0)
    
    L=len(slope_q)
    for i,g in enumerate(gene_labels_idx):
        for l in range(L):
            if np.abs(slope_mat_all[i,l]) > slope_q[l]:
                #if g not in cont_genes:
                #    cont_genes[g]=[l]
                #else:
                cont_genes[g].append(l)
    
    if not ct_attributable:
        return cont_genes
    
    cont_genes_domain_ct={g: [] for g in cont_genes} # dict gene -> [(domain,ct)]

    for g in cont_genes:
        for l in cont_genes[g]:
            other=True
            for ct in domain_cts[l]:
                if np.abs( pw_fit_dict[ct][0][gene_labels_idx==g,l] ) / np.abs(pw_fit_dict['all_cell_types'][0][gene_labels_idx==g,l]) > ct_perc:
                    other=False
                    cont_genes_domain_ct[g].append( (l,ct) )
                
            if other:
                cont_genes_domain_ct[g].append( (l, 'Other') )
                
    return cont_genes_domain_ct