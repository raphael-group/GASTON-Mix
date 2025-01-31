#############################################
# See GASTON code: https://github.com/raphael-group/GASTON/blob/main/src/gaston/binning_and_plotting.py
#############################################

from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def bin_data(counts_mat, gaston_labels, gaston_isodepth, 
              cell_type_df, gene_labels, num_bins=7, idx_kept=None, umi_threshold=500, pc=0, 
             pc_exposure=True, extra_data=[]):
    
    if idx_kept is None:
        idx_kept=np.where(np.sum(counts_mat,0) > umi_threshold)[0]
    gene_labels_idx=gene_labels[idx_kept]
    
    exposure=np.sum(counts_mat,axis=1)
    
    cmat=counts_mat[:,idx_kept]
    N,G=cmat.shape
    if cell_type_df is not None:
        cell_type_mat=cell_type_df.to_numpy()
        cell_type_names=np.array(cell_type_df.columns)
    else:
        cell_type_mat=np.ones((N,1))
        cell_type_names=['All']

    


    # BINNING
    num_bins_per_domain=[num_bins]
    bins=np.array([])
    L=len(np.unique(gaston_labels))
    
    for l in range(L):
        isodepth_l=gaston_isodepth[np.where(gaston_labels==l)[0]]
        
        if l>0:
            isodepth_lm1=gaston_isodepth[np.where(gaston_labels==l-1)[0]]
            isodepth_left=0.5*(np.min(isodepth_l) + np.max(isodepth_lm1))
        else:
            isodepth_left=np.min(isodepth_l)-0.01
            
        if l<L-1:
            isodepth_lp1=gaston_isodepth[np.where(gaston_labels==l+1)[0]]
            isodepth_right=0.5*(np.max(isodepth_l) + np.min(isodepth_lp1))
        else:
            isodepth_right=np.max(isodepth_l)+0.01
        
        bins_l=np.linspace(isodepth_left, isodepth_right, num=num_bins_per_domain[l]+1)
        if l!=0:
            bins_l=bins_l[1:]
        bins=np.concatenate((bins, bins_l))


    unique_binned_isodepths=np.array( [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)] )
    binned_isodepth_inds=np.digitize(gaston_isodepth, bins)-1 #ie [1,0,3,15,...]
    binned_isodepths=unique_binned_isodepths[binned_isodepth_inds]
    
    # remove bins not used
    unique_binned_isodepths=np.delete(unique_binned_isodepths,
                                   [np.where(unique_binned_isodepths==t)[0][0] for t in unique_binned_isodepths if t not in binned_isodepths])

    N_1d=len(unique_binned_isodepths)
    binned_count=np.zeros( (N_1d,G) )
    binned_exposure=np.zeros( N_1d )
    to_subtract=np.zeros( N_1d )
    binned_labels=np.zeros(N_1d)
    binned_cell_type_mat=np.zeros((N_1d, len(cell_type_names)))
    binned_number_spots=np.zeros(N_1d)

    binned_count_per_ct={ct: np.zeros( (N_1d,G) ) for ct in cell_type_names}
    binned_exposure_per_ct={ct: np.zeros( N_1d ) for ct in cell_type_names}
    to_subtract_per_ct={ct:np.zeros( N_1d ) for ct in cell_type_names}
    binned_extra_data=[np.zeros(N_1d) for i in range(len(extra_data))]
    map_1d_bins_to_2d={} # map b -> [list of cells in bin b]
    for ind, b in enumerate(unique_binned_isodepths):
        bin_pts=np.where(binned_isodepths==b)[0]
        
        binned_count[ind,:]=np.sum(cmat[bin_pts,:],axis=0)
        binned_exposure[ind]=np.sum(exposure[bin_pts])
        if pc>0:
            to_subtract[ind]=np.log(10**6 * (len(bin_pts)/np.sum(exposure[bin_pts])))
        binned_labels[ind]= int(mode( gaston_labels[bin_pts],keepdims=False).mode)
        binned_cell_type_mat[ind,:] = np.sum( cell_type_mat[bin_pts,:], axis=0)
        binned_number_spots[ind]=len(bin_pts)
        map_1d_bins_to_2d[b]=bin_pts

        for i, eb in enumerate(extra_data):
            binned_extra_data[i][ind]=np.mean(extra_data[i][bin_pts])
        
        for ct_ind, ct in enumerate(cell_type_names):
            
            ct_spots=np.where(cell_type_mat[:,ct_ind] > 0)[0]
            ct_spots_bin = [t for t in ct_spots if t in bin_pts]
            ct_spots_bin_proportions=cell_type_mat[ct_spots_bin,ct_ind]
            
            if len(ct_spots_bin)>0:
                binned_count_per_ct[ct][ind,:]=np.sum(cmat[ct_spots_bin,:] * np.tile(ct_spots_bin_proportions,(G,1)).T, axis=0)
                binned_exposure_per_ct[ct][ind]=np.sum(exposure[ct_spots_bin] * ct_spots_bin_proportions)
                if pc>0:
                    to_subtract_per_ct[ct]=np.log(10**6 * len(ct_spots_bin) / np.sum(exposure[ct_spots_bin]))
            
    # subtract single constant if we add PC
    to_subtract=np.median(to_subtract)
    to_subtract_per_ct={ct:np.median(to_subtract_per_ct[ct]) for ct in cell_type_names}
            
    L=len(np.unique(gaston_labels))
    segs=[np.where(binned_labels==i)[0] for i in range(L)]

    to_return={}
    
    to_return['L']=len(np.unique(gaston_labels))
    to_return['umi_threshold']=umi_threshold
    to_return['gaston_labels']=gaston_labels
    to_return['counts_mat_idx']=cmat
    to_return['cell_type_mat']=cell_type_mat
    to_return['cell_type_names']=cell_type_names
    to_return['idx_kept']=idx_kept
    to_return['gene_labels_idx']=gene_labels_idx
    
    to_return['binned_isodepths']=binned_isodepths
    to_return['unique_binned_isodepths']=unique_binned_isodepths
    to_return['binned_count']=binned_count
    to_return['binned_exposure']=binned_exposure
    to_return['to_subtract']=to_subtract
    to_return['binned_labels']=binned_labels
    to_return['binned_cell_type_mat']=binned_cell_type_mat
    to_return['binned_number_spots']=binned_number_spots
    
    to_return['binned_count_per_ct']=binned_count_per_ct
    to_return['binned_exposure_per_ct']=binned_exposure_per_ct
    to_return['to_subtract_per_ct']=to_subtract_per_ct
    to_return['binned_extra_data']=binned_extra_data
    
    to_return['map_1d_bins_to_2d']=map_1d_bins_to_2d
    to_return['segs']=segs

    return to_return