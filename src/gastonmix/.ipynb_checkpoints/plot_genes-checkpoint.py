from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_gene_pwlinear(gene_name, pw_fit_dict, expert_ind, gastonmix_labels, isodepth, binning_output,
                       cell_type_list=None, ct_colors=None, spot_threshold=0.25, pt_size=10, 
                       colors=None, linear_fit=True, lw=2, domain_list=None, ticksize=20, figsize=(7,3),
                      offset=10**6, xticks=None, yticks=None, alpha=1, domain_boundary_plotting=False, 
                      save=False, save_dir="./", variable_spot_size=False, show_lgd=False,
                      lgd_bbox=(1.05,1), extract_values = False):
    
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    
    binned_count_list=[]
    binned_exposure_list=[]
    to_subtract_list=[]
    ct_ind_list=[]
    
    if cell_type_list is None:
        binned_count_list.append(binning_output['binned_count'])
        binned_exposure_list.append(binning_output['binned_exposure'])
        to_subtract_list.append(binning_output['to_subtract'])
        
    else:
        for ct in cell_type_list:
            binned_count_list.append(binning_output['binned_count_per_ct'][ct])
            binned_exposure_list.append(binning_output['binned_exposure_per_ct'][ct])
            to_subtract_list.append(binning_output['to_subtract_per_ct'][ct])
            ct_ind_list.append( np.where(binning_output['cell_type_names']==ct)[0][0] )
    
    segs=binning_output['segs']
    L=len(segs)

    fig,ax=plt.subplots(figsize=figsize)

    if domain_list is None:
        domain_list=range(L)

    values_list = []
    for seg in domain_list:
        for i in range(len(binned_count_list)):
            pts_seg=np.where(binned_labels==seg)[0]
            binned_count=binned_count_list[i]
            binned_exposure=binned_exposure_list[i]
            to_subtract=np.log( offset*1 / np.mean(binned_exposure) )
            ct=None
            if cell_type_list is not None:
                ct=cell_type_list[i]
                # if restricting cell types, then restrict spots also
                binned_cell_type_mat=binning_output['binned_cell_type_mat']
                ct_ind=ct_ind_list[i]
                pts_seg=[p for p in pts_seg if binned_cell_type_mat[p,ct_ind] / binned_cell_type_mat[p,:].sum() > spot_threshold]
                
                # set colors for cell types
                if ct_colors is None:
                    c=None
                else:
                    c=ct_colors[ct]
            else:
                # set colors for domains
                if colors is None:
                    c=None
                else:
                    c=colors[seg]
                
            xax=unique_binned_isodepths[pts_seg]
            # print(binned_count.shape)
            yax=np.log((binned_count[pts_seg,gene] / binned_exposure[pts_seg]) * offset + 1)

            if extract_values:
                values_list.append(np.column_stack((xax, yax)))
            
            s=pt_size
            if variable_spot_size:
                s=s*binning_output['binned_number_spots'][pts_seg]
            plt.scatter(xax, yax, color=c, s=s, alpha=alpha,label=ct)

            if linear_fit:
                if ct is None:
                    slope_mat, intercept_mat, _, _ = pw_fit_dict['all_cell_types']
                else:
                    slope_mat, intercept_mat, _, _ = pw_fit_dict[ct]

                slope=slope_mat[gene,seg]
                intercept=intercept_mat[gene,seg]
                plt.plot(unique_binned_isodepths[pts_seg], np.log(offset) + intercept + slope*unique_binned_isodepths[pts_seg], color='grey', alpha=1, lw=lw )

    if xticks is None:
        plt.xticks(fontsize=ticksize)
    else:
        plt.xticks(xticks,fontsize=ticksize)
        
    if yticks is None:
        plt.yticks(fontsize=ticksize)
    else:
        plt.yticks(yticks,fontsize=ticksize)
        
    if domain_boundary_plotting and len(domain_list)>1:
        binned_labels=binning_output['binned_labels']
        
        left_bps=[]
        right_bps=[]

        for i in range(len(binned_labels)-1):
            if binned_labels[i] != binned_labels[i+1]:
                left_bps.append(unique_binned_isodepths[i])
                right_bps.append(unique_binned_isodepths[i+1])
        
        for i in domain_list[:-1]:
            plt.axvline((left_bps[i]+right_bps[i])*0.5, color='black', ls='--', linewidth=1.5, alpha=0.2)

    sns.despine()
    if show_lgd:
        plt.legend(bbox_to_anchor=lgd_bbox)
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{gene_name}_pwlinear.pdf", bbox_inches="tight")
        plt.close()

    if extract_values:
        all_values = np.vstack(values_list)
        values_filename = f"{save_dir}/{gene_name}_raw_all.txt"
        save_values({gene_name: all_values}, values_filename)

def save_values(values_dict, filename):
    with open(filename, 'w') as file:
        for key, values in values_dict.items():
            file.write(f"{key}\n")
            np.savetxt(file, values, delimiter='\t', fmt='%.6f')

def get_gene_plot_values(gene_name, binning_output, offset=10**6):
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    unique_binned_isodepths=binning_output['unique_binned_isodepths']
    binned_labels=binning_output['binned_labels']
    
    binned_count_list=binning_output['binned_count']
    binned_exposure_list=binning_output['binned_exposure']

    domain_list=range(len(binning_output['segs']))

    values = []
        
    for seg in domain_list:
        for i in range(len(binned_count_list)):
            pts_seg=np.where(binned_labels==seg)[0]
            binned_count=binned_count_list[i]
            binned_exposure=binned_exposure_list[i]
                
            xax=unique_binned_isodepths[pts_seg]
            yax=np.log((binned_count[gene,pts_seg] / binned_exposure[pts_seg]) * offset + 1)

            values.append(np.column_stack((xax, yax)))
    
    return np.vstack(values)

# NxG counts matrix
# plot raw expression values of gene
def plot_gene_raw(gene_name, gene_labels, counts_mat, coords_mat, 
                       offset=10**6, figsize=(6,6), colorbar=True, vmax=None, vmin=None, s=16, rotate=None,cmap='Blues'):

    if rotate is not None:
        coords_mat=rotate_by_theta(coords_mat,rotate)
    gene_idx=np.where(gene_labels==gene_name)[0]

    exposure = np.sum(counts_mat, axis=1, keepdims=False)
    raw_expression = np.squeeze(counts_mat[:, gene_idx])

    expression = np.log((raw_expression / exposure) * offset + 1)

    fig,ax=plt.subplots(figsize=figsize)

    im1 = ax.scatter(coords_mat[:, 0], 
        coords_mat[:, 1],
        c = expression,
        cmap = cmap, s=s, vmax=vmax, vmin=vmin)

    if colorbar:
        cbar=plt.colorbar(im1)
        cbar.ax.tick_params(labelsize=10)

    plt.axis('off')

# plot piecewise linear gene function learned by GASTON
def plot_gene_function(gene_name, coords_mat, pw_fit_dict, gaston_labels, gaston_isodepth, 
                       binning_output, offset=10**6, figsize=(6,6), colorbar=True, 
                       contours=False, contour_levels=4, contour_lw=1, contour_fs=10, s=16,
                      rotate=None,cmap='Blues'):

    if rotate is not None:
        coords_mat=rotate_by_theta(coords_mat,rotate)
    
    gene_labels_idx=binning_output['gene_labels_idx']
    if gene_name in gene_labels_idx:
        gene=np.where(gene_labels_idx==gene_name)[0]
    else:
        umi_threshold=binning_output['umi_threshold']
        raise ValueError(f'gene does not have UMI count above threshold {umi_threshold}')
    
    slope_mat, intercept_mat, _, _ = pw_fit_dict['all_cell_types']
    if gene_name in binning_output['gene_labels_idx']:
        gene=np.where(gene_labels_idx==gene_name)[0]

    outputs = np.zeros(gaston_isodepth.shape[0])
    for i in range(gaston_isodepth.shape[0]):
        dom = int(gaston_labels[i])
        slope=slope_mat[gene,dom]
        intercept=intercept_mat[gene,dom]
        outputs[i] = np.log(offset) + intercept + slope * gaston_isodepth[i]

    fig,ax=plt.subplots(figsize=figsize)

    im1 = ax.scatter(coords_mat[:, 0], 
        coords_mat[:, 1],
        c = outputs,
        cmap = cmap, s=s)


    if contours:
        CS=ax.tricontour(coords_mat[:,0], coords_mat[:,1], outputs, levels=contour_levels, linewidths=contour_lw, colors='k', linestyles='solid')
        ax.clabel(CS, CS.levels, inline=True, fontsize=contour_fs)
    if colorbar:
        cbar=plt.colorbar(im1)
        cbar.ax.tick_params(labelsize=10)

    plt.axis('off')