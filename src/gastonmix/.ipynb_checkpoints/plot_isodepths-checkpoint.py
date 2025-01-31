from scipy.stats import zscore
import numpy as np
from math import ceil
import torch

from run_moe_script import *
from isodepth_scaling import adjust_isodepth

# outlier threshold doesnt show points that are far away
# set to np.inf if you want to show all points
def plot_all_isodepths(output_folder,seed,model='final_model.pt',outlier_threshold=np.inf,levels=3):
    moe_model=torch.load(output_folder + f'seed{seed}/' + model)
    Atorch=torch.load(output_folder + f'seed{seed}/' + 'Atorch.pt')
    Storch=torch.load(output_folder + f'seed{seed}/' + 'Storch.pt')
    coords_mat=Storch.cpu().detach().numpy()
    
    moe_labels=torch.argmax(moe_model(Storch)[2],1).detach().cpu().numpy()
    isodepth_list=[moe_model.isodepths_list[i](Storch).flatten().cpu().detach().numpy()[moe_labels==i] for i in range(len(np.unique(moe_labels)))]
    N=Storch.shape[0]
    num_isodepths=len(np.unique(moe_labels))
    

    R = int(ceil(num_isodepths / 2))
    C = 2
    fig, axs = plt.subplots(R, C, figsize=(5 * C, 5 * R))
    
    for r in range(R):
        for c in range(C):
            ind = r * C + c
            if ind < len(isodepth_list):
                # Mask for current label
                current_label_mask = (moe_labels == ind)
    
                # Extract current coordinates and isodepth values
                current_coords = coords_mat[current_label_mask]
                current_isodepth = isodepth_list[ind]
    
                # Compute Z-scores for current coordinates
                z_scores = np.abs(zscore(current_coords, axis=0))
                
                # Create a mask for non-outliers based on Z-scores
                non_outliers_mask = (z_scores[:, 0] < outlier_threshold) & (z_scores[:, 1] < outlier_threshold)
                
                # Filter out the outliers
                filtered_coords = current_coords[non_outliers_mask]
                filtered_isodepth = current_isodepth[non_outliers_mask]
    
                # Scatter plot for non-outlier points
                sc = axs[r, c].scatter(filtered_coords[:, 0], filtered_coords[:, 1],
                                       c=filtered_isodepth, s=1, cmap='Reds')
    
                # Add contours for non-outlier points
                CS = axs[r, c].tricontour(filtered_coords[:, 0], filtered_coords[:, 1],
                                          filtered_isodepth, levels=levels, colors='k', linestyles='solid')
                axs[r, c].clabel(CS, CS.levels, inline=True, fontsize=10)
    
                # Set plot limits
                axs[r, c].set_xlim([np.min(coords_mat[:, 0]), np.max(coords_mat[:, 0])])
                axs[r, c].set_ylim([np.min(coords_mat[:, 1]), np.max(coords_mat[:, 1])])
                axs[r, c].set_title(f'Isodepth {ind}')
    
    plt.tight_layout()
    plt.show()
    return moe_labels,isodepth_list

def plot_individual_isodepth(ind,output_folder,seed,model='final_model.pt',outlier_threshold=np.inf,levels=7,
                            linewidth=2.5,density=0.6,arrowsize=2.5,figsize=(4.5,5)):
    moe_model=torch.load(output_folder + f'seed{seed}/' + model)
    Atorch=torch.load(output_folder + f'seed{seed}/' + 'Atorch.pt')
    Storch=torch.load(output_folder + f'seed{seed}/' + 'Storch.pt')
    coords_mat=Storch.cpu().detach().numpy()

    moe_labels=torch.argmax(moe_model(Storch)[2],1).detach().cpu().numpy()
    isodepth_list=[moe_model.isodepths_list[i](Storch).flatten().cpu().detach().numpy()[moe_labels==i] for i in range(len(np.unique(moe_labels)))]
    N=Storch.shape[0]
    num_isodepths=len(np.unique(moe_labels))

    # scale isodepth using physical distance
    coords_ind = coords_mat[moe_labels == ind]
    isodepth_ind = isodepth_list[ind]
    
    isodepth_ind_adjusted=adjust_isodepth(isodepth_ind, np.zeros(len(isodepth_ind)), coords_ind, q_vals=[0.05], visualize=False, figsize=(5,5))

    ######################################
    # PLOTTING
    ######################################
    
    fig,axs=plt.subplots(figsize=figsize)
    
    St=Storch.detach().cpu().numpy()
    St_ind=St[moe_labels==ind]
    
    # plot
    # Extract current coordinates and isodepth values
    # coords_ind = coords_mat[moe_labels == ind]
    isodepth_ind=isodepth_ind_adjusted
    
    # Compute Z-scores for current coordinates
    z_scores = np.abs(zscore(St_ind, axis=0))
    
    # Create a mask for non-outliers based on Z-scores
    non_outliers_mask = (z_scores[:, 0] < outlier_threshold) & (z_scores[:, 1] < outlier_threshold)
    
    # Filter out the outliers
    filtered_coords = St_ind[non_outliers_mask,:]
    filtered_isodepth = isodepth_ind[non_outliers_mask]
    
    # Scatter plot for non-outlier points
    plt.scatter(filtered_coords[:, 0], filtered_coords[:, 1],
                           c=filtered_isodepth, s=5, cmap='Reds')
    
    # Add contours for non-outlier points
    CS=plt.tricontour(filtered_coords[:, 0], filtered_coords[:, 1],
                              filtered_isodepth, levels=levels, colors='dimgray', linestyles='solid',
                     linewidths=3)
    plt.clabel(CS, CS.levels, inline=True, fontsize=15)
    
    ######
    # streamlines
    
    St_ind=St_ind[non_outliers_mask,:]
    x=torch.tensor(St_ind,requires_grad=True).float()
    G=torch.autograd.grad(outputs=moe_model.isodepths_list[ind].cpu()(x).flatten(),inputs=x, grad_outputs=torch.ones_like(x[:,0]))[0]
    G=G.detach().cpu().numpy()
    
    # CODE FROM scVelo
    smooth=None
    min_mass=None
    if St_ind.shape[0]>1000:
        n_neighbors=1000
    else:
        n_neighbors=10
    cutoff_perc=0
    
    X_grid, V_grid = compute_velocity_on_grid(
                X_emb=St_ind,
                V_emb=G,
                density=1,
                smooth=smooth,
                min_mass=min_mass,
                n_neighbors=n_neighbors,
                adjust_for_stream=True,
                cutoff_perc=cutoff_perc,
            )
    lengths = np.sqrt((V_grid**2).sum(0))
    linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
    
    stream_kwargs = {
            "linewidth": linewidth,
            "density": density,
            "zorder": 3,
            "color": "k",
            "arrowsize": arrowsize,
            "arrowstyle": "-|>",
            "maxlength": 1000,
            "integration_direction": "both",
        }
    
    plt.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
    
    # Set plot limits
    plt.xlim([np.min(St_ind[:, 0]), np.max(St_ind[:, 0])])
    plt.ylim([np.min(St_ind[:, 1]), np.max(St_ind[:, 1])])
    # plt.title(f'Isodepth {ind}')
    
    plt.tight_layout()
    plt.axis('off')
    return isodepth_ind

#######################################################
# streamlines code taken from scVelo

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

def compute_velocity_on_grid(
    X_emb,
    V_emb,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    """TODO."""
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid**2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

    return X_grid, V_grid