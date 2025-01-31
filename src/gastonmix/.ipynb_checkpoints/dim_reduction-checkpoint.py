import numpy as np
import scanpy as sc
import pandas as pd 

def get_top_pearson_residuals(num_pcs, counts_mat, coords_mat, gene_labels=None, n_top_genes=5000, clip=0.01):
    df=pd.DataFrame(counts_mat, columns=gene_labels)
    adata=sc.AnnData(df)
    adata.obsm["coords"] = coords_mat
    
    sc.experimental.pp.highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=n_top_genes
    )
    
    adata = adata[:, adata.var["highly_variable"]]
    adata.layers["raw"] = adata.X.copy()
    adata.layers["sqrt_norm"] = np.sqrt(
        sc.pp.normalize_total(adata, inplace=False)["X"]
    )
    
    theta=np.inf
    sc.experimental.pp.normalize_pearson_residuals(adata, clip=clip, theta=theta)
    sc.pp.pca(adata, n_comps=num_pcs)
    return adata.obsm['X_pca']