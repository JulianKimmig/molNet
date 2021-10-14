import sys
import os

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,modp)
sys.path.append(modp)

from tools.ecdf import ecdf_conf

import pickle
import numpy as np
import matplotlib.pyplot as plt

available_ds=[]
for d in os.listdir(ecdf_conf.DATADIR):
    try:
        intd=int(d)
    except ValueError:
        continue
    if not os.path.isdir(os.path.join(ecdf_conf.DATADIR,d)):
        continue
        
    available_ds.append(intd)
    

available_ds=sorted(available_ds,reverse=True) 

samples_dir=os.path.join(ecdf_conf.DATADIR,str(available_ds[0]))


def generate_ecdf(data, res=None, smooth=False, unique_only=False):
    if data.ndim > 1:
        data = np.squeeze(data)
        if data.ndim > 1:
            return [
                ecdf(data[..., i], res=res, smooth=smooth, unique_only=unique_only)
                for i in range(data.shape[-1])
            ]
    x = np.sort(data)
    n = len(data)
    y = np.arange(1, n + 1) / n
    if smooth:
        x, uindices = np.unique(x, return_index=True)
        y = np.array([a.mean() for a in np.split(y, uindices[1:])])
        y[0] = 0
        y[-1] = 1
    if res:
        dp = (np.linspace(0, 1, res) * (len(x) - 1)).astype(int)
        n = res
        x = x[dp]
        y = y[dp]

    if unique_only:
        x, uindices = np.unique(x, return_index=True)
        y = y[uindices]
    return x, y



def work_sub_feat(dist,n,feat_dist_file):
    print(feat_dist_file)
    need_run=False
    
    ecdf_plot_file=feat_dist_file.replace("_feature_dist.pckl",f"_ecdf_{n}.png")
    uecdf_plot_file=feat_dist_file.replace("_feature_dist.pckl",f"_uecdf_{n}.png")
    if not os.path.exists(ecdf_plot_file) or not os.path.exists(uecdf_plot_file) or ecdf_conf.REDRAW:
        need_run=True
    
    if not need_run:
        return
    
    det_ecdf_x,det_ecdf_y=generate_ecdf(dist,smooth=False)
    smooth_ecdf_x,smooth_ecdf_y=generate_ecdf(dist,smooth=True)
    
    udet_ecdf_x,udet_ecdf_y=generate_ecdf(dist,smooth=False,unique_only=True)
    usmooth_ecdf_x,usmooth_ecdf_y=generate_ecdf(dist,smooth=True,unique_only=True)
    
    if not os.path.exists(uecdf_plot_file) or ecdf_conf.REDRAW:
        plt.plot(udet_ecdf_x,udet_ecdf_y,label="ECDF")
        plt.plot(usmooth_ecdf_x,usmooth_ecdf_y,label="smoothed ECDF")
        plt.legend()
        plt.savefig(uecdf_plot_file)
        plt.close()
        
    
    if not os.path.exists(ecdf_plot_file) or ecdf_conf.REDRAW:
        plt.plot(det_ecdf_x,det_ecdf_y,label="ECDF")
        plt.plot(smooth_ecdf_x,smooth_ecdf_y,label="smoothed ECDF")
        plt.legend()
        plt.savefig(ecdf_plot_file)
        plt.close()

    
    
def work_feat(feat_dist_file):
    with open(feat_dist_file,"rb") as f:
        feat_dist=pickle.load(f)
    
    if len(feat_dist.shape)>2:
        raise NotImplementedError("cannot work with larger than 1d features, do they make sense? can you split them?")
        
    feat_dist=feat_dist[~np.isnan(feat_dist).any(1)]
    
    for k in range(feat_dist.shape[1]):
        work_sub_feat(feat_dist[:,k],k,feat_dist_file)
    
    

for feat_group in os.listdir(samples_dir):
    if not feat_group.startswith("_autogen_"):
        continue
    fg_dir=os.path.join(samples_dir,feat_group)
    if not os.path.isdir(fg_dir):
        continue
        
    
    feature_dists=[os.path.join(fg_dir,f) for f in os.listdir(fg_dir) if f.endswith("_feature_dist.pckl")]
    
    
    for feature_dist in feature_dists:
        work_feat(feature_dist)
        
        
    