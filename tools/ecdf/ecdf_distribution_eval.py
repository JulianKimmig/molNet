import sys
import os

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,modp)
sys.path.append(modp)

from tools.ecdf import ecdf_conf

import pickle
import numpy as np
import matplotlib.pyplot as plt


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

class ECDF:
    def __init__(self,feat_dist_file,n=0,dist_data=None):
        self.dirname=os.path.dirname(os.path.abspath(feat_dist_file))
        self.dist_pickle=os.path.basename(feat_dist_file)
        self.basename=self.dist_pickle.replace("_feature_dist.pckl","")
        self._n=n
        self._dist_data=dist_data
    
    def __str__(self):
        return f"{self.basename}_{self._n}"
    
    @property
    def dist_data(self):
        if self._dist_data is not None:
            return self._dist_data
        
        print(self,"load dist_data")
        with open(os.path.join(self.dirname,self.dist_pickle),"rb") as f:
            feat_dist=pickle.load(f)
            
        if len(feat_dist.shape)>2:
            raise NotImplementedError("cannot work with larger than 1d features, do they make sense? can you split them?")
        
        feat_dist=feat_dist[~np.isnan(feat_dist).any(1)]
    
        self._dist_data = feat_dist[:,self._n]
        
        return self._dist_data
    
    @property
    def full_ecdf(self):
        path=os.path.join(self.dirname,str(self)+"_full_ecdf.pckl")
        x,y=None,None
        if os.path.exists(path):
            try:
                with open(path,"rb") as f:
                    x,y=pickle.load(f)
                if self.dist_data.shape[0] != x.shape[0] or self.dist_data.shape[0] != y.shape[0]:
                    x,y=None,None
            except Exception:
                x,y=None,None
                
        if x is None or y is None:
            print(self,"generate full_ecdf")
            x,y = generate_ecdf(self.dist_data, res=None, smooth=False, unique_only=False)
            with open(path,"w+b") as f:
                pickle.dump((x,y),f)
        return x,y
                
    @property
    def smooth_ecdf(self):
        path=os.path.join(self.dirname,str(self)+"_smooth_ecdf.pckl")
        x,y=None,None
        if os.path.exists(path):
            try:
                with open(path,"rb") as f:
                    x,y=pickle.load(f)
                if self.dist_data.max() != x.max() or self.dist_data.min() != x.min() or y.max()!=1 or y.min()!=0:
                    x,y=None,None
            except Exception:
                x,y=None,None
                
        if x is None or y is None:
            print(self,"generate smooth_ecdf")
            x,y = generate_ecdf(self.dist_data, res=None, smooth=True, unique_only=False)
            with open(path,"w+b") as f:
                pickle.dump((x,y),f)
        return x,y     
    
    def get_ecdf_img_path(self):
        path=os.path.join(self.dirname,str(self)+"_ecdf.png")
        if not os.path.exists(path):
            plt.plot(*self.full_ecdf,label="ECDF")
            plt.plot(*self.smooth_ecdf,label="smoothed ECDF")
            plt.legend()
            plt.savefig(path)
            plt.close()
        return path
        
        
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


def work_ecdf(ecdf):
    print(ecdf)
    #ecdf.full_ecdf
    #ecdf.smooth_ecdf
    return

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
    print(feat_dist_file)
    with open(feat_dist_file,"rb") as f:
        feat_dist=pickle.load(f)
    
    if len(feat_dist.shape)>2:
        raise NotImplementedError("cannot work with larger than 1d features, do they make sense? can you split them?")
        
    feat_dist=feat_dist[~np.isnan(feat_dist).any(1)]
    
    
    
    for k in range(feat_dist.shape[1]):
        ecdf_obj = ECDF(
            feat_dist_file=feat_dist_file,
            n=k,
            dist_data=feat_dist[:,k]
        )
        work_ecdf(ecdf_obj)
        
for feat_group in sorted(os.listdir(samples_dir)):
    if not feat_group.startswith("_autogen_"):
        continue
    fg_dir=os.path.join(samples_dir,feat_group)
    if not os.path.isdir(fg_dir):
        continue
        
    
    feature_dists=[os.path.join(fg_dir,f) for f in os.listdir(fg_dir) if f.endswith("_feature_dist.pckl")]
    
    
    for feature_dist in feature_dists:
        work_feat(feature_dist)
        
        
      #  break
    #break