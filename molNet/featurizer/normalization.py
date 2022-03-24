import inspect
from copy import deepcopy
from functools import partial
from typing import Callable

import numpy as np
import numba
from numba import jit
from numba.extending import register_jitable,overload

from molNet.featurizer import get_normalization_data


def normalization_method(name,arguments=[]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.norm_name = name
        wrapper.norm_arguments = arguments
        return wrapper
    return decorator

@register_jitable
def _jitclip_array(x,minval,maxval):
    xn = np.zeros_like(x,dtype=np.float64)
    for i in range(len(x)):
        xn[i] = jitclip(x[i],minval,maxval)
    return xn

@register_jitable
def _jitclip_value(x,minval,maxval):
    if x<minval:
        return minval
    elif x>maxval:
        return maxval
    return x

def jitclip(x,minval,maxval):
    raise NotImplementedError

@overload(jitclip)
def ov_jitclip(x,minval,maxval):
    if isinstance(x, numba.types.Number):
        return _jitclip_value#(x,minval,maxval)
    elif  isinstance(x, numba.types.Array):
        return _jitclip_array#(x,minval,maxval)
    raise NotImplementedError()

@normalization_method(name="tanh_norm",arguments=["m","d"])
@jit(nopython=True)
def tanh_norm(x,m:float=0,d:float=1):
    dx=d*(x-m)
    ex = np.exp(dx)
    enx= np.exp(-dx)
    r=(ex-enx)/(ex+enx)
    r[np.isnan(r)] = 0
    r[np.isposinf(r)] = 1e32
    r[np.isneginf(r)] = -1e32
    return r

@jit(nopython=True)
def _linear_norm(x, m: float = 1, c: float = 0):
    return x * m + c

@normalization_method(name="linear_norm",arguments=["m","c"])
def linear_norm(x, m: float = 1, c: float = 0):
    return _linear_norm(x, m, c)

@normalization_method(name="min_max_norm",arguments=["min","max"])
@jit(nopython=True)
def min_max_norm(x, min: float = 0, max: float = 1):
    if min>max:
        max,min=min,max
    if min == max:
        max+= 1e-32
        max*=(1+1e-6)
    ll:float = 0.0
    ul:float = 1.0
    return jitclip(_linear_norm(x, m=1 / (max - min), c=-min / (max - min)),ll,ul)

@jit(nopython=True)
def _sig_norm(x, m: float = 0, d: float = 1):
    div = (1 + np.exp(-d * (x - m)))
    div[div==0]=1e-32
    div[np.isposinf(div)]=1e32
    div[np.isneginf(div)]=-1e32
    return 1 /div

@normalization_method(name="sig_norm",arguments=["m","d"])
def sig_norm(x, m: float = 0, d: float = 1):
    return _sig_norm(x, m, d)

@normalization_method(name="dual_sig_norm",arguments=["m","d1","d2"])
@jit(nopython=True)
def dual_sig_norm(x, m: float = 0, d1: float = 1, d2: float = 1):
    if d2==0:
        d2=1e-32
    if d1==0:
        d1=1e-32
    dt=max(d1/d2,d2/d1)*10
    s1=_sig_norm(x, m, d1)
    s2=_sig_norm(x, m, d2)
    st=_sig_norm(x, m, dt)
    return s1*(1-st)+s2*st

@normalization_method(name="genlog_norm",arguments=["B","M","Q","v"])
@jit(nopython=True)
def genlog_norm(x, B: float = 1, M: float = 0, Q: float = 1, v: float = 0.1):
    # B=growth rate (-np.inf,np.inf)
    # M=shifts horizontally (-np.inf,np.inf)
    # Q=urvibess/stepness (0,np.inf)
    # v=stepness (1e-12,np.inf)
    div= (1 + Q * np.exp(-B * (x - M)))
    div[div==0]+=1e-32
    div=div** (1 / v)
    div[np.isnan(div)] = 1e-32
    div[np.isposinf(div)] = 1e32
    div[np.isneginf(div)] = -1e32
    return 1 / div

@normalization_method(name="weibull_norm",arguments=["l","k","m"])
@jit(nopython=True)
def weibull_norm(x, l: float = 1.0, k: float = 1.0,m: float=0):
    if l<=0:
        l=1e-32
    if k<=0:
        k=1
    x=x+m
    x[x<0]=0
    r= 1  - np.exp(-(l*(x))**k)
    r[np.isnan(r)] = 0
    r[np.isposinf(r)] = 1e32
    r[np.isneginf(r)] = -1e32
    return r

@normalization_method(name="unity_norm",arguments=[])
def unity_norm(x):
    return x

_t_array = np.linspace(0, 1, 10_000)






class NormalizationException(Exception):
    pass


NORMMAP = {
        "unity_norm": unity_norm,
        "tanh_norm": tanh_norm,
        "linear_norm": linear_norm,
        "min_max_norm": min_max_norm,
        "sig_norm": sig_norm,
        "dual_sig_norm": dual_sig_norm,
        "genlog_norm": genlog_norm,
    "weibull_norm":weibull_norm,
    }


NORM_NAMES = {}
for k, v in {
    ("unity","unity_norm"):"unity_norm",
    ("tanh","tanh_norm"):"tanh_norm",
    ("linear","linear_norm"):"linear_norm",
    ("min_max","min_max_norm"):"min_max_norm",
    ("sig","sig_norm"):"sig_norm",
    ("dual_sig","dual_sig_norm"):"dual_sig_norm",
    ("genlog","genlog_norm"):"genlog_norm",
    ("weibull","weibull_norm"):"weibull_norm",
}.items():
    for key in k:
        NORM_NAMES[key] = v


class NormalizationClass:
    preferred_normalization = "unity"

    def __init__(
        self,
        preferred_normalization=None,
        normalization_data=None,
    ):

        if normalization_data is None:
            normalization_data=[]


        feat_name=str(self).lower()
        nd=get_normalization_data()
        if feat_name in nd:
            normalization_data = [
                {**deepcopy(nd[feat_name][i]),**normalization_data[i]} if len(normalization_data)>i else deepcopy(nd[feat_name][i])
                for i in range(len(nd[feat_name]))
            ]

        self.normalization_data = normalization_data

        self._preferred_norm_name="unity"
        self._preferred_norm = unity_norm
        self._preferred_norm_params = []
        if preferred_normalization is None:
            preferred_normalization = self.preferred_normalization
        self.preferred_norm = preferred_normalization


    @property
    def preferred_norm(self):
        return self._preferred_norm

    @property
    def preferred_norm_name(self):
        return self._preferred_norm_name

    @preferred_norm.setter
    def preferred_norm(self, normalization):
        self.set_preferred_norm(normalization)

    def get_norm(self):
        return self._preferred_norm,self._preferred_norm_name,self._preferred_norm_params

    def reset_norm(self,norm_data):
        self._preferred_norm,self._preferred_norm_name,self._preferred_norm_params = norm_data

    def set_preferred_norm(self, normalization=None):
        if normalization is None:
            normalization="unity"

        #store prvious norm data in case something went wrong
        prenorm= self.get_norm()

        try:
            self._preferred_norm_name = normalization
            if isinstance(normalization,list):
                for n in normalization:
                    if n not in NORM_NAMES:
                        raise NormalizationException(
                            f"normalization '{n}' is not defined"
                        )

                if len(self.normalization_data)!= len(normalization):
                    raise NormalizationException(
                        f"try to set a list of normalizations but it seems the length ({len(normalization)}) did't match the normalization_data length ({len(self.normalization_data)})"
                    )

                normalization=[NORM_NAMES[n] for n in normalization]
                self._preferred_norm = [NORMMAP[n] for n in normalization]

                self._preferred_norm_params = []
                for i in range(len(self.normalization_data)):
                    try:
                        self._preferred_norm_params.append(self.normalization_data[i][normalization[i]]['params'])
                    except KeyError as e:
                        raise NormalizationException(f"try to set global normalization ({normalization[i]}), "
                                                     "but the parameters are not defined for every dimension in the "
                                                     f"normalization data (missing dim: {i} of {len(self.normalization_data)} dims). "
                                                     "Try to set via list and define missing dim as unity or use global "
                                                     "paramets (not defined as list in normalization_data)")

            else:
                if normalization not in NORM_NAMES:
                    raise NormalizationException(
                        f"normalization '{normalization}' is not defined"
                    )
                normalization=NORM_NAMES[normalization]
                self._preferred_norm = NORMMAP[normalization]

                self._preferred_norm_params = []
                for i in range(len(self.normalization_data)):
                    if normalization=="unity_norm":
                        self._preferred_norm_params.append([])
                    else:
                        try:
                            self._preferred_norm_params.append(self.normalization_data[i][normalization]['params'])
                        except KeyError as e:
                            raise NormalizationException(f"try to set global normalization ({normalization}), "
                                                         "but the parameters are not defined for every dimension in the "
                                                         f"normalization data (missing dim: {i} of {len(self.normalization_data)} dims). "
                                                         "Try to set via list and define missing dim as unity or use global "
                                                         "paramets (not defined as list in normalization_data)")

        except Exception as e:
            self.reset_norm(prenorm)
            raise e
            #if np.isnan(self.normalize(_t_array[:len(self._preferred_norm_params)])).any():
            #    self._preferred_norm,self._preferred_norm_name=prenorm
            #    raise NormalizationException(
            #        f"cannot set normalization to '{normalization}', there is nan in the result"
            #    )

    def get_best_norm(self):
        best_norms=[]
        for i,d in enumerate(self.normalization_data):
            best_norm="unity"
            best_norm_data_r=np.inf
            for norm, norm_data in d.items():
                if norm_data["R2"]<best_norm_data_r:
                    best_norm=norm
                    best_norm_data_r = norm_data["R2"]
            best_norms.append(best_norm)

        return best_norms

    def set_best_norm(self):
        bn=self.get_best_norm()
        if len(bn)==0:
            bn="unity"
        self.set_preferred_norm(bn)
        return bn

    def normalize(self, x):
        if isinstance(self._preferred_norm,list):
            if x.shape[0] != len(self._preferred_norm):
                raise NormalizationException(f"normalization is a list ({len(self._preferred_norm)}), but the data length didn't match ({x.shape[0]})")

        xf=x.reshape(x.shape[0],-1)
        xn = np.zeros_like(xf,dtype=float)

        print(self._preferred_norm,self._preferred_norm_params)

        if isinstance(self._preferred_norm,list):
            for i in range(len(x)):
                xn[i]= self._preferred_norm[i](xf[i],*self._preferred_norm_params[i])
        else:
            for i in range(xf.shape[0]):
                xn[i]= self._preferred_norm(xf[i],*self._preferred_norm_params[i])
        return xn.reshape(x.shape)






def generate_norm(norm_names,norm_paramss):
    norms=[]
    paramss=[]
    for i,norm_name in enumerate(norm_names):
        if norm_name not in NORM_NAMES:
            raise NormalizationException(
                f"normalization '{norm_name}' is not defined"
            )
        norm_name=NORM_NAMES[norm_name]
        params=norm_paramss[i]
        norm_func=NORMMAP[norm_name]

        params = dict(zip(norm_func.norm_arguments,params))
        paramss.append(params)
        norms.append(
            partial(NORMMAP[norm_name],**params)
        )
    @normalization_method(".".join(norm_names))
    def _n(x):
        _x=x.copy()
        if _x.ndim==1:
            _x=_x[:,np.newaxis]
        print(_x.shape,x.shape,len(norm_names),len(norm_paramss))
        _x = np.array([norms[i](_x[i]) for i in range(_x.shape[0])])
        return _x.reshape(x.shape)
    _n.params=paramss
    return _n


def detect_featurizer_best_norm(featurizer,allow_mixed=True):
    nd=get_normalization_data()
    feat_name=str(featurizer).lower()
    if feat_name not in nd:
        return unity_norm
    normalization_data = nd[feat_name]


    if allow_mixed:
        best_norms=[]
        best_norms_params=[]
        for i,d in enumerate(normalization_data):
            best_norm="unity"
            best_norm_data_r=np.inf
            best_norm_params=[]
            for norm, norm_data in d.items():
                if norm_data["R2"]<best_norm_data_r:
                    best_norm=norm
                    best_norm_data_r = norm_data["R2"]
                    best_norm_params=d[best_norm]["params"]
            best_norms.append(best_norm)
            best_norms_params.append(best_norm_params)
            print(len(best_norms))
        #print(best_norms,best_norms_params)
        return generate_norm(best_norms,best_norms_params)
    else:
        raise NotImplementedError("unmixed best norms not implemented yet")

class Normalization(object):
    def __init__(self, ):
        self._norm=unity_norm

    def set_norm(self, norm:Callable):
        self._norm=norm

    def set_best_featurizer_norm(self,featurizer,**kwargs) -> Callable:
        bn=detect_featurizer_best_norm(featurizer,**kwargs)
        self.set_norm(bn)
        return bn


    def __call__(self,featurizer:"Featurizer", x):
        return self._norm(x)