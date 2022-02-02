import numpy as np
import numba
from numba import jit
from numba.extending import register_jitable,overload

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

@jit(nopython=True)
def tanh_norm(x,m:float=0,d:float=1):
    dx=d*(x-m)
    ex = np.exp(dx)
    enx= np.exp(-dx)
    return (ex-enx)/(ex+enx)

@jit(nopython=True)
def linear_norm(x, m: float = 1, c: float = 0):
    return x * m + c

@jit(nopython=True)
def min_max_norm(x, min: float = 0, max: float = 1):
    # if min>max:
    #    max,min=min,max
    # if min == max:
    #    max*= 1+1e-6
    ll:float = 0.0
    ul:float = 1.0
    return jitclip(linear_norm(x, m=1 / (max - min), c=-min / (max - min)),ll,ul)

@jit(nopython=True)
def sig_norm(x, m: float = 0, d: float = 1):
    return 1 / (1 + np.exp(-d * (x - m)))

@jit(nopython=True)
def dual_sig_norm(x, m: float = 0, d1: float = 1, d2: float = 1):
    dt=max(d1/d2,d2/d1)*10
    s1=sig_norm(x, m, d1)
    s2=sig_norm(x, m, d2)
    st=sig_norm(x, m, dt)
    return s1*(1-st)+s2*st

@jit(nopython=True)
def genlog_norm(x, B: float = 1, M: float = 0, Q: float = 1, v: float = 0.1):
    # B=growth rate (-np.inf,np.inf)
    # M=shifts horizontally (-np.inf,np.inf)
    # Q=urvibess/stepness (0,np.inf)
    # v=stepness (1e-12,np.inf)
    return 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / v)


_t_array = np.linspace(0, 1, 5)



class NormalizationException(Exception):
    pass


class NormalizationClass:
    linear_norm_parameter = (1, 0)
    min_max_norm_parameter = (np.nan, np.nan)
    sigmoidal_norm_parameter = (np.nan, np.nan)
    dual_sigmoidal_norm_parameter = (np.nan, np.nan, np.nan)
    genlog_norm_parameter = (np.nan, np.nan, np.nan)
    preferred_normalization = "unity"

    def __init__(
        self,
        preferred_normalization=None,
        linear_norm_parameter=None,
        min_max_norm_parameter=None,
        sigmoidal_norm_parameter=None,
        dual_sigmoidal_norm_parameter=None,
        genlog_norm_parameter=None,
    ):
        if linear_norm_parameter is None:
            linear_norm_parameter = self.linear_norm_parameter
        self._linear_norm_parameter = linear_norm_parameter

        if min_max_norm_parameter is None:
            min_max_norm_parameter = self.min_max_norm_parameter
        self._min_max_norm_parameter = min_max_norm_parameter

        if sigmoidal_norm_parameter is None:
            sigmoidal_norm_parameter = self.sigmoidal_norm_parameter
        self._sigmoidal_norm_parameter = sigmoidal_norm_parameter

        if dual_sigmoidal_norm_parameter is None:
            dual_sigmoidal_norm_parameter = self.dual_sigmoidal_norm_parameter
        self._dual_sigmoidal_norm_parameter = dual_sigmoidal_norm_parameter

        if genlog_norm_parameter is None:
            genlog_norm_parameter = self.genlog_norm_parameter
        self._genlog_norm_parameter = genlog_norm_parameter

        self._norm_map = {
            None: self.unity_norm,
            "None": self.unity_norm,
            "unity": self.unity_norm,
            "linear": self.linear_norm,
            "min_max": self.min_max_norm,
            "sig": self.sig_norm,
            "dual_sig": self.dual_sig_norm,
            "genlog": self.genlog_norm,
        }
        self._preferred_norm_name="unity"
        self._preferred_norm = self.unity_norm
        if preferred_normalization is None:
            preferred_normalization = self.preferred_normalization
        self.preferred_norm = preferred_normalization

    def unity_norm(self, x):
        return x

    def linear_norm(self, x):
        return linear_norm(
            x, m=self._linear_norm_parameter[0], c=self._linear_norm_parameter[1]
        )

    def min_max_norm(self, x):
        return min_max_norm(
            x, min=self._min_max_norm_parameter[0], max=self._min_max_norm_parameter[1]
        )

    def sig_norm(self, x):
        return sig_norm(
            x, m=self._sigmoidal_norm_parameter[0], d=self._sigmoidal_norm_parameter[1]
        )

    def dual_sig_norm(self, x):
        return dual_sig_norm(
            x,
            m=self._dual_sigmoidal_norm_parameter[0],
            d1=self._dual_sigmoidal_norm_parameter[1],
            d2=self._dual_sigmoidal_norm_parameter[2],
        )

    def genlog_norm(self, x):
        return genlog_norm(
            x,
            B=self._genlog_norm_parameter[0],
            M=self._genlog_norm_parameter[1],
            Q=self._genlog_norm_parameter[2],
            v=self._genlog_norm_parameter[3],
        )

    @property
    def preferred_norm(self):
        return self._preferred_norm

    @property
    def preferred_norm_name(self):
        return self._preferred_norm_name

    @preferred_norm.setter
    def preferred_norm(self, normalization):
        self.set_preferred_norm(normalization)

    def set_preferred_norm(self, normalization=None):
        if normalization is None:
            normalization="unity"
        if normalization not in self._norm_map:
            raise NormalizationException(
                f"normalization '{normalization}' is not defined"
            )
        prenorm= self._preferred_norm,self._preferred_norm_name
        self._preferred_norm = self._norm_map[normalization]
        self._preferred_norm_name = normalization
        if np.isnan(self.normalize(_t_array)).any():
            self._preferred_norm,self._preferred_norm_name=prenorm
            raise NormalizationException(
                f"cannot set normalization to '{normalization}', there is nan in the result"
            )
            
    def normalize(self, x):
        return self._preferred_norm(x)
