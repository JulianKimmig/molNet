import numpy as np


def linear_norm(x, m: float = 1, c: float = 0):
    return x * m + c


def min_max_norm(x, min: float = 0, max: float = 1):
    # if min>max:
    #    max,min=min,max
    # if min == max:
    #    max*= 1+1e-6
    return np.clip(linear_norm(x, m=1 / (max - min), c=-min / (max - min)), 0, 1)


def sig_norm(x, m: float = 0, d: float = 1):
    return 1 / (1 + np.exp(-d * (x - m)))

def scaled_sig_norm(x,m: float = 0, d: float = 1,ml: float = 1,cl: float = 0):
    return linear_norm(sig_norm(x,m, d),ml,cl)

def dual_sig_norm(x, m: float = 0, d1: float = 1, d2: float = 1):
    li = x <= m
    # mx = np.argmin(np.abs(x - m))
    return np.concatenate((sig_norm(x[li], m=m, d=d1), sig_norm(x[~li], m=m, d=d2)))

def scaled_dual_sig_norm(x,m: float = 0, d1: float = 1, d2: float = 1,ml: float = 1,cl: float = 0):
    return linear_norm(dual_sig_norm(x,m, d1,d2),ml,cl)

def genlog_norm(x, B, M, Q, v):
    # B=growth rate (-np.inf,np.inf)
    # M=shifts horizontally (-np.inf,np.inf)
    # Q=urvibess/stepness (0,np.inf)
    # v=stepness (1e-12,np.inf)
    return np.nan_to_num(1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / v), nan=np.nan)

def scaled_genlog_norm(x, B, M, Q, v,m,c):
    return linear_norm(genlog_norm(x, B, M, Q, v),m,c)

_t_array = np.arange(-4, 4)


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
        self._preferred_norm = self.preferred_normalization
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

    @preferred_norm.setter
    def preferred_norm(self, normalization):
        self._preferred_norm = self._norm_map[normalization]
        if np.isnan(self.normalize(_t_array)).any():
            raise NormalizationException(
                f"cannot set normalization to '{normalization}', there is nan in the result"
            )

    def normalize(self, x):
        return self._preferred_norm(x)
