from warnings import warn
AS_NUMPY_ARRY = False
import numpy as np   

from .normalization import solve_normalization

class Featurizer:
    def __init__(
        self, length=None, pre_featurize=None, name=None, feature_descriptions=None,
        normalization="default",
    ):
        self.feature_descriptions = feature_descriptions
       
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.pre_featurize = pre_featurize
        self._len = length
        self.set_normalization(normalization)
       
    
    def set_normalization(self,normalization="default",**params):
         self._normalization=solve_normalization(self,normalization,**params)
            
    def __len__(self):
        if self._len is None:
            warn(
                "length for featurizer '{}' not defined please run at least oes for autodetermination".format(
                    self
                )
            )
        return self._len

    def __call__(self, to_featurize):
        if self.pre_featurize is not None:
            to_featurize = self.pre_featurize(to_featurize)
        f=self.featurize(to_featurize)
        f=np.array(f)
        if self._normalization is not None:
            f = self._normalization(f)
        if self._len is None:
            self._len = len(f)
        #if AS_NUMPY_ARRY:
        #    return 
        return f
    
    def featurize(self, to_featurize):
        return [to_featurize]

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

    def describe_features(self):
        if self.feature_descriptions is None:
            return [self._name] * len(self)

        return [self.feature_descriptions[i] for i in range(len(self))]

    def __add__(self, other):
        if isinstance(other, FeaturizerList):
            return other + self
        return FeaturizerList([self, other])


class OneHotEncodingException(Exception):
    pass


class OneHotFeaturizer(Featurizer):
    def __init__(self, possible_values, *args, **kwargs):
        super().__init__(*args, **kwargs,normalization=None)
        self._len = len(possible_values)
        self.possible_values = possible_values

    def __len__(self):
        return len(self.possible_values)

    def featurize(self, to_featurize):
        if None in self.possible_values and to_featurize not in self.possible_values:
            to_featurize = None
        if to_featurize not in self.possible_values:
            raise OneHotEncodingException(
                "cannot one hot encode '{}' in '{}', allowed values are {}".format(
                    to_featurize, self, self.possible_values
                )
            )
        # return list(map(lambda v: to_featurize == v, self.possible_values))
        return [v == to_featurize for v in self.possible_values]

    def describe_features(self):
        if self.feature_descriptions is None:
            n = str(self)
            return ["{}: {}".format(n, v) for v in self.possible_values]

        return super().describe_features()


class LambdaFeaturizer(Featurizer):
    def __init__(self, lamda_call, length, *args, **kwargs):
        super().__init__(length=length, *args, **kwargs)
        self.featurize = lamda_call


class FeaturizerList(Featurizer):
    def __init__(self, feature_list, name=None, *args, **kwargs):
        if name == None:
            name = "FeatureList({})".format(",".join([str(f) for f in feature_list]))
        self._feature_list = feature_list
        super().__init__(length=None, name=name, *args, **kwargs)
        

    def __len__(self):
        return sum([len(f) for f in self._feature_list])

    def __repr__(self):
        return "{}({})".format(
            str(self), ",".join([repr(f) for f in self._feature_list])
        )

    def featurize(self, to_featurize):
        features = []
        for f in self._feature_list:
            features.extend(f(to_featurize))
        return features
    
    def set_normalization(self,normalization="linear",**params):
        for f in self._feature_list:
            f.set_normalization(normalization,**params)
        self._normalization=None
            
    def describe_features(self):
        fl = []
        for f in self._feature_list:
            fl.extend(f.describe_features())
        return fl

    def __add__(self, other):
        if isinstance(other, FeaturizerList):
            return FeaturizerList(self._feature_list + other._feature_list)
        return FeaturizerList(self._feature_list + [other])
