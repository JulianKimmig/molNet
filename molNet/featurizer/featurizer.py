from typing import List, Optional, Union
from warnings import warn

AS_NUMPY_ARRY = False
import numpy as np

from .normalization import NormalizationClass


class OneHotEncodingException(Exception):
    pass


class Featurizer(NormalizationClass):
    dtype = object
    NAME = None
    NORMALIZATION = None
    DESCRIPTION=None

    def pre_featurize(self, x):
        return x

    def __init__(self, name=None, pre_featurize=None, dtype=None, feature_descriptions=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if name is None:
            if self.NAME is None:
                name = self.__class__.__name__
            else:
                name = self.NAME
        if pre_featurize is None:
            pre_featurize = self.pre_featurize

        if dtype is None:
            dtype = self.dtype
        self._dtype = dtype
        self._pre_featurize = pre_featurize
        self._name = name

        if feature_descriptions is None and self.DESCRIPTION is not None:
            feature_descriptions = self.DESCRIPTION
        self._feature_descriptions = feature_descriptions

    def __str__(self):
        return self._name

    def __repr__(self):
        return {"name":self._name,
                "description":self._feature_descriptions,
                "dtype":self._dtype,
                "norm":self._preferred_norm_name,
                }

    def __add__(self, other):
        if isinstance(other, FeaturizerList):
            return other + self
        return FeaturizerList([self, other])

    def get_dtype(self):
        return self._dtype

    def featurize(self, to_featurize):
        return to_featurize

    def __call__(self, to_featurize):
        if self._pre_featurize is not None:
            to_featurize = self._pre_featurize(to_featurize)
        f = self.featurize(to_featurize)
        f = np.array(f, dtype=self._dtype)
        if not f.ndim:
            f = np.expand_dims(f, 0)
        f = self.normalize(f)
        return f

    def describe_features(self):
        if self._feature_descriptions is None:
            return self.__repr__
        return self._feature_descriptions

class FixedSizeFeaturizer(Featurizer):
    LENGTH: int = -1
    DESCRIPTION: Union[str,List[str]] = None

    def __init__(self, length=None, *args, **kwargs):
        if length is None and self.LENGTH > 0:
            length = self.LENGTH
        self._length = length

        super().__init__(*args, **kwargs)

    def __len__(self):
        if self._length is None:
            warn(
                "length for featurizer '{}' not defined please run at least oes for autodetermination".format(
                    self
                )
            )
        return self._length

    def __call__(self, to_featurize):
        f = super().__call__(to_featurize)
        if self._length is None:
            self._len = len(f)
        return f

    def describe_features(self):
        if self._feature_descriptions is None:
            return [str(self)] * len(self)

        if isinstance(self._feature_descriptions,str):
            return super().describe_features()

        return [self._feature_descriptions[i] for i in range(len(self))]


class OneHotFeaturizer(FixedSizeFeaturizer):
    dtype = np.bool_

    def __init__(self, possible_values, *args, **kwargs):
        kwargs["length"] = len(possible_values)
        super().__init__(*args, **kwargs)
        self._possible_values = possible_values

    def featurize(self, to_featurize):
        if None in self._possible_values and to_featurize not in self._possible_values:
            to_featurize = None
        if to_featurize not in self._possible_values:
            raise OneHotEncodingException(
                "cannot one hot encode '{}' in '{}', allowed values are {}".format(
                    to_featurize, self, self._possible_values
                )
            )
        # return list(map(lambda v: to_featurize == v, self.possible_values))
        return [v == to_featurize for v in self._possible_values]

    def describe_features(self):
        if self._feature_descriptions is None:
            n = str(self)
            return ["{}: {}".format(n, v) for v in self._possible_values]

        return super().describe_features()


class FeaturizerList(Featurizer):
    def __init__(self, feature_list, name=None, *args, **kwargs):
        if name == None:
            name = "FeatureList({})".format(",".join([str(f) for f in feature_list]))
        if not "dtype" in kwargs and len(feature_list) > 0:
            kwargs["dtype"] = (
                np.concatenate([np.ones(1, dtype=f.get_dtype()) for f in feature_list])
            ).dtype

        self._feature_list = feature_list
        super().__init__(name=name, *args, **kwargs)

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

    def describe_features(self):
        fl = []
        for f in self._feature_list:
            fl.extend(f.describe_features())
        return fl

    def __add__(self, other):
        if isinstance(other, FeaturizerList):
            return FeaturizerList(self._feature_list + other._feature_list)
        return FeaturizerList(self._feature_list + [other])
