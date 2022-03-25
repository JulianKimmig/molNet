from typing import List, Union, Callable, Any
from warnings import warn

AS_NUMPY_ARRY = False
import numpy as np




class OneHotEncodingException(Exception):
    pass


class Featurizer():
    dtype = object
    NAME = None
    NORMALIZATION = None
    DESCRIPTION = None

    def prepend_postfeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        if name in self._post_featurizer_names:
            raise ValueError(f"{name} is already in the list of postfeaturizers")
        self._post_featurizer.insert(0, pre_featurizer)
        self._post_featurizer_names.insert(0, name)

    def append_postfeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        if name in self._post_featurizer_names:
            raise ValueError(f"{name} is already in the list of postfeaturizers")
        self._post_featurizer.append(pre_featurizer)
        self._post_featurizer_names.append(name)

    def add_postfeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        self.append_postfeaturizer(pre_featurizer,name=name)


    def prepend_prefeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        if name in self._pre_featurizer_names:
            raise ValueError(f"{name} is already in the list of prefeaturizers")
        self._pre_featurizer.insert(0, pre_featurizer)
        self._pre_featurizer_names.insert(0, name)

    def append_prefeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        if name in self._pre_featurizer_names:
            raise ValueError(f"{name} is already in the list of prefeaturizers")
        self._pre_featurizer.append(pre_featurizer)
        self._pre_featurizer_names.append(name)

    def add_prefeaturizer(self, pre_featurizer:Callable[["Featurizer",Any], Any],name:str):
        self.append_prefeaturizer(pre_featurizer,name=name)

    def __init__(self, name=None, dtype=None, feature_descriptions=None, *args,
                 **kwargs):

        if name is None:
            if self.NAME is None:
                name = f'{self.__class__.__module__}.{self.__class__.__name__}'
            else:
                name = self.NAME
        self._name = name
        super().__init__(*args, **kwargs)

        if dtype is None:
            dtype = self.dtype
        self._dtype = dtype
        self._pre_featurizer:List[Callable[["Featurizer",Any], Any]] = []
        self._post_featurizer:List[Callable[["Featurizer",Any], Any]] = []
        self._post_featurizer_names:List[str] = []
        self._pre_featurizer_names:List[str] = []


        if feature_descriptions is None and self.DESCRIPTION is not None:
            feature_descriptions = self.DESCRIPTION
        self._feature_descriptions = feature_descriptions

    def __str__(self):
        return self._name

    def __repr__(self):
        return repr(self.dict)

    def as_dict(self):
        return {"name": self._name,
                "description": self._feature_descriptions,
                "dtype": self._dtype,
                "norm": self._preferred_norm_name,
                }

    @property
    def dict(self):
        return self.as_dict()

    def __add__(self, other):
        if isinstance(other, FeaturizerList):
            return other + self
        return FeaturizerList([self, other])

    def get_dtype(self):
        return self._dtype

    def featurize(self, to_featurize):
        return to_featurize

    def prefeaturize(self, to_featurize:Any,ignored_prefeaturizers:List[str]=None):
        if ignored_prefeaturizers is None:
            ignored_prefeaturizers = []
        for i,pref in enumerate(self._pre_featurizer):
            if self._pre_featurizer_names[i] in ignored_prefeaturizers:
                continue
            to_featurize = pref(self,to_featurize)
        return to_featurize

    def postfeaturize(self, to_featurize:Any,ignored_postfeaturizers:List[str]=None):
        if ignored_postfeaturizers is None:
            ignored_postfeaturizers = []
        for i,postf in enumerate(self._post_featurizer):
            if self._post_featurizer_names[i] in ignored_postfeaturizers:
                continue
            to_featurize = postf(self,to_featurize)
        return to_featurize

    def __call__(self, to_featurize,ignored_prefeaturizers=None,ignored_postfeaturizers=None):
        if ignored_prefeaturizers is None:
            ignored_prefeaturizers = []
        if ignored_postfeaturizers is None:
            ignored_postfeaturizers = []

        to_featurize = self.prefeaturize(to_featurize,ignored_prefeaturizers=ignored_prefeaturizers)
        for pref in self._pre_featurizer:
            to_featurize = pref(self,to_featurize)
        f = self.featurize(to_featurize)
        f = np.array(f, dtype=self._dtype)
        if not f.ndim:
            f = np.expand_dims(f, 0)
        f= self.postfeaturize(f,ignored_postfeaturizers=ignored_postfeaturizers)
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
        if length is None:
            raise ValueError(
                f"no length given to {self}, please define via 'LENGTH' as class attribute or via keyword 'length' during initialization")
        self._length = length
        if len(self)<=0:
            raise ValueError(
                f"length of {self}, is 0 or smaller, which seems unreasonable")

        super().__init__(*args, **kwargs)

    def __len__(self):
        if self._length is None:
            warn(
                "length for featurizer '{}' not defined please run at least oes for autodetermination".format(
                    self
                )
            )
        return self._length

    def __call__(self, to_featurize, **kwargs):
        f = super().__call__(to_featurize, **kwargs)
        if self._length is None:
            self._len = len(f)
        return f

    def as_dict(self):
        d = super(FixedSizeFeaturizer, self).as_dict()
        d["length"] = self._length
        return d

    def describe_features(self):
        if self._feature_descriptions is None:
            return [str(self)] * len(self)

        if isinstance(self._feature_descriptions, str):
            return super().describe_features()

        return [self._feature_descriptions[i] for i in range(len(self))]


class OneHotFeaturizer(FixedSizeFeaturizer):
    dtype = bool
    POSSIBLE_VALUES = []

    def __init__(self, possible_values=None, *args, **kwargs):
        if possible_values is None and len(self.POSSIBLE_VALUES) > 0:
            possible_values = self.POSSIBLE_VALUES
        if possible_values is None:
            raise ValueError(
                f"no possible values given to {self}, please define via 'POSSIBLE_VALUES' as class attribute or via keyword 'possible_values' during initialization")
        self._possible_values = possible_values
        kwargs["length"] = len(possible_values)
        super().__init__(*args, **kwargs)
        self._ofeaturize=self.featurize

        self.featurize = self._oh_featurize

    def as_dict(self):
        d = super(FixedSizeFeaturizer, self).as_dict()
        d["possible_values"] = self._possible_values
        return d

    def _oh_featurize(self, x):
        x=self._ofeaturize(x)
        if None in self._possible_values and x not in self._possible_values:
            x = None
        if x not in self._possible_values:
            raise OneHotEncodingException(
                "cannot one hot encode '{}' in '{}', allowed values are {}".format(
                    x, self, self._possible_values
                )
            )
        return [v == x for v in self._possible_values]

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


class StringFeaturizer(Featurizer):
    dtype = str
