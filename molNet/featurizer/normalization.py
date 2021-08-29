def linear_norm(x, m=1, c=0):
    return x * m + c


def linear_norm_gen(m=1, c=0):
    def _norm(x):
        return x * m + c

    return _norm


def min_max_norm(x, min, max):
    return linear_norm(x, m=1 / (max - min), c=-min / (max - min))


def min_max_norm_gen(min, max):
    return linear_norm_gen(m=1 / (max - min), c=-min / (max - min))


def solve_normalization(featurizer, name, **params):
    if name == None:
        return None
    elif name == "default":
        return None
    elif name == "linear":
        return linear_norm_gen(**params)
    elif name == "minmax":
        return min_max_norm_gen(**params)
    else:
        raise ValueError("unknown norm '{}'".format(name))
