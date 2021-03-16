import numba

try:
    from numba import jit, njit

    JIT_AVAILABLE = True
except:
    JIT_AVAILABLE = False

import numpy as np


class Substance:
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name

        return super().__str__()

    def __repr__(self):
        return str(self)


class Reaction:
    def __init__(self, k=0, reactants=[], products=[]):
        self._reactants = []
        self._products = []
        self.k = k
        for r in reactants:
            self.add_reactant(r[0], r[1])
        for r in products:
            self.add_product(r[0], r[1])

    def get_back_reaction(self, k_back=0):
        br = Reaction()
        br._reactants = self._products
        br._products = self._reactants
        br.k = k_back

    def add_reactant(self, substance, stoichiometry):
        for e in self.get_reactants(with_stoichiometry=True):
            if e[0] == substance:
                e[1] += stoichiometry
                return
        self._reactants.append([substance, stoichiometry])

    def add_product(self, substance, stoichiometry):
        for p in self.get_products(with_stoichiometry=True):
            if p[0] == substance:
                p[1] += stoichiometry
                return
        self._products.append([substance, stoichiometry])

    def get_product_stoichiometry(self, substance):
        for p in self.get_products(with_stoichiometry=True):
            if p[0] == substance:
                return p[1]
        return 0

    def get_reactant_stoichiometry(self, substance):
        for e in self.get_reactants(with_stoichiometry=True):
            if e[0] == substance:
                return e[1]
        return 0

    def get_reactants(self, with_stoichiometry=False):
        if with_stoichiometry:
            return [ed for ed in self._reactants]
        return [ed[0] for ed in self._reactants]

    def get_products(self, with_stoichiometry=False):
        if with_stoichiometry:
            return [pd for pd in self._products]
        return [pd[0] for pd in self._products]

    @property
    def reactants(self):
        return self.get_reactants()

    @property
    def products(self):
        return self.get_products()

    def __str__(self):
        s = "{} --> {}".format(
            " + ".join(
                [
                    "{} {}".format(e[1], e[0])
                    for e in self.get_reactants(with_stoichiometry=True)
                ]
            ),
            " + ".join(
                [
                    "{} {}".format(p[1], p[0])
                    for p in self.get_products(with_stoichiometry=True)
                ]
            ),
        )
        return s

    def _repr_latex_(self):
        s = "${} \\xrightarrow{{ {} }}  {}$".format(
            " + ".join(
                [
                    "{} {}".format(e[1], e[0])
                    for e in self.get_reactants(with_stoichiometry=True)
                ]
            ),
            self.k,
            " + ".join(
                [
                    "{} {}".format(p[1], p[0])
                    for p in self.get_products(with_stoichiometry=True)
                ]
            ),
        )
        return s


class ReactionSet:
    def __init__(self):
        self._reactions = []

    def add_reaction(self, reaction: Reaction):
        self._reactions.append(reaction)

    @property
    def reactions(self):
        """

        :rtype: List[Reaction]
        """
        return self._reactions

    def get_differential_function(self):

        reactions = self.reactions
        ks = np.array([r.k for r in reactions], dtype=np.float)
        all_substances = self.all_substances()

        res_prod = np.array(
            [
                [r.get_product_stoichiometry(p) for p in all_substances]
                for r in reactions
            ],
            dtype=np.float,
        )
        res_ed = np.array(
            [
                [r.get_reactant_stoichiometry(e) for e in all_substances]
                for r in reactions
            ],
            dtype=np.float,
        )
        exp = res_ed
        res = res_prod - res_ed

        def _step_diff(y, _exp, _res, _ks):
            # x = y ** _exp
            # x = x.prod(axis=1.0)
            # x = (x[:, None] * _res).T
            # x = (x * _ks)
            # x = np.sum(x, axis=1)
            # return x
            return np.sum(((np.prod(y ** exp, axis=1)[:, None] * res).T * ks), axis=1)

        def _step_diff_jit(y, _exp, _res, _ks):
            x1 = y ** _exp
            x2 = np.empty(x1.shape[0])
            for i in range(len(x2)):
                x2[i] = np.prod(x1[i, :])
            x3 = np.empty((x2.shape[0], 1))
            for i in range(len(x2)):
                x3[i, 0] = x2[i]
            x3 = (x3 * _res).T
            x3 = x3 * _ks
            x4 = np.empty(x3.shape[0], dtype=y.dtype)
            for i in range(len(x4)):
                x4[i] = np.sum(x3[i, :])
            # x = np.sum(x, axis=1)
            return x4

        if JIT_AVAILABLE:
            _step_diff = numba.njit(
                [
                    "float64[:](float64[:],float64[:,:],float64[:,:],float64[:])",
                    "float32[:](float32[:],float32[:,:],float32[:,:],float32[:])",
                ]
            )(_step_diff_jit)

        def step_diff(y, t):
            if not isinstance(y, np.ndarray):
                y = np.array(y, dtype=np.float)
            if not y.dtype == np.float:
                y = y.astype(np.float)
            return _step_diff(y, exp, res, ks)

        return step_diff, {
            "substances": all_substances,
            "reactions": reactions,
            "res_prod": res_prod,
            "res_ed": res_ed,
            "exp": exp,
            "res": res,
            "ks": ks,
            "_step_diff": _step_diff,
        }

    def all_reactants(self):
        d = []
        for r in self._reactions:
            d.extend(r.reactants)
        return list(set(d))

    def all_substances(self):
        return list(set(self.all_reactants() + self.all_products()))

    def all_products(self):
        d = []
        for r in self._reactions:
            d.extend(r.products)
        return list(set(d))

    def _repr_latex_(self):
        return (
            "$" + " \\\\ ".join([r._repr_latex_()[1:-1] for r in self._reactions]) + "$"
        )
