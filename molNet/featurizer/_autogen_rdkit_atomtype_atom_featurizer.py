from molNet.featurizer._atom_featurizer import SingleValueAtomFeaturizer, OneHotAtomFeaturizer


class Atom_IsRgroup_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 0


class Atom_IsH_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 1


class Atom_IsHe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 2


class Atom_IsLi_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 3


class Atom_IsBe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 4


class Atom_IsB_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 5


class Atom_IsC_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 6


class Atom_IsN_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 7


class Atom_IsO_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 8


class Atom_IsF_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 9


class Atom_IsNe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 10


class Atom_IsNa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 11


class Atom_IsMg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 12


class Atom_IsAl_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 13


class Atom_IsSi_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 14


class Atom_IsP_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 15


class Atom_IsS_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 16


class Atom_IsCl_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 17


class Atom_IsAr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 18


class Atom_IsK_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 19


class Atom_IsCa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 20


class Atom_IsSc_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 21


class Atom_IsTi_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 22


class Atom_IsV_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 23


class Atom_IsCr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 24


class Atom_IsMn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 25


class Atom_IsFe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 26


class Atom_IsCo_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 27


class Atom_IsNi_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 28


class Atom_IsCu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 29


class Atom_IsZn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 30


class Atom_IsGa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 31


class Atom_IsGe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 32


class Atom_IsAs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 33


class Atom_IsSe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 34


class Atom_IsBr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 35


class Atom_IsKr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 36


class Atom_IsRb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 37


class Atom_IsSr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 38


class Atom_IsY_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 39


class Atom_IsZr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 40


class Atom_IsNb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 41


class Atom_IsMo_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 42


class Atom_IsTc_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 43


class Atom_IsRu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 44


class Atom_IsRh_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 45


class Atom_IsPd_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 46


class Atom_IsAg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 47


class Atom_IsCd_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 48


class Atom_IsIn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 49


class Atom_IsSn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 50


class Atom_IsSb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 51


class Atom_IsTe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 52


class Atom_IsI_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 53


class Atom_IsXe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 54


class Atom_IsCs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 55


class Atom_IsBa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 56


class Atom_IsLa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 57


class Atom_IsCe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 58


class Atom_IsPr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 59


class Atom_IsNd_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 60


class Atom_IsPm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 61


class Atom_IsSm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 62


class Atom_IsEu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 63


class Atom_IsGd_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 64


class Atom_IsTb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 65


class Atom_IsDy_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 66


class Atom_IsHo_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 67


class Atom_IsEr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 68


class Atom_IsTm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 69


class Atom_IsYb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 70


class Atom_IsLu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 71


class Atom_IsHf_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 72


class Atom_IsTa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 73


class Atom_IsW_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 74


class Atom_IsRe_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 75


class Atom_IsOs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 76


class Atom_IsIr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 77


class Atom_IsPt_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 78


class Atom_IsAu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 79


class Atom_IsHg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 80


class Atom_IsTl_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 81


class Atom_IsPb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 82


class Atom_IsBi_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 83


class Atom_IsPo_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 84


class Atom_IsAt_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 85


class Atom_IsRn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 86


class Atom_IsFr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 87


class Atom_IsRa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 88


class Atom_IsAc_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 89


class Atom_IsTh_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 90


class Atom_IsPa_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 91


class Atom_IsU_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 92


class Atom_IsNp_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 93


class Atom_IsPu_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 94


class Atom_IsAm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 95


class Atom_IsCm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 96


class Atom_IsBk_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 97


class Atom_IsCf_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 98


class Atom_IsEs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 99


class Atom_IsFm_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 100


class Atom_IsMd_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 101


class Atom_IsNo_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 102


class Atom_IsLr_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 103


class Atom_IsRf_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 104


class Atom_IsDb_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 105


class Atom_IsSg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 106


class Atom_IsBh_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 107


class Atom_IsHs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 108


class Atom_IsMt_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 109


class Atom_IsDs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 110


class Atom_IsRg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 111


class Atom_IsCn_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 112


class Atom_IsNh_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 113


class Atom_IsFl_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 114


class Atom_IsMc_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 115


class Atom_IsLv_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 116


class Atom_IsTs_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 117


class Atom_IsOg_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool

    def featurize(self, atom):
        return atom.GetAtomicNum() == 118


class Atom_AllSymbolOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = ['*', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
                       'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                       'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                       'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                       'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                       'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                       'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
                       'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    def featurize(self, atom):
        return atom.GetSymbol()


atom_IsRgroup_featurizer = Atom_IsRgroup_Featurizer()
atom_IsH_featurizer = Atom_IsH_Featurizer()
atom_IsHe_featurizer = Atom_IsHe_Featurizer()
atom_IsLi_featurizer = Atom_IsLi_Featurizer()
atom_IsBe_featurizer = Atom_IsBe_Featurizer()
atom_IsB_featurizer = Atom_IsB_Featurizer()
atom_IsC_featurizer = Atom_IsC_Featurizer()
atom_IsN_featurizer = Atom_IsN_Featurizer()
atom_IsO_featurizer = Atom_IsO_Featurizer()
atom_IsF_featurizer = Atom_IsF_Featurizer()
atom_IsNe_featurizer = Atom_IsNe_Featurizer()
atom_IsNa_featurizer = Atom_IsNa_Featurizer()
atom_IsMg_featurizer = Atom_IsMg_Featurizer()
atom_IsAl_featurizer = Atom_IsAl_Featurizer()
atom_IsSi_featurizer = Atom_IsSi_Featurizer()
atom_IsP_featurizer = Atom_IsP_Featurizer()
atom_IsS_featurizer = Atom_IsS_Featurizer()
atom_IsCl_featurizer = Atom_IsCl_Featurizer()
atom_IsAr_featurizer = Atom_IsAr_Featurizer()
atom_IsK_featurizer = Atom_IsK_Featurizer()
atom_IsCa_featurizer = Atom_IsCa_Featurizer()
atom_IsSc_featurizer = Atom_IsSc_Featurizer()
atom_IsTi_featurizer = Atom_IsTi_Featurizer()
atom_IsV_featurizer = Atom_IsV_Featurizer()
atom_IsCr_featurizer = Atom_IsCr_Featurizer()
atom_IsMn_featurizer = Atom_IsMn_Featurizer()
atom_IsFe_featurizer = Atom_IsFe_Featurizer()
atom_IsCo_featurizer = Atom_IsCo_Featurizer()
atom_IsNi_featurizer = Atom_IsNi_Featurizer()
atom_IsCu_featurizer = Atom_IsCu_Featurizer()
atom_IsZn_featurizer = Atom_IsZn_Featurizer()
atom_IsGa_featurizer = Atom_IsGa_Featurizer()
atom_IsGe_featurizer = Atom_IsGe_Featurizer()
atom_IsAs_featurizer = Atom_IsAs_Featurizer()
atom_IsSe_featurizer = Atom_IsSe_Featurizer()
atom_IsBr_featurizer = Atom_IsBr_Featurizer()
atom_IsKr_featurizer = Atom_IsKr_Featurizer()
atom_IsRb_featurizer = Atom_IsRb_Featurizer()
atom_IsSr_featurizer = Atom_IsSr_Featurizer()
atom_IsY_featurizer = Atom_IsY_Featurizer()
atom_IsZr_featurizer = Atom_IsZr_Featurizer()
atom_IsNb_featurizer = Atom_IsNb_Featurizer()
atom_IsMo_featurizer = Atom_IsMo_Featurizer()
atom_IsTc_featurizer = Atom_IsTc_Featurizer()
atom_IsRu_featurizer = Atom_IsRu_Featurizer()
atom_IsRh_featurizer = Atom_IsRh_Featurizer()
atom_IsPd_featurizer = Atom_IsPd_Featurizer()
atom_IsAg_featurizer = Atom_IsAg_Featurizer()
atom_IsCd_featurizer = Atom_IsCd_Featurizer()
atom_IsIn_featurizer = Atom_IsIn_Featurizer()
atom_IsSn_featurizer = Atom_IsSn_Featurizer()
atom_IsSb_featurizer = Atom_IsSb_Featurizer()
atom_IsTe_featurizer = Atom_IsTe_Featurizer()
atom_IsI_featurizer = Atom_IsI_Featurizer()
atom_IsXe_featurizer = Atom_IsXe_Featurizer()
atom_IsCs_featurizer = Atom_IsCs_Featurizer()
atom_IsBa_featurizer = Atom_IsBa_Featurizer()
atom_IsLa_featurizer = Atom_IsLa_Featurizer()
atom_IsCe_featurizer = Atom_IsCe_Featurizer()
atom_IsPr_featurizer = Atom_IsPr_Featurizer()
atom_IsNd_featurizer = Atom_IsNd_Featurizer()
atom_IsPm_featurizer = Atom_IsPm_Featurizer()
atom_IsSm_featurizer = Atom_IsSm_Featurizer()
atom_IsEu_featurizer = Atom_IsEu_Featurizer()
atom_IsGd_featurizer = Atom_IsGd_Featurizer()
atom_IsTb_featurizer = Atom_IsTb_Featurizer()
atom_IsDy_featurizer = Atom_IsDy_Featurizer()
atom_IsHo_featurizer = Atom_IsHo_Featurizer()
atom_IsEr_featurizer = Atom_IsEr_Featurizer()
atom_IsTm_featurizer = Atom_IsTm_Featurizer()
atom_IsYb_featurizer = Atom_IsYb_Featurizer()
atom_IsLu_featurizer = Atom_IsLu_Featurizer()
atom_IsHf_featurizer = Atom_IsHf_Featurizer()
atom_IsTa_featurizer = Atom_IsTa_Featurizer()
atom_IsW_featurizer = Atom_IsW_Featurizer()
atom_IsRe_featurizer = Atom_IsRe_Featurizer()
atom_IsOs_featurizer = Atom_IsOs_Featurizer()
atom_IsIr_featurizer = Atom_IsIr_Featurizer()
atom_IsPt_featurizer = Atom_IsPt_Featurizer()
atom_IsAu_featurizer = Atom_IsAu_Featurizer()
atom_IsHg_featurizer = Atom_IsHg_Featurizer()
atom_IsTl_featurizer = Atom_IsTl_Featurizer()
atom_IsPb_featurizer = Atom_IsPb_Featurizer()
atom_IsBi_featurizer = Atom_IsBi_Featurizer()
atom_IsPo_featurizer = Atom_IsPo_Featurizer()
atom_IsAt_featurizer = Atom_IsAt_Featurizer()
atom_IsRn_featurizer = Atom_IsRn_Featurizer()
atom_IsFr_featurizer = Atom_IsFr_Featurizer()
atom_IsRa_featurizer = Atom_IsRa_Featurizer()
atom_IsAc_featurizer = Atom_IsAc_Featurizer()
atom_IsTh_featurizer = Atom_IsTh_Featurizer()
atom_IsPa_featurizer = Atom_IsPa_Featurizer()
atom_IsU_featurizer = Atom_IsU_Featurizer()
atom_IsNp_featurizer = Atom_IsNp_Featurizer()
atom_IsPu_featurizer = Atom_IsPu_Featurizer()
atom_IsAm_featurizer = Atom_IsAm_Featurizer()
atom_IsCm_featurizer = Atom_IsCm_Featurizer()
atom_IsBk_featurizer = Atom_IsBk_Featurizer()
atom_IsCf_featurizer = Atom_IsCf_Featurizer()
atom_IsEs_featurizer = Atom_IsEs_Featurizer()
atom_IsFm_featurizer = Atom_IsFm_Featurizer()
atom_IsMd_featurizer = Atom_IsMd_Featurizer()
atom_IsNo_featurizer = Atom_IsNo_Featurizer()
atom_IsLr_featurizer = Atom_IsLr_Featurizer()
atom_IsRf_featurizer = Atom_IsRf_Featurizer()
atom_IsDb_featurizer = Atom_IsDb_Featurizer()
atom_IsSg_featurizer = Atom_IsSg_Featurizer()
atom_IsBh_featurizer = Atom_IsBh_Featurizer()
atom_IsHs_featurizer = Atom_IsHs_Featurizer()
atom_IsMt_featurizer = Atom_IsMt_Featurizer()
atom_IsDs_featurizer = Atom_IsDs_Featurizer()
atom_IsRg_featurizer = Atom_IsRg_Featurizer()
atom_IsCn_featurizer = Atom_IsCn_Featurizer()
atom_IsNh_featurizer = Atom_IsNh_Featurizer()
atom_IsFl_featurizer = Atom_IsFl_Featurizer()
atom_IsMc_featurizer = Atom_IsMc_Featurizer()
atom_IsLv_featurizer = Atom_IsLv_Featurizer()
atom_IsTs_featurizer = Atom_IsTs_Featurizer()
atom_IsOg_featurizer = Atom_IsOg_Featurizer()
atom_AllSymbolOneHot_featurizer = Atom_AllSymbolOneHot_Featurizer()
_available_featurizer = {
    'atom_IsRgroup_featurizer': atom_IsRgroup_featurizer,
    'atom_IsH_featurizer': atom_IsH_featurizer,
    'atom_IsHe_featurizer': atom_IsHe_featurizer,
    'atom_IsLi_featurizer': atom_IsLi_featurizer,
    'atom_IsBe_featurizer': atom_IsBe_featurizer,
    'atom_IsB_featurizer': atom_IsB_featurizer,
    'atom_IsC_featurizer': atom_IsC_featurizer,
    'atom_IsN_featurizer': atom_IsN_featurizer,
    'atom_IsO_featurizer': atom_IsO_featurizer,
    'atom_IsF_featurizer': atom_IsF_featurizer,
    'atom_IsNe_featurizer': atom_IsNe_featurizer,
    'atom_IsNa_featurizer': atom_IsNa_featurizer,
    'atom_IsMg_featurizer': atom_IsMg_featurizer,
    'atom_IsAl_featurizer': atom_IsAl_featurizer,
    'atom_IsSi_featurizer': atom_IsSi_featurizer,
    'atom_IsP_featurizer': atom_IsP_featurizer,
    'atom_IsS_featurizer': atom_IsS_featurizer,
    'atom_IsCl_featurizer': atom_IsCl_featurizer,
    'atom_IsAr_featurizer': atom_IsAr_featurizer,
    'atom_IsK_featurizer': atom_IsK_featurizer,
    'atom_IsCa_featurizer': atom_IsCa_featurizer,
    'atom_IsSc_featurizer': atom_IsSc_featurizer,
    'atom_IsTi_featurizer': atom_IsTi_featurizer,
    'atom_IsV_featurizer': atom_IsV_featurizer,
    'atom_IsCr_featurizer': atom_IsCr_featurizer,
    'atom_IsMn_featurizer': atom_IsMn_featurizer,
    'atom_IsFe_featurizer': atom_IsFe_featurizer,
    'atom_IsCo_featurizer': atom_IsCo_featurizer,
    'atom_IsNi_featurizer': atom_IsNi_featurizer,
    'atom_IsCu_featurizer': atom_IsCu_featurizer,
    'atom_IsZn_featurizer': atom_IsZn_featurizer,
    'atom_IsGa_featurizer': atom_IsGa_featurizer,
    'atom_IsGe_featurizer': atom_IsGe_featurizer,
    'atom_IsAs_featurizer': atom_IsAs_featurizer,
    'atom_IsSe_featurizer': atom_IsSe_featurizer,
    'atom_IsBr_featurizer': atom_IsBr_featurizer,
    'atom_IsKr_featurizer': atom_IsKr_featurizer,
    'atom_IsRb_featurizer': atom_IsRb_featurizer,
    'atom_IsSr_featurizer': atom_IsSr_featurizer,
    'atom_IsY_featurizer': atom_IsY_featurizer,
    'atom_IsZr_featurizer': atom_IsZr_featurizer,
    'atom_IsNb_featurizer': atom_IsNb_featurizer,
    'atom_IsMo_featurizer': atom_IsMo_featurizer,
    'atom_IsTc_featurizer': atom_IsTc_featurizer,
    'atom_IsRu_featurizer': atom_IsRu_featurizer,
    'atom_IsRh_featurizer': atom_IsRh_featurizer,
    'atom_IsPd_featurizer': atom_IsPd_featurizer,
    'atom_IsAg_featurizer': atom_IsAg_featurizer,
    'atom_IsCd_featurizer': atom_IsCd_featurizer,
    'atom_IsIn_featurizer': atom_IsIn_featurizer,
    'atom_IsSn_featurizer': atom_IsSn_featurizer,
    'atom_IsSb_featurizer': atom_IsSb_featurizer,
    'atom_IsTe_featurizer': atom_IsTe_featurizer,
    'atom_IsI_featurizer': atom_IsI_featurizer,
    'atom_IsXe_featurizer': atom_IsXe_featurizer,
    'atom_IsCs_featurizer': atom_IsCs_featurizer,
    'atom_IsBa_featurizer': atom_IsBa_featurizer,
    'atom_IsLa_featurizer': atom_IsLa_featurizer,
    'atom_IsCe_featurizer': atom_IsCe_featurizer,
    'atom_IsPr_featurizer': atom_IsPr_featurizer,
    'atom_IsNd_featurizer': atom_IsNd_featurizer,
    'atom_IsPm_featurizer': atom_IsPm_featurizer,
    'atom_IsSm_featurizer': atom_IsSm_featurizer,
    'atom_IsEu_featurizer': atom_IsEu_featurizer,
    'atom_IsGd_featurizer': atom_IsGd_featurizer,
    'atom_IsTb_featurizer': atom_IsTb_featurizer,
    'atom_IsDy_featurizer': atom_IsDy_featurizer,
    'atom_IsHo_featurizer': atom_IsHo_featurizer,
    'atom_IsEr_featurizer': atom_IsEr_featurizer,
    'atom_IsTm_featurizer': atom_IsTm_featurizer,
    'atom_IsYb_featurizer': atom_IsYb_featurizer,
    'atom_IsLu_featurizer': atom_IsLu_featurizer,
    'atom_IsHf_featurizer': atom_IsHf_featurizer,
    'atom_IsTa_featurizer': atom_IsTa_featurizer,
    'atom_IsW_featurizer': atom_IsW_featurizer,
    'atom_IsRe_featurizer': atom_IsRe_featurizer,
    'atom_IsOs_featurizer': atom_IsOs_featurizer,
    'atom_IsIr_featurizer': atom_IsIr_featurizer,
    'atom_IsPt_featurizer': atom_IsPt_featurizer,
    'atom_IsAu_featurizer': atom_IsAu_featurizer,
    'atom_IsHg_featurizer': atom_IsHg_featurizer,
    'atom_IsTl_featurizer': atom_IsTl_featurizer,
    'atom_IsPb_featurizer': atom_IsPb_featurizer,
    'atom_IsBi_featurizer': atom_IsBi_featurizer,
    'atom_IsPo_featurizer': atom_IsPo_featurizer,
    'atom_IsAt_featurizer': atom_IsAt_featurizer,
    'atom_IsRn_featurizer': atom_IsRn_featurizer,
    'atom_IsFr_featurizer': atom_IsFr_featurizer,
    'atom_IsRa_featurizer': atom_IsRa_featurizer,
    'atom_IsAc_featurizer': atom_IsAc_featurizer,
    'atom_IsTh_featurizer': atom_IsTh_featurizer,
    'atom_IsPa_featurizer': atom_IsPa_featurizer,
    'atom_IsU_featurizer': atom_IsU_featurizer,
    'atom_IsNp_featurizer': atom_IsNp_featurizer,
    'atom_IsPu_featurizer': atom_IsPu_featurizer,
    'atom_IsAm_featurizer': atom_IsAm_featurizer,
    'atom_IsCm_featurizer': atom_IsCm_featurizer,
    'atom_IsBk_featurizer': atom_IsBk_featurizer,
    'atom_IsCf_featurizer': atom_IsCf_featurizer,
    'atom_IsEs_featurizer': atom_IsEs_featurizer,
    'atom_IsFm_featurizer': atom_IsFm_featurizer,
    'atom_IsMd_featurizer': atom_IsMd_featurizer,
    'atom_IsNo_featurizer': atom_IsNo_featurizer,
    'atom_IsLr_featurizer': atom_IsLr_featurizer,
    'atom_IsRf_featurizer': atom_IsRf_featurizer,
    'atom_IsDb_featurizer': atom_IsDb_featurizer,
    'atom_IsSg_featurizer': atom_IsSg_featurizer,
    'atom_IsBh_featurizer': atom_IsBh_featurizer,
    'atom_IsHs_featurizer': atom_IsHs_featurizer,
    'atom_IsMt_featurizer': atom_IsMt_featurizer,
    'atom_IsDs_featurizer': atom_IsDs_featurizer,
    'atom_IsRg_featurizer': atom_IsRg_featurizer,
    'atom_IsCn_featurizer': atom_IsCn_featurizer,
    'atom_IsNh_featurizer': atom_IsNh_featurizer,
    'atom_IsFl_featurizer': atom_IsFl_featurizer,
    'atom_IsMc_featurizer': atom_IsMc_featurizer,
    'atom_IsLv_featurizer': atom_IsLv_featurizer,
    'atom_IsTs_featurizer': atom_IsTs_featurizer,
    'atom_IsOg_featurizer': atom_IsOg_featurizer,
    'atom_AllSymbolOneHot_featurizer': atom_AllSymbolOneHot_featurizer,
}

__all__ = [
    'Atom_IsRgroup_Featurizer',
    'atom_IsRgroup_featurizer',
    'Atom_IsH_Featurizer',
    'atom_IsH_featurizer',
    'Atom_IsHe_Featurizer',
    'atom_IsHe_featurizer',
    'Atom_IsLi_Featurizer',
    'atom_IsLi_featurizer',
    'Atom_IsBe_Featurizer',
    'atom_IsBe_featurizer',
    'Atom_IsB_Featurizer',
    'atom_IsB_featurizer',
    'Atom_IsC_Featurizer',
    'atom_IsC_featurizer',
    'Atom_IsN_Featurizer',
    'atom_IsN_featurizer',
    'Atom_IsO_Featurizer',
    'atom_IsO_featurizer',
    'Atom_IsF_Featurizer',
    'atom_IsF_featurizer',
    'Atom_IsNe_Featurizer',
    'atom_IsNe_featurizer',
    'Atom_IsNa_Featurizer',
    'atom_IsNa_featurizer',
    'Atom_IsMg_Featurizer',
    'atom_IsMg_featurizer',
    'Atom_IsAl_Featurizer',
    'atom_IsAl_featurizer',
    'Atom_IsSi_Featurizer',
    'atom_IsSi_featurizer',
    'Atom_IsP_Featurizer',
    'atom_IsP_featurizer',
    'Atom_IsS_Featurizer',
    'atom_IsS_featurizer',
    'Atom_IsCl_Featurizer',
    'atom_IsCl_featurizer',
    'Atom_IsAr_Featurizer',
    'atom_IsAr_featurizer',
    'Atom_IsK_Featurizer',
    'atom_IsK_featurizer',
    'Atom_IsCa_Featurizer',
    'atom_IsCa_featurizer',
    'Atom_IsSc_Featurizer',
    'atom_IsSc_featurizer',
    'Atom_IsTi_Featurizer',
    'atom_IsTi_featurizer',
    'Atom_IsV_Featurizer',
    'atom_IsV_featurizer',
    'Atom_IsCr_Featurizer',
    'atom_IsCr_featurizer',
    'Atom_IsMn_Featurizer',
    'atom_IsMn_featurizer',
    'Atom_IsFe_Featurizer',
    'atom_IsFe_featurizer',
    'Atom_IsCo_Featurizer',
    'atom_IsCo_featurizer',
    'Atom_IsNi_Featurizer',
    'atom_IsNi_featurizer',
    'Atom_IsCu_Featurizer',
    'atom_IsCu_featurizer',
    'Atom_IsZn_Featurizer',
    'atom_IsZn_featurizer',
    'Atom_IsGa_Featurizer',
    'atom_IsGa_featurizer',
    'Atom_IsGe_Featurizer',
    'atom_IsGe_featurizer',
    'Atom_IsAs_Featurizer',
    'atom_IsAs_featurizer',
    'Atom_IsSe_Featurizer',
    'atom_IsSe_featurizer',
    'Atom_IsBr_Featurizer',
    'atom_IsBr_featurizer',
    'Atom_IsKr_Featurizer',
    'atom_IsKr_featurizer',
    'Atom_IsRb_Featurizer',
    'atom_IsRb_featurizer',
    'Atom_IsSr_Featurizer',
    'atom_IsSr_featurizer',
    'Atom_IsY_Featurizer',
    'atom_IsY_featurizer',
    'Atom_IsZr_Featurizer',
    'atom_IsZr_featurizer',
    'Atom_IsNb_Featurizer',
    'atom_IsNb_featurizer',
    'Atom_IsMo_Featurizer',
    'atom_IsMo_featurizer',
    'Atom_IsTc_Featurizer',
    'atom_IsTc_featurizer',
    'Atom_IsRu_Featurizer',
    'atom_IsRu_featurizer',
    'Atom_IsRh_Featurizer',
    'atom_IsRh_featurizer',
    'Atom_IsPd_Featurizer',
    'atom_IsPd_featurizer',
    'Atom_IsAg_Featurizer',
    'atom_IsAg_featurizer',
    'Atom_IsCd_Featurizer',
    'atom_IsCd_featurizer',
    'Atom_IsIn_Featurizer',
    'atom_IsIn_featurizer',
    'Atom_IsSn_Featurizer',
    'atom_IsSn_featurizer',
    'Atom_IsSb_Featurizer',
    'atom_IsSb_featurizer',
    'Atom_IsTe_Featurizer',
    'atom_IsTe_featurizer',
    'Atom_IsI_Featurizer',
    'atom_IsI_featurizer',
    'Atom_IsXe_Featurizer',
    'atom_IsXe_featurizer',
    'Atom_IsCs_Featurizer',
    'atom_IsCs_featurizer',
    'Atom_IsBa_Featurizer',
    'atom_IsBa_featurizer',
    'Atom_IsLa_Featurizer',
    'atom_IsLa_featurizer',
    'Atom_IsCe_Featurizer',
    'atom_IsCe_featurizer',
    'Atom_IsPr_Featurizer',
    'atom_IsPr_featurizer',
    'Atom_IsNd_Featurizer',
    'atom_IsNd_featurizer',
    'Atom_IsPm_Featurizer',
    'atom_IsPm_featurizer',
    'Atom_IsSm_Featurizer',
    'atom_IsSm_featurizer',
    'Atom_IsEu_Featurizer',
    'atom_IsEu_featurizer',
    'Atom_IsGd_Featurizer',
    'atom_IsGd_featurizer',
    'Atom_IsTb_Featurizer',
    'atom_IsTb_featurizer',
    'Atom_IsDy_Featurizer',
    'atom_IsDy_featurizer',
    'Atom_IsHo_Featurizer',
    'atom_IsHo_featurizer',
    'Atom_IsEr_Featurizer',
    'atom_IsEr_featurizer',
    'Atom_IsTm_Featurizer',
    'atom_IsTm_featurizer',
    'Atom_IsYb_Featurizer',
    'atom_IsYb_featurizer',
    'Atom_IsLu_Featurizer',
    'atom_IsLu_featurizer',
    'Atom_IsHf_Featurizer',
    'atom_IsHf_featurizer',
    'Atom_IsTa_Featurizer',
    'atom_IsTa_featurizer',
    'Atom_IsW_Featurizer',
    'atom_IsW_featurizer',
    'Atom_IsRe_Featurizer',
    'atom_IsRe_featurizer',
    'Atom_IsOs_Featurizer',
    'atom_IsOs_featurizer',
    'Atom_IsIr_Featurizer',
    'atom_IsIr_featurizer',
    'Atom_IsPt_Featurizer',
    'atom_IsPt_featurizer',
    'Atom_IsAu_Featurizer',
    'atom_IsAu_featurizer',
    'Atom_IsHg_Featurizer',
    'atom_IsHg_featurizer',
    'Atom_IsTl_Featurizer',
    'atom_IsTl_featurizer',
    'Atom_IsPb_Featurizer',
    'atom_IsPb_featurizer',
    'Atom_IsBi_Featurizer',
    'atom_IsBi_featurizer',
    'Atom_IsPo_Featurizer',
    'atom_IsPo_featurizer',
    'Atom_IsAt_Featurizer',
    'atom_IsAt_featurizer',
    'Atom_IsRn_Featurizer',
    'atom_IsRn_featurizer',
    'Atom_IsFr_Featurizer',
    'atom_IsFr_featurizer',
    'Atom_IsRa_Featurizer',
    'atom_IsRa_featurizer',
    'Atom_IsAc_Featurizer',
    'atom_IsAc_featurizer',
    'Atom_IsTh_Featurizer',
    'atom_IsTh_featurizer',
    'Atom_IsPa_Featurizer',
    'atom_IsPa_featurizer',
    'Atom_IsU_Featurizer',
    'atom_IsU_featurizer',
    'Atom_IsNp_Featurizer',
    'atom_IsNp_featurizer',
    'Atom_IsPu_Featurizer',
    'atom_IsPu_featurizer',
    'Atom_IsAm_Featurizer',
    'atom_IsAm_featurizer',
    'Atom_IsCm_Featurizer',
    'atom_IsCm_featurizer',
    'Atom_IsBk_Featurizer',
    'atom_IsBk_featurizer',
    'Atom_IsCf_Featurizer',
    'atom_IsCf_featurizer',
    'Atom_IsEs_Featurizer',
    'atom_IsEs_featurizer',
    'Atom_IsFm_Featurizer',
    'atom_IsFm_featurizer',
    'Atom_IsMd_Featurizer',
    'atom_IsMd_featurizer',
    'Atom_IsNo_Featurizer',
    'atom_IsNo_featurizer',
    'Atom_IsLr_Featurizer',
    'atom_IsLr_featurizer',
    'Atom_IsRf_Featurizer',
    'atom_IsRf_featurizer',
    'Atom_IsDb_Featurizer',
    'atom_IsDb_featurizer',
    'Atom_IsSg_Featurizer',
    'atom_IsSg_featurizer',
    'Atom_IsBh_Featurizer',
    'atom_IsBh_featurizer',
    'Atom_IsHs_Featurizer',
    'atom_IsHs_featurizer',
    'Atom_IsMt_Featurizer',
    'atom_IsMt_featurizer',
    'Atom_IsDs_Featurizer',
    'atom_IsDs_featurizer',
    'Atom_IsRg_Featurizer',
    'atom_IsRg_featurizer',
    'Atom_IsCn_Featurizer',
    'atom_IsCn_featurizer',
    'Atom_IsNh_Featurizer',
    'atom_IsNh_featurizer',
    'Atom_IsFl_Featurizer',
    'atom_IsFl_featurizer',
    'Atom_IsMc_Featurizer',
    'atom_IsMc_featurizer',
    'Atom_IsLv_Featurizer',
    'atom_IsLv_featurizer',
    'Atom_IsTs_Featurizer',
    'atom_IsTs_featurizer',
    'Atom_IsOg_Featurizer',
    'atom_IsOg_featurizer',
    'Atom_AllSymbolOneHot_Featurizer',
    'atom_AllSymbolOneHot_featurizer',
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    testdata = Chem.MolFromSmiles('c1ccccc1').GetAtoms()[0]
    for n, f in get_available_featurizer().items():
        print(n, f(testdata))
    print(len(get_available_featurizer()))


if __name__ == '__main__':
    main()
