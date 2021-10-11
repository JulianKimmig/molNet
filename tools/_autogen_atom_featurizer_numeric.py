from molNet.featurizer._atom_featurizer import AtomFeaturizer, SingleValueAtomFeaturizer
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import (
    Chi1,
)
from rdkit.Chem.GraphDescriptors import (
    Chi1,
)
from rdkit.Chem.rdMolDescriptors import (
    GetAtomPairAtomCode,
)


class Chi1_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi1)
    # normalization
    linear_norm_parameter = (
        0.5241743105814027,
        0.10216132967690017,
    )  # error of 4.16E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (1.02E-01,1.30E+00)
    min_max_norm_parameter = (
        2.0937646859667335e-17,
        1.6335455017279354,
    )  # error of 4.17E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.7936228857227259,
        2.7914074053861078,
    )  # error of 4.49E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (9.84E-02,9.84E-01)
    dual_sigmoidal_norm_parameter = (
        0.8411881903275521,
        1.5908783391030816,
        3.1245365087621155,
    )  # error of 4.35E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (2.08E-01,9.89E-01)
    genlog_norm_parameter = (
        6.49085138617289,
        1.259995898660127,
        4.4863032977662165,
        6.661851485103663,
    )  # error of 3.77E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (2.34E-01,9.99E-01)
    preferred_normalization = "genlog"
    # functions


class Chi1_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi1)
    # normalization
    linear_norm_parameter = (
        0.5241743105814027,
        0.10216132967690017,
    )  # error of 4.16E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (1.02E-01,1.30E+00)
    min_max_norm_parameter = (
        2.0937646859667335e-17,
        1.6335455017279354,
    )  # error of 4.17E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.7936228857227259,
        2.7914074053861078,
    )  # error of 4.49E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (9.84E-02,9.84E-01)
    dual_sigmoidal_norm_parameter = (
        0.8411881903275521,
        1.5908783391030816,
        3.1245365087621155,
    )  # error of 4.35E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (2.08E-01,9.89E-01)
    genlog_norm_parameter = (
        6.49085138617289,
        1.259995898660127,
        4.4863032977662165,
        6.661851485103663,
    )  # error of 3.77E-02 with sample range (0.00E+00,2.28E+00) resulting in fit range (2.34E-01,9.99E-01)
    preferred_normalization = "genlog"
    # functions


class AtomPairAtomCode_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(GetAtomPairAtomCode)
    # normalization
    linear_norm_parameter = (
        0.0005943853768189955,
        0.36806553203764436,
    )  # error of 7.07E-02 with sample range (1.00E+00,4.99E+02) resulting in fit range (3.69E-01,6.65E-01)
    min_max_norm_parameter = (
        1.0000000000000002,
        491.8593074706766,
    )  # error of 1.72E-01 with sample range (1.00E+00,4.99E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        210.2001702344892,
        0.00582475542923338,
    )  # error of 1.33E-01 with sample range (1.00E+00,4.99E+02) resulting in fit range (2.28E-01,8.43E-01)
    dual_sigmoidal_norm_parameter = (
        78.69097737805613,
        0.043409834497287005,
        0.0025859937189602117,
    )  # error of 1.15E-01 with sample range (1.00E+00,4.99E+02) resulting in fit range (3.32E-02,7.48E-01)
    genlog_norm_parameter = (
        0.004335415607514263,
        -614.0108269990006,
        0.0007341529411965586,
        3.116787814184606e-05,
    )  # error of 1.30E-01 with sample range (1.00E+00,4.99E+02) resulting in fit range (1.95E-01,8.28E-01)
    preferred_normalization = "dual_sig"
    # functions


class AtomMapNum_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetAtomMapNum()


class AtomicNum_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.009087878464170251,
        0.7596300578436055,
    )  # error of 1.14E-01 with sample range (1.00E+00,1.04E+02) resulting in fit range (7.69E-01,1.70E+00)
    min_max_norm_parameter = (
        1.0,
        8.209328260330016,
    )  # error of 2.75E-02 with sample range (1.00E+00,1.04E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        4.438473593321477,
        0.6021034304168726,
    )  # error of 3.17E-02 with sample range (1.00E+00,1.04E+02) resulting in fit range (1.12E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        5.4466553290184425,
        0.06194278787774897,
        1.2336403796172142,
    )  # error of 1.04E-02 with sample range (1.00E+00,1.04E+02) resulting in fit range (4.32E-01,1.00E+00)
    genlog_norm_parameter = (
        2.7647461513749936,
        7.1364329171323755,
        10.881150912883296,
        14.617629080229056,
    )  # error of 2.81E-02 with sample range (1.00E+00,1.04E+02) resulting in fit range (2.66E-01,1.00E+00)
    preferred_normalization = "min_max"
    # functions
    def featurize(self, atom):
        return atom.GetAtomicNum()


class ChiralTag_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.4971581730383765,
        0.4971581730383764,
    )  # error of 3.33E-16 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.97E-01,1.49E+00)
    min_max_norm_parameter = (
        5.02858074974806e-11,
        1.0057161498490403,
    )  # error of 4.20E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        -1.2725417783910495,
        2.2725417620984265,
    )  # error of 8.47E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.47E-01,9.99E-01)
    dual_sigmoidal_norm_parameter = (
        -1.2725417783910495,
        1.0,
        2.2725417620984265,
    )  # error of 8.47E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.47E-01,9.99E-01)
    genlog_norm_parameter = (
        2.0134247792522393,
        -1.0461344023149428,
        1.0042191161311258,
        2.8396377685948124,
    )  # error of 2.65E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.60E-01,9.99E-01)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetChiralTag()


class Degree_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.21510347394011986,
        0.07042296989744401,
    )  # error of 2.62E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.04E-02,1.36E+00)
    min_max_norm_parameter = (
        0.0,
        4.191504874207649,
    )  # error of 2.94E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        1.9480446575884445,
        1.0386012197927859,
    )  # error of 3.01E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.17E-01,9.85E-01)
    dual_sigmoidal_norm_parameter = (
        1.8783054670996062,
        1.1949738673146972,
        0.9632130237262879,
    )  # error of 2.79E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.58E-02,9.81E-01)
    genlog_norm_parameter = (
        0.9629257465127755,
        1.6269099278663834,
        0.8579344914576629,
        0.7152347683470208,
    )  # error of 2.99E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.02E-01,9.82E-01)
    preferred_normalization = "linear"
    # functions
    def featurize(self, atom):
        return atom.GetDegree()


class ExplicitValence_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.17923265379642528,
        0.08909943356358807,
    )  # error of 4.04E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.91E-02,1.16E+00)
    min_max_norm_parameter = (
        0.0,
        4.832680136118196,
    )  # error of 4.57E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        2.240070810763014,
        0.851076484107633,
    )  # error of 6.11E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.29E-01,9.61E-01)
    dual_sigmoidal_norm_parameter = (
        2.642658101834148,
        0.5431504905803756,
        1.1640070412245809,
    )  # error of 5.95E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.92E-01,9.80E-01)
    genlog_norm_parameter = (
        7.165199957404065,
        4.642562583727061,
        1.189311186214621,
        22.645542010176428,
    )  # error of 5.04E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (2.28E-01,1.00E+00)
    preferred_normalization = "linear"
    # functions
    def featurize(self, atom):
        return atom.GetExplicitValence()


class FormalCharge_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.4987622900746891,
        0.5000335024142983,
    )  # error of 2.76E-05 with sample range (-2.00E+00,4.00E+00) resulting in fit range (-4.97E-01,2.50E+00)
    min_max_norm_parameter = (
        -1.0025487338270875,
        1.0024143916187587,
    )  # error of 2.76E-05 with sample range (-2.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        -4.34449804066779e-05,
        6.6932508101710075,
    )  # error of 1.11E-05 with sample range (-2.00E+00,4.00E+00) resulting in fit range (1.54E-06,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        -4.3292208591961126e-05,
        6.682303151119855,
        6.704331282449926,
    )  # error of 2.50E-19 with sample range (-2.00E+00,4.00E+00) resulting in fit range (1.57E-06,1.00E+00)
    genlog_norm_parameter = (
        6.7060573772447185,
        0.14316608576162523,
        0.3851293892097991,
        1.0044754547744847,
    )  # error of 2.83E-11 with sample range (-2.00E+00,4.00E+00) resulting in fit range (1.58E-06,1.00E+00)
    preferred_normalization = "dual_sig"
    # functions
    def featurize(self, atom):
        return atom.GetFormalCharge()


class Hybridization_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.14756938813571396,
        0.24990799389634322,
    )  # error of 6.60E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (2.50E-01,1.14E+00)
    min_max_norm_parameter = (
        0.0,
        4.331403424814428,
    )  # error of 9.19E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        1.6862276042276507,
        0.6223006149330815,
    )  # error of 7.65E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (2.59E-01,9.36E-01)
    dual_sigmoidal_norm_parameter = (
        2.662352221138141,
        0.09545492194749604,
        1.6650618258506025,
    )  # error of 8.26E-03 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.37E-01,9.96E-01)
    genlog_norm_parameter = (
        20.946173814469617,
        4.457607111820098,
        4.005082090741002,
        80.93487501096364,
    )  # error of 4.75E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (3.10E-01,1.00E+00)
    preferred_normalization = "linear"
    # functions
    def featurize(self, atom):
        return atom.GetHybridization()


class ImplicitValence_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetImplicitValence()


class IsAromatic_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.bool
    # normalization
    linear_norm_parameter = (
        1.0000000000000002,
        2.220446049250313e-16,
    )  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    min_max_norm_parameter = (
        1e-10,
        0.9999999999,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.5027041798086037,
        46.14043898879842,
    )  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        0.49954642862905374,
        18.83817716761341,
        18.88166255189391,
    )  # error of 8.03E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.18E-05,1.00E+00)
    genlog_norm_parameter = (
        12.011496984688954,
        -0.47958199521284717,
        1.1726250998531205,
        0.0003411801343887883,
    )  # error of 4.87E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.03E-05,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetIsAromatic()


class Isotope_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.029493590375225188,
        0.6004761650946653,
    )  # error of 2.58E-01 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.00E-01,1.13E+00)
    min_max_norm_parameter = (
        6.895111112897188e-09,
        2.000245873323907,
    )  # error of 2.47E-05 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        1.2831390901502053,
        12.56009418804053,
    )  # error of 2.47E-05 with sample range (0.00E+00,1.80E+01) resulting in fit range (1.00E-07,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        0.0017089977049821869,
        9868.732954500145,
        4.502962331040455,
    )  # error of 2.44E-05 with sample range (0.00E+00,1.80E+01) resulting in fit range (4.74E-08,1.00E+00)
    genlog_norm_parameter = (
        9.058114615966034,
        1.0005388958791346,
        0.9998917858992039,
        0.8637373152840924,
    )  # error of 2.67E-05 with sample range (0.00E+00,1.80E+01) resulting in fit range (2.77E-05,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetIsotope()


class Mass_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.float32
    # normalization
    linear_norm_parameter = (
        0.0052764480368310196,
        0.7051728546204995,
    )  # error of 1.45E-01 with sample range (1.01E+00,2.39E+02) resulting in fit range (7.10E-01,1.97E+00)
    min_max_norm_parameter = (
        1.0080000162124636,
        16.421243017727168,
    )  # error of 7.04E-02 with sample range (1.01E+00,2.39E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        7.38704762106774,
        0.21789296783391315,
    )  # error of 6.18E-02 with sample range (1.01E+00,2.39E+02) resulting in fit range (1.99E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        10.812000236362453,
        0.01889106708112699,
        0.7147951319959395,
    )  # error of 3.27E-02 with sample range (1.01E+00,2.39E+02) resulting in fit range (4.54E-01,1.00E+00)
    genlog_norm_parameter = (
        1.342888160493401,
        13.505780673674437,
        18.683441716266238,
        13.929276142424545,
    )  # error of 5.42E-02 with sample range (1.01E+00,2.39E+02) resulting in fit range (2.43E-01,1.00E+00)
    preferred_normalization = "genlog"
    # functions
    def featurize(self, atom):
        return atom.GetMass()


class NoImplicit_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.bool
    # normalization
    linear_norm_parameter = (
        0.5000000000000002,
        0.5,
    )  # error of 2.22E-16 with sample range (1.00E+00,1.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        -2.204574051588843,
        3.2045762347336524,
    )  # error of 3.47E-05 with sample range (1.00E+00,1.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        -2.204574051588843,
        1.0,
        3.2045762347336524,
    )  # error of 3.47E-05 with sample range (1.00E+00,1.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter = (
        1.1094116429570557,
        0.27672595780434966,
        0.0,
        1.8479760859001104,
    )  # error of 0.00E+00 with sample range (1.00E+00,1.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetNoImplicit()


class NumExplicitHs_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetNumExplicitHs()


class NumImplicitHs_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetNumImplicitHs()


class NumRadicalElectrons_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.20001830461402503,
        0.3999117300045151,
    )  # error of 2.83E-01 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.00E-01,1.20E+00)
    min_max_norm_parameter = (
        5.7159811313222985e-09,
        1.000195222581766,
    )  # error of 2.34E-05 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.677151562296285,
        26.456244730682627,
    )  # error of 2.34E-05 with sample range (0.00E+00,4.00E+00) resulting in fit range (1.66E-08,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        0.028355681961328934,
        571.9813356311003,
        8.790580791815003,
    )  # error of 2.34E-05 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.04E-08,1.00E+00)
    genlog_norm_parameter = (
        9.193079467327067,
        -0.988196206861674,
        1.4626396133889485,
        2.0171938209017733e-05,
    )  # error of 3.12E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (2.69E-04,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetNumRadicalElectrons()


class TotalDegree_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.2152339489171452,
        0.07028846843175673,
    )  # error of 2.62E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.03E-02,1.36E+00)
    min_max_norm_parameter = (
        0.0,
        4.190000809228329,
    )  # error of 2.94E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        1.9473829991859444,
        1.0397355172680758,
    )  # error of 3.01E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.17E-01,9.85E-01)
    dual_sigmoidal_norm_parameter = (
        1.8778289232473346,
        1.1958938244175974,
        0.9644106718674091,
    )  # error of 2.79E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.57E-02,9.82E-01)
    genlog_norm_parameter = (
        0.964190295894188,
        1.4351443015517322,
        1.0334044207076438,
        0.7160145552455551,
    )  # error of 2.99E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.02E-01,9.83E-01)
    preferred_normalization = "linear"
    # functions
    def featurize(self, atom):
        return atom.GetTotalDegree()


class TotalNumHs_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.GetTotalNumHs()


class TotalValence_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.17934599363141746,
        0.08859334072593106,
    )  # error of 4.04E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.86E-02,1.16E+00)
    min_max_norm_parameter = (
        0.0,
        4.834430353944814,
    )  # error of 4.56E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        2.2415731463905733,
        0.8514804233169152,
    )  # error of 6.11E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.29E-01,9.61E-01)
    dual_sigmoidal_norm_parameter = (
        2.645914809758109,
        0.5424327295654923,
        1.1670910362795361,
    )  # error of 5.95E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.92E-01,9.80E-01)
    genlog_norm_parameter = (
        7.314002682276351,
        4.3801650414542515,
        8.144865093797915,
        23.097608758230557,
    )  # error of 5.04E-02 with sample range (0.00E+00,6.00E+00) resulting in fit range (2.28E-01,1.00E+00)
    preferred_normalization = "linear"
    # functions
    def featurize(self, atom):
        return atom.GetTotalValence()


class IsInRing_Featurizer(SingleValueAtomFeaturizer):
    # statics
    dtype = np.bool
    # normalization
    linear_norm_parameter = (
        1.0000000000000002,
        2.220446049250313e-16,
    )  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    min_max_norm_parameter = (
        1e-10,
        0.9999999999,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        0.5027041798086037,
        46.14043898879842,
    )  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        0.49954642862905374,
        18.83817716761341,
        18.88166255189391,
    )  # error of 8.03E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.18E-05,1.00E+00)
    genlog_norm_parameter = (
        12.011496984688954,
        -0.47958199521284717,
        1.1726250998531205,
        0.0003411801343887883,
    )  # error of 4.87E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.03E-05,1.00E+00)
    preferred_normalization = "unity"
    # functions
    def featurize(self, atom):
        return atom.IsInRing()


atom_Chi1 = Chi1_Featurizer()
atom_Chi1 = Chi1_Featurizer()
atom_AtomPairAtomCode = AtomPairAtomCode_Featurizer()
atom_AtomMapNum = AtomMapNum_Featurizer()
atom_AtomicNum = AtomicNum_Featurizer()
atom_ChiralTag = ChiralTag_Featurizer()
atom_Degree = Degree_Featurizer()
atom_ExplicitValence = ExplicitValence_Featurizer()
atom_FormalCharge = FormalCharge_Featurizer()
atom_Hybridization = Hybridization_Featurizer()
atom_ImplicitValence = ImplicitValence_Featurizer()
atom_IsAromatic = IsAromatic_Featurizer()
atom_Isotope = Isotope_Featurizer()
atom_Mass = Mass_Featurizer()
atom_NoImplicit = NoImplicit_Featurizer()
atom_NumExplicitHs = NumExplicitHs_Featurizer()
atom_NumImplicitHs = NumImplicitHs_Featurizer()
atom_NumRadicalElectrons = NumRadicalElectrons_Featurizer()
atom_TotalDegree = TotalDegree_Featurizer()
atom_TotalNumHs = TotalNumHs_Featurizer()
atom_TotalValence = TotalValence_Featurizer()
atom_IsInRing = IsInRing_Featurizer()

_available_featurizer = {
    "atom_Chi1": atom_Chi1,
    "atom_Chi1": atom_Chi1,
    "atom_AtomPairAtomCode": atom_AtomPairAtomCode,
    "atom_AtomMapNum": atom_AtomMapNum,
    "atom_AtomicNum": atom_AtomicNum,
    "atom_ChiralTag": atom_ChiralTag,
    "atom_Degree": atom_Degree,
    "atom_ExplicitValence": atom_ExplicitValence,
    "atom_FormalCharge": atom_FormalCharge,
    "atom_Hybridization": atom_Hybridization,
    "atom_ImplicitValence": atom_ImplicitValence,
    "atom_IsAromatic": atom_IsAromatic,
    "atom_Isotope": atom_Isotope,
    "atom_Mass": atom_Mass,
    "atom_NoImplicit": atom_NoImplicit,
    "atom_NumExplicitHs": atom_NumExplicitHs,
    "atom_NumImplicitHs": atom_NumImplicitHs,
    "atom_NumRadicalElectrons": atom_NumRadicalElectrons,
    "atom_TotalDegree": atom_TotalDegree,
    "atom_TotalNumHs": atom_TotalNumHs,
    "atom_TotalValence": atom_TotalValence,
    "atom_IsInRing": atom_IsInRing,
}


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k)
        f(testmol.GetAtoms()[0])


if __name__ == "__main__":
    main()
