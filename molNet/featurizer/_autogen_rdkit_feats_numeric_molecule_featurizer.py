import numpy as np
from numpy import inf
from rdkit.Chem import (
    rdqueries,
)
from rdkit.Chem.Descriptors import (
    VSA_EState4,
    fr_C_O,
    SlogP_VSA5,
    EState_VSA7,
    fr_benzene,
    HeavyAtomMolWt,
    fr_priamide,
    fr_lactam,
    fr_oxazole,
    fr_guanido,
    MolLogP,
    fr_unbrch_alkane,
    SlogP_VSA9,
    MolMR,
    fr_benzodiazepine,
    fr_NH1,
    fr_oxime,
    fr_thiophene,
    Chi3n,
    fr_piperzine,
    BCUT2D_LOGPLOW,
    SlogP_VSA2,
    EState_VSA8,
    fr_Al_OH,
    fr_C_O_noCOO,
    BCUT2D_MWHI,
    SMR_VSA1,
    VSA_EState5,
    MinPartialCharge,
    Chi1,
    EState_VSA1,
    EState_VSA3,
    SMR_VSA6,
    NOCount,
    fr_azide,
    fr_imidazole,
    Chi4v,
    MinAbsEStateIndex,
    BCUT2D_MRHI,
    fr_aryl_methyl,
    fr_lactone,
    SlogP_VSA7,
    fr_ether,
    PEOE_VSA11,
    SlogP_VSA11,
    Kappa1,
    fr_Nhpyrrole,
    NumSaturatedRings,
    fr_hdrzone,
    BCUT2D_CHGHI,
    fr_phos_acid,
    fr_term_acetylene,
    NumAliphaticHeterocycles,
    fr_para_hydroxylation,
    fr_phenol_noOrthoHbond,
    fr_morpholine,
    fr_COO2,
    fr_Ar_OH,
    PEOE_VSA1,
    PEOE_VSA12,
    qed,
    VSA_EState8,
    fr_HOCCN,
    fr_prisulfonamd,
    fr_imide,
    VSA_EState6,
    FractionCSP3,
    VSA_EState3,
    fr_bicyclic,
    MaxEStateIndex,
    VSA_EState2,
    fr_barbitur,
    PEOE_VSA10,
    NumValenceElectrons,
    EState_VSA11,
    fr_dihydropyridine,
    SlogP_VSA12,
    fr_sulfide,
    fr_Ar_COO,
    SlogP_VSA10,
    fr_azo,
    fr_Ar_NH,
    PEOE_VSA7,
    fr_N_O,
    Chi1v,
    fr_nitro_arom_nonortho,
    MaxAbsPartialCharge,
    BCUT2D_CHGLO,
    fr_pyridine,
    fr_nitro,
    fr_thiazole,
    BCUT2D_LOGPHI,
    EState_VSA4,
    fr_Imine,
    EState_VSA5,
    fr_Ndealkylation1,
    fr_isothiocyan,
    fr_ketone,
    MinEStateIndex,
    EState_VSA10,
    HeavyAtomCount,
    SMR_VSA7,
    NHOHCount,
    fr_halogen,
    VSA_EState10,
    fr_piperdine,
    EState_VSA2,
    SMR_VSA2,
    NumAliphaticRings,
    SMR_VSA9,
    EState_VSA6,
    fr_alkyl_carbamate,
    fr_hdrzine,
    RingCount,
    SlogP_VSA1,
    fr_nitrile,
    SMR_VSA4,
    fr_COO,
    fr_urea,
    fr_ArN,
    fr_aldehyde,
    fr_Ar_N,
    VSA_EState1,
    FpDensityMorgan3,
    fr_Al_COO,
    fr_allylic_oxid,
    VSA_EState7,
    SlogP_VSA4,
    PEOE_VSA5,
    fr_SH,
    SMR_VSA10,
    VSA_EState9,
    MolWt,
    fr_tetrazole,
    BCUT2D_MWLOW,
    fr_ketone_Topliss,
    PEOE_VSA4,
    fr_Al_OH_noTert,
    FpDensityMorgan2,
    fr_phenol,
    NumSaturatedHeterocycles,
    fr_NH2,
    fr_sulfonamd,
    fr_thiocyan,
    fr_amidine,
    fr_sulfone,
    SlogP_VSA6,
    PEOE_VSA13,
    fr_epoxide,
    Chi0,
    fr_diazo,
    fr_aniline,
    fr_C_S,
    fr_NH0,
    PEOE_VSA6,
    NumRadicalElectrons,
    SMR_VSA3,
    SMR_VSA8,
    MinAbsPartialCharge,
    fr_methoxy,
    SMR_VSA5,
    fr_nitro_arom,
    TPSA,
    FpDensityMorgan1,
    fr_isocyan,
    fr_quatN,
    SlogP_VSA3,
    fr_ester,
    PEOE_VSA14,
    fr_nitroso,
    NumHAcceptors,
    fr_Ndealkylation2,
    fr_amide,
    NumHDonors,
    PEOE_VSA9,
    MaxPartialCharge,
    SlogP_VSA8,
    MaxAbsEStateIndex,
    PEOE_VSA2,
    BCUT2D_MRLOW,
    fr_phos_ester,
    fr_furan,
    Ipc,
    NumSaturatedCarbocycles,
    fr_alkyl_halide,
    PEOE_VSA8,
    PEOE_VSA3,
    EState_VSA9,
    NumRotatableBonds,
)
from rdkit.Chem.Descriptors3D import (
    SpherocityIndex,
    RadiusOfGyration,
    PMI3,
    PMI1,
    NPR2,
)
from rdkit.Chem.GraphDescriptors import (
    Chi4n,
    Chi0n,
    Kappa2,
    BalabanJ,
    Chi3v,
    Kappa3,
    BertzCT,
    Chi1n,
    HallKierAlpha,
)
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHeterocycles,
    CalcPMI2,
    CalcNumHBD,
    CalcAsphericity,
    CalcEccentricity,
    CalcNumLipinskiHBD,
    CalcChi2v,
    CalcNumSpiroAtoms,
    CalcChi2n,
    CalcNumHeteroatoms,
    CalcNPR1,
    CalcNumAromaticRings,
    CalcNumLipinskiHBA,
    CalcNumAliphaticCarbocycles,
    CalcNumAromaticCarbocycles,
    CalcNumHBA,
    CalcLabuteASA,
    CalcExactMolWt,
    CalcPBF,
    CalcNumBridgeheadAtoms,
    CalcNumRings,
    CalcInertialShapeFactor,
    CalcNumAmideBonds,
    CalcChi0v,
    CalcNumAromaticHeterocycles,
    CalcPhi,
)
from rdkit.Chem.rdmolops import (
    GetFormalCharge,
    GetSSSR,
)

from molNet.featurizer._molecule_featurizer import (
    SingleValueMoleculeFeaturizer,
)


class Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcAsphericity)
    # normalization
    linear_norm_parameter = (
        1.4721378049698663,
        -0.11228706292547166,
    )  # error of 2.90E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-1.12E-01,1.36E+00)
    linear_norm_parameter_normdata = {"error": 0.028962171585907876}
    min_max_norm_parameter = (
        0.09005620590041596,
        0.7355219615267257,
    )  # error of 1.68E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.016794434794666063}
    sigmoidal_norm_parameter = (
        0.40970204866580884,
        7.643961121304289,
    )  # error of 2.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.18E-02,9.89E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.022019959631773577}
    dual_sigmoidal_norm_parameter = (
        0.39938009612685643,
        8.419338230908746,
        6.969158525926429,
    )  # error of 1.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (3.35E-02,9.85E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.019882996117645317}
    genlog_norm_parameter = (
        6.206324202534631,
        0.2751042956173693,
        0.693881632805456,
        0.396584639577746,
    )  # error of 1.81E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.89E-02,9.81E-01)
    genlog_norm_parameter_normdata = {"error": 0.01807082490258409}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [0.018889321268164994, 0.980803887328162]],
        "sample_bounds99": [
            [0.0, 0.7981497049331665],
            [0.03746188537738742, 0.9450478288397537],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class BCUT2D_CHGHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGHI)
    # normalization
    linear_norm_parameter = (
        1.0601079754933302,
        0.40787344122731317,
    )  # error of 1.66E-01 with sample range (-1.56E+00,3.00E+00) resulting in fit range (-1.24E+00,3.59E+00)
    linear_norm_parameter_normdata = {"error": 0.16588688478022787}
    min_max_norm_parameter = (
        -0.09795833235878999,
        0.13496394474519627,
    )  # error of 8.61E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.08612694253715643}
    sigmoidal_norm_parameter = (
        0.015414812733986287,
        19.410301428357236,
    )  # error of 8.02E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (5.54E-14,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.08019579552551206}
    dual_sigmoidal_norm_parameter = (
        -0.00807343779216827,
        43.35367999823987,
        6.98718326256343,
    )  # error of 3.60E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (6.81E-30,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.03603525374776613}
    genlog_norm_parameter = (
        10.493839669900748,
        -1.3802885737772868,
        1.0884738608627027,
        6.495230094877747e-07,
    )  # error of 7.25E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0724558749076135}
    autogen_normdata = {
        "sample_bounds": [[-1.5571801662445068, 3.0023117065429688], [0.0, 1.0]],
        "sample_bounds99": [
            [-1.5571801662445068, 0.6839888095855713],
            [0.00016265514524111847, 0.9999024251691604],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class BCUT2D_CHGLO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGLO)
    # normalization
    linear_norm_parameter = (
        1.7653570916472574,
        5.065686201381148,
    )  # error of 5.60E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (-4.74E-01,2.57E+00)
    linear_norm_parameter_normdata = {"error": 0.05604662825701627}
    min_max_norm_parameter = (
        -2.8372498211712145,
        -2.348799925638046,
    )  # error of 3.18E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03182842007660005}
    sigmoidal_norm_parameter = (
        -2.597093268685726,
        10.060474784720004,
    )  # error of 2.27E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (4.32E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02273406766810744}
    dual_sigmoidal_norm_parameter = (
        -2.610242935462682,
        12.02832694648774,
        8.465657724472262,
    )  # error of 1.37E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (1.75E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013683293171572517}
    genlog_norm_parameter = (
        5.193530459418791,
        0.1453888097019428,
        6.141574044323054e-08,
        0.13043665465404497,
    )  # error of 6.19E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (7.37E-04,9.98E-01)
    genlog_norm_parameter_normdata = {"error": 0.061885906119964834}
    autogen_normdata = {
        "sample_bounds": [
            [-3.1378395557403564, -1.4145506620407104],
            [0.0007373030198308448, 0.9984476989664517],
        ],
        "sample_bounds99": [
            [-3.1378395557403564, -2.246455669403076],
            [0.04460306948004584, 0.9078039323878682],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class BCUT2D_LOGPHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPHI)
    # normalization
    linear_norm_parameter = (
        0.9221221942827517,
        0.3462995547380183,
    )  # error of 1.18E-01 with sample range (-1.76E+00,2.74E+00) resulting in fit range (-1.27E+00,2.88E+00)
    linear_norm_parameter_normdata = {"error": 0.11825029626886543}
    min_max_norm_parameter = (
        -0.2074553870185299,
        0.45765097151660844,
    )  # error of 5.19E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05188646032013021}
    sigmoidal_norm_parameter = (
        0.1195132064019319,
        7.293530987809198,
    )  # error of 3.57E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (1.14E-06,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0357090774835128}
    dual_sigmoidal_norm_parameter = (
        0.1022300315565414,
        8.561773964095105,
        6.034403781508101,
    )  # error of 3.18E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (1.23E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.03183616481322525}
    genlog_norm_parameter = (
        5.109862650525073,
        -0.6107396029217007,
        1.187734747333248,
        0.0428023289659632,
    )  # error of 2.84E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (6.73E-62,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.028406290956909275}
    autogen_normdata = {
        "sample_bounds": [
            [-1.756446361541748, 2.7425131797790527],
            [6.730503472817432e-62, 0.9999989959321832],
        ],
        "sample_bounds99": [
            [-1.756446361541748, 0.8799106478691101],
            [0.003507278007523965, 0.9938760169404552],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class BCUT2D_LOGPLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPLOW)
    # normalization
    linear_norm_parameter = (
        1.7051602905317909,
        4.918269713274668,
    )  # error of 6.24E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (-9.14E-01,2.89E+00)
    linear_norm_parameter_normdata = {"error": 0.06244251482007325}
    min_max_norm_parameter = (
        -2.845292207717101,
        -2.3529603749416776,
    )  # error of 3.19E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.031877589534722486}
    sigmoidal_norm_parameter = (
        -2.603156460495485,
        9.951733084541459,
    )  # error of 2.24E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (2.94E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0224054519558156}
    dual_sigmoidal_norm_parameter = (
        -2.616876569972246,
        11.992922320816978,
        8.31768491859481,
    )  # error of 1.22E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (6.54E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.012203969919723046}
    genlog_norm_parameter = (
        5.184707528824923,
        0.14549139613726714,
        6.191112419316223e-08,
        0.1323972302992638,
    )  # error of 5.98E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (2.19E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.05977339629404716}
    autogen_normdata = {
        "sample_bounds": [
            [-3.420210838317871, -1.1916742324829102],
            [2.1882793196230041e-07, 0.999520648384412],
        ],
        "sample_bounds99": [
            [-3.420210838317871, -2.2184712886810303],
            [0.03012141619963804, 0.9226401060489251],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class BCUT2D_MRHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRHI)
    # normalization
    linear_norm_parameter = (
        0.2760716898709984,
        -0.05891671770132878,
    )  # error of 3.51E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (-2.01E-01,3.86E+00)
    linear_norm_parameter_normdata = {"error": 0.0351029871782958}
    min_max_norm_parameter = (
        0.3604159078021756,
        3.7266268387929946,
    )  # error of 1.95E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019473664706093585}
    sigmoidal_norm_parameter = (
        2.042877009089624,
        1.4588708297468977,
    )  # error of 2.02E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.34E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.020160591172873606}
    dual_sigmoidal_norm_parameter = (
        2.0470603177042705,
        1.4477543462395674,
        1.4702724365676079,
    )  # error of 2.01E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.39E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.020146447895633596}
    genlog_norm_parameter = (
        1.472662885867806,
        1.1093757659778625,
        4.1222192700056075,
        1.0286052647630917,
    )  # error of 2.02E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.42E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0201536475203664}
    autogen_normdata = {
        "sample_bounds": [
            [-0.5141971707344055, 14.200179100036621],
            [0.024165286849448346, 0.9999999830016973],
        ],
        "sample_bounds99": [
            [-0.5141971707344055, 3.6541318893432617],
            [0.04927767657189255, 0.9345802683789772],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class BCUT2D_MRLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRLOW)
    # normalization
    linear_norm_parameter = (
        1.1473576459333634,
        3.176799142930221,
    )  # error of 5.86E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (-3.35E-01,9.95E+00)
    linear_norm_parameter_normdata = {"error": 0.05855995315621425}
    min_max_norm_parameter = (
        -2.7210155512608267,
        -1.9615807430024825,
    )  # error of 2.64E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.026446697108850236}
    sigmoidal_norm_parameter = (
        -2.3468285836092604,
        6.523147526382408,
    )  # error of 1.96E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (9.42E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.019611762692767622}
    dual_sigmoidal_norm_parameter = (
        -2.3658089847913875,
        7.708768374105313,
        5.6046994729986075,
    )  # error of 1.15E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (4.70E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011464737322801377}
    genlog_norm_parameter = (
        5.026719131818521,
        -0.15480604800294004,
        2.914165119287385e-06,
        0.2480961925505442,
    )  # error of 9.54E-03 with sample range (-3.06E+00,5.91E+00) resulting in fit range (3.09E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009536972098255395}
    autogen_normdata = {
        "sample_bounds": [
            [-3.0605123043060303, 5.9063401222229],
            [0.00030932829203077103, 1.0],
        ],
        "sample_bounds99": [
            [-3.0605123043060303, -1.816205620765686],
            [0.014634330591433209, 0.9704266659736424],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class BCUT2D_MWHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWHI)
    # normalization
    linear_norm_parameter = (
        0.06382980925203052,
        0.0914070582631521,
    )  # error of 7.74E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (8.35E-02,8.19E+00)
    linear_norm_parameter_normdata = {"error": 0.07742742784060243}
    min_max_norm_parameter = (
        -0.12367738038301467,
        13.619949109255183,
    )  # error of 9.41E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.09410831068091396}
    sigmoidal_norm_parameter = (
        5.983255681107075,
        0.32288774435232176,
    )  # error of 8.34E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (1.22E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.08337918526777212}
    dual_sigmoidal_norm_parameter = (
        2.7970938163841925,
        1.1803039065034748,
        0.20362210160668412,
    )  # error of 6.07E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (3.08E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.06074502902519052}
    genlog_norm_parameter = (
        0.24662148087773003,
        -11.03145426668446,
        0.0018951413133585643,
        5.2108085494788376e-05,
    )  # error of 7.49E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (8.47E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.07491246038368224}
    autogen_normdata = {
        "sample_bounds": [
            [-0.12367738038301468, 126.92240905761719],
            [0.08471988715450136, 1.0],
        ],
        "sample_bounds99": [
            [-0.12367738038301468, 12.870870590209961],
            [0.0912405247807563, 0.9283442024639362],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class BCUT2D_MWLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWLOW)
    # normalization
    linear_norm_parameter = (
        1.018429176169886,
        2.7560478377646005,
    )  # error of 6.41E-02 with sample range (-3.04E+00,3.04E+01) resulting in fit range (-3.35E-01,3.38E+01)
    linear_norm_parameter_normdata = {"error": 0.06411159701409647}
    min_max_norm_parameter = (
        -2.64231464861815,
        -1.799802847378942,
    )  # error of 2.41E-02 with sample range (-3.04E+00,3.04E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.024137921551886397}
    sigmoidal_norm_parameter = (
        -2.2221798051728805,
        5.9554575147054996,
    )  # error of 9.16E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (7.81E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.009158707113873408}
    dual_sigmoidal_norm_parameter = (
        -2.2278580763025553,
        6.237162365182138,
        5.69565286208031,
    )  # error of 7.91E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (6.45E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.007905003961711855}
    genlog_norm_parameter = (
        5.497327278489337,
        -0.3469721711960865,
        2.2754995989973485e-05,
        0.7612352369837492,
    )  # error of 7.69E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (4.55E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007694037649734914}
    autogen_normdata = {
        "sample_bounds": [
            [-3.035536289215088, 30.4495849609375],
            [0.00454553999172402, 1.0],
        ],
        "sample_bounds99": [
            [-3.035536289215088, -1.6671724319458008],
            [0.02848681057703966, 0.9830797991743503],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BalabanJ)
    # normalization
    linear_norm_parameter = (
        0.17660976361946912,
        0.011623897998747967,
    )  # error of 1.60E-01 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.16E-02,8.95E+00)
    linear_norm_parameter_normdata = {"error": 0.15973504817337922}
    min_max_norm_parameter = (
        1.51069451962931,
        3.7663482034024947,
    )  # error of 4.43E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04432071908453417}
    sigmoidal_norm_parameter = (
        2.6275405979312487,
        2.067375370708902,
    )  # error of 2.69E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (4.35E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02693968746344809}
    dual_sigmoidal_norm_parameter = (
        2.5486119874079707,
        2.6623154651762686,
        1.6193606097747983,
    )  # error of 1.47E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.13E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.014749481272315054}
    genlog_norm_parameter = (
        1.4473269011318037,
        -3.423175107360796,
        0.4668648634358428,
        0.00011138512904900963,
    )  # error of 1.47E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.52E-13,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.014735712181616116}
    autogen_normdata = {
        "sample_bounds": [
            [-7.44194258004427e-05, 50.58443832397461],
            [1.5237535890560646e-13, 1.0],
        ],
        "sample_bounds99": [
            [-7.44194258004427e-05, 7.896667957305908],
            [1.52861806529418e-13, 0.9998633763465247],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(BertzCT)
    # normalization
    linear_norm_parameter = (
        0.0002435689316791905,
        0.08402337833593164,
    )  # error of 1.62E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (8.40E-02,1.15E+01)
    linear_norm_parameter_normdata = {"error": 0.16229801806921698}
    min_max_norm_parameter = (
        589.3484428971769,
        2551.6280739737376,
    )  # error of 3.81E-02 with sample range (0.00E+00,4.67E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03805862869115655}
    sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
    )  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.573752388363361}
    dual_sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
        1.0,
    )  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.573752388363361}
    genlog_norm_parameter = (
        0.0017098593375205584,
        -843.1415613946119,
        0.0010786968654881758,
        2.7454687544914185e-05,
    )  # error of 9.82E-03 with sample range (0.00E+00,4.67E+04) resulting in fit range (9.21E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009821748253533272}
    autogen_normdata = {
        "sample_bounds": [[0.0, 46697.1484375], [9.212669327697183e-05, 1.0]],
        "sample_bounds99": [
            [0.0, 4777.64453125],
            [0.010694724668148848, 0.9995771355136788],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi0)
    # normalization
    linear_norm_parameter = (
        0.007513574566740955,
        0.2957974579215942,
    )  # error of 2.01E-01 with sample range (0.00E+00,7.19E+02) resulting in fit range (2.96E-01,5.70E+00)
    linear_norm_parameter_normdata = {"error": 0.2009157348352737}
    min_max_norm_parameter = (
        17.447753846956868,
        55.71754607510952,
    )  # error of 4.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04702712109798194}
    sigmoidal_norm_parameter = (
        36.35211468883877,
        0.1295884036148426,
    )  # error of 2.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (8.92E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02698700982591915}
    dual_sigmoidal_norm_parameter = (
        34.781883667706126,
        0.18372919698120913,
        0.1052982438114165,
    )  # error of 1.38E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (1.67E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013765009473759587}
    genlog_norm_parameter = (
        0.09445935703723156,
        -40.68840587473802,
        0.052847704469068994,
        5.66023636095279e-05,
    )  # error of 1.10E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (2.09E-09,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.01101534681220656}
    autogen_normdata = {
        "sample_bounds": [[0.0, 719.1281127929688], [2.0851897391681524e-09, 1.0]],
        "sample_bounds99": [
            [0.0, 106.84493255615234],
            [0.0051956291074414995, 0.9997024207098082],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi0n)
    # normalization
    linear_norm_parameter = (
        0.009618083281699108,
        0.17347984942891348,
    )  # error of 2.14E-01 with sample range (7.63E-01,6.43E+02) resulting in fit range (1.81E-01,6.36E+00)
    linear_norm_parameter_normdata = {"error": 0.21447791682097878}
    min_max_norm_parameter = (
        16.004022253403853,
        51.11689241252734,
    )  # error of 4.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04311719433749951}
    sigmoidal_norm_parameter = (
        33.37586607594709,
        0.1432287247282441,
    )  # error of 2.67E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.28E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026654493203100007}
    dual_sigmoidal_norm_parameter = (
        31.988365552286066,
        0.18812033153864993,
        0.11360901005668378,
    )  # error of 1.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (2.80E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013091155898891408}
    genlog_norm_parameter = (
        0.09991792382506794,
        -24.750452023618326,
        0.8473827278488687,
        0.003956246904308439,
    )  # error of 1.01E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.17E-08,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.010106091422280545}
    autogen_normdata = {
        "sample_bounds": [
            [0.7634413838386536, 643.3276977539062],
            [9.165519374139487e-08, 1.0],
        ],
        "sample_bounds99": [
            [0.7634413838386536, 88.63299560546875],
            [0.003827904879572938, 0.99969464239547],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcChi0v)
    # normalization
    linear_norm_parameter = (
        0.027431175492895643,
        0.1959231987976101,
    )  # error of 1.84E-01 with sample range (1.00E+00,2.77E+02) resulting in fit range (2.23E-01,7.80E+00)
    linear_norm_parameter_normdata = {"error": 0.18435911083822928}
    min_max_norm_parameter = (
        5.833583645428592,
        19.149899507438242,
    )  # error of 4.23E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0423165024732474}
    sigmoidal_norm_parameter = (
        12.391158701361956,
        0.36836959542101233,
    )  # error of 2.62E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (1.48E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02620473856261326}
    dual_sigmoidal_norm_parameter = (
        11.842303403027104,
        0.5140666711896098,
        0.2991216617631474,
    )  # error of 1.16E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (3.78E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011572965050369354}
    genlog_norm_parameter = (
        0.2681312003670772,
        -18.83996729447344,
        0.18053251723602642,
        6.445742816927413e-05,
    )  # error of 7.72E-03 with sample range (1.00E+00,2.77E+02) resulting in fit range (1.12E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007721812904196733}
    autogen_normdata = {
        "sample_bounds": [[1.0, 277.3277282714844], [1.120763376208642e-06, 1.0]],
        "sample_bounds99": [
            [1.0, 33.21092224121094],
            [0.010061172965595765, 0.9987700087476209],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi1)
    # normalization
    linear_norm_parameter = (
        0.021402601422805922,
        0.057358382906918415,
    )  # error of 1.75E-01 with sample range (0.00E+00,4.28E+02) resulting in fit range (5.74E-02,9.23E+00)
    linear_norm_parameter_normdata = {"error": 0.17466614265327193}
    min_max_norm_parameter = (
        10.35770293280183,
        30.318616722974067,
    )  # error of 4.02E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.040197831953247984}
    sigmoidal_norm_parameter = (
        20.218648442527503,
        0.24321989118039597,
    )  # error of 2.60E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (7.26E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0260019712753007}
    dual_sigmoidal_norm_parameter = (
        19.497970716081007,
        0.31854028517260596,
        0.1945431362682854,
    )  # error of 1.08E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (2.00E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010777766870155912}
    genlog_norm_parameter = (
        0.17155312446420867,
        -25.562386440585016,
        0.11328083867494013,
        6.744819013669663e-05,
    )  # error of 6.97E-03 with sample range (0.00E+00,4.28E+02) resulting in fit range (8.29E-10,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006972462739463424}
    autogen_normdata = {
        "sample_bounds": [[0.0, 428.4106140136719], [8.292217328103303e-10, 1.0]],
        "sample_bounds99": [
            [0.0, 53.370033264160156],
            [0.004867140847930712, 0.9995846495813856],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi1n)
    # normalization
    linear_norm_parameter = (
        0.02460656481380441,
        0.05696382112467857,
    )  # error of 1.83E-01 with sample range (0.00E+00,3.32E+02) resulting in fit range (5.70E-02,8.22E+00)
    linear_norm_parameter_normdata = {"error": 0.18291393291527094}
    min_max_norm_parameter = (
        8.215243805842091,
        25.568763302868422,
    )  # error of 3.90E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.038979148512106314}
    sigmoidal_norm_parameter = (
        16.800375876069985,
        0.2846180730802336,
    )  # error of 2.44E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (8.31E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02441944444484631}
    dual_sigmoidal_norm_parameter = (
        16.22854448035761,
        0.3557006902802079,
        0.22864396656407704,
    )  # error of 1.13E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (3.10E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011309864829293758}
    genlog_norm_parameter = (
        0.20081899416624355,
        -3.9155988161228468,
        2.447646307417477,
        0.05743409500456496,
    )  # error of 7.92E-03 with sample range (0.00E+00,3.32E+02) resulting in fit range (2.17E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007916880713483804}
    autogen_normdata = {
        "sample_bounds": [[0.0, 331.6630859375], [2.168463774955951e-06, 1.0]],
        "sample_bounds99": [
            [0.0, 43.43079376220703],
            [0.005005294570965555, 0.9996657899932929],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi1v)
    # normalization
    linear_norm_parameter = (
        0.0627452397151258,
        0.08691967075790497,
    )  # error of 1.60E-01 with sample range (0.00E+00,1.66E+02) resulting in fit range (8.69E-02,1.05E+01)
    linear_norm_parameter_normdata = {"error": 0.15977372838461698}
    min_max_norm_parameter = (
        2.910758257183072,
        10.518729378759696,
    )  # error of 3.99E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039869239162642046}
    sigmoidal_norm_parameter = (
        6.661928594684052,
        0.6389477840131871,
    )  # error of 2.54E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (1.40E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.025429672953429426}
    dual_sigmoidal_norm_parameter = (
        6.384313111721113,
        0.855658675292274,
        0.5214673948038986,
    )  # error of 1.05E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (4.22E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01053044473974626}
    genlog_norm_parameter = (
        0.46026801832546155,
        -8.095863453866041,
        0.035194021485851955,
        6.0705507117096385e-05,
    )  # error of 5.40E-03 with sample range (0.00E+00,1.66E+02) resulting in fit range (8.69E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0054043850231123795}
    autogen_normdata = {
        "sample_bounds": [[0.0, 166.4846954345703], [8.689704479646525e-07, 1.0]],
        "sample_bounds99": [
            [0.0, 17.79261589050293],
            [0.009344479363126993, 0.9985030616467082],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcChi2n)
    # normalization
    linear_norm_parameter = (
        0.11014910768339037,
        0.02748881833733474,
    )  # error of 1.49E-01 with sample range (0.00E+00,1.02E+02) resulting in fit range (2.75E-02,1.13E+01)
    linear_norm_parameter_normdata = {"error": 0.14882496284018512}
    min_max_norm_parameter = (
        1.6850371430659572,
        6.553874507410937,
    )  # error of 3.80E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03803702365442874}
    sigmoidal_norm_parameter = (
        4.081549676171799,
        0.9904095492145635,
    )  # error of 2.83E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (1.73E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.028309812384088532}
    dual_sigmoidal_norm_parameter = (
        3.872385716112142,
        1.3409368307127125,
        0.7675552000980483,
    )  # error of 1.08E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (5.53E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010801044847858121}
    genlog_norm_parameter = (
        0.6988519183533591,
        -5.131211636610489,
        0.02667312124620046,
        6.549046586159772e-05,
    )  # error of 7.78E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (1.26E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007780744106234158}
    autogen_normdata = {
        "sample_bounds": [[0.0, 102.18376922607422], [1.2601634282948772e-05, 1.0]],
        "sample_bounds99": [
            [0.0, 11.08464527130127],
            [0.008217298182089881, 0.9987327007021163],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcChi2v)
    # normalization
    linear_norm_parameter = (
        0.09656674011706068,
        0.026475463133179833,
    )  # error of 1.43E-01 with sample range (0.00E+00,1.68E+02) resulting in fit range (2.65E-02,1.62E+01)
    linear_norm_parameter_normdata = {"error": 0.1433450398709747}
    min_max_norm_parameter = (
        1.85040714882094,
        7.55574015613779,
    )  # error of 3.72E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.037213638027546986}
    sigmoidal_norm_parameter = (
        4.659367062804354,
        0.84727788625193,
    )  # error of 2.67E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (1.89E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026665340230128252}
    dual_sigmoidal_norm_parameter = (
        4.4502084375740605,
        1.1067186253730108,
        0.6703972503948212,
    )  # error of 1.05E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (7.21E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010507367144383972}
    genlog_norm_parameter = (
        0.5958613338398937,
        -6.162941168270009,
        0.024553039197838925,
        5.9488626931918526e-05,
    )  # error of 4.82E-03 with sample range (0.00E+00,1.68E+02) resulting in fit range (2.79E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.004820173884202131}
    autogen_normdata = {
        "sample_bounds": [[0.0, 167.90440368652344], [2.7867794576901104e-05, 1.0]],
        "sample_bounds99": [
            [0.0, 12.48746109008789],
            [0.010623137145489153, 0.9983057600452417],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi3n)
    # normalization
    linear_norm_parameter = (
        0.16045617502782894,
        0.06275167455712582,
    )  # error of 1.47E-01 with sample range (0.00E+00,6.23E+01) resulting in fit range (6.28E-02,1.01E+01)
    linear_norm_parameter_normdata = {"error": 0.1467187220413775}
    min_max_norm_parameter = (
        0.8698290958911558,
        4.249166143207709,
    )  # error of 3.75E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03747229058474156}
    sigmoidal_norm_parameter = (
        2.5307853903255304,
        1.4301006733612187,
    )  # error of 2.83E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (2.61E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02833355770371523}
    dual_sigmoidal_norm_parameter = (
        2.39001284835464,
        1.9077778839291673,
        1.1008446280194297,
    )  # error of 1.03E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (1.04E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010338586193370981}
    genlog_norm_parameter = (
        1.0015555418961968,
        -6.993589109683919,
        0.6539760540894591,
        7.215853648046741e-05,
    )  # error of 6.81E-03 with sample range (0.00E+00,6.23E+01) resulting in fit range (2.68E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006807848867536977}
    autogen_normdata = {
        "sample_bounds": [[0.0, 62.314781188964844], [0.0002678339660345653, 1.0]],
        "sample_bounds99": [
            [0.0, 7.702393054962158],
            [0.008353507171546724, 0.9990037203769909],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi3v)
    # normalization
    linear_norm_parameter = (
        0.13641142111605653,
        0.06481842365937796,
    )  # error of 1.43E-01 with sample range (0.00E+00,1.56E+02) resulting in fit range (6.48E-02,2.13E+01)
    linear_norm_parameter_normdata = {"error": 0.14324855083703938}
    min_max_norm_parameter = (
        0.9571111913222607,
        5.030476311519345,
    )  # error of 3.74E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03735973529791995}
    sigmoidal_norm_parameter = (
        2.9588780643186574,
        1.1889256254049316,
    )  # error of 2.82E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (2.88E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.028195893463175727}
    dual_sigmoidal_norm_parameter = (
        2.798221903514353,
        1.5668857301099821,
        0.9198749149438118,
    )  # error of 1.06E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (1.23E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010582126143067557}
    genlog_norm_parameter = (
        0.8306394357062581,
        -8.216329250032336,
        0.5115200078187916,
        7.287918580914142e-05,
    )  # error of 5.78E-03 with sample range (0.00E+00,1.56E+02) resulting in fit range (4.89E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005784909062420294}
    autogen_normdata = {
        "sample_bounds": [[0.0, 155.71441650390625], [0.000488759335530553, 1.0]],
        "sample_bounds99": [
            [0.0, 8.783188819885254],
            [0.011920355611963635, 0.9985455134519171],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi4n)
    # normalization
    linear_norm_parameter = (
        0.2235474556767324,
        0.12057837229443968,
    )  # error of 1.52E-01 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.21E-01,8.63E+00)
    linear_norm_parameter_normdata = {"error": 0.15222099244306292}
    min_max_norm_parameter = (
        0.40708839430881494,
        2.71836618230583,
    )  # error of 3.94E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03939655652763707}
    sigmoidal_norm_parameter = (
        1.5414132407381294,
        2.0927265327645506,
    )  # error of 3.03E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (3.82E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.030256595482341535}
    dual_sigmoidal_norm_parameter = (
        1.4410261354054712,
        2.823564070788039,
        1.5793253269271639,
    )  # error of 1.12E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.68E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011186659825690531}
    genlog_norm_parameter = (
        1.4612311355483178,
        -3.1076538629942276,
        0.036869609927097875,
        6.329448915489527e-05,
    )  # error of 8.27E-03 with sample range (0.00E+00,3.81E+01) resulting in fit range (2.01E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00827292069878479}
    autogen_normdata = {
        "sample_bounds": [[0.0, 38.074581146240234], [0.0020094042543737406, 1.0]],
        "sample_bounds99": [
            [0.0, 5.326207160949707],
            [0.011537014447146662, 0.9993217233815135],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Chi4v)
    # normalization
    linear_norm_parameter = (
        0.1846796920584458,
        0.12308022240634398,
    )  # error of 1.49E-01 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.23E-01,4.41E+01)
    linear_norm_parameter_normdata = {"error": 0.149173411406505}
    min_max_norm_parameter = (
        0.4365031459643245,
        3.319224643809378,
    )  # error of 3.97E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039736569580275584}
    sigmoidal_norm_parameter = (
        1.8494300166801143,
        1.6807292424962799,
    )  # error of 3.13E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (4.28E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.031295890988254055}
    dual_sigmoidal_norm_parameter = (
        1.7217931490660436,
        2.2865005312485036,
        1.2589856485533308,
    )  # error of 1.11E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.91E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011133542768661025}
    genlog_norm_parameter = (
        1.1734746168371561,
        -1.7962003612591946,
        0.0029925667547978287,
        6.355988886134693e-05,
    )  # error of 8.73E-03 with sample range (0.00E+00,2.38E+02) resulting in fit range (3.28E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008734935297766193}
    autogen_normdata = {
        "sample_bounds": [[0.0, 237.9375], [0.003280498218902248, 1.0]],
        "sample_bounds99": [
            [0.0, 6.234282970428467],
            [0.01611400383585105, 0.9989935769817007],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class EState_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA1)
    # normalization
    linear_norm_parameter = (
        0.004092014874662153,
        -0.010698392745672747,
    )  # error of 1.49E-01 with sample range (0.00E+00,3.00E+03) resulting in fit range (-1.07E-02,1.23E+01)
    linear_norm_parameter_normdata = {"error": 0.14934620073488075}
    min_max_norm_parameter = (
        52.7894991169631,
        182.85287865533604,
    )  # error of 3.50E-02 with sample range (0.00E+00,3.00E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.035004668419589914}
    sigmoidal_norm_parameter = (
        117.01950459885676,
        0.03736764559445686,
    )  # error of 2.35E-02 with sample range (0.00E+00,3.00E+03) resulting in fit range (1.25E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02350368145283961}
    dual_sigmoidal_norm_parameter = (
        112.59767673519065,
        0.047080361931645665,
        0.029919335405386053,
    )  # error of 8.64E-03 with sample range (0.00E+00,3.00E+03) resulting in fit range (4.96E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.008643693942784013}
    genlog_norm_parameter = (
        0.026207059091876064,
        -29.32208667290742,
        0.8017806351559116,
        0.026267232610707802,
    )  # error of 5.55E-03 with sample range (0.00E+00,3.00E+03) resulting in fit range (5.93E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005553489185310075}
    autogen_normdata = {
        "sample_bounds": [[0.0, 3003.141845703125], [5.93138718398977e-06, 1.0]],
        "sample_bounds99": [
            [0.0, 312.777099609375],
            [0.0028517984389769503, 0.9991308470156686],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class EState_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA10)
    # normalization
    linear_norm_parameter = (
        0.004005936989037917,
        0.7781446305553839,
    )  # error of 1.04E-01 with sample range (0.00E+00,5.61E+02) resulting in fit range (7.78E-01,3.02E+00)
    linear_norm_parameter_normdata = {"error": 0.10424765084860484}
    min_max_norm_parameter = (
        2.7254874994650347e-24,
        19.011338996425444,
    )  # error of 5.00E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0500331651315478}
    sigmoidal_norm_parameter = (
        7.65981607322896,
        0.1802261811598023,
    )  # error of 1.90E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (2.01E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01898386647826381}
    dual_sigmoidal_norm_parameter = (
        6.86218123444723,
        0.35061452177966596,
        0.16362367621429438,
    )  # error of 1.63E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (8.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.016345315914826627}
    genlog_norm_parameter = (
        0.15082696872418874,
        -47.33232943640999,
        0.21307465523667943,
        7.684743905258856e-05,
    )  # error of 1.54E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (1.11E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.015407689557835549}
    autogen_normdata = {
        "sample_bounds": [[0.0, 560.70068359375], [0.11079290503885982, 1.0]],
        "sample_bounds99": [
            [0.0, 46.7266731262207],
            [0.16295961503868264, 0.9980959803137086],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class EState_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA11)
    # normalization
    linear_norm_parameter = (
        0.0013586330466129892,
        0.9547790258899351,
    )  # error of 1.61E-02 with sample range (0.00E+00,2.06E+02) resulting in fit range (9.55E-01,1.23E+00)
    linear_norm_parameter_normdata = {"error": 0.016096203500796658}
    min_max_norm_parameter = (
        1.272927585246758e-08,
        1.5189995462591486,
    )  # error of 2.36E-02 with sample range (0.00E+00,2.06E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.023570804175567735}
    sigmoidal_norm_parameter = (
        -10.572305720334622,
        0.17246009308288568,
    )  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.61E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.006050626221514024}
    dual_sigmoidal_norm_parameter = (
        -10.572306217753033,
        1.0,
        0.17246008806155144,
    )  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.61E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.006050626221513832}
    genlog_norm_parameter = (
        0.16825568253446274,
        -18.685708494376684,
        0.021169686173775844,
        0.0059838039730952235,
    )  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.59E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006047183917905931}
    autogen_normdata = {
        "sample_bounds": [[0.0, 206.16510009765625], [0.858601875522268, 1.0]],
        "sample_bounds99": [
            [0.0, 17.380144119262695],
            [0.8859801561294588, 0.991899040197748],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class EState_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA2)
    # normalization
    linear_norm_parameter = (
        0.004010356366189516,
        0.8323097045695795,
    )  # error of 5.49E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (8.32E-01,8.07E+00)
    linear_norm_parameter_normdata = {"error": 0.05492405200000982}
    min_max_norm_parameter = (
        1.0419551447906305e-18,
        11.80200135547047,
    )  # error of 7.44E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.07435602748967914}
    sigmoidal_norm_parameter = (
        -0.1763739551036555,
        0.14514432511172065,
    )  # error of 1.12E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (5.06E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.011228232294457042}
    dual_sigmoidal_norm_parameter = (
        6.05707558822192e-07,
        18237830.635071214,
        0.14716311435942953,
    )  # error of 1.11E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (1.59E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011149632046344093}
    genlog_norm_parameter = (
        0.13263666358003515,
        -25.816811140188687,
        0.06712692476088679,
        0.0028752983368743124,
    )  # error of 1.11E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.68E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.011068999263567103}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1804.5626220703125], [0.4678378552628887, 1.0]],
        "sample_bounds99": [
            [0.0, 35.68024826049805],
            [0.594979903984448, 0.9934733734115093],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class EState_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA3)
    # normalization
    linear_norm_parameter = (
        0.003554939219269828,
        0.9223002361310408,
    )  # error of 1.14E-02 with sample range (0.00E+00,2.03E+02) resulting in fit range (9.22E-01,1.65E+00)
    linear_norm_parameter_normdata = {"error": 0.011439471617019224}
    min_max_norm_parameter = (
        1.1454288690296624e-20,
        3.325963599839221,
    )  # error of 3.62E-02 with sample range (0.00E+00,2.03E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03622341361124755}
    sigmoidal_norm_parameter = (
        -12.712321107593572,
        0.14776745606907848,
    )  # error of 7.76E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (8.67E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.007762639492589844}
    dual_sigmoidal_norm_parameter = (
        -12.712321897413275,
        0.999999999999936,
        0.14776744978227158,
    )  # error of 7.76E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (8.67E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0077626394925912}
    genlog_norm_parameter = (
        0.37951708290975456,
        12.653272129728492,
        2.688183939622898,
        55.41423775634062,
    )  # error of 7.16E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (9.01E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00715603116548685}
    autogen_normdata = {
        "sample_bounds": [[0.0, 203.31784057617188], [0.9007217871119829, 1.0]],
        "sample_bounds99": [
            [0.0, 16.099586486816406],
            [0.9185886468897587, 0.9901958134361775],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class EState_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA4)
    # normalization
    linear_norm_parameter = (
        0.0016155123555318786,
        0.9703629575801614,
    )  # error of 5.51E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.70E-01,1.08E+00)
    linear_norm_parameter_normdata = {"error": 0.00550989445842681}
    min_max_norm_parameter = (
        4.536992969157554e-16,
        2.4512528381627163,
    )  # error of 1.42E-02 with sample range (0.00E+00,6.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.01416412073412546}
    sigmoidal_norm_parameter = (
        -13.655427016167506,
        0.20068528350641354,
    )  # error of 3.15E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.39E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0031492435108688276}
    dual_sigmoidal_norm_parameter = (
        -13.65542695721612,
        1.0,
        0.20068528388954926,
    )  # error of 3.15E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0031492435108688333}
    genlog_norm_parameter = (
        0.21363444138978052,
        1.043942200221003,
        0.4496119693144669,
        7.510938179283622,
    )  # error of 3.15E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.42E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0031454150647766944}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 67.77261352539062],
            [0.9423569656313454, 0.9999999614485963],
        ],
        "sample_bounds99": [
            [0.0, 6.286160469055176],
            [0.9619545696450033, 0.9823390866585728],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class EState_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA5)
    # normalization
    linear_norm_parameter = (
        0.0017540134931710583,
        0.9543908360985814,
    )  # error of 8.35E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.54E-01,1.07E+00)
    linear_norm_parameter_normdata = {"error": 0.008348281812370035}
    min_max_norm_parameter = (
        5.383565720355907e-09,
        1.3949325598805027,
    )  # error of 2.48E-02 with sample range (0.00E+00,6.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.024800440271381168}
    sigmoidal_norm_parameter = (
        -27.554098729991185,
        0.09771212063268499,
    )  # error of 7.09E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.37E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.007085781952052848}
    dual_sigmoidal_norm_parameter = (
        -27.55410005194067,
        0.9999999999998883,
        0.097712116945909,
    )  # error of 7.09E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.37E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0070857819520516755}
    genlog_norm_parameter = (
        0.09642505280468441,
        -45.182144979315076,
        0.0035062065496121453,
        0.0006817441387180094,
    )  # error of 7.08E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.36E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007076435870885517}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 67.77261352539062],
            [0.9361912066153693, 0.999904279770274],
        ],
        "sample_bounds99": [
            [0.0, 22.040956497192383],
            [0.9436330696237515, 0.9925617096983916],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class EState_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA6)
    # normalization
    linear_norm_parameter = (
        0.0007744620095372357,
        0.9845100004329747,
    )  # error of 3.11E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.85E-01,1.05E+00)
    linear_norm_parameter_normdata = {"error": 0.003106423502403562}
    min_max_norm_parameter = (
        2.427726210776916e-14,
        4.450565380641523,
    )  # error of 7.79E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0077886665261108975}
    sigmoidal_norm_parameter = (
        -15.161431554355305,
        0.2124627730460597,
    )  # error of 2.28E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.62E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.002275837743308425}
    dual_sigmoidal_norm_parameter = (
        -15.161441858605182,
        1.0,
        0.2124626681786106,
    )  # error of 2.28E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0022758377432803346}
    genlog_norm_parameter = (
        0.21176166741260735,
        -3.0958232038380884,
        0.00011081057180114139,
        0.0014560393017236052,
    )  # error of 2.27E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.61E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.002273681337951208}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 90.36347961425781],
            [0.9612625064799714, 0.9999999998066311],
        ],
        "sample_bounds99": [
            [0.0, 6.0083794593811035],
            [0.9844380364163404, 0.9890748850029636],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class EState_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA7)
    # normalization
    linear_norm_parameter = (
        0.0009047456761177575,
        0.988557120440686,
    )  # error of 2.13E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (9.89E-01,1.03E+00)
    linear_norm_parameter_normdata = {"error": 0.0021346565382597514}
    min_max_norm_parameter = (
        2.1257783917477867e-14,
        4.312791487510295,
    )  # error of 4.52E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00451615959689124}
    sigmoidal_norm_parameter = (
        -2.0761099576023025,
        0.6793696501735206,
    )  # error of 1.30E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (8.04E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0013018716353242912}
    dual_sigmoidal_norm_parameter = (
        1.944517606033578e-06,
        6834910.858068878,
        0.9641921467533802,
    )  # error of 1.26E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (1.69E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0012643659180491926}
    genlog_norm_parameter = (
        0.676393961032863,
        -5.070793607019569,
        0.2971380059555156,
        0.04019749142435954,
    )  # error of 1.30E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (7.88E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0013017252114583683}
    autogen_normdata = {
        "sample_bounds": [[0.0, 47.789833068847656], [0.7879719704140621, 1.0]],
        "sample_bounds99": [
            [0.0, 4.829702377319336],
            [0.9867127056451099, 0.9910068711456511],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class EState_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA8)
    # normalization
    linear_norm_parameter = (
        0.007671915193011093,
        0.672048083837521,
    )  # error of 1.07E-01 with sample range (0.00E+00,3.49E+02) resulting in fit range (6.72E-01,3.35E+00)
    linear_norm_parameter_normdata = {"error": 0.1067521173200459}
    min_max_norm_parameter = (
        1.8002683601870467e-16,
        21.232186746380208,
    )  # error of 5.81E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05806004521523183}
    sigmoidal_norm_parameter = (
        8.641715400832636,
        0.17831995658486827,
    )  # error of 2.66E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (1.76E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026612197557234825}
    dual_sigmoidal_norm_parameter = (
        8.12775159056054,
        0.28166778919387403,
        0.167117877201916,
    )  # error of 2.50E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (9.20E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.024974811617751114}
    genlog_norm_parameter = (
        0.14956173559855754,
        -14.922836621848374,
        0.0011044146975219356,
        4.6839956847277765e-05,
    )  # error of 2.40E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (7.96E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.024034749815382152}
    autogen_normdata = {
        "sample_bounds": [[0.0, 348.95831298828125], [0.07962471111339213, 1.0]],
        "sample_bounds99": [
            [0.0, 40.842254638671875],
            [0.18396175692839173, 0.9944113066133891],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class EState_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(EState_VSA9)
    # normalization
    linear_norm_parameter = (
        0.008676493919004956,
        0.275505806742827,
    )  # error of 1.93E-01 with sample range (0.00E+00,6.11E+02) resulting in fit range (2.76E-01,5.57E+00)
    linear_norm_parameter_normdata = {"error": 0.19294633548768453}
    min_max_norm_parameter = (
        14.505227188937377,
        52.02218632660705,
    )  # error of 4.07E-02 with sample range (0.00E+00,6.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04065108757700309}
    sigmoidal_norm_parameter = (
        33.12486658378458,
        0.13300161576607572,
    )  # error of 1.91E-02 with sample range (0.00E+00,6.11E+02) resulting in fit range (1.21E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.019085974936984}
    dual_sigmoidal_norm_parameter = (
        32.05805124413192,
        0.16877196262466987,
        0.11492183140109144,
    )  # error of 9.95E-03 with sample range (0.00E+00,6.11E+02) resulting in fit range (4.45E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.009950892608105756}
    genlog_norm_parameter = (
        0.10208126159435472,
        -17.176379138009892,
        17.329332364534014,
        0.14881011107521142,
    )  # error of 7.12E-03 with sample range (0.00E+00,6.11E+02) resulting in fit range (8.98E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007122793679116591}
    autogen_normdata = {
        "sample_bounds": [[0.0, 610.7322998046875], [8.980349201465476e-05, 1.0]],
        "sample_bounds99": [
            [0.0, 91.637939453125],
            [0.006576283413603363, 0.9989275511741401],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcEccentricity)
    # normalization
    linear_norm_parameter = (
        4.5706147462074735,
        -3.8232476425495223,
    )  # error of 1.44E-01 with sample range (0.00E+00,1.00E+00) resulting in fit range (-3.82E+00,7.47E-01)
    linear_norm_parameter_normdata = {"error": 0.14378004757224336}
    min_max_norm_parameter = (
        0.9128972962156874,
        1.0,
    )  # error of 8.67E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.08670948517205165}
    sigmoidal_norm_parameter = (
        0.9575243366977794,
        40.76860157160623,
    )  # error of 6.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.11E-17,8.50E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.062043541808452354}
    dual_sigmoidal_norm_parameter = (
        0.9687599765083621,
        23.52840921810807,
        80.91404488539182,
    )  # error of 2.49E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.26E-10,9.26E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.024887733219507782}
    genlog_norm_parameter = (
        13.011559378552663,
        -0.17556161013905733,
        0.7479247294393261,
        4.680504264769145e-07,
    )  # error of 1.29E-01 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,6.95E-01)
    genlog_norm_parameter_normdata = {"error": 0.1285550603254054}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [0.0, 0.6951570738995947]],
        "sample_bounds99": [
            [0.0, 0.9972704648971558],
            [8.970866898366992e-05, 0.688777386223641],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcExactMolWt)
    # normalization
    linear_norm_parameter = (
        0.0012840291612334909,
        0.046753757513299354,
    )  # error of 1.65E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (8.92E-02,1.78E+01)
    linear_norm_parameter_normdata = {"error": 0.16488932516193885}
    min_max_norm_parameter = (
        175.3029880379195,
        537.2110757327455,
    )  # error of 3.98E-02 with sample range (3.31E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039783622105194066}
    sigmoidal_norm_parameter = (
        1.0,
        1.0,
    )  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.5012523548781626}
    dual_sigmoidal_norm_parameter = (
        1.0,
        1.0,
        1.0,
    )  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5012523548781626}
    genlog_norm_parameter = (
        1.0,
        1.0,
        1.0,
        1.000000000001,
    )  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.5012523548781626}
    autogen_normdata = {
        "sample_bounds": [[33.06145095825195, 13806.9453125], [0.999999999999988, 1.0]],
        "sample_bounds99": [[33.06145095825195, 881.330322265625], [1.0, 1.0]],
    }
    preferred_normalization = "min_max"
    # functions


class FormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(GetFormalCharge)
    # normalization
    # functions


class FpDensityMorgan1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan1)
    # normalization
    linear_norm_parameter = (
        0.6114489442416012,
        -0.2500130662147515,
    )  # error of 1.07E-01 with sample range (1.72E-02,4.50E+00) resulting in fit range (-2.39E-01,2.50E+00)
    linear_norm_parameter_normdata = {"error": 0.10653932039920164}
    min_max_norm_parameter = (
        0.7464667046271726,
        1.5888733629158822,
    )  # error of 3.23E-02 with sample range (1.72E-02,4.50E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0322929190431465}
    sigmoidal_norm_parameter = (
        1.167586022382926,
        6.06701731964724,
    )  # error of 7.87E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (9.30E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.007870342169414703}
    dual_sigmoidal_norm_parameter = (
        1.1745399912761414,
        5.851158150793598,
        6.523356772700546,
    )  # error of 7.03E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (1.14E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.007034181194737896}
    genlog_norm_parameter = (
        7.0169407190623465,
        1.0074906567566424,
        5.008686569277984,
        1.3640353929467133,
    )  # error of 5.29E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (1.88E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005291522059231368}
    autogen_normdata = {
        "sample_bounds": [
            [0.017241379246115685, 4.5],
            [0.001882013044250549, 0.9999999999164859],
        ],
        "sample_bounds99": [
            [0.017241379246115685, 0.9743589758872986],
            [0.008202090559724306, 0.9917400916990834],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class FpDensityMorgan2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan2)
    # normalization
    linear_norm_parameter = (
        0.3914967754398223,
        -0.3064699812712066,
    )  # error of 1.22E-01 with sample range (1.72E-02,5.00E+00) resulting in fit range (-3.00E-01,1.65E+00)
    linear_norm_parameter_normdata = {"error": 0.12192812806313}
    min_max_norm_parameter = (
        1.3833635629107808,
        2.497866940325602,
    )  # error of 3.66E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.036566087949340326}
    sigmoidal_norm_parameter = (
        1.9426344460243463,
        4.4864204391015186,
    )  # error of 1.50E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (1.77E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.015028323808818017}
    dual_sigmoidal_norm_parameter = (
        1.9697213352362348,
        4.019684859363329,
        5.576682871615822,
    )  # error of 1.08E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (3.90E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010835836378995337}
    genlog_norm_parameter = (
        6.474407856468799,
        2.28789522252963,
        0.36195135033063697,
        2.021031899463606,
    )  # error of 6.57E-03 with sample range (1.72E-02,5.00E+00) resulting in fit range (1.15E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006567045672523755}
    autogen_normdata = {
        "sample_bounds": [
            [0.017241379246115685, 5.0],
            [0.00114627085145234, 0.9999999957617958],
        ],
        "sample_bounds99": [
            [0.017241379246115685, 1.7241379022598267],
            [0.0076090596694183615, 0.9903536247212675],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class FpDensityMorgan3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan3)
    # normalization
    linear_norm_parameter = (
        0.30167675052364684,
        -0.3594212717972325,
    )  # error of 1.33E-01 with sample range (1.72E-02,5.00E+00) resulting in fit range (-3.54E-01,1.15E+00)
    linear_norm_parameter_normdata = {"error": 0.1333500607142603}
    min_max_norm_parameter = (
        2.026192791712374,
        3.2639324204504474,
    )  # error of 3.99E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03987385004419746}
    sigmoidal_norm_parameter = (
        2.6480344039052963,
        3.9835088427839342,
    )  # error of 2.03E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (2.81E-05,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.020345708607287766}
    dual_sigmoidal_norm_parameter = (
        2.6901285326989717,
        3.4012200092787612,
        5.4248939800812845,
    )  # error of 1.41E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (1.13E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.014095327496953981}
    genlog_norm_parameter = (
        6.765823684601774,
        2.8933763878591408,
        1.1387498165232823,
        2.6037005967210907,
    )  # error of 8.55E-03 with sample range (1.72E-02,5.00E+00) resulting in fit range (5.40E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008549730565698155}
    autogen_normdata = {
        "sample_bounds": [
            [0.017241379246115685, 5.0],
            [0.0005401529537692825, 0.9999997176298872],
        ],
        "sample_bounds99": [
            [0.017241379246115685, 2.4146342277526855],
            [0.006472771750348853, 0.9893759495017781],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(FractionCSP3)
    # normalization
    linear_norm_parameter = (
        1.1210114366616568,
        0.024238385885020208,
    )  # error of 6.63E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.42E-02,1.15E+00)
    linear_norm_parameter_normdata = {"error": 0.06629610917198944}
    min_max_norm_parameter = (
        0.030077967569291073,
        0.7816451738469055,
    )  # error of 3.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039915557978113345}
    sigmoidal_norm_parameter = (
        0.4036392556168974,
        6.810271411900117,
    )  # error of 2.19E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (6.01E-02,9.83E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.021893396525646636}
    dual_sigmoidal_norm_parameter = (
        0.3793638964895536,
        8.367627232005448,
        5.736046185011766,
    )  # error of 1.04E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.01E-02,9.72E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010390460984598808}
    genlog_norm_parameter = (
        5.121344220451963,
        0.06761202500583105,
        0.7177465382525524,
        0.18748114357432705,
    )  # error of 4.98E-03 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.38E-02,9.68E-01)
    genlog_norm_parameter_normdata = {"error": 0.004975425584248969}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [0.023842034333237446, 0.9683089421215593]],
        "sample_bounds99": [
            [0.0, 0.9846153855323792],
            [0.027347836838996668, 0.9662561915761914],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(HallKierAlpha)
    # normalization
    linear_norm_parameter = (
        0.10941027797520916,
        0.8206630827667591,
    )  # error of 1.86E-01 with sample range (-7.86E+01,6.10E+01) resulting in fit range (-7.78E+00,7.50E+00)
    linear_norm_parameter_normdata = {"error": 0.18589725616762204}
    min_max_norm_parameter = (
        -4.025047303628864,
        -0.27476699527518833,
    )  # error of 2.37E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02373235861155972}
    sigmoidal_norm_parameter = (
        -2.136641082271498,
        1.3958219191517867,
    )  # error of 1.43E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (4.45E-47,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.014252265465030198}
    dual_sigmoidal_norm_parameter = (
        -2.0451007382705924,
        1.188738004380637,
        1.5856962115538982,
    )  # error of 8.43E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (3.00E-40,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.008427760098605581}
    genlog_norm_parameter = (
        1.7619009617820294,
        -1.7315732634329195,
        1.3960318965437657,
        1.8402714405695966,
    )  # error of 7.26E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (9.11E-33,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0072648523242426315}
    autogen_normdata = {
        "sample_bounds": [
            [-78.5999984741211, 61.01298522949219],
            [9.107825457921401e-33, 1.0],
        ],
        "sample_bounds99": [
            [-78.5999984741211, -0.6376623511314392],
            [0.0041931197943515875, 0.9792781553058847],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class HeavyAtomCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(HeavyAtomCount)
    # normalization
    linear_norm_parameter = (
        0.005752556775892259,
        0.43633424647546626,
    )  # error of 2.06E-01 with sample range (2.00E+00,5.72E+02) resulting in fit range (4.48E-01,3.73E+00)
    linear_norm_parameter_normdata = {"error": 0.20599697046185259}
    min_max_norm_parameter = (
        11.422941214588905,
        38.12137726479023,
    )  # error of 2.20E-02 with sample range (2.00E+00,5.72E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.021959650165684363}
    sigmoidal_norm_parameter = (
        24.563511621430862,
        0.19297073304914714,
    )  # error of 1.69E-02 with sample range (2.00E+00,5.72E+02) resulting in fit range (1.27E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0169093696731723}
    dual_sigmoidal_norm_parameter = (
        22.896243586651543,
        0.2844117231776177,
        0.13983227405587295,
    )  # error of 7.75E-03 with sample range (2.00E+00,5.72E+02) resulting in fit range (2.62E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00775341742034876}
    genlog_norm_parameter = (
        0.13285606133429728,
        -37.31121013727694,
        0.198159127715709,
        8.426352275764387e-05,
    )  # error of 6.51E-03 with sample range (2.00E+00,5.72E+02) resulting in fit range (3.13E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006506632799032513}
    autogen_normdata = {
        "sample_bounds": [[2.0, 572.0], [3.1292198700740895e-06, 1.0]],
        "sample_bounds99": [[2.0, 63.0], [0.012521942019214978, 0.9986766465562286]],
    }
    preferred_normalization = "genlog"
    # functions


class HeavyAtomMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(HeavyAtomMolWt)
    # normalization
    linear_norm_parameter = (
        0.001100187734004554,
        0.17368773471419774,
    )  # error of 1.74E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (2.00E-01,1.54E+01)
    linear_norm_parameter_normdata = {"error": 0.17361738501525223}
    min_max_norm_parameter = (
        154.3124682840099,
        516.5882134868882,
    )  # error of 4.25E-02 with sample range (2.40E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04251843621338545}
    sigmoidal_norm_parameter = (
        1.0,
        1.0,
    )  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.434128830880842}
    dual_sigmoidal_norm_parameter = (
        1.0,
        1.0,
        1.0,
    )  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.434128830880842}
    genlog_norm_parameter = (
        1.0,
        1.0,
        1.0,
        1.000000000001,
    )  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.434128830880842}
    autogen_normdata = {
        "sample_bounds": [
            [24.02199935913086, 13805.681640625],
            [0.9999999998996141, 1.0],
        ],
        "sample_bounds99": [[24.02199935913086, 868.9030151367188], [1.0, 1.0]],
    }
    preferred_normalization = "min_max"
    # functions


class InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcInertialShapeFactor)
    # normalization
    linear_norm_parameter = (
        249.802812058766,
        0.2626729794953536,
    )  # error of 1.70E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (2.63E-01,2.41E+06)
    linear_norm_parameter_normdata = {"error": 0.17046180041647568}
    min_max_norm_parameter = (
        3.4403992924852414e-29,
        0.0015516770560768962,
    )  # error of 5.89E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05887555931240933}
    sigmoidal_norm_parameter = (
        0.0007537390827077178,
        2772.3791520268146,
    )  # error of 5.07E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (1.10E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.05073955975989971}
    dual_sigmoidal_norm_parameter = (
        8.477210431460079e-09,
        1.0226616162563986,
        2.5040477356755724,
    )  # error of 2.88E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (5.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.28764466720535214}
    genlog_norm_parameter = (
        10.739020490365299,
        -1.3568177149110616,
        0.8192065779630017,
        5.476913749206345e-07,
    )  # error of 2.85E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (4.95E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.28533219545622224}
    autogen_normdata = {
        "sample_bounds": [[0.0, 9630.4208984375], [0.4952231248484652, 1.0]],
        "sample_bounds99": [
            [0.0, 0.00370579375885427],
            [0.4954010214975587, 0.5126043938160759],
        ],
    }
    preferred_normalization = "sig"
    # functions


class Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Ipc)
    # normalization
    linear_norm_parameter = (
        1.5595572574111734e-38,
        0.4932082097701287,
    )  # error of 2.81E-01 with sample range (0.00E+00,INF) resulting in fit range (4.93E-01,INF)
    linear_norm_parameter_normdata = {"error": 0.28130736278670065}
    min_max_norm_parameter = (
        5e-324,
        1794244951.795763,
    )  # error of 2.42E-01 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.24180427535895055}
    sigmoidal_norm_parameter = (
        1.0,
        1.0,
    )  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.5793297328485066}
    dual_sigmoidal_norm_parameter = (
        1.0,
        1.0,
        1.0,
    )  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5793297328485066}
    genlog_norm_parameter = (
        1.0,
        1.0,
        1.0,
        1.000000000001,
    )  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.5793297328485066}
    autogen_normdata = {
        "sample_bounds": [[0.0, inf], [0.2689414213703484, 1.0]],
        "sample_bounds99": [[0.0, 3.660618012617995e30], [1.0, 1.0]],
    }
    preferred_normalization = "min_max"
    # functions


class Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Kappa1)
    # normalization
    linear_norm_parameter = (
        0.07531844296316315,
        0.10272005612925916,
    )  # error of 1.58E-01 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.06E-01,3.83E+01)
    linear_norm_parameter_normdata = {"error": 0.15810130669881933}
    min_max_norm_parameter = (
        1.58553325314983,
        8.60938851018364,
    )  # error of 3.82E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03824292449888255}
    sigmoidal_norm_parameter = (
        5.045698563226252,
        0.6990022831903797,
    )  # error of 2.71E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (2.94E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.027082321270926062}
    dual_sigmoidal_norm_parameter = (
        4.767739315887101,
        0.9158440851290796,
        0.5590280737588555,
    )  # error of 1.25E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.30E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01249947022528059}
    genlog_norm_parameter = (
        0.4925387826065551,
        -8.85306821854755,
        0.03629063704060047,
        5.9933878301094394e-05,
    )  # error of 4.77E-03 with sample range (4.08E-02,5.07E+02) resulting in fit range (5.12E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.004767715866603624}
    autogen_normdata = {
        "sample_bounds": [
            [0.040816325694322586, 506.85345458984375],
            [0.0005115304587048784, 1.0],
        ],
        "sample_bounds99": [
            [0.040816325694322586, 13.940459251403809],
            [0.019219828125048063, 0.9967491488697972],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Kappa2)
    # normalization
    linear_norm_parameter = (
        0.030418143185359847,
        0.2871593951766924,
    )  # error of 2.34E-01 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.87E-01,1.08E+03)
    linear_norm_parameter_normdata = {"error": 0.23429253315089843}
    min_max_norm_parameter = (
        3.446066777200142,
        12.00058941506709,
    )  # error of 4.58E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.045835035388305335}
    sigmoidal_norm_parameter = (
        7.66772112289306,
        0.5795843116540953,
    )  # error of 3.08E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (1.16E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.030784863703215513}
    dual_sigmoidal_norm_parameter = (
        7.272013631840285,
        0.8076799655613323,
        0.44608338323349067,
    )  # error of 1.32E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.81E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013227753344240065}
    genlog_norm_parameter = (
        0.4077659253828925,
        -9.791338200341348,
        0.05323800602728675,
        6.686361666192702e-05,
    )  # error of 1.05E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (4.19E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0105000416878865}
    autogen_normdata = {
        "sample_bounds": [[0.0, 35659.2421875], [4.1928109866545946e-07, 1.0]],
        "sample_bounds99": [
            [0.0, 23.003252029418945],
            [0.007535675172971146, 0.9999584044865643],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(Kappa3)
    # normalization
    linear_norm_parameter = (
        0.03561704845372893,
        0.3431665835165363,
    )  # error of 2.45E-01 with sample range (-4.32E+00,2.80E+04) resulting in fit range (1.89E-01,9.97E+02)
    linear_norm_parameter_normdata = {"error": 0.24538028229872727}
    min_max_norm_parameter = (
        1.6209280367339833,
        6.869092211937303,
    )  # error of 4.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04422617388183231}
    sigmoidal_norm_parameter = (
        4.207644140181935,
        0.935275527719581,
    )  # error of 3.22E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (3.42E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.03217554074352283}
    dual_sigmoidal_norm_parameter = (
        3.9792546917484937,
        1.265521274567412,
        0.6994418134155707,
    )  # error of 1.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (2.73E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.014239041156764966}
    genlog_norm_parameter = (
        0.6458423546818326,
        -5.942656398999168,
        0.03022130405947887,
        6.593687949417948e-05,
    )  # error of 1.21E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (2.37E-70,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.012078409215171363}
    autogen_normdata = {
        "sample_bounds": [
            [-4.324386119842529, 27984.41796875],
            [2.367304459384336e-70, 1.0],
        ],
        "sample_bounds99": [
            [-4.324386119842529, 15.822834014892578],
            [0.010368133997053362, 0.9999985050724834],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcLabuteASA)
    # normalization
    linear_norm_parameter = (
        0.0029906658775537,
        -0.06099902360508458,
    )  # error of 1.58E-01 with sample range (1.50E+01,3.79E+03) resulting in fit range (-1.62E-02,1.13E+01)
    linear_norm_parameter_normdata = {"error": 0.1575192378522443}
    min_max_norm_parameter = (
        93.68303193611298,
        261.4054366254348,
    )  # error of 3.76E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.037596041565658364}
    sigmoidal_norm_parameter = (
        176.2509725704667,
        0.028811070902678346,
    )  # error of 2.90E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (9.50E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.029039719767854237}
    dual_sigmoidal_norm_parameter = (
        168.88110152589715,
        0.0387688192218933,
        0.021877792004091706,
    )  # error of 1.06E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (2.55E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010635486087794264}
    genlog_norm_parameter = (
        0.02007957443756673,
        -204.06805973664225,
        0.0668114933774804,
        4.9438112644249285e-05,
    )  # error of 8.27E-03 with sample range (1.50E+01,3.79E+03) resulting in fit range (6.07E-08,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008266373133760602}
    autogen_normdata = {
        "sample_bounds": [
            [14.964731216430664, 3791.76025390625],
            [6.073319326050133e-08, 1.0],
        ],
        "sample_bounds99": [
            [14.964731216430664, 427.30023193359375],
            [0.007384653339473905, 0.999376383619305],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MaxAbsEStateIndex)
    # normalization
    linear_norm_parameter = (
        0.11609490354923989,
        -0.965852149695141,
    )  # error of 9.45E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (-9.66E-01,2.80E+00)
    linear_norm_parameter_normdata = {"error": 0.0945427358976128}
    min_max_norm_parameter = (
        10.794835917382866,
        15.53613807930635,
    )  # error of 7.37E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.07370859085311829}
    sigmoidal_norm_parameter = (
        13.16028708089213,
        0.9139468145600386,
    )  # error of 6.32E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (5.98E-06,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.06322094637748296}
    dual_sigmoidal_norm_parameter = (
        13.445192620610834,
        0.4908521805658206,
        1.4572576693025465,
    )  # error of 3.13E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (1.36E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.031299859924642306}
    genlog_norm_parameter = (
        2.9717890080241913,
        14.430056485891708,
        4.791780998388575,
        7.282273604915751,
    )  # error of 3.39E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (2.23E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.033883858865600296}
    autogen_normdata = {
        "sample_bounds": [[0.0, 32.462745666503906], [0.0022342248383460605, 1.0]],
        "sample_bounds99": [
            [0.0, 16.346118927001953],
            [0.05373153420855222, 0.9994915170954386],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class MaxAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MaxAbsPartialCharge)
    # normalization
    linear_norm_parameter = (
        3.0434418303394315,
        -0.7918951311620821,
    )  # error of 1.02E-01 with sample range (0.00E+00,INF) resulting in fit range (-7.92E-01,INF)
    linear_norm_parameter_normdata = {"error": 0.10245767203525469}
    min_max_norm_parameter = (
        0.2952082545313489,
        0.5504603722073848,
    )  # error of 4.52E-02 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04517076608490933}
    sigmoidal_norm_parameter = (
        0.425391072059637,
        18.052436340913527,
    )  # error of 5.24E-02 with sample range (0.00E+00,INF) resulting in fit range (4.62E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.05241345451072119}
    dual_sigmoidal_norm_parameter = (
        0.44710956517413125,
        13.358900058018621,
        30.95122279557429,
    )  # error of 4.77E-02 with sample range (0.00E+00,INF) resulting in fit range (2.54E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.04768713125196702}
    genlog_norm_parameter = (
        676.3056523936937,
        0.5153389266359548,
        0.9444653266062369,
        73.15316513583674,
    )  # error of 4.71E-02 with sample range (0.00E+00,INF) resulting in fit range (8.54E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.047100998777310506}
    autogen_normdata = {
        "sample_bounds": [[0.0, inf], [0.008535169883055159, 1.0]],
        "sample_bounds99": [[0.0, 0.5430407524108887], [0.03226743887265661, 1.0]],
    }
    preferred_normalization = "min_max"
    # functions


class MaxEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MaxEStateIndex)
    # normalization
    linear_norm_parameter = (
        0.1160824744026907,
        -0.9656546887853172,
    )  # error of 9.45E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (-9.66E-01,2.80E+00)
    linear_norm_parameter_normdata = {"error": 0.09452784161835764}
    min_max_norm_parameter = (
        10.793338979130617,
        15.53686323514367,
    )  # error of 7.37E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.07374160800715936}
    sigmoidal_norm_parameter = (
        13.160091893100835,
        0.9136778762386883,
    )  # error of 6.33E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (6.00E-06,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.06325658987522365}
    dual_sigmoidal_norm_parameter = (
        13.445167722011659,
        0.4906292568720514,
        1.4572136129982889,
    )  # error of 3.13E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (1.36E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.03130964978955604}
    genlog_norm_parameter = (
        2.973438748100483,
        14.43830672678955,
        4.686060882765243,
        7.288908309142506,
    )  # error of 3.39E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (2.24E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.033906668614697044}
    autogen_normdata = {
        "sample_bounds": [[0.0, 32.462745666503906], [0.002238664555878225, 1.0]],
        "sample_bounds99": [
            [0.0, 16.346118927001953],
            [0.05377745476142734, 0.9994928686961123],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class MaxPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MaxPartialCharge)
    # normalization
    linear_norm_parameter = (
        3.0906307507635584,
        -0.334505311086143,
    )  # error of 7.24E-02 with sample range (-4.12E-01,INF) resulting in fit range (-1.61E+00,INF)
    linear_norm_parameter_normdata = {"error": 0.07241799813450908}
    min_max_norm_parameter = (
        0.1437213632361377,
        0.3915749702129172,
    )  # error of 3.92E-02 with sample range (-4.12E-01,INF) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03921670656399854}
    sigmoidal_norm_parameter = (
        0.26834153843781333,
        19.306170677985925,
    )  # error of 1.77E-02 with sample range (-4.12E-01,INF) resulting in fit range (1.99E-06,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01767560246090192}
    dual_sigmoidal_norm_parameter = (
        0.26691979313947295,
        20.14067256838992,
        18.55398222869742,
    )  # error of 1.72E-02 with sample range (-4.12E-01,INF) resulting in fit range (1.16E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.017204213419227444}
    genlog_norm_parameter = (
        10.920843919005986,
        -1.1781659552061643,
        1.0402844953559025,
        2.1279094136284055e-07,
    )  # error of 4.30E-02 with sample range (-4.12E-01,INF) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0429949570590489}
    autogen_normdata = {
        "sample_bounds": [[-0.41150951385498047, inf], [0.0, 1.0]],
        "sample_bounds99": [
            [-0.41150951385498047, 0.4695388376712799],
            [0.0018276845938865307, 0.9635735840437843],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class MinAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MinAbsEStateIndex)
    # normalization
    linear_norm_parameter = (
        0.43546020000054764,
        0.22371772226149234,
    )  # error of 1.33E-01 with sample range (0.00E+00,6.27E+00) resulting in fit range (2.24E-01,2.95E+00)
    linear_norm_parameter_normdata = {"error": 0.13312116937642537}
    min_max_norm_parameter = (
        1.5407439555097887e-33,
        1.18171530209488,
    )  # error of 6.48E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06478832930969794}
    sigmoidal_norm_parameter = (
        0.5580567270616906,
        3.2920268348402595,
    )  # error of 3.85E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (1.37E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.03845875371000878}
    dual_sigmoidal_norm_parameter = (
        0.48049021393760116,
        4.528460770199798,
        2.473169687745443,
    )  # error of 2.70E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (1.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.026961922528310645}
    genlog_norm_parameter = (
        2.32359863435508,
        -3.035193161685175,
        0.19234318567209244,
        7.070812679807398e-05,
    )  # error of 2.37E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (9.51E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.023683708243495315}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 6.266533374786377],
            [0.09505418361468589, 0.9999988831567012],
        ],
        "sample_bounds99": [
            [0.0, 2.9175925254821777],
            [0.09543219770630726, 0.9978052943288531],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class MinAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MinAbsPartialCharge)
    # normalization
    linear_norm_parameter = (
        3.1682344592574303,
        -0.3438979216493595,
    )  # error of 8.02E-02 with sample range (0.00E+00,INF) resulting in fit range (-3.44E-01,INF)
    linear_norm_parameter_normdata = {"error": 0.08020702001161399}
    min_max_norm_parameter = (
        0.14478720365913814,
        0.3864847550968621,
    )  # error of 3.67E-02 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.036700059960331044}
    sigmoidal_norm_parameter = (
        0.266281069473945,
        20.060338031364914,
    )  # error of 1.73E-02 with sample range (0.00E+00,INF) resulting in fit range (4.76E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.017345167167046788}
    dual_sigmoidal_norm_parameter = (
        0.26630944700484976,
        20.042744120617208,
        20.075933144268717,
    )  # error of 1.73E-02 with sample range (0.00E+00,INF) resulting in fit range (4.78E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.017344968022429963}
    genlog_norm_parameter = (
        11.090863298621167,
        -1.1570477862580686,
        1.0539258556125684,
        2.192085103218304e-07,
    )  # error of 4.89E-02 with sample range (0.00E+00,INF) resulting in fit range (2.63E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.04887453776925912}
    autogen_normdata = {
        "sample_bounds": [[0.0, inf], [2.6342212398417388e-06, 1.0]],
        "sample_bounds99": [
            [0.0, 0.4315127432346344],
            [0.0017065300859320071, 0.9333410613860469],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MinEStateIndex)
    # normalization
    linear_norm_parameter = (
        0.22449008413602467,
        1.529064938673871,
    )  # error of 9.84E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (-3.78E+00,1.79E+00)
    linear_norm_parameter_normdata = {"error": 0.09839052793649566}
    min_max_norm_parameter = (
        -6.065158419096168,
        -2.9821960881671177,
    )  # error of 4.20E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04196731226075694}
    sigmoidal_norm_parameter = (
        -4.49507192502996,
        1.6028612470816288,
    )  # error of 2.52E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (4.77E-14,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0252182112598502}
    dual_sigmoidal_norm_parameter = (
        -4.402227844870071,
        1.2805552108237557,
        2.027604138125001,
    )  # error of 8.88E-03 with sample range (-2.36E+01,1.17E+00) resulting in fit range (2.02E-11,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00888334834331836}
    genlog_norm_parameter = (
        2.588653604531449,
        -0.7719180383003792,
        0.00041704155880015604,
        2.712978138811103,
    )  # error of 6.76E-03 with sample range (-2.36E+01,1.17E+00) resulting in fit range (5.93E-09,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006761739487275899}
    autogen_normdata = {
        "sample_bounds": [
            [-23.6320858001709, 1.171296238899231],
            [5.92534479911652e-09, 0.9999989951599305],
        ],
        "sample_bounds99": [
            [-23.6320858001709, -1.6612099409103394],
            [0.0193892401257079, 0.9995940367151389],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class MinPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MinPartialCharge)
    # normalization
    linear_norm_parameter = (
        2.948628224298744,
        1.7394347203560183,
    )  # error of 1.05E-01 with sample range (-2.00E+00,INF) resulting in fit range (-4.16E+00,INF)
    linear_norm_parameter_normdata = {"error": 0.10452852544583144}
    min_max_norm_parameter = (
        -0.5479342566156457,
        -0.28873718774420354,
    )  # error of 4.48E-02 with sample range (-2.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04480668091491009}
    sigmoidal_norm_parameter = (
        -0.420139769859882,
        17.629128066695653,
    )  # error of 5.34E-02 with sample range (-2.00E+00,INF) resulting in fit range (8.02E-13,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.053364554403492}
    dual_sigmoidal_norm_parameter = (
        -0.4397372036237273,
        26.874835355850397,
        13.461649276273022,
    )  # error of 5.23E-02 with sample range (-2.00E+00,INF) resulting in fit range (6.16E-19,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.05231292072567235}
    genlog_norm_parameter = (
        12.826344774463708,
        0.020758794415196596,
        3.261876078798542e-06,
        0.0014671976142966849,
    )  # error of 5.17E-02 with sample range (-2.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.05169765925291506}
    autogen_normdata = {
        "sample_bounds": [[-2.0, inf], [0.0, 1.0]],
        "sample_bounds99": [
            [-2.0, -0.20695531368255615],
            [0.0003479239881094209, 0.9829320273570826],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class MolLogP_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MolLogP)
    # normalization
    linear_norm_parameter = (
        0.08784472608787079,
        0.1981370072747647,
    )  # error of 1.64E-01 with sample range (-5.89E+01,9.69E+01) resulting in fit range (-4.98E+00,8.71E+00)
    linear_norm_parameter_normdata = {"error": 0.16371601005122985}
    min_max_norm_parameter = (
        0.46977870811162403,
        5.966058231862306,
    )  # error of 3.39E-02 with sample range (-5.89E+01,9.69E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03394293957251461}
    sigmoidal_norm_parameter = (
        3.206291545692704,
        0.8985363235365765,
    )  # error of 1.25E-02 with sample range (-5.89E+01,9.69E+01) resulting in fit range (5.76E-25,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.012463911204746434}
    dual_sigmoidal_norm_parameter = (
        3.1239114243134525,
        0.9965869299386755,
        0.8112374648461028,
    )  # error of 7.33E-03 with sample range (-5.89E+01,9.69E+01) resulting in fit range (1.42E-27,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.007327316800313644}
    genlog_norm_parameter = (
        0.757148865450231,
        1.3856248715993458,
        1.6174647792081618,
        0.5076711936739915,
    )  # error of 6.47E-03 with sample range (-5.89E+01,9.69E+01) resulting in fit range (3.42E-40,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0064693538957135465}
    autogen_normdata = {
        "sample_bounds": [
            [-58.9098014831543, 96.88919830322266],
            [3.423492662752597e-40, 1.0],
        ],
        "sample_bounds99": [
            [-58.9098014831543, 10.831500053405762],
            [0.0034131831497609616, 0.9998324926239983],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class MolMR_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MolMR)
    # normalization
    linear_norm_parameter = (
        0.005557720699731816,
        -0.061030996122217296,
    )  # error of 1.53E-01 with sample range (0.00E+00,2.20E+03) resulting in fit range (-6.10E-02,1.22E+01)
    linear_norm_parameter_normdata = {"error": 0.15262165688044133}
    min_max_norm_parameter = (
        48.121442195030106,
        142.36977751407525,
    )  # error of 3.62E-02 with sample range (0.00E+00,2.20E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03615501177358133}
    sigmoidal_norm_parameter = (
        94.48683995909174,
        0.05138768339127881,
    )  # error of 2.83E-02 with sample range (0.00E+00,2.20E+03) resulting in fit range (7.73E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.028323575886008857}
    dual_sigmoidal_norm_parameter = (
        90.41107725469143,
        0.068563492805142,
        0.03926937503461646,
    )  # error of 1.05E-02 with sample range (0.00E+00,2.20E+03) resulting in fit range (2.03E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010524087327864131}
    genlog_norm_parameter = (
        0.03578835034273679,
        -116.09899952608899,
        0.05782641450323655,
        4.7341909551770326e-05,
    )  # error of 7.72E-03 with sample range (0.00E+00,2.20E+03) resulting in fit range (4.82E-09,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007715273707135446}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2199.830078125], [4.817086067682376e-09, 1.0]],
        "sample_bounds99": [
            [0.0, 232.58419799804688],
            [0.006812586438708956, 0.9992044679627353],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class MolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(MolWt)
    # normalization
    linear_norm_parameter = (
        0.0012559144823710566,
        0.056127425145544585,
    )  # error of 1.68E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (9.77E-02,1.74E+01)
    linear_norm_parameter_normdata = {"error": 0.16836555013248564}
    min_max_norm_parameter = (
        175.4052910230421,
        538.3280010265217,
    )  # error of 4.00E-02 with sample range (3.31E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0400150097919368}
    sigmoidal_norm_parameter = (
        1.0,
        1.0,
    )  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.5017887630143759}
    dual_sigmoidal_norm_parameter = (
        1.0,
        1.0,
        1.0,
    )  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5017887630143759}
    genlog_norm_parameter = (
        1.0,
        1.0,
        1.0,
        1.000000000001,
    )  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.5017887630143759}
    autogen_normdata = {
        "sample_bounds": [
            [33.08415222167969, 13805.681640625],
            [0.9999999999999885, 1.0],
        ],
        "sample_bounds99": [[33.08415222167969, 880.4979858398438], [1.0, 1.0]],
    }
    preferred_normalization = "min_max"
    # functions


class NHOHCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NHOHCount)
    # normalization
    linear_norm_parameter = (
        0.011451805303745677,
        0.8103893376864633,
    )  # error of 1.09E-01 with sample range (0.00E+00,8.20E+01) resulting in fit range (8.10E-01,1.75E+00)
    linear_norm_parameter_normdata = {"error": 0.1093936401741258}
    min_max_norm_parameter = (
        9.860761315262648e-32,
        3.187051718282465,
    )  # error of 1.94E-02 with sample range (0.00E+00,8.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019444687584919223}
    sigmoidal_norm_parameter = (
        1.3178308349589882,
        1.1160540358868796,
    )  # error of 6.16E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.87E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.006159518941504394}
    dual_sigmoidal_norm_parameter = (
        1.0743143001638746,
        5.214591634885128,
        0.9332858170955084,
    )  # error of 3.15E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (3.68E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.003147962480995746}
    genlog_norm_parameter = (
        1.0491738144723946,
        -4.348296718164784,
        0.017728446734426147,
        6.444988862716029e-05,
    )  # error of 9.99E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (5.66E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009992878038903702}
    autogen_normdata = {
        "sample_bounds": [[0.0, 82.0], [0.0566152982336769, 1.0]],
        "sample_bounds99": [[0.0, 6.0], [0.36577793806998565, 0.998145659024973]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NOCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NOCount)
    # normalization
    linear_norm_parameter = (
        0.0021474686509596053,
        0.8235464950362099,
    )  # error of 1.92E-01 with sample range (0.00E+00,1.96E+02) resulting in fit range (8.24E-01,1.24E+00)
    linear_norm_parameter_normdata = {"error": 0.1921585081293036}
    min_max_norm_parameter = (
        1.2880062839960937,
        9.187763624570907,
    )  # error of 1.38E-02 with sample range (0.00E+00,1.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.013846698638782243}
    sigmoidal_norm_parameter = (
        5.184102720727361,
        0.6674369952484219,
    )  # error of 7.25E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (3.05E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.007251569533280246}
    dual_sigmoidal_norm_parameter = (
        5.01372760473936,
        0.7649977970154859,
        0.5850210479926004,
    )  # error of 5.05E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (2.11E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.005050968151679584}
    genlog_norm_parameter = (
        0.5268273795803626,
        1.5540721740088757,
        1.7263012695643103,
        0.3474832434074276,
    )  # error of 3.62E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (1.02E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.003624850298533195}
    autogen_normdata = {
        "sample_bounds": [[0.0, 196.0], [0.01023326561734317, 1.0]],
        "sample_bounds99": [[0.0, 15.0], [0.031876227895358995, 0.9975438175592366]],
    }
    preferred_normalization = "genlog"
    # functions


class NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcNPR1)
    # normalization
    linear_norm_parameter = (
        1.993532814687622,
        -0.07291478792771389,
    )  # error of 6.17E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (-7.29E-02,1.83E+00)
    linear_norm_parameter_normdata = {"error": 0.061728982485812586}
    min_max_norm_parameter = (
        0.06225581266603365,
        0.49572101730415397,
    )  # error of 2.98E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.029796855243874387}
    sigmoidal_norm_parameter = (
        0.27514214399673687,
        11.301895844887198,
    )  # error of 2.74E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (4.27E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.027352449556509766}
    dual_sigmoidal_norm_parameter = (
        0.2598822642874916,
        14.136432045747458,
        9.153928411437114,
    )  # error of 1.65E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (2.48E-02,9.98E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.016506918848526703}
    genlog_norm_parameter = (
        7.873560517478882,
        -0.9842677258585575,
        1.1168374338530573,
        8.53760240249483e-05,
    )  # error of 1.08E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (3.57E-03,9.97E-01)
    genlog_norm_parameter_normdata = {"error": 0.01075571562176027}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 0.9535223841667175],
            [0.0035703351151573316, 0.9969110007503373],
        ],
        "sample_bounds99": [
            [0.0, 0.6135377287864685],
            [0.03136893806380091, 0.968971394175979],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(NPR2)
    # normalization
    linear_norm_parameter = (
        3.0363848124092283,
        -2.1050145083551772,
    )  # error of 7.55E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-2.11E+00,9.31E-01)
    linear_norm_parameter_normdata = {"error": 0.07547980695973905}
    min_max_norm_parameter = (
        0.7328377706938303,
        0.9989929606321605,
    )  # error of 3.95E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03945681388392586}
    sigmoidal_norm_parameter = (
        0.8690167433696966,
        18.125099021117805,
    )  # error of 3.57E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.44E-07,9.15E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.035723535153184105}
    dual_sigmoidal_norm_parameter = (
        0.8826306733966545,
        13.438687377055738,
        25.1975217333764,
    )  # error of 1.76E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (7.06E-06,9.51E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.017621678627210136}
    genlog_norm_parameter = (
        41.22968146938368,
        0.9555513699693968,
        0.8329988798867941,
        4.4083555329656905,
    )  # error of 1.41E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.37E-04,9.72E-01)
    genlog_norm_parameter_normdata = {"error": 0.01405438439828333}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [0.00013700937456992732, 0.9720179859647797]],
        "sample_bounds99": [
            [0.0, 0.9843385815620422],
            [0.04063864286909488, 0.9576813145628001],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticCarbocycles)
    # normalization
    linear_norm_parameter = (
        0.029386355676372537,
        0.8723594863010827,
    )  # error of 2.62E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (8.72E-01,1.40E+00)
    linear_norm_parameter_normdata = {"error": 0.02619488960538958}
    min_max_norm_parameter = (
        7.2035108451548135e-09,
        1.1561266295897044,
    )  # error of 1.01E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.010078326862442933}
    sigmoidal_norm_parameter = (
        -0.4231823522456474,
        1.309702732778145,
    )  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0038627930443218446}
    dual_sigmoidal_norm_parameter = (
        -0.4231823921093463,
        1.0,
        1.3097026985565412,
    )  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.003862793044326242}
    genlog_norm_parameter = (
        1.2612530269741857,
        -4.6087319696557145,
        0.3180813147293475,
        0.0018673678985071268,
    )  # error of 3.67E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.01E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.003668784080520261}
    autogen_normdata = {
        "sample_bounds": [[0.0, 18.0], [0.6011477519809362, 0.9999999999296065]],
        "sample_bounds99": [[0.0, 2.0], [0.865693816674613, 0.9884901557981153]],
    }
    preferred_normalization = "min_max"
    # functions


class NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumAliphaticHeterocycles)
    # normalization
    linear_norm_parameter = (
        0.07182403555851259,
        0.6979616844087828,
    )  # error of 6.50E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (6.98E-01,3.71E+00)
    linear_norm_parameter_normdata = {"error": 0.06501190929792677}
    min_max_norm_parameter = (
        6.816327279061287e-09,
        1.4697587576089106,
    )  # error of 1.83E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.018297669482265272}
    sigmoidal_norm_parameter = (
        0.5242081959438346,
        1.5883082643053523,
    )  # error of 6.06E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (3.03E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0006057908383317623}
    dual_sigmoidal_norm_parameter = (
        0.5241697895237789,
        32.139936612408746,
        1.5882035086803503,
    )  # error of 3.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (4.83E-08,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0003628683527523974}
    genlog_norm_parameter = (
        1.5563961403496074,
        0.06313351849368842,
        1.5020948194793298,
        0.7783007105283993,
    )  # error of 5.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.85E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0005627506710425015}
    autogen_normdata = {
        "sample_bounds": [[0.0, 42.0], [0.28489256061061674, 1.0]],
        "sample_bounds99": [[0.0, 2.0], [0.680378574283369, 0.9803767405000863]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumAliphaticRings)
    # normalization
    linear_norm_parameter = (
        0.04716081590128,
        0.7025071278256165,
    )  # error of 8.86E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (7.03E-01,2.68E+00)
    linear_norm_parameter_normdata = {"error": 0.0885886948242506}
    min_max_norm_parameter = (
        1.5777218104420236e-29,
        2.223682308863912,
    )  # error of 2.88E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02881039066665504}
    sigmoidal_norm_parameter = (
        0.7813916309499217,
        1.3224930670990482,
    )  # error of 4.20E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.62E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.004203589934143805}
    dual_sigmoidal_norm_parameter = (
        0.7813895517185011,
        19.442917588992998,
        1.322487361093461,
    )  # error of 2.34E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.52E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0023407868651397798}
    genlog_norm_parameter = (
        1.1513899435108788,
        -4.640594483150161,
        0.1786885365072345,
        0.00048121717016260063,
    )  # error of 2.04E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (1.70E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0020439716934064246}
    autogen_normdata = {
        "sample_bounds": [[0.0, 42.0], [0.1695677970330986, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.5704920037626812, 0.9824109136751766]],
    }
    preferred_normalization = "genlog"
    # functions


class NumAmideBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumAmideBonds)
    # normalization
    linear_norm_parameter = (
        0.025379291619204047,
        0.8062087745432029,
    )  # error of 7.32E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.06E-01,2.41E+00)
    linear_norm_parameter_normdata = {"error": 0.07316365477809328}
    min_max_norm_parameter = (
        9.860761315262648e-31,
        2.0799891850295853,
    )  # error of 3.30E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.033048308726217036}
    sigmoidal_norm_parameter = (
        0.4876800061173155,
        1.2673698495352894,
    )  # error of 2.03E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (3.50E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0020290401868979994}
    dual_sigmoidal_norm_parameter = (
        0.4876792517661068,
        32.508328557898885,
        1.267368546681463,
    )  # error of 1.10E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (1.30E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.001100491743573129}
    genlog_norm_parameter = (
        1.140969782171021,
        -4.316137506939921,
        0.3881750500000892,
        0.0021382809875235952,
    )  # error of 1.45E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (2.68E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0014547429761732264}
    autogen_normdata = {
        "sample_bounds": [[0.0, 63.0], [0.26790024758247083, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.6562314668309573, 0.9863482444482209]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticCarbocycles)
    # normalization
    linear_norm_parameter = (
        0.029344509343809166,
        0.7342345249533615,
    )  # error of 1.10E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (7.34E-01,2.52E+00)
    linear_norm_parameter_normdata = {"error": 0.1103343753066689}
    min_max_norm_parameter = (
        9.860761315262648e-32,
        2.4631545244076647,
    )  # error of 1.96E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019586692320725813}
    sigmoidal_norm_parameter = (
        1.0872508968285497,
        1.3660589721714311,
    )  # error of 6.47E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.85E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0064662961947873675}
    dual_sigmoidal_norm_parameter = (
        1.0872509124487726,
        13.4357718685885,
        1.366059068627403,
    )  # error of 4.24E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.53E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.004237534896887361}
    genlog_norm_parameter = (
        1.1361653376687364,
        -3.222488739271155,
        0.3316020448352798,
        0.00359737796009675,
    )  # error of 5.54E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (9.45E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005543065609843302}
    autogen_normdata = {
        "sample_bounds": [[0.0, 61.0], [0.09452214188344103, 1.0]],
        "sample_bounds99": [[0.0, 5.0], [0.4678955094522694, 0.9974093412036721]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticHeterocycles)
    # normalization
    linear_norm_parameter = (
        0.06608705216326816,
        0.7231759141671211,
    )  # error of 6.12E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (7.23E-01,1.71E+00)
    linear_norm_parameter_normdata = {"error": 0.061166618492081375}
    min_max_norm_parameter = (
        2.425436375356735e-09,
        1.4183360150961999,
    )  # error of 2.28E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.022818659362760136}
    sigmoidal_norm_parameter = (
        0.4595690682034204,
        1.6119949186776596,
    )  # error of 6.29E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.23E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0006289089312844855}
    dual_sigmoidal_norm_parameter = (
        0.4595442280127375,
        36.76321468072523,
        1.6119316552199183,
    )  # error of 4.35E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (4.60E-08,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0004352269580539154}
    genlog_norm_parameter = (
        1.671881279088869,
        0.4712983057709566,
        1.5877180316534256,
        1.4432222740017289,
    )  # error of 5.24E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.53E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0005236082183723774}
    autogen_normdata = {
        "sample_bounds": [[0.0, 15.0], [0.3531672130703787, 0.9999999999689322]],
        "sample_bounds99": [[0.0, 2.0], [0.7050476896281848, 0.984261728267582]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticRings)
    # normalization
    linear_norm_parameter = (
        0.034866795939075525,
        0.6345283877004492,
    )  # error of 1.43E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (6.35E-01,2.76E+00)
    linear_norm_parameter_normdata = {"error": 0.14320732353899343}
    min_max_norm_parameter = (
        4.733165431326071e-30,
        3.555582078939516,
    )  # error of 1.41E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.014104553778604489}
    sigmoidal_norm_parameter = (
        1.7284808042419078,
        1.2169190611775202,
    )  # error of 6.86E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.09E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.006864483402826702}
    dual_sigmoidal_norm_parameter = (
        1.686248103154103,
        1.3303663811394981,
        1.1588291961663748,
    )  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (9.59E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.006131762284204077}
    genlog_norm_parameter = (
        1.0818715704403212,
        0.8316808527919441,
        1.181216074235139,
        0.5468718534498849,
    )  # error of 5.78E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (8.28E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0057771476764493825}
    autogen_normdata = {
        "sample_bounds": [[0.0, 61.0], [0.08284028655371821, 1.0]],
        "sample_bounds99": [[0.0, 7.0], [0.28555811062787634, 0.999075105694155]],
    }
    preferred_normalization = "genlog"
    # functions


class NumAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.0028101191309833906,
        0.4586559034152283,
    )  # error of 2.08E-01 with sample range (2.00E+00,9.38E+02) resulting in fit range (4.64E-01,3.09E+00)
    linear_norm_parameter_normdata = {"error": 0.2082344752244387}
    min_max_norm_parameter = (
        22.699274586285846,
        70.39472667366697,
    )  # error of 2.45E-02 with sample range (2.00E+00,9.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02453646041713264}
    sigmoidal_norm_parameter = (
        46.31674761573806,
        0.10711279271638556,
    )  # error of 1.75E-02 with sample range (2.00E+00,9.38E+02) resulting in fit range (8.60E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.017498605638079444}
    dual_sigmoidal_norm_parameter = (
        43.70797609285935,
        0.1511085487560505,
        0.07900109330436363,
    )  # error of 9.28E-03 with sample range (2.00E+00,9.38E+02) resulting in fit range (1.83E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.009282817416233123}
    genlog_norm_parameter = (
        0.07326085064819542,
        -44.92533678413176,
        0.022954022788335537,
        4.515441426070382e-05,
    )  # error of 8.12E-03 with sample range (2.00E+00,9.38E+02) resulting in fit range (8.09E-08,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008116740032060915}
    autogen_normdata = {
        "sample_bounds": [[2.0, 938.0], [8.090719806700145e-08, 1.0]],
        "sample_bounds99": [[2.0, 130.0], [0.006354682073333458, 0.9995720080545568]],
    }
    preferred_normalization = "genlog"

    # functions
    def featurize(self, mol):
        return mol.GetNumAtoms()


class NumBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.002743260112020973,
        0.45085043244012823,
    )  # error of 2.10E-01 with sample range (0.00E+00,9.97E+02) resulting in fit range (4.51E-01,3.19E+00)
    linear_norm_parameter_normdata = {"error": 0.20980875134901514}
    min_max_norm_parameter = (
        22.92431994469887,
        73.53427413100934,
    )  # error of 2.46E-02 with sample range (0.00E+00,9.97E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.024586358347791767}
    sigmoidal_norm_parameter = (
        48.01236350848081,
        0.10071768927745786,
    )  # error of 1.76E-02 with sample range (0.00E+00,9.97E+02) resulting in fit range (7.88E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0175895779424767}
    dual_sigmoidal_norm_parameter = (
        45.26546665522718,
        0.1416051615875594,
        0.07451723131457108,
    )  # error of 9.14E-03 with sample range (0.00E+00,9.97E+02) resulting in fit range (1.64E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0091429014271612}
    genlog_norm_parameter = (
        0.06893983960267623,
        -54.16701644496038,
        0.03480866141021429,
        4.781043915143292e-05,
    )  # error of 7.91E-03 with sample range (0.00E+00,9.97E+02) resulting in fit range (2.81E-08,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007911960577461026}
    autogen_normdata = {
        "sample_bounds": [[0.0, 997.0], [2.8126839300004762e-08, 1.0]],
        "sample_bounds99": [[0.0, 133.0], [0.006550172458051944, 0.999476024561193]],
    }
    preferred_normalization = "genlog"

    # functions
    def featurize(self, mol):
        return mol.GetNumBonds()


class NumBridgeheadAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumBridgeheadAtoms)
    # normalization
    linear_norm_parameter = (
        0.0022274780962232565,
        0.983622366844093,
    )  # error of 2.88E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.84E-01,1.03E+00)
    linear_norm_parameter_normdata = {"error": 0.002876503196729235}
    min_max_norm_parameter = (
        5.24640715369184e-09,
        2.0351023563443684,
    )  # error of 2.27E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0022655624646996764}
    sigmoidal_norm_parameter = (
        -4.164793009240712,
        0.6606716515917349,
    )  # error of 8.24E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.40E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0008240589235670398}
    dual_sigmoidal_norm_parameter = (
        -4.164793009240712,
        1.0,
        0.6606716515917349,
    )  # error of 8.24E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.40E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0008240589235670398}
    genlog_norm_parameter = (
        0.6577017025023064,
        -7.340551820620397,
        0.038705820798677966,
        0.0049219713882773025,
    )  # error of 8.20E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.39E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.000819992171773689}
    autogen_normdata = {
        "sample_bounds": [[0.0, 20.0], [0.9390112795870728, 0.9999998780490362]],
        "sample_bounds99": [[0.0, 0.0], [0.9832522348529897, 0.9832522348529897]],
    }
    preferred_normalization = "min_max"
    # functions


class NumHAcceptors_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumHAcceptors)
    # normalization
    linear_norm_parameter = (
        0.018653528399554697,
        0.5574729713710027,
    )  # error of 1.93E-01 with sample range (0.00E+00,1.71E+02) resulting in fit range (5.57E-01,3.75E+00)
    linear_norm_parameter_normdata = {"error": 0.19331234976840278}
    min_max_norm_parameter = (
        1.2630439561525542,
        7.515443068519551,
    )  # error of 1.49E-02 with sample range (0.00E+00,1.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.014916965114520562}
    sigmoidal_norm_parameter = (
        4.367102956502307,
        0.8007786530186061,
    )  # error of 9.05E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (2.94E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.009052151867426368}
    dual_sigmoidal_norm_parameter = (
        4.1534479232479296,
        0.9876358257117317,
        0.6617786555883881,
    )  # error of 4.81E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (1.63E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.004812422686568668}
    genlog_norm_parameter = (
        0.5934186785372179,
        0.2070623538578317,
        1.481913928120067,
        0.18363056680011278,
    )  # error of 3.35E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (4.70E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0033525886273823996}
    autogen_normdata = {
        "sample_bounds": [[0.0, 171.0], [0.004702506571571493, 1.0]],
        "sample_bounds99": [[0.0, 13.0], [0.028197857960620926, 0.9977529884904796]],
    }
    preferred_normalization = "genlog"
    # functions


class NumHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumHBA)
    # normalization
    linear_norm_parameter = (
        0.018653528399554697,
        0.5574729713710027,
    )  # error of 1.93E-01 with sample range (0.00E+00,1.71E+02) resulting in fit range (5.57E-01,3.75E+00)
    linear_norm_parameter_normdata = {"error": 0.19331234976840278}
    min_max_norm_parameter = (
        1.2630439561525542,
        7.515443068519551,
    )  # error of 1.49E-02 with sample range (0.00E+00,1.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.014916965114520562}
    sigmoidal_norm_parameter = (
        4.367102956502307,
        0.8007786530186061,
    )  # error of 9.05E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (2.94E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.009052151867426368}
    dual_sigmoidal_norm_parameter = (
        4.1534479232479296,
        0.9876358257117317,
        0.6617786555883881,
    )  # error of 4.81E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (1.63E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.004812422686568668}
    genlog_norm_parameter = (
        0.5934186785372179,
        0.2070623538578317,
        1.481913928120067,
        0.18363056680011278,
    )  # error of 3.35E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (4.70E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0033525886273823996}
    autogen_normdata = {
        "sample_bounds": [[0.0, 171.0], [0.004702506571571493, 1.0]],
        "sample_bounds99": [[0.0, 13.0], [0.028197857960620926, 0.9977529884904796]],
    }
    preferred_normalization = "genlog"
    # functions


class NumHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumHBD)
    # normalization
    linear_norm_parameter = (
        0.012771534772396786,
        0.8154539249362293,
    )  # error of 1.08E-01 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.15E-01,1.62E+00)
    linear_norm_parameter_normdata = {"error": 0.1079948061142216}
    min_max_norm_parameter = (
        4.930380657631324e-32,
        2.5821101687257144,
    )  # error of 1.47E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.014740703371160535}
    sigmoidal_norm_parameter = (
        1.192951047473294,
        1.324568319014574,
    )  # error of 5.53E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (1.71E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.005531306820770444}
    dual_sigmoidal_norm_parameter = (
        1.0347358874847317,
        7.806096732465813,
        1.1607918751317259,
    )  # error of 2.70E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (3.10E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0027039624550877803}
    genlog_norm_parameter = (
        1.2342450292077596,
        -3.1860939154279575,
        0.010540153782624365,
        6.628023223700797e-05,
    )  # error of 8.35E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.43E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008352134919810262}
    autogen_normdata = {
        "sample_bounds": [[0.0, 63.0], [0.04433667261063756, 1.0]],
        "sample_bounds99": [[0.0, 5.0], [0.40374354066600154, 0.9981073591839424]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumHDonors_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumHDonors)
    # normalization
    linear_norm_parameter = (
        0.012771534772396786,
        0.8154539249362293,
    )  # error of 1.08E-01 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.15E-01,1.62E+00)
    linear_norm_parameter_normdata = {"error": 0.1079948061142216}
    min_max_norm_parameter = (
        4.930380657631324e-32,
        2.5821101687257144,
    )  # error of 1.47E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.014740703371160535}
    sigmoidal_norm_parameter = (
        1.192951047473294,
        1.324568319014574,
    )  # error of 5.53E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (1.71E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.005531306820770444}
    dual_sigmoidal_norm_parameter = (
        1.0347358874847317,
        7.806096732465813,
        1.1607918751317259,
    )  # error of 2.70E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (3.10E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0027039624550877803}
    genlog_norm_parameter = (
        1.2342450292077596,
        -3.1860939154279575,
        0.010540153782624365,
        6.628023223700797e-05,
    )  # error of 8.35E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.43E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008352134919810262}
    autogen_normdata = {
        "sample_bounds": [[0.0, 63.0], [0.04433667261063756, 1.0]],
        "sample_bounds99": [[0.0, 5.0], [0.40374354066600154, 0.9981073591839424]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumHeavyAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32

    # normalization
    # functions
    def featurize(self, mol):
        return mol.GetNumHeavyAtoms()


class NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumHeteroatoms)
    # normalization
    linear_norm_parameter = (
        0.016077495185775725,
        0.5033528543882042,
    )  # error of 2.02E-01 with sample range (0.00E+00,2.15E+02) resulting in fit range (5.03E-01,3.96E+00)
    linear_norm_parameter_normdata = {"error": 0.2024606514251501}
    min_max_norm_parameter = (
        1.9439247544979925,
        10.63914346171618,
    )  # error of 1.53E-02 with sample range (0.00E+00,2.15E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.015335579239956843}
    sigmoidal_norm_parameter = (
        6.309633287283373,
        0.5961308359128846,
    )  # error of 8.73E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (2.27E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.008729951760481431}
    dual_sigmoidal_norm_parameter = (
        6.055674915647298,
        0.7152110168610164,
        0.5023619314244931,
    )  # error of 5.00E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (1.30E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.004996061706823595}
    genlog_norm_parameter = (
        0.45068032906750594,
        3.1700201560253234,
        0.6797360113219059,
        0.23628960935552892,
    )  # error of 3.24E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (3.38E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0032379205009833148}
    autogen_normdata = {
        "sample_bounds": [[0.0, 215.0], [0.003378088839170628, 1.0]],
        "sample_bounds99": [[0.0, 18.0], [0.012667187524391483, 0.9977096060465817]],
    }
    preferred_normalization = "genlog"
    # functions


class NumHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumHeterocycles)
    # normalization
    linear_norm_parameter = (
        0.08010064808252315,
        0.5457901645984795,
    )  # error of 1.04E-01 with sample range (0.00E+00,5.10E+01) resulting in fit range (5.46E-01,4.63E+00)
    linear_norm_parameter_normdata = {"error": 0.10416067120179412}
    min_max_norm_parameter = (
        2.3665827156630354e-30,
        2.5826903013211857,
    )  # error of 2.27E-02 with sample range (0.00E+00,5.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.022664869373203837}
    sigmoidal_norm_parameter = (
        1.1686560826020225,
        1.2821472679988912,
    )  # error of 9.61E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (1.83E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0009607625164504321}
    dual_sigmoidal_norm_parameter = (
        1.1481821648217836,
        1.4759403526269972,
        1.260097047842649,
    )  # error of 4.81E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (1.55E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00048104744459832427}
    genlog_norm_parameter = (
        1.2460769316097229,
        0.8043155234342317,
        1.2559544507804246,
        0.847538805817889,
    )  # error of 4.68E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (1.73E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00046843741557276884}
    autogen_normdata = {
        "sample_bounds": [[0.0, 51.0], [0.17309267384586693, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.4455401684971587, 0.9730555366817043]],
    }
    preferred_normalization = "genlog"
    # functions


class NumLipinskiHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBA)
    # normalization
    linear_norm_parameter = (
        0.0021474686509596053,
        0.8235464950362099,
    )  # error of 1.92E-01 with sample range (0.00E+00,1.96E+02) resulting in fit range (8.24E-01,1.24E+00)
    linear_norm_parameter_normdata = {"error": 0.1921585081293036}
    min_max_norm_parameter = (
        1.2880062839960937,
        9.187763624570907,
    )  # error of 1.38E-02 with sample range (0.00E+00,1.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.013846698638782243}
    sigmoidal_norm_parameter = (
        5.184102720727361,
        0.6674369952484219,
    )  # error of 7.25E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (3.05E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.007251569533280246}
    dual_sigmoidal_norm_parameter = (
        5.01372760473936,
        0.7649977970154859,
        0.5850210479926004,
    )  # error of 5.05E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (2.11E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.005050968151679584}
    genlog_norm_parameter = (
        0.5268273795803626,
        1.5540721740088757,
        1.7263012695643103,
        0.3474832434074276,
    )  # error of 3.62E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (1.02E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.003624850298533195}
    autogen_normdata = {
        "sample_bounds": [[0.0, 196.0], [0.01023326561734317, 1.0]],
        "sample_bounds99": [[0.0, 15.0], [0.031876227895358995, 0.9975438175592366]],
    }
    preferred_normalization = "genlog"
    # functions


class NumLipinskiHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBD)
    # normalization
    linear_norm_parameter = (
        0.011451805303745677,
        0.8103893376864633,
    )  # error of 1.09E-01 with sample range (0.00E+00,8.20E+01) resulting in fit range (8.10E-01,1.75E+00)
    linear_norm_parameter_normdata = {"error": 0.1093936401741258}
    min_max_norm_parameter = (
        9.860761315262648e-32,
        3.187051718282465,
    )  # error of 1.94E-02 with sample range (0.00E+00,8.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019444687584919223}
    sigmoidal_norm_parameter = (
        1.3178308349589882,
        1.1160540358868796,
    )  # error of 6.16E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.87E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.006159518941504394}
    dual_sigmoidal_norm_parameter = (
        1.0743143001638746,
        5.214591634885128,
        0.9332858170955084,
    )  # error of 3.15E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (3.68E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.003147962480995746}
    genlog_norm_parameter = (
        1.0491738144723946,
        -4.348296718164784,
        0.017728446734426147,
        6.444988862716029e-05,
    )  # error of 9.99E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (5.66E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009992878038903702}
    autogen_normdata = {
        "sample_bounds": [[0.0, 82.0], [0.0566152982336769, 1.0]],
        "sample_bounds99": [[0.0, 6.0], [0.36577793806998565, 0.998145659024973]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumRadicalElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumRadicalElectrons)
    # normalization
    linear_norm_parameter = (
        0.0019698227137791013,
        0.9930256276934915,
    )  # error of 3.18E-04 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.93E-01,1.24E+00)
    linear_norm_parameter_normdata = {"error": 0.0003181694162865134}
    min_max_norm_parameter = (
        7.363905475264589e-09,
        1.0052570210496823,
    )  # error of 7.40E-04 with sample range (0.00E+00,1.26E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0007396612332801185}
    sigmoidal_norm_parameter = (
        -6.451319539635213,
        0.7043741012907843,
    )  # error of 5.68E-06 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.89E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 5.675729854534536e-06}
    dual_sigmoidal_norm_parameter = (
        -6.45131932074973,
        1.0,
        0.7043741212164152,
    )  # error of 5.68E-06 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.89E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.6757298545262665e-06}
    genlog_norm_parameter = (
        0.7169297047257565,
        -2.139202514537995,
        0.611771246584949,
        11.916658171809683,
    )  # error of 9.45E-06 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.90E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 9.45363897789996e-06}
    autogen_normdata = {"sample_bounds": [[0.0, 126.0], [0.9896504946137776, 1.0]]}
    preferred_normalization = "unity"
    # functions


class NumRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumRings)
    # normalization
    linear_norm_parameter = (
        0.03860180770005206,
        0.5113290906759573,
    )  # error of 1.73E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (5.11E-01,2.87E+00)
    linear_norm_parameter_normdata = {"error": 0.17329582361047302}
    min_max_norm_parameter = (
        0.334517739633367,
        5.208998938983079,
    )  # error of 1.60E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.01600245280122771}
    sigmoidal_norm_parameter = (
        2.7238161788458215,
        1.1018114352614388,
    )  # error of 1.08E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.74E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.010810851867941116}
    dual_sigmoidal_norm_parameter = (
        2.5715740496764488,
        1.283226429101575,
        0.9174925881562721,
    )  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (3.56E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0061287005299005375}
    genlog_norm_parameter = (
        0.813320479300628,
        -1.296799001575922,
        3.1048252219308727,
        0.173898726052091,
    )  # error of 4.52E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.48E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.004520854109330362}
    autogen_normdata = {
        "sample_bounds": [[0.0, 61.0], [0.014767969276556474, 1.0]],
        "sample_bounds99": [[0.0, 9.0], [0.10515038913140289, 0.998176048141036]],
    }
    preferred_normalization = "genlog"
    # functions


class NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumRotatableBonds)
    # normalization
    linear_norm_parameter = (
        0.0016857886272905187,
        0.8125456140318361,
    )  # error of 1.82E-01 with sample range (0.00E+00,2.48E+02) resulting in fit range (8.13E-01,1.23E+00)
    linear_norm_parameter_normdata = {"error": 0.1817541602892939}
    min_max_norm_parameter = (
        1.6292528997825728,
        13.174098358632333,
    )  # error of 1.89E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.018912500281580716}
    sigmoidal_norm_parameter = (
        7.304462270693269,
        0.4542440447080067,
    )  # error of 1.34E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (3.50E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.013383312721439285}
    dual_sigmoidal_norm_parameter = (
        6.70900145999476,
        0.6333818528506181,
        0.3400706824599175,
    )  # error of 8.05E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (1.41E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.008053243446220083}
    genlog_norm_parameter = (
        0.31189865208337914,
        -9.114334199023174,
        0.006438735255284094,
        6.04991692778764e-05,
    )  # error of 6.90E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (2.03E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006897938936309171}
    autogen_normdata = {
        "sample_bounds": [[0.0, 248.0], [0.0020297781209821183, 1.0]],
        "sample_bounds99": [[0.0, 34.0], [0.010685707622442048, 0.9998874147561215]],
    }
    preferred_normalization = "genlog"
    # functions


class NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumSaturatedCarbocycles)
    # normalization
    linear_norm_parameter = (
        0.020972112507738894,
        0.9098571128596467,
    )  # error of 1.95E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (9.10E-01,1.29E+00)
    linear_norm_parameter_normdata = {"error": 0.019497660894072805}
    min_max_norm_parameter = (
        8.1643240157463e-09,
        1.1069248512275087,
    )  # error of 6.94E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.006944331170706256}
    sigmoidal_norm_parameter = (
        -0.6176260310959208,
        1.385111849563034,
    )  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0025959251447008032}
    dual_sigmoidal_norm_parameter = (
        -0.6176260310959208,
        1.0,
        1.385111849563034,
    )  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0025959251447008032}
    genlog_norm_parameter = (
        1.34934732668474,
        -4.239877299828134,
        0.3156576472934097,
        0.0026523319732207028,
    )  # error of 2.51E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.77E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.002510577728125684}
    autogen_normdata = {
        "sample_bounds": [[0.0, 18.0], [0.6772565652979902, 0.9999999999889493]],
        "sample_bounds99": [[0.0, 2.0], [0.9038138014803501, 0.9932167596465881]],
    }
    preferred_normalization = "min_max"
    # functions


class NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumSaturatedHeterocycles)
    # normalization
    linear_norm_parameter = (
        0.06338179563839252,
        0.7789048985591301,
    )  # error of 4.13E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.79E-01,2.68E+00)
    linear_norm_parameter_normdata = {"error": 0.041274621876909896}
    min_max_norm_parameter = (
        5.3308083108781935e-09,
        1.2539605902115512,
    )  # error of 8.89E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00888924278300703}
    sigmoidal_norm_parameter = (
        0.24911412739948294,
        1.825541426846478,
    )  # error of 5.15E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.88E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0005148763310441382}
    dual_sigmoidal_norm_parameter = (
        0.24895656142426995,
        70.18656709264866,
        1.8251784944463916,
    )  # error of 3.32E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (2.58E-08,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00033202723822699947}
    genlog_norm_parameter = (
        3.040454535261453,
        -1.8346527987667436,
        0.2657189128955128,
        0.00020823617864693198,
    )  # error of 6.76E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (8.06E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006758222733162049}
    autogen_normdata = {
        "sample_bounds": [[0.0, 30.0], [0.00806417751586015, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.7940689077655179, 0.9890353059522397]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumSaturatedRings)
    # normalization
    linear_norm_parameter = (
        0.037486269092924296,
        0.7906195585254473,
    )  # error of 6.21E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.91E-01,1.92E+00)
    linear_norm_parameter_normdata = {"error": 0.06209440565791554}
    min_max_norm_parameter = (
        5.730855425109739e-09,
        1.404946368678208,
    )  # error of 1.95E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019504724572593377}
    sigmoidal_norm_parameter = (
        0.36617461217911906,
        1.4317359459562997,
    )  # error of 2.83E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.72E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.002826421160208341}
    dual_sigmoidal_norm_parameter = (
        0.36616672566972636,
        42.588115306732064,
        1.4317211909754344,
    )  # error of 1.65E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (1.69E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0016458490370013074}
    genlog_norm_parameter = (
        1.3158763699929032,
        -4.224399447110491,
        0.23204276022801856,
        0.0007067117749192951,
    )  # error of 2.06E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (2.82E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0020622735171694833}
    autogen_normdata = {
        "sample_bounds": [[0.0, 30.0], [0.2823294185339797, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.7122346710518005, 0.9934710852353208]],
    }
    preferred_normalization = "dual_sig"
    # functions


class NumSpiroAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(CalcNumSpiroAtoms)
    # normalization
    linear_norm_parameter = (
        0.4935155835974765,
        0.49351558359747627,
    )  # error of 2.22E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.94E-01,1.97E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        1.3202332968823136e-08,
        1.0131392331517604,
    )  # error of 4.00E-04 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0003999640032396612}
    sigmoidal_norm_parameter = (
        -1.0813823993563174,
        2.081382361017821,
    )  # error of 2.19E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.05E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 2.187305891965252e-11}
    dual_sigmoidal_norm_parameter = (
        -1.0813823993563174,
        1.0,
        2.081382361017821,
    )  # error of 2.19E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.05E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.187305891965252e-11}
    genlog_norm_parameter = (
        1.1030658072817703,
        0.29757561708535163,
        0.05236067323455242,
        1.8263575666538847,
    )  # error of 1.08E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.62E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 1.0774575009975251e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 3.0], [0.9623011850125086, 0.9985481670134284]],
        "sample_bounds99": [[0.0, 0.0], [0.9870311779695276, 0.9870311779695276]],
    }
    preferred_normalization = "min_max"
    # functions


class NumValenceElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(NumValenceElectrons)
    # normalization
    linear_norm_parameter = (
        0.0012640047970202684,
        0.37615455634302164,
    )  # error of 2.30E-01 with sample range (9.00E+00,2.83E+03) resulting in fit range (3.88E-01,3.96E+00)
    linear_norm_parameter_normdata = {"error": 0.2296027541571038}
    min_max_norm_parameter = (
        65.29068287368149,
        197.1901127519803,
    )  # error of 2.92E-02 with sample range (9.00E+00,2.83E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.029207388884198614}
    sigmoidal_norm_parameter = (
        130.58040996825486,
        0.038386270577942194,
    )  # error of 2.24E-02 with sample range (9.00E+00,2.83E+03) resulting in fit range (9.31E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02244995759756559}
    dual_sigmoidal_norm_parameter = (
        122.45399947791547,
        0.05607235692152875,
        0.02764386723353996,
    )  # error of 9.38E-03 with sample range (9.00E+00,2.83E+03) resulting in fit range (1.72E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.009383551758683696}
    genlog_norm_parameter = (
        0.02628541624509447,
        -166.9267768792304,
        0.0743891542250422,
        4.704587090742882e-05,
    )  # error of 8.31E-03 with sample range (9.00E+00,2.83E+03) resulting in fit range (1.84E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.008307502980540072}
    autogen_normdata = {
        "sample_bounds": [[9.0, 2834.0], [1.8430932604587123e-07, 1.0]],
        "sample_bounds99": [[9.0, 317.0], [0.009762038989055137, 0.9991878321756301]],
    }
    preferred_normalization = "genlog"
    # functions


class NumberAtomsAc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(89)))


class NumberAtomsAg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(47)))


class NumberAtomsAl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(13)))


class NumberAtomsAm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(95)))


class NumberAtomsAr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(18)))


class NumberAtomsAs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(33)))


class NumberAtomsAt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(85)))


class NumberAtomsAu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(79)))


class NumberAtomsB_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(5)))


class NumberAtomsBa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(56)))


class NumberAtomsBe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(4)))


class NumberAtomsBh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(107)))


class NumberAtomsBi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(83)))


class NumberAtomsBk_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(97)))


class NumberAtomsBr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(35)))


class NumberAtomsC_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6)))


class NumberAtomsCa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(20)))


class NumberAtomsCd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(48)))


class NumberAtomsCe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(58)))


class NumberAtomsCf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(98)))


class NumberAtomsCl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(17)))


class NumberAtomsCm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(96)))


class NumberAtomsCn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(112)))


class NumberAtomsCo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(27)))


class NumberAtomsCr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(24)))


class NumberAtomsCs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(55)))


class NumberAtomsCu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(29)))


class NumberAtomsDb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(105)))


class NumberAtomsDs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(110)))


class NumberAtomsDy_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(66)))


class NumberAtomsEr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(68)))


class NumberAtomsEs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(99)))


class NumberAtomsEu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(63)))


class NumberAtomsF_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(9)))


class NumberAtomsFe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(26)))


class NumberAtomsFl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(114)))


class NumberAtomsFm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(100)))


class NumberAtomsFr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(87)))


class NumberAtomsGa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(31)))


class NumberAtomsGd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(64)))


class NumberAtomsGe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(32)))


class NumberAtomsH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1)))


class NumberAtomsHe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(2)))


class NumberAtomsHf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(72)))


class NumberAtomsHg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(80)))


class NumberAtomsHo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(67)))


class NumberAtomsHs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(108)))


class NumberAtomsI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(53)))


class NumberAtomsIn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(49)))


class NumberAtomsIr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(77)))


class NumberAtomsK_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(19)))


class NumberAtomsKr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(36)))


class NumberAtomsLa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(57)))


class NumberAtomsLi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(3)))


class NumberAtomsLr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(103)))


class NumberAtomsLu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(71)))


class NumberAtomsLv_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(116)))


class NumberAtomsMc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(115)))


class NumberAtomsMd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(101)))


class NumberAtomsMg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(12)))


class NumberAtomsMn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(25)))


class NumberAtomsMo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(42)))


class NumberAtomsMt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(109)))


class NumberAtomsN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(7)))


class NumberAtomsNa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(11)))


class NumberAtomsNb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(41)))


class NumberAtomsNd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(60)))


class NumberAtomsNe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(10)))


class NumberAtomsNh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(113)))


class NumberAtomsNi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(28)))


class NumberAtomsNo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(102)))


class NumberAtomsNp_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(93)))


class NumberAtomsO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(8)))


class NumberAtomsOg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(118)))


class NumberAtomsOs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(76)))


class NumberAtomsP_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(15)))


class NumberAtomsPa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(91)))


class NumberAtomsPb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(82)))


class NumberAtomsPd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(46)))


class NumberAtomsPm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(61)))


class NumberAtomsPo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(84)))


class NumberAtomsPr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(59)))


class NumberAtomsPt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(78)))


class NumberAtomsPu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(94)))


class NumberAtomsRa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(88)))


class NumberAtomsRb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(37)))


class NumberAtomsRe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(75)))


class NumberAtomsRf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(104)))


class NumberAtomsRg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(111)))


class NumberAtomsRh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(45)))


class NumberAtomsRn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(86)))


class NumberAtomsRu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(44)))


class NumberAtomsS_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(16)))


class NumberAtomsSb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(51)))


class NumberAtomsSc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(21)))


class NumberAtomsSe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(34)))


class NumberAtomsSg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(106)))


class NumberAtomsSi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(14)))


class NumberAtomsSm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(62)))


class NumberAtomsSn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(50)))


class NumberAtomsSr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(38)))


class NumberAtomsTa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(73)))


class NumberAtomsTb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(65)))


class NumberAtomsTc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(43)))


class NumberAtomsTe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(52)))


class NumberAtomsTh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(90)))


class NumberAtomsTi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(22)))


class NumberAtomsTl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(81)))


class NumberAtomsTm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(69)))


class NumberAtomsTs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(117)))


class NumberAtomsU_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(92)))


class NumberAtomsV_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(23)))


class NumberAtomsW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(74)))


class NumberAtomsXe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(54)))


class NumberAtomsY_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(39)))


class NumberAtomsYb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(70)))


class NumberAtomsZn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(30)))


class NumberAtomsZr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.uint32

    # normalization
    # functions
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(40)))


class PBF_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcPBF)
    # normalization
    linear_norm_parameter = (
        0.8002336508637942,
        -0.27832921776549496,
    )  # error of 1.06E-01 with sample range (0.00E+00,4.59E+00) resulting in fit range (-2.78E-01,3.39E+00)
    linear_norm_parameter_normdata = {"error": 0.10613828372857644}
    min_max_norm_parameter = (
        0.5464675217564985,
        1.3692618279142443,
    )  # error of 3.39E-02 with sample range (0.00E+00,4.59E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03387838859347443}
    sigmoidal_norm_parameter = (
        0.9571543856358279,
        5.963418540555978,
    )  # error of 8.11E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (3.31E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.008105692980017449}
    dual_sigmoidal_norm_parameter = (
        0.9519198242022207,
        6.247989180945802,
        5.682171774716921,
    )  # error of 6.60E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (2.61E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.006603729479400456}
    genlog_norm_parameter = (
        5.524918699735243,
        0.9719868061024474,
        0.6460728455481474,
        0.7751201578131506,
    )  # error of 6.72E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (1.71E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006719819784728546}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 4.585230827331543],
            [0.0017056190576018558, 0.99999999821703],
        ],
        "sample_bounds99": [
            [0.0, 1.809682846069336],
            [0.004775500328810231, 0.9977558993776061],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA1)
    # normalization
    linear_norm_parameter = (
        0.005573163277374748,
        0.6734703650713099,
    )  # error of 1.59E-01 with sample range (0.00E+00,4.11E+02) resulting in fit range (6.73E-01,2.97E+00)
    linear_norm_parameter_normdata = {"error": 0.15940926074855274}
    min_max_norm_parameter = (
        1.2124620489038405,
        22.986430313946542,
    )  # error of 4.38E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.043769349196678485}
    sigmoidal_norm_parameter = (
        12.034307131924646,
        0.21632262910541178,
    )  # error of 2.81E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (6.89E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.028111238324454}
    dual_sigmoidal_norm_parameter = (
        11.236970449779486,
        0.3464811225654301,
        0.18910464232260876,
    )  # error of 2.51E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (2.00E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.02510161141335076}
    genlog_norm_parameter = (
        0.17343207450751735,
        -3.946557704540659,
        0.865536383761228,
        0.0790595972609236,
    )  # error of 2.56E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (1.02E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.025560221574969476}
    autogen_normdata = {
        "sample_bounds": [[0.0, 411.38494873046875], [0.010235789461828855, 1.0]],
        "sample_bounds99": [
            [0.0, 49.62247085571289],
            [0.06868584214543348, 0.9989983811029488],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA10)
    # normalization
    linear_norm_parameter = (
        0.008224721394031964,
        0.6823272493410806,
    )  # error of 9.26E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (6.82E-01,4.36E+00)
    linear_norm_parameter_normdata = {"error": 0.09261880428634882}
    min_max_norm_parameter = (
        3.4024230320080053e-25,
        18.337088617478788,
    )  # error of 7.24E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0724310196412757}
    sigmoidal_norm_parameter = (
        6.490192767810458,
        0.173177943098159,
    )  # error of 1.90E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (2.45E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.018981852239564542}
    dual_sigmoidal_norm_parameter = (
        6.250108397838072,
        0.3448712479918515,
        0.1680578064754544,
    )  # error of 1.70E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (1.04E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.016967648912387605}
    genlog_norm_parameter = (
        0.14813140069086117,
        -40.294705469197034,
        0.08035258299320733,
        0.00010994575203479595,
    )  # error of 1.74E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (1.54E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.017443822736165078}
    autogen_normdata = {
        "sample_bounds": [[0.0, 446.54827880859375], [0.15434273980068333, 1.0]],
        "sample_bounds99": [
            [0.0, 39.10489273071289],
            [0.24489156633602807, 0.99446793765598],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA11)
    # normalization
    linear_norm_parameter = (
        0.00828695135755142,
        0.7644503184456655,
    )  # error of 5.96E-02 with sample range (0.00E+00,1.99E+02) resulting in fit range (7.64E-01,2.41E+00)
    linear_norm_parameter_normdata = {"error": 0.0596310156837373}
    min_max_norm_parameter = (
        7.634541570684105e-30,
        11.947029795426264,
    )  # error of 7.48E-02 with sample range (0.00E+00,1.99E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.07478432967341819}
    sigmoidal_norm_parameter = (
        2.445806162262592,
        0.204409252101571,
    )  # error of 7.55E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (3.78E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00755352638321553}
    dual_sigmoidal_norm_parameter = (
        2.416316669281992,
        0.37652639481703293,
        0.2036970883656592,
    )  # error of 7.44E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (2.87E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.007435575328623936}
    genlog_norm_parameter = (
        0.188105450084389,
        -5.123490506719317,
        0.7826648784562164,
        0.23410080884517526,
    )  # error of 7.17E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (3.28E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007167224087189206}
    autogen_normdata = {
        "sample_bounds": [[0.0, 198.5148162841797], [0.32759562425646765, 1.0]],
        "sample_bounds99": [
            [0.0, 26.611743927001953],
            [0.41455780361262257, 0.9916401361591761],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA12)
    # normalization
    linear_norm_parameter = (
        0.006473147723920936,
        0.807818201399682,
    )  # error of 7.84E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (8.08E-01,2.45E+00)
    linear_norm_parameter_normdata = {"error": 0.07835812368470209}
    min_max_norm_parameter = (
        1.6316394719639418e-20,
        8.845548026863177,
    )  # error of 6.74E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06744206022668735}
    sigmoidal_norm_parameter = (
        2.364978109571876,
        0.25437535485941726,
    )  # error of 2.74E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (3.54E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.027386895294874484}
    dual_sigmoidal_norm_parameter = (
        2.4728143895684482,
        -0.14054045028871964,
        0.25875435969354915,
    )  # error of 2.72E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (5.86E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.02718526476633664}
    genlog_norm_parameter = (
        0.23768272482675293,
        0.5325488077852092,
        0.4927515507994007,
        0.38011078236238405,
    )  # error of 2.73E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (3.11E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.027277672793453698}
    autogen_normdata = {
        "sample_bounds": [[0.0, 253.8263397216797], [0.31079970963902614, 1.0]],
        "sample_bounds99": [
            [0.0, 23.09967613220215],
            [0.4127739039634941, 0.9940027518241187],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class PEOE_VSA13_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA13)
    # normalization
    linear_norm_parameter = (
        0.014179141123108696,
        0.7394291555972963,
    )  # error of 6.57E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (7.39E-01,2.50E+00)
    linear_norm_parameter_normdata = {"error": 0.06566889709056924}
    min_max_norm_parameter = (
        3.944304526105059e-30,
        7.6449284829185435,
    )  # error of 6.49E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06488608045856947}
    sigmoidal_norm_parameter = (
        1.5958556965144552,
        0.30276265905030647,
    )  # error of 3.71E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (3.82E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.037098161282851656}
    dual_sigmoidal_norm_parameter = (
        2.218412985546247,
        -0.46896144663283507,
        0.34298637709403673,
    )  # error of 3.52E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (7.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.03517704895082876}
    genlog_norm_parameter = (
        0.3872544233666115,
        8.961606530328575,
        0.37919178378095403,
        3.2511536595656523,
    )  # error of 3.62E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (4.52E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.03622002877919463}
    autogen_normdata = {
        "sample_bounds": [[0.0, 124.05077362060547], [0.4522933139694446, 1.0]],
        "sample_bounds99": [
            [0.0, 17.22934913635254],
            [0.5047804899315141, 0.9953145072095371],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class PEOE_VSA14_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA14)
    # normalization
    linear_norm_parameter = (
        0.0006563772490514073,
        0.9227826193892867,
    )  # error of 7.32E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (9.23E-01,1.26E+00)
    linear_norm_parameter_normdata = {"error": 0.07316637800712694}
    min_max_norm_parameter = (
        2.415003722890725e-18,
        8.09692280816151,
    )  # error of 5.41E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05414658809686194}
    sigmoidal_norm_parameter = (
        0.8016925269391764,
        0.23235287767416216,
    )  # error of 3.04E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (4.54E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.030386458516873646}
    dual_sigmoidal_norm_parameter = (
        0.8017086331085084,
        13.229489438503897,
        0.2323534820925018,
    )  # error of 2.97E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (2.48E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.029695136752406693}
    genlog_norm_parameter = (
        0.22222598504567376,
        0.3101295335912875,
        0.574317312660232,
        0.5782220229296028,
    )  # error of 3.04E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (4.36E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.030362626892270143}
    autogen_normdata = {
        "sample_bounds": [[0.0, 509.8097229003906], [0.43635634574372073, 1.0]],
        "sample_bounds99": [
            [0.0, 33.057403564453125],
            [0.5080296235971603, 0.9993387683587578],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA2)
    # normalization
    linear_norm_parameter = (
        0.009229128833342298,
        0.6965819780408643,
    )  # error of 1.14E-01 with sample range (0.00E+00,4.96E+02) resulting in fit range (6.97E-01,5.27E+00)
    linear_norm_parameter_normdata = {"error": 0.11376510393881116}
    min_max_norm_parameter = (
        0.0,
        15.384470706540956,
    )  # error of 6.59E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06588214388768879}
    sigmoidal_norm_parameter = (
        6.006780263970044,
        0.24361243133391997,
    )  # error of 3.95E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (1.88E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.03946741942274695}
    dual_sigmoidal_norm_parameter = (
        5.098681834550195,
        0.9896101152028739,
        0.21213507995825584,
    )  # error of 3.59E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (6.40E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.035902132880379425}
    genlog_norm_parameter = (
        0.20264108204920955,
        -20.285121176464397,
        0.0423527936394588,
        0.0002989437005061263,
    )  # error of 3.87E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (9.80E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.03868531021684446}
    autogen_normdata = {
        "sample_bounds": [[0.0, 496.0393371582031], [0.09804092324073793, 1.0]],
        "sample_bounds99": [
            [0.0, 28.915647506713867],
            [0.1694768724181753, 0.9934578465875649],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class PEOE_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA3)
    # normalization
    linear_norm_parameter = (
        0.011713317780700283,
        0.7004807757872287,
    )  # error of 7.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (7.00E-01,3.33E+00)
    linear_norm_parameter_normdata = {"error": 0.07382649718780694}
    min_max_norm_parameter = (
        8.166483837005377e-17,
        11.680862968336768,
    )  # error of 5.66E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05662678244898566}
    sigmoidal_norm_parameter = (
        3.964877055534785,
        0.24576276616405693,
    )  # error of 2.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (2.74E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02377467696990656}
    dual_sigmoidal_norm_parameter = (
        3.970655618436759,
        0.20163568378595578,
        0.2460410246957036,
    )  # error of 2.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (3.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.023756220307555725}
    genlog_norm_parameter = (
        0.21940802278396696,
        -0.5798243214656876,
        0.3524908089908593,
        0.17056850428594234,
    )  # error of 2.35E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (2.05E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.023498904738141313}
    autogen_normdata = {
        "sample_bounds": [[0.0, 224.58233642578125], [0.2049859526454392, 1.0]],
        "sample_bounds99": [
            [0.0, 23.439016342163086],
            [0.29743936509593155, 0.9894926249208685],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PEOE_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA4)
    # normalization
    linear_norm_parameter = (
        0.005054555707632469,
        0.8464248056257232,
    )  # error of 4.88E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (8.46E-01,1.98E+00)
    linear_norm_parameter_normdata = {"error": 0.048842094781865325}
    min_max_norm_parameter = (
        4.919925289519856e-27,
        6.338586899034091,
    )  # error of 6.33E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06328882844110144}
    sigmoidal_norm_parameter = (
        -1.5531847169925812,
        0.19004989659369026,
    )  # error of 1.60E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.73E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.015957260461116812}
    dual_sigmoidal_norm_parameter = (
        -1.553184502245005,
        2.4359009370888938,
        0.19004990078391787,
    )  # error of 1.60E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.73E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01595726046111681}
    genlog_norm_parameter = (
        0.20017422745235175,
        -3.6779586876784567,
        3.0860639286955327,
        1.7123190047115404,
    )  # error of 1.59E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.89E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.015928354122895356}
    autogen_normdata = {
        "sample_bounds": [[0.0, 223.91116333007812], [0.5886368470694118, 1.0]],
        "sample_bounds99": [
            [0.0, 24.863502502441406],
            [0.6343276846777406, 0.9941777960895415],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class PEOE_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA5)
    # normalization
    linear_norm_parameter = (
        0.006570381366511557,
        0.7983008331386301,
    )  # error of 4.66E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (7.98E-01,1.71E+00)
    linear_norm_parameter_normdata = {"error": 0.046553062855522125}
    min_max_norm_parameter = (
        5e-324,
        6.468041922682696,
    )  # error of 8.78E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.08784119333864027}
    sigmoidal_norm_parameter = (
        -3.9038317058155574,
        0.13911276842288417,
    )  # error of 2.41E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.33E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.024085271963182508}
    dual_sigmoidal_norm_parameter = (
        -3.903831899586245,
        1.0,
        0.1391127663161195,
    )  # error of 2.41E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.33E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.02408527196317994}
    genlog_norm_parameter = (
        0.3631163541168652,
        13.097694456290501,
        3.7300629160720957,
        16.89236467331696,
    )  # error of 2.06E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.98E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.02061127002636795}
    autogen_normdata = {
        "sample_bounds": [[0.0, 138.81100463867188], [0.6979463305829587, 1.0]],
        "sample_bounds99": [
            [0.0, 23.729320526123047],
            [0.7193872999036939, 0.9956098745316877],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class PEOE_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA6)
    # normalization
    linear_norm_parameter = (
        0.0023302771680984535,
        0.6632167817066577,
    )  # error of 1.34E-01 with sample range (0.00E+00,1.31E+03) resulting in fit range (6.63E-01,3.70E+00)
    linear_norm_parameter_normdata = {"error": 0.1344211049776361}
    min_max_norm_parameter = (
        1.94951131475506e-20,
        53.568112487136325,
    )  # error of 6.28E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.06276965367360464}
    sigmoidal_norm_parameter = (
        22.943177031905776,
        0.06530117675670048,
    )  # error of 2.26E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (1.83E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.022642971442177993}
    dual_sigmoidal_norm_parameter = (
        20.773256554969635,
        0.11533427848602933,
        0.058051651917743946,
    )  # error of 1.56E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (8.35E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.015559964766865624}
    genlog_norm_parameter = (
        0.052967436982639095,
        -41.27761902140869,
        0.0007740146403896999,
        3.780519374358048e-05,
    )  # error of 1.43E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (1.00E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.014303497345667276}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1305.1624755859375], [0.10029989005167886, 1.0]],
        "sample_bounds99": [
            [0.0, 164.87229919433594],
            [0.15731358786177269, 0.9996551586193185],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PEOE_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA7)
    # normalization
    linear_norm_parameter = (
        0.006731394986636996,
        0.31295466008048334,
    )  # error of 1.50E-01 with sample range (0.00E+00,1.25E+03) resulting in fit range (3.13E-01,8.73E+00)
    linear_norm_parameter_normdata = {"error": 0.1495084351015147}
    min_max_norm_parameter = (
        7.51983930256869,
        68.45113819370312,
    )  # error of 4.09E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04086672694664189}
    sigmoidal_norm_parameter = (
        37.76437056260887,
        0.08135882094622694,
    )  # error of 1.76E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (4.43E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01756874566693887}
    dual_sigmoidal_norm_parameter = (
        36.4821270671721,
        0.10096914890593979,
        0.07316926419086144,
    )  # error of 1.14E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (2.45E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01141732313594319}
    genlog_norm_parameter = (
        0.0646092930736572,
        -11.183615770310546,
        3.0745155731673646,
        0.18492971323280805,
    )  # error of 9.38E-03 with sample range (0.00E+00,1.25E+03) resulting in fit range (7.16E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009382872344320657}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1250.0850830078125], [0.007161919921330218, 1.0]],
        "sample_bounds99": [
            [0.0, 119.45008087158203],
            [0.01869547814356935, 0.9965974083273145],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PEOE_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA8)
    # normalization
    linear_norm_parameter = (
        0.009235349219404814,
        0.2024712092074698,
    )  # error of 1.20E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.02E-01,1.69E+01)
    linear_norm_parameter_normdata = {"error": 0.12008086851370327}
    min_max_norm_parameter = (
        4.887388416091266,
        65.78861410255793,
    )  # error of 3.62E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03623975985294657}
    sigmoidal_norm_parameter = (
        35.028275267622924,
        0.07929972923744978,
    )  # error of 1.39E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (5.85E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.013857972632048741}
    dual_sigmoidal_norm_parameter = (
        33.92323584081409,
        0.0959836874096441,
        0.07177350962439769,
    )  # error of 5.71E-03 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.71E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.005709869990248599}
    genlog_norm_parameter = (
        0.06293663002863742,
        -24.020003909413585,
        5.494310398432799,
        0.1879364280593064,
    )  # error of 1.81E-03 with sample range (0.00E+00,1.80E+03) resulting in fit range (1.46E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0018077267887142707}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1804.5626220703125], [0.01464846519219847, 1.0]],
        "sample_bounds99": [
            [0.0, 113.23653411865234],
            [0.039123446122229726, 0.9954672318007999],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PEOE_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA9)
    # normalization
    linear_norm_parameter = (
        0.009582946298100165,
        0.3577672573591545,
    )  # error of 1.18E-01 with sample range (0.00E+00,9.81E+02) resulting in fit range (3.58E-01,9.76E+00)
    linear_norm_parameter_normdata = {"error": 0.11825998938362119}
    min_max_norm_parameter = (
        7.644079543108024e-15,
        47.75350503201181,
    )  # error of 3.98E-02 with sample range (0.00E+00,9.81E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039769773350963954}
    sigmoidal_norm_parameter = (
        23.587460909634668,
        0.10069901844268675,
    )  # error of 1.27E-02 with sample range (0.00E+00,9.81E+02) resulting in fit range (8.51E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.012687573709107922}
    dual_sigmoidal_norm_parameter = (
        22.77525947240647,
        0.12980123075088157,
        0.09300622071060506,
    )  # error of 5.01E-03 with sample range (0.00E+00,9.81E+02) resulting in fit range (4.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.005006769569429538}
    genlog_norm_parameter = (
        0.08096934062384088,
        -10.659741320803331,
        1.222084000815295,
        0.10931895128869719,
    )  # error of 1.35E-03 with sample range (0.00E+00,9.81E+02) resulting in fit range (2.23E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0013484264878322414}
    autogen_normdata = {
        "sample_bounds": [[0.0, 981.4776000976562], [0.02229825329009178, 1.0]],
        "sample_bounds99": [
            [0.0, 80.62095642089844],
            [0.06577837869966403, 0.9934155711904766],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PMI1)
    # normalization
    linear_norm_parameter = (
        6.013420622833213e-05,
        0.37438762357376315,
    )  # error of 2.26E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (3.74E-01,3.55E+01)
    linear_norm_parameter_normdata = {"error": 0.2264379152766898}
    min_max_norm_parameter = (
        5.253076648348887e-17,
        2920.8284746927293,
    )  # error of 6.82E-02 with sample range (0.00E+00,5.85E+05) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0682415071915629}
    sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
    )  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.576586544882377}
    dual_sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
        1.0,
    )  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.576586544882377}
    genlog_norm_parameter = (
        0.0010903005704858534,
        -1491.8542049268044,
        0.00016755497187333072,
        1.07461994166649e-05,
    )  # error of 4.23E-02 with sample range (0.00E+00,5.85E+05) resulting in fit range (4.66E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.04234266537421787}
    autogen_normdata = {
        "sample_bounds": [[0.0, 584913.1875], [0.0466358544778456, 1.0]],
        "sample_bounds99": [
            [0.0, 10495.0009765625],
            [0.08072263670225632, 0.9999998854464567],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcPMI2)
    # normalization
    linear_norm_parameter = (
        3.364102817740711e-05,
        0.2861109449057394,
    )  # error of 1.90E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.86E-01,2.45E+01)
    linear_norm_parameter_normdata = {"error": 0.19023694320611}
    min_max_norm_parameter = (
        44.665447243295176,
        10171.614030140174,
    )  # error of 5.66E-02 with sample range (0.00E+00,7.21E+05) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.056605366971435034}
    sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
    )  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.5766077877462175}
    dual_sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
        1.0,
    )  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5766077877462175}
    genlog_norm_parameter = (
        0.0003278945802835982,
        -2022.7431105250037,
        4.99666114488119e-05,
        7.74007368127996e-06,
    )  # error of 3.02E-02 with sample range (0.00E+00,7.21E+05) resulting in fit range (3.59E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.030199367847976676}
    autogen_normdata = {
        "sample_bounds": [[0.0, 721074.0625], [0.03594683846941451, 1.0]],
        "sample_bounds99": [
            [0.0, 25639.033203125],
            [0.060501976927303865, 0.9999664067115498],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(PMI3)
    # normalization
    linear_norm_parameter = (
        2.8952313203676817e-05,
        0.2866572166015228,
    )  # error of 1.92E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.87E-01,2.52E+01)
    linear_norm_parameter_normdata = {"error": 0.19225831135617755}
    min_max_norm_parameter = (
        147.52756421954018,
        11758.510463576746,
    )  # error of 5.52E-02 with sample range (0.00E+00,8.59E+05) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05524107671183394}
    sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
    )  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.5766238637518415}
    dual_sigmoidal_norm_parameter = (
        2.8655544226961647,
        2.8655544226961647,
        1.0,
    )  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5766238637518415}
    genlog_norm_parameter = (
        0.0002859890919949101,
        -7568.186233062281,
        0.0002326237441419841,
        7.8102180557447e-06,
    )  # error of 2.91E-02 with sample range (0.00E+00,8.59E+05) resulting in fit range (3.27E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.02913446208915581}
    autogen_normdata = {
        "sample_bounds": [[0.0, 859342.6875], [0.03272174193613442, 1.0]],
        "sample_bounds99": [
            [0.0, 29880.111328125],
            [0.058977674007790616, 0.9999681760061229],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class Phi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(CalcPhi)
    # normalization
    linear_norm_parameter = (
        0.22667929243030271,
        0.11757079564019635,
    )  # error of 1.70E-01 with sample range (0.00E+00,2.25E+04) resulting in fit range (1.18E-01,5.10E+03)
    linear_norm_parameter_normdata = {"error": 0.16998764561004706}
    min_max_norm_parameter = (
        0.5556463225317848,
        2.6045119614877015,
    )  # error of 3.59E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.035937440036152485}
    sigmoidal_norm_parameter = (
        1.5662194643242748,
        2.3640636223858147,
    )  # error of 2.70E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (2.41E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02702566359991152}
    dual_sigmoidal_norm_parameter = (
        1.4899535288187378,
        3.041861538022715,
        1.857040602496807,
    )  # error of 1.21E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (1.06E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.012087945303030468}
    genlog_norm_parameter = (
        1.642527678707246,
        -3.563939646758198,
        0.21450571151231135,
        7.208305385073792e-05,
    )  # error of 4.86E-03 with sample range (0.00E+00,2.25E+04) resulting in fit range (1.97E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.004855690062784112}
    autogen_normdata = {
        "sample_bounds": [[0.0, 22500.017578125], [0.00019660403526402764, 1.0]],
        "sample_bounds99": [
            [0.0, 4.511241912841797],
            [0.015197802089403155, 0.9995888219593494],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(RadiusOfGyration)
    # normalization
    linear_norm_parameter = (
        0.274743032958996,
        -0.6363152066427643,
    )  # error of 8.52E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (-6.36E-01,2.85E+00)
    linear_norm_parameter_normdata = {"error": 0.08516029139429206}
    min_max_norm_parameter = (
        2.68976145372615,
        5.459084333173108,
    )  # error of 2.97E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02966791333550533}
    sigmoidal_norm_parameter = (
        4.059081941807601,
        1.764235395780906,
    )  # error of 1.73E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (7.76E-04,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01734440432664209}
    dual_sigmoidal_norm_parameter = (
        3.992230990546881,
        2.0750846967324876,
        1.5067797752888388,
    )  # error of 7.09E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (2.52E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.007092821402940244}
    genlog_norm_parameter = (
        1.347041034484673,
        2.109976974858853,
        2.3535858732123165,
        0.237797029415594,
    )  # error of 2.95E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (1.59E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.002949381497062385}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 12.672340393066406],
            [1.589518087213331e-07, 0.9999934474950445],
        ],
        "sample_bounds99": [
            [0.0, 6.605759620666504],
            [0.006238970231155407, 0.9887240756960017],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class RelativeContentAc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(89)))
                / mol.GetNumAtoms()
        )


class RelativeContentAg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(47)))
                / mol.GetNumAtoms()
        )


class RelativeContentAl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(13)))
                / mol.GetNumAtoms()
        )


class RelativeContentAm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(95)))
                / mol.GetNumAtoms()
        )


class RelativeContentAr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(18)))
                / mol.GetNumAtoms()
        )


class RelativeContentAs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(33)))
                / mol.GetNumAtoms()
        )


class RelativeContentAt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(85)))
                / mol.GetNumAtoms()
        )


class RelativeContentAu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(79)))
                / mol.GetNumAtoms()
        )


class RelativeContentB_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(5)))
                / mol.GetNumAtoms()
        )


class RelativeContentBa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(56)))
                / mol.GetNumAtoms()
        )


class RelativeContentBe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(4)))
                / mol.GetNumAtoms()
        )


class RelativeContentBh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(107)))
                / mol.GetNumAtoms()
        )


class RelativeContentBi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(83)))
                / mol.GetNumAtoms()
        )


class RelativeContentBk_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(97)))
                / mol.GetNumAtoms()
        )


class RelativeContentBr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(35)))
                / mol.GetNumAtoms()
        )


class RelativeContentC_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6)))
                / mol.GetNumAtoms()
        )


class RelativeContentCa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(20)))
                / mol.GetNumAtoms()
        )


class RelativeContentCd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(48)))
                / mol.GetNumAtoms()
        )


class RelativeContentCe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(58)))
                / mol.GetNumAtoms()
        )


class RelativeContentCf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(98)))
                / mol.GetNumAtoms()
        )


class RelativeContentCl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(17)))
                / mol.GetNumAtoms()
        )


class RelativeContentCm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(96)))
                / mol.GetNumAtoms()
        )


class RelativeContentCn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(112)))
                / mol.GetNumAtoms()
        )


class RelativeContentCo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(27)))
                / mol.GetNumAtoms()
        )


class RelativeContentCr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(24)))
                / mol.GetNumAtoms()
        )


class RelativeContentCs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(55)))
                / mol.GetNumAtoms()
        )


class RelativeContentCu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(29)))
                / mol.GetNumAtoms()
        )


class RelativeContentDb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(105)))
                / mol.GetNumAtoms()
        )


class RelativeContentDs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(110)))
                / mol.GetNumAtoms()
        )


class RelativeContentDy_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(66)))
                / mol.GetNumAtoms()
        )


class RelativeContentEr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(68)))
                / mol.GetNumAtoms()
        )


class RelativeContentEs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(99)))
                / mol.GetNumAtoms()
        )


class RelativeContentEu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(63)))
                / mol.GetNumAtoms()
        )


class RelativeContentF_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(9)))
                / mol.GetNumAtoms()
        )


class RelativeContentFe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(26)))
                / mol.GetNumAtoms()
        )


class RelativeContentFl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(114)))
                / mol.GetNumAtoms()
        )


class RelativeContentFm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(100)))
                / mol.GetNumAtoms()
        )


class RelativeContentFr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(87)))
                / mol.GetNumAtoms()
        )


class RelativeContentGa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(31)))
                / mol.GetNumAtoms()
        )


class RelativeContentGd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(64)))
                / mol.GetNumAtoms()
        )


class RelativeContentGe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(32)))
                / mol.GetNumAtoms()
        )


class RelativeContentH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1)))
                / mol.GetNumAtoms()
        )


class RelativeContentHe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(2)))
                / mol.GetNumAtoms()
        )


class RelativeContentHf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(72)))
                / mol.GetNumAtoms()
        )


class RelativeContentHg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(80)))
                / mol.GetNumAtoms()
        )


class RelativeContentHo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(67)))
                / mol.GetNumAtoms()
        )


class RelativeContentHs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(108)))
                / mol.GetNumAtoms()
        )


class RelativeContentI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(53)))
                / mol.GetNumAtoms()
        )


class RelativeContentIn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(49)))
                / mol.GetNumAtoms()
        )


class RelativeContentIr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(77)))
                / mol.GetNumAtoms()
        )


class RelativeContentK_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(19)))
                / mol.GetNumAtoms()
        )


class RelativeContentKr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(36)))
                / mol.GetNumAtoms()
        )


class RelativeContentLa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(57)))
                / mol.GetNumAtoms()
        )


class RelativeContentLi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(3)))
                / mol.GetNumAtoms()
        )


class RelativeContentLr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(103)))
                / mol.GetNumAtoms()
        )


class RelativeContentLu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(71)))
                / mol.GetNumAtoms()
        )


class RelativeContentLv_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(116)))
                / mol.GetNumAtoms()
        )


class RelativeContentMc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(115)))
                / mol.GetNumAtoms()
        )


class RelativeContentMd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(101)))
                / mol.GetNumAtoms()
        )


class RelativeContentMg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(12)))
                / mol.GetNumAtoms()
        )


class RelativeContentMn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(25)))
                / mol.GetNumAtoms()
        )


class RelativeContentMo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(42)))
                / mol.GetNumAtoms()
        )


class RelativeContentMt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(109)))
                / mol.GetNumAtoms()
        )


class RelativeContentN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(7)))
                / mol.GetNumAtoms()
        )


class RelativeContentNa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(11)))
                / mol.GetNumAtoms()
        )


class RelativeContentNb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(41)))
                / mol.GetNumAtoms()
        )


class RelativeContentNd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(60)))
                / mol.GetNumAtoms()
        )


class RelativeContentNe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(10)))
                / mol.GetNumAtoms()
        )


class RelativeContentNh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(113)))
                / mol.GetNumAtoms()
        )


class RelativeContentNi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(28)))
                / mol.GetNumAtoms()
        )


class RelativeContentNo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(102)))
                / mol.GetNumAtoms()
        )


class RelativeContentNp_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(93)))
                / mol.GetNumAtoms()
        )


class RelativeContentO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(8)))
                / mol.GetNumAtoms()
        )


class RelativeContentOg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(118)))
                / mol.GetNumAtoms()
        )


class RelativeContentOs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(76)))
                / mol.GetNumAtoms()
        )


class RelativeContentP_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(15)))
                / mol.GetNumAtoms()
        )


class RelativeContentPa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(91)))
                / mol.GetNumAtoms()
        )


class RelativeContentPb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(82)))
                / mol.GetNumAtoms()
        )


class RelativeContentPd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(46)))
                / mol.GetNumAtoms()
        )


class RelativeContentPm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(61)))
                / mol.GetNumAtoms()
        )


class RelativeContentPo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(84)))
                / mol.GetNumAtoms()
        )


class RelativeContentPr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(59)))
                / mol.GetNumAtoms()
        )


class RelativeContentPt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(78)))
                / mol.GetNumAtoms()
        )


class RelativeContentPu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(94)))
                / mol.GetNumAtoms()
        )


class RelativeContentRa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(88)))
                / mol.GetNumAtoms()
        )


class RelativeContentRb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(37)))
                / mol.GetNumAtoms()
        )


class RelativeContentRe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(75)))
                / mol.GetNumAtoms()
        )


class RelativeContentRf_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(104)))
                / mol.GetNumAtoms()
        )


class RelativeContentRg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(111)))
                / mol.GetNumAtoms()
        )


class RelativeContentRh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(45)))
                / mol.GetNumAtoms()
        )


class RelativeContentRn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(86)))
                / mol.GetNumAtoms()
        )


class RelativeContentRu_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(44)))
                / mol.GetNumAtoms()
        )


class RelativeContentS_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(16)))
                / mol.GetNumAtoms()
        )


class RelativeContentSb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(51)))
                / mol.GetNumAtoms()
        )


class RelativeContentSc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(21)))
                / mol.GetNumAtoms()
        )


class RelativeContentSe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(34)))
                / mol.GetNumAtoms()
        )


class RelativeContentSg_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(106)))
                / mol.GetNumAtoms()
        )


class RelativeContentSi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(14)))
                / mol.GetNumAtoms()
        )


class RelativeContentSm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(62)))
                / mol.GetNumAtoms()
        )


class RelativeContentSn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(50)))
                / mol.GetNumAtoms()
        )


class RelativeContentSr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(38)))
                / mol.GetNumAtoms()
        )


class RelativeContentTa_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(73)))
                / mol.GetNumAtoms()
        )


class RelativeContentTb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(65)))
                / mol.GetNumAtoms()
        )


class RelativeContentTc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(43)))
                / mol.GetNumAtoms()
        )


class RelativeContentTe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(52)))
                / mol.GetNumAtoms()
        )


class RelativeContentTh_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(90)))
                / mol.GetNumAtoms()
        )


class RelativeContentTi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(22)))
                / mol.GetNumAtoms()
        )


class RelativeContentTl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(81)))
                / mol.GetNumAtoms()
        )


class RelativeContentTm_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(69)))
                / mol.GetNumAtoms()
        )


class RelativeContentTs_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(117)))
                / mol.GetNumAtoms()
        )


class RelativeContentU_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(92)))
                / mol.GetNumAtoms()
        )


class RelativeContentV_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(23)))
                / mol.GetNumAtoms()
        )


class RelativeContentW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(74)))
                / mol.GetNumAtoms()
        )


class RelativeContentXe_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(54)))
                / mol.GetNumAtoms()
        )


class RelativeContentY_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(39)))
                / mol.GetNumAtoms()
        )


class RelativeContentYb_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(70)))
                / mol.GetNumAtoms()
        )


class RelativeContentZn_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(30)))
                / mol.GetNumAtoms()
        )


class RelativeContentZr_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32

    # normalization
    # functions
    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(40)))
                / mol.GetNumAtoms()
        )


class RingCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(RingCount)
    # normalization
    linear_norm_parameter = (
        0.03860180770005206,
        0.5113290906759573,
    )  # error of 1.73E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (5.11E-01,2.87E+00)
    linear_norm_parameter_normdata = {"error": 0.17329582361047302}
    min_max_norm_parameter = (
        0.334517739633367,
        5.208998938983079,
    )  # error of 1.60E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.01600245280122771}
    sigmoidal_norm_parameter = (
        2.7238161788458215,
        1.1018114352614388,
    )  # error of 1.08E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.74E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.010810851867941116}
    dual_sigmoidal_norm_parameter = (
        2.5715740496764488,
        1.283226429101575,
        0.9174925881562721,
    )  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (3.56E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0061287005299005375}
    genlog_norm_parameter = (
        0.813320479300628,
        -1.296799001575922,
        3.1048252219308727,
        0.173898726052091,
    )  # error of 4.52E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.48E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.004520854109330362}
    autogen_normdata = {
        "sample_bounds": [[0.0, 61.0], [0.014767969276556474, 1.0]],
        "sample_bounds99": [[0.0, 9.0], [0.10515038913140289, 0.998176048141036]],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA1)
    # normalization
    linear_norm_parameter = (
        0.0060832031355171345,
        0.29372966007484347,
    )  # error of 1.94E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.94E-01,1.13E+01)
    linear_norm_parameter_normdata = {"error": 0.1943813544608048}
    min_max_norm_parameter = (
        18.58970931537026,
        67.49192460811703,
    )  # error of 4.61E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.046115470278111574}
    sigmoidal_norm_parameter = (
        42.77007350322991,
        0.09980774119785008,
    )  # error of 2.58E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (1.38E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.025774125422552165}
    dual_sigmoidal_norm_parameter = (
        41.1131338380977,
        0.1350440206008133,
        0.08256623106509779,
    )  # error of 1.34E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.86E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013368665897985996}
    genlog_norm_parameter = (
        0.07260260182895871,
        -58.05913383073333,
        0.059350163104591735,
        6.019674609549131e-05,
    )  # error of 1.05E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.78E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.010458670484832143}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1804.5626220703125], [4.77531349722592e-07, 1.0]],
        "sample_bounds99": [
            [0.0, 144.67449951171875],
            [0.005263838611181123, 0.9998275266925875],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA10)
    # normalization
    linear_norm_parameter = (
        0.00805950974671309,
        0.47073915323460086,
    )  # error of 1.02E-01 with sample range (0.00E+00,9.17E+02) resulting in fit range (4.71E-01,7.86E+00)
    linear_norm_parameter_normdata = {"error": 0.10203568126861637}
    min_max_norm_parameter = (
        1.750898428174181e-15,
        43.23543148663909,
    )  # error of 4.79E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.047886001904314676}
    sigmoidal_norm_parameter = (
        20.019352651600183,
        0.09701956716011305,
    )  # error of 1.52E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (1.25E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.015197671864322224}
    dual_sigmoidal_norm_parameter = (
        19.446527523828802,
        0.12516516798549934,
        0.09265699049870557,
    )  # error of 1.34E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (8.06E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013440204509775337}
    genlog_norm_parameter = (
        0.08397540299203315,
        -29.079145155656267,
        13.392489270965957,
        0.2851200950108372,
    )  # error of 1.34E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (6.66E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.01338987432965727}
    autogen_normdata = {
        "sample_bounds": [[0.0, 917.13037109375], [0.0665913328192721, 1.0]],
        "sample_bounds99": [
            [0.0, 76.09814453125],
            [0.15251902032243586, 0.9934713437812029],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA2)
    # normalization
    linear_norm_parameter = (
        0.010815720584295,
        0.8621985340234535,
    )  # error of 5.64E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (8.62E-01,1.62E+00)
    linear_norm_parameter_normdata = {"error": 0.056411144346572004}
    min_max_norm_parameter = (
        1.833608967890303e-18,
        1.5590020213009668,
    )  # error of 5.87E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.058713432010427086}
    sigmoidal_norm_parameter = (
        -0.7174803678465178,
        0.5694423092539833,
    )  # error of 1.98E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (6.01E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.019817940577625412}
    dual_sigmoidal_norm_parameter = (
        -0.717479882031807,
        0.22963196789745544,
        0.5694424255830062,
    )  # error of 1.98E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (6.01E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.019817940577578925}
    genlog_norm_parameter = (
        0.5276645515124948,
        -10.71840839055069,
        0.010015624270769853,
        6.411100330456761e-05,
    )  # error of 1.92E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (5.79E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.01917982876827545}
    autogen_normdata = {
        "sample_bounds": [[0.0, 70.1287841796875], [0.5790470649251049, 1.0]],
        "sample_bounds99": [
            [0.0, 8.328696250915527],
            [0.644044314655474, 0.9934807369715062],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class SMR_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA3)
    # normalization
    linear_norm_parameter = (
        0.011429994095032314,
        0.595492552884596,
    )  # error of 1.14E-01 with sample range (0.00E+00,2.58E+02) resulting in fit range (5.95E-01,3.55E+00)
    linear_norm_parameter_normdata = {"error": 0.11398371922018284}
    min_max_norm_parameter = (
        0.0,
        18.19914049080423,
    )  # error of 5.20E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05197523488125678}
    sigmoidal_norm_parameter = (
        8.390853699132085,
        0.20684364498704194,
    )  # error of 2.62E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (1.50E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026229621945632352}
    dual_sigmoidal_norm_parameter = (
        8.015272681994086,
        0.25797153237552817,
        0.19541050070073113,
    )  # error of 2.55E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (1.12E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.025498687383927087}
    genlog_norm_parameter = (
        0.17646114389930065,
        -4.665984213322554,
        1.6877261310032854,
        0.23010151418720523,
    )  # error of 2.52E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (8.99E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.025156318980394023}
    autogen_normdata = {
        "sample_bounds": [[0.0, 258.0604553222656], [0.08988639136571477, 1.0]],
        "sample_bounds99": [
            [0.0, 31.52606964111328],
            [0.20385899051196943, 0.988039372692567],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA4)
    # normalization
    linear_norm_parameter = (
        0.004491810149551356,
        0.8258291767126574,
    )  # error of 7.18E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (8.26E-01,1.68E+00)
    linear_norm_parameter_normdata = {"error": 0.07182784726410202}
    min_max_norm_parameter = (
        7.03458954642263e-19,
        11.924815721181979,
    )  # error of 5.26E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05261870315921639}
    sigmoidal_norm_parameter = (
        3.0253911905893234,
        0.22133635579572117,
    )  # error of 1.52E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (3.39E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.015192662029293834}
    dual_sigmoidal_norm_parameter = (
        3.0253926225003913,
        3.7948488488645316,
        0.2213364264257019,
    )  # error of 1.45E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (1.03E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.014454172956683465}
    genlog_norm_parameter = (
        0.19622444380256232,
        -22.51736834567983,
        0.005239935347448878,
        4.7114007922905926e-05,
    )  # error of 1.38E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (2.62E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.013759795429778891}
    autogen_normdata = {
        "sample_bounds": [[0.0, 190.1898956298828], [0.26172820172377204, 1.0]],
        "sample_bounds99": [
            [0.0, 33.212158203125],
            [0.48939017445992056, 0.9981100419257615],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA5)
    # normalization
    linear_norm_parameter = (
        0.002902392328484771,
        0.5504163136192717,
    )  # error of 1.80E-01 with sample range (0.00E+00,1.49E+03) resulting in fit range (5.50E-01,4.89E+00)
    linear_norm_parameter_normdata = {"error": 0.18035710031787433}
    min_max_norm_parameter = (
        2.783727923618457e-16,
        60.67206561974408,
    )  # error of 4.76E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04761311948362417}
    sigmoidal_norm_parameter = (
        29.231452891359663,
        0.07160204674656424,
    )  # error of 2.47E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (1.10E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.024678526660424294}
    dual_sigmoidal_norm_parameter = (
        27.029412112232396,
        0.10378791856638528,
        0.06092739825372446,
    )  # error of 1.53E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (5.70E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.015279515534089735}
    genlog_norm_parameter = (
        0.05458670512702919,
        -81.00086947132154,
        0.011563905524825425,
        4.2620247317584376e-05,
    )  # error of 1.36E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (3.84E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.013576847398411357}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1494.324462890625], [0.038394998189259774, 1.0]],
        "sample_bounds99": [
            [0.0, 171.39308166503906],
            [0.07203701818612461, 0.9997329468309722],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA6)
    # normalization
    linear_norm_parameter = (
        0.007356009115044526,
        0.5604258193276979,
    )  # error of 1.01E-01 with sample range (0.00E+00,7.02E+02) resulting in fit range (5.60E-01,5.72E+00)
    linear_norm_parameter_normdata = {"error": 0.10100105254731877}
    min_max_norm_parameter = (
        4.58167181735802e-19,
        33.13532929564598,
    )  # error of 5.79E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0579186816187386}
    sigmoidal_norm_parameter = (
        13.927321244136747,
        0.10991207620783901,
    )  # error of 1.58E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (1.78E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01578045792672207}
    dual_sigmoidal_norm_parameter = (
        13.134406044570465,
        0.17414098398967184,
        0.10254276120653481,
    )  # error of 1.06E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (9.22E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010603216136901683}
    genlog_norm_parameter = (
        0.09044407004882077,
        -49.73733011721062,
        0.0107227727601361,
        4.9257981828589994e-05,
    )  # error of 1.05E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (8.88E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.010498817220485313}
    autogen_normdata = {
        "sample_bounds": [[0.0, 701.5629272460938], [0.08875388916213116, 1.0]],
        "sample_bounds99": [
            [0.0, 59.44770812988281],
            [0.1718645300117853, 0.9889559772521227],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA7)
    # normalization
    linear_norm_parameter = (
        0.004721924676151512,
        0.3518106416338208,
    )  # error of 1.44E-01 with sample range (0.00E+00,2.00E+03) resulting in fit range (3.52E-01,9.80E+00)
    linear_norm_parameter_normdata = {"error": 0.1442667177668769}
    min_max_norm_parameter = (
        0.027182952918228913,
        90.6963985858831,
    )  # error of 3.59E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.035901853270376}
    sigmoidal_norm_parameter = (
        45.068602585969984,
        0.05384149034564499,
    )  # error of 1.60E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (8.12E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.016008183224336486}
    dual_sigmoidal_norm_parameter = (
        43.693610417543745,
        0.06291260195656685,
        0.04995417054949297,
    )  # error of 1.34E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (6.02E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013384988164861776}
    genlog_norm_parameter = (
        0.045747951039014585,
        -31.989628190375154,
        10.397849271952843,
        0.396132822643155,
    )  # error of 1.31E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (4.53E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.013121019209396316}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2001.5146484375], [0.04531731989094163, 1.0]],
        "sample_bounds99": [
            [0.0, 178.96360778808594],
            [0.06993880980882695, 0.9983967104061189],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SMR_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA8)
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.5}
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5}
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0}
    autogen_normdata = {"sample_bounds": [[0.0, 0.0], [1.0, 1.0]]}
    preferred_normalization = "unity"
    # functions


class SMR_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SMR_VSA9)
    # normalization
    linear_norm_parameter = (
        0.002463279237957446,
        0.8723998320669256,
    )  # error of 5.34E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (8.72E-01,1.54E+00)
    linear_norm_parameter_normdata = {"error": 0.05342003212482407}
    min_max_norm_parameter = (
        5e-324,
        13.161474983617241,
    )  # error of 4.95E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.049470620536279736}
    sigmoidal_norm_parameter = (
        1.1525795833229362,
        0.15186000443615003,
    )  # error of 1.83E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (4.56E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01825868558914517}
    dual_sigmoidal_norm_parameter = (
        1.1525943308892048,
        9.547259158595873,
        0.15186024720256838,
    )  # error of 1.76E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (1.66E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01757079370761303}
    genlog_norm_parameter = (
        0.1391658630525705,
        -38.716528759081775,
        0.006166018170748839,
        3.075454268494904e-05,
    )  # error of 1.79E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (4.00E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.017916190069777124}
    autogen_normdata = {
        "sample_bounds": [[0.0, 272.8189392089844], [0.39994165654303954, 1.0]],
        "sample_bounds99": [
            [0.0, 39.264892578125],
            [0.6332275967349426, 0.9961535886303725],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class SSSR_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(GetSSSR)
    # normalization
    # functions


class SlogP_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA1)
    # normalization
    linear_norm_parameter = (
        0.007436684822960693,
        0.6872466373090912,
    )  # error of 1.45E-01 with sample range (0.00E+00,4.14E+02) resulting in fit range (6.87E-01,3.76E+00)
    linear_norm_parameter_normdata = {"error": 0.14513379240880545}
    min_max_norm_parameter = (
        1.9721522630525295e-31,
        16.894446647493005,
    )  # error of 4.50E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04501768199135168}
    sigmoidal_norm_parameter = (
        7.760808021810809,
        0.25639853999337753,
    )  # error of 3.00E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (1.20E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.029970991550821443}
    dual_sigmoidal_norm_parameter = (
        7.393606686015775,
        0.31725188648189856,
        0.2380832859822617,
    )  # error of 2.94E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (8.74E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.02943805894390255}
    genlog_norm_parameter = (
        0.21913039835407314,
        -4.436671418072039,
        3.264645969091995,
        0.30033106818669153,
    )  # error of 2.92E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (6.87E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.02918533936031894}
    autogen_normdata = {
        "sample_bounds": [[0.0, 413.802001953125], [0.0687271348580626, 1.0]],
        "sample_bounds99": [
            [0.0, 31.18532371520996],
            [0.18681793589715526, 0.9956485124359113],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA10)
    # normalization
    linear_norm_parameter = (
        0.005233706932712693,
        0.7982300755527156,
    )  # error of 6.65E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (7.98E-01,2.58E+00)
    linear_norm_parameter_normdata = {"error": 0.06650981767721549}
    min_max_norm_parameter = (
        2.633367426076831e-17,
        12.74842203329568,
    )  # error of 4.94E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04935860041869704}
    sigmoidal_norm_parameter = (
        3.6455899379715135,
        0.1901649776805424,
    )  # error of 1.29E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (3.33E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.012865741925579124}
    dual_sigmoidal_norm_parameter = (
        3.61535917881715,
        0.2741058052371704,
        0.18951197796421562,
    )  # error of 1.28E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (2.71E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.012846046544175786}
    genlog_norm_parameter = (
        0.16768815401308138,
        -38.62371767303691,
        0.09022718543848567,
        0.00010267516869384561,
    )  # error of 1.16E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (2.59E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.011606993920544236}
    autogen_normdata = {
        "sample_bounds": [[0.0, 341.2431640625], [0.2586798219341158, 1.0]],
        "sample_bounds99": [
            [0.0, 31.445770263671875],
            [0.43340889841835345, 0.9931224877657844],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA11)
    # normalization
    linear_norm_parameter = (
        0.0055772293963739505,
        0.8513122937313707,
    )  # error of 5.02E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (8.51E-01,1.88E+00)
    linear_norm_parameter_normdata = {"error": 0.05023914272942102}
    min_max_norm_parameter = (
        1.6665175605263984,
        6.577923952521674,
    )  # error of 2.87E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02867633460919674}
    sigmoidal_norm_parameter = (
        4.411498413610341,
        1.231868331024134,
    )  # error of 2.88E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (4.34E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.028838335958735424}
    dual_sigmoidal_norm_parameter = (
        3.2776387723523693e-06,
        3358906.662488689,
        0.25601634916947247,
    )  # error of 2.52E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (1.65E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.025237217885336827}
    genlog_norm_parameter = (
        1.1493152290391706,
        5.127398807774413,
        0.19594466870004,
        0.5193970251551439,
    )  # error of 2.88E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (2.65E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.02882644711478666}
    autogen_normdata = {
        "sample_bounds": [[0.0, 183.984375], [0.0002652984747177803, 1.0]],
        "sample_bounds99": [
            [0.0, 21.625438690185547],
            [0.6917643888527882, 0.9999999991408026],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class SlogP_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA12)
    # normalization
    linear_norm_parameter = (
        0.006154770006633981,
        0.6983876686482481,
    )  # error of 8.20E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (6.98E-01,2.37E+00)
    linear_norm_parameter_normdata = {"error": 0.08202962727816064}
    min_max_norm_parameter = (
        5e-324,
        16.317969218729832,
    )  # error of 7.75E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.07751537809340323}
    sigmoidal_norm_parameter = (
        4.978333217945518,
        0.12756642613388391,
    )  # error of 2.43E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (3.46E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02425617442054257}
    dual_sigmoidal_norm_parameter = (
        4.674100438413901,
        1.175288360979358,
        0.1242634938303602,
    )  # error of 2.35E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (4.10E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.02352697211679736}
    genlog_norm_parameter = (
        0.12015613183504709,
        -7.034013238843647,
        1.9472746789635476,
        0.5328246516809809,
    )  # error of 2.42E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (3.20E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.02419542939105498}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 270.8090515136719],
            [0.3196143118574142, 0.9999999999999882],
        ],
        "sample_bounds99": [
            [0.0, 45.10428237915039],
            [0.4809978497030905, 0.9930909928424386],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class SlogP_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA2)
    # normalization
    linear_norm_parameter = (
        0.007179097183038996,
        0.31036360002709074,
    )  # error of 1.50E-01 with sample range (0.00E+00,1.09E+03) resulting in fit range (3.10E-01,8.13E+00)
    linear_norm_parameter_normdata = {"error": 0.15006808388180573}
    min_max_norm_parameter = (
        2.8149378256505875,
        62.887170739624814,
    )  # error of 3.91E-02 with sample range (0.00E+00,1.09E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03906385027345499}
    sigmoidal_norm_parameter = (
        32.49634012337661,
        0.07987637026985167,
    )  # error of 1.84E-02 with sample range (0.00E+00,1.09E+03) resulting in fit range (6.94E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01842103271614385}
    dual_sigmoidal_norm_parameter = (
        31.064566298185326,
        0.10301047742185264,
        0.06990769312823802,
    )  # error of 7.32E-03 with sample range (0.00E+00,1.09E+03) resulting in fit range (3.92E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0073170230561812955}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1088.590576171875], [0.03916663329707189, 1.0]],
        "sample_bounds99": [
            [0.0, 118.87759399414062],
            [0.06200103297444043, 0.9981503163165847],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class SlogP_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA3)
    # normalization
    linear_norm_parameter = (
        0.005572864820903245,
        0.7183501823787287,
    )  # error of 1.09E-01 with sample range (0.00E+00,4.79E+02) resulting in fit range (7.18E-01,3.39E+00)
    linear_norm_parameter_normdata = {"error": 0.1089511244958868}
    min_max_norm_parameter = (
        2.408585330451039e-20,
        22.439152351441383,
    )  # error of 5.09E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.050860062015352044}
    sigmoidal_norm_parameter = (
        9.49158537571278,
        0.16960656906498323,
    )  # error of 1.48E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (1.67E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.014795263316906532}
    dual_sigmoidal_norm_parameter = (
        8.968468865831698,
        0.2604162096282373,
        0.1593241753703142,
    )  # error of 1.16E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (8.82E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01158314029443211}
    genlog_norm_parameter = (
        0.1413969571337046,
        -41.552127286264174,
        0.058402954914476224,
        6.214988652289824e-05,
    )  # error of 9.62E-03 with sample range (0.00E+00,4.79E+02) resulting in fit range (7.15E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00962394172282809}
    autogen_normdata = {
        "sample_bounds": [[0.0, 478.8599853515625], [0.07148048883677417, 1.0]],
        "sample_bounds99": [
            [0.0, 46.55548095703125],
            [0.18117072293003697, 0.9964187196078913],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA4)
    # normalization
    linear_norm_parameter = (
        0.0055106885127717,
        0.760669380441174,
    )  # error of 7.89E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (7.61E-01,1.07E+01)
    linear_norm_parameter_normdata = {"error": 0.07894025064492369}
    min_max_norm_parameter = (
        0.0,
        14.831828070749904,
    )  # error of 5.76E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05755612306017365}
    sigmoidal_norm_parameter = (
        4.655549562955406,
        0.16917248467895607,
    )  # error of 1.66E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.13E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01662213593241825}
    dual_sigmoidal_norm_parameter = (
        4.616140037567237,
        0.20613311394450484,
        0.16841100802007053,
    )  # error of 1.66E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.79E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.016587084930612384}
    genlog_norm_parameter = (
        0.14664312645928354,
        -30.075254041938358,
        0.012062431404949683,
        0.00010270053051938266,
    )  # error of 1.58E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.40E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.015832088834298628}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1804.5626220703125], [0.24000844667168697, 1.0]],
        "sample_bounds99": [
            [0.0, 38.76947021484375],
            [0.3412585576266998, 0.9952311856133698],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA5)
    # normalization
    linear_norm_parameter = (
        0.0030832580459571788,
        0.41517180790701097,
    )  # error of 1.77E-01 with sample range (0.00E+00,2.18E+03) resulting in fit range (4.15E-01,7.12E+00)
    linear_norm_parameter_normdata = {"error": 0.17660486588662325}
    min_max_norm_parameter = (
        13.274337247983418,
        99.35584702819898,
    )  # error of 4.24E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04238871244147601}
    sigmoidal_norm_parameter = (
        55.95034092538948,
        0.05594705841606074,
    )  # error of 2.08E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (4.19E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.020829196979360223}
    dual_sigmoidal_norm_parameter = (
        53.71656201320551,
        0.0743286650853182,
        0.048589611246767174,
    )  # error of 1.11E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (1.81E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011129728838095779}
    genlog_norm_parameter = (
        0.041857300165139694,
        -104.73487770928024,
        0.024956991610754424,
        4.5297139161081066e-05,
    )  # error of 7.53E-03 with sample range (0.00E+00,2.18E+03) resulting in fit range (1.04E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007531655574018605}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2175.591552734375], [0.0010354386230284006, 1.0]],
        "sample_bounds99": [
            [0.0, 236.63450622558594],
            [0.012988898821409797, 0.9996961851798565],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA6)
    # normalization
    linear_norm_parameter = (
        0.0046552542351420145,
        0.4184194987223895,
    )  # error of 1.54E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.18E-01,8.79E+00)
    linear_norm_parameter_normdata = {"error": 0.15387015088735315}
    min_max_norm_parameter = (
        5.914526578091133e-16,
        72.98691414016453,
    )  # error of 4.02E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.040243710275165316}
    sigmoidal_norm_parameter = (
        35.2283623788487,
        0.06244680638294594,
    )  # error of 1.72E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (9.98E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.017236331737933076}
    dual_sigmoidal_norm_parameter = (
        33.58491481319086,
        0.07805361314794311,
        0.05646183503476589,
    )  # error of 1.20E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (6.78E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.012038259635700578}
    genlog_norm_parameter = (
        0.0501622034533546,
        -55.89821242891373,
        12.849846035306,
        0.18739916749450045,
    )  # error of 1.17E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.63E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.011747425619691889}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1797.5101318359375], [0.04633667872224092, 1.0]],
        "sample_bounds99": [
            [0.0, 163.1742706298828],
            [0.07572708754389622, 0.9989228844180713],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class SlogP_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA7)
    # normalization
    linear_norm_parameter = (
        0.0036630233160711567,
        0.8928265874311769,
    )  # error of 5.13E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (8.93E-01,3.81E+00)
    linear_norm_parameter_normdata = {"error": 0.051306834093120125}
    min_max_norm_parameter = (
        5e-324,
        6.023897866072181,
    )  # error of 4.56E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04557746783376814}
    sigmoidal_norm_parameter = (
        0.4655912538215367,
        0.3384327438176033,
    )  # error of 1.07E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (4.61E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01074429479844314}
    dual_sigmoidal_norm_parameter = (
        0.4655844230945186,
        26.16053644691805,
        0.33843220244046884,
    )  # error of 1.00E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (5.13E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010009171162598814}
    genlog_norm_parameter = (
        0.3105497920408985,
        -8.003062292293896,
        0.0005277821750136437,
        4.8408124103588315e-05,
    )  # error of 9.68E-03 with sample range (0.00E+00,7.96E+02) resulting in fit range (4.03E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.009684156854452917}
    autogen_normdata = {
        "sample_bounds": [[0.0, 795.9664916992188], [0.40327309791059285, 1.0]],
        "sample_bounds99": [
            [0.0, 17.963537216186523],
            [0.5566662051528054, 0.9966013957800864],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class SlogP_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA8)
    # normalization
    linear_norm_parameter = (
        0.0012320682276687166,
        0.9018392839757955,
    )  # error of 5.04E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (9.02E-01,1.34E+00)
    linear_norm_parameter_normdata = {"error": 0.050431027654289175}
    min_max_norm_parameter = (
        1.6753303243403e-23,
        12.492939268423935,
    )  # error of 5.71E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.05708119519262628}
    sigmoidal_norm_parameter = (
        -0.9543059714466142,
        0.13070084470487742,
    )  # error of 1.44E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (5.31E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.014438116125692386}
    dual_sigmoidal_norm_parameter = (
        -0.9543065457922881,
        1.0,
        0.1307008395582034,
    )  # error of 1.44E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (5.31E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.014438116125706517}
    genlog_norm_parameter = (
        0.12040058020793103,
        -28.875724704142296,
        0.0010455368290572217,
        4.5865565007565374e-05,
    )  # error of 1.40E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (4.94E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.014037050828442466}
    autogen_normdata = {
        "sample_bounds": [[0.0, 357.2527770996094], [0.49429721124643655, 1.0]],
        "sample_bounds99": [
            [0.0, 56.104217529296875],
            [0.6070101419171942, 0.9991917638510758],
        ],
    }
    preferred_normalization = "min_max"
    # functions


class SlogP_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA9)
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.5}
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5}
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0}
    autogen_normdata = {"sample_bounds": [[0.0, 0.0], [1.0, 1.0]]}
    preferred_normalization = "unity"
    # functions


class SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(SpherocityIndex)
    # normalization
    linear_norm_parameter = (
        2.014040859270808,
        -0.004922239185380645,
    )  # error of 8.41E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (-4.92E-03,1.89E+00)
    linear_norm_parameter_normdata = {"error": 0.08411362308789125}
    min_max_norm_parameter = (
        0.04763405868523997,
        0.42911985390392243,
    )  # error of 3.56E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.035620890912944805}
    sigmoidal_norm_parameter = (
        0.2351671867619151,
        12.82810781848393,
    )  # error of 2.66E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (4.67E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026608119787476327}
    dual_sigmoidal_norm_parameter = (
        0.22125633366340652,
        16.421918195920984,
        10.082160667790887,
    )  # error of 1.03E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (2.57E-02,9.99E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.010281833492414159}
    genlog_norm_parameter = (
        8.900273970395503,
        -0.9610538652694148,
        1.12231002232635,
        4.096671219695612e-05,
    )  # error of 2.88E-03 with sample range (0.00E+00,9.39E-01) resulting in fit range (5.08E-03,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 0.0028831891267616333}
    autogen_normdata = {
        "sample_bounds": [
            [0.0, 0.9388075470924377],
            [0.005079999415383075, 0.9987589136149658],
        ],
        "sample_bounds99": [
            [0.0, 0.6009262800216675],
            [0.012549610123851232, 0.9847516342450336],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(TPSA)
    # normalization
    linear_norm_parameter = (
        0.0021678074691265925,
        0.4909310697599548,
    )  # error of 1.90E-01 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.91E-01,6.20E+00)
    linear_norm_parameter_normdata = {"error": 0.18977395418071857}
    min_max_norm_parameter = (
        14.599527973028703,
        118.60656324496115,
    )  # error of 3.84E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03835472675388331}
    sigmoidal_norm_parameter = (
        66.26559293182696,
        0.047786843726724,
    )  # error of 1.40E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.04E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.01400868518265227}
    dual_sigmoidal_norm_parameter = (
        64.41269425069702,
        0.05696911342563409,
        0.04375673621418519,
    )  # error of 9.63E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (2.49E-02,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.009625801372653755}
    genlog_norm_parameter = (
        0.03939646267682107,
        -5.601968508575941,
        3.883656715190027,
        0.3094072416673409,
    )  # error of 7.60E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (1.03E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007597944115971997}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2632.610107421875], [0.010340085309999867, 1.0]],
        "sample_bounds99": [
            [0.0, 235.77999877929688],
            [0.02004096728289619, 0.9991108244619733],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState1)
    # normalization
    linear_norm_parameter = (
        0.0021288093550339005,
        0.07050165770753636,
    )  # error of 1.68E-01 with sample range (-7.93E+00,6.18E+03) resulting in fit range (5.36E-02,1.32E+01)
    linear_norm_parameter_normdata = {"error": 0.16780139610629397}
    min_max_norm_parameter = (
        79.6151847284827,
        292.20287575940097,
    )  # error of 3.66E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.03660778168903182}
    sigmoidal_norm_parameter = (
        184.75792508436405,
        0.022940041643693208,
    )  # error of 2.38E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (1.19E-02,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.023770784444087645}
    dual_sigmoidal_norm_parameter = (
        178.29195170701303,
        0.02827751749368,
        0.01835803091582772,
    )  # error of 1.08E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (5.14E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0108011039425151}
    genlog_norm_parameter = (
        0.016043494356278568,
        -65.07670678628045,
        1.7558484413584392,
        0.047741917755167965,
    )  # error of 7.31E-03 with sample range (-7.93E+00,6.18E+03) resulting in fit range (1.45E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007312254873831111}
    autogen_normdata = {
        "sample_bounds": [
            [-7.931442737579346, 6184.9267578125],
            [1.4536693786705851e-05, 1.0],
        ],
        "sample_bounds99": [
            [-7.931442737579346, 520.8341064453125],
            [0.006563464026039269, 0.999766441978703],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState10)
    # normalization
    linear_norm_parameter = (
        0.05651052449377392,
        0.4477716191034624,
    )  # error of 2.35E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (-1.74E+00,7.28E+00)
    linear_norm_parameter_normdata = {"error": 0.2350683415812613}
    min_max_norm_parameter = (
        -1.6960388214922417,
        3.3674968629997517,
    )  # error of 1.76E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.1763122404698758}
    sigmoidal_norm_parameter = (
        0.09768644304876672,
        6.040221189979557,
    )  # error of 1.42E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (1.42E-102,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.14206283626478766}
    dual_sigmoidal_norm_parameter = (
        0.0035845937533911135,
        223.13480568444166,
        0.3998340930016602,
    )  # error of 7.98E-02 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.07977376497122787}
    genlog_norm_parameter = (
        4.164671303859201,
        -1.5247008100660342,
        0.042910346826479207,
        7.419051336584222e-05,
    )  # error of 1.38E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.13820631948330714}
    autogen_normdata = {
        "sample_bounds": [[-38.72703170776367, 120.85002136230469], [0.0, 1.0]],
        "sample_bounds99": [
            [-38.72703170776367, 11.966391563415527],
            [3.543585746598224e-73, 1.0],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class VSA_EState2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState2)
    # normalization
    linear_norm_parameter = (
        0.01528826607315403,
        0.2670303270450643,
    )  # error of 1.35E-01 with sample range (-3.29E+01,8.56E+02) resulting in fit range (-2.36E-01,1.34E+01)
    linear_norm_parameter_normdata = {"error": 0.13477681618384887}
    min_max_norm_parameter = (
        -4.878450507072114,
        34.03022413712193,
    )  # error of 4.20E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.041972354897797035}
    sigmoidal_norm_parameter = (
        14.37762800683558,
        0.12195442376720393,
    )  # error of 4.00E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (3.12E-03,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.039990902266508754}
    dual_sigmoidal_norm_parameter = (
        13.67578622158198,
        0.14469289317841083,
        0.10641187063656972,
    )  # error of 3.56E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (1.18E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.035564321417806855}
    genlog_norm_parameter = (
        0.08922289149859944,
        3.6614572002216086,
        0.17396397275448758,
        0.09811430866673368,
    )  # error of 3.45E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (2.60E-08,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0345221935108625}
    autogen_normdata = {
        "sample_bounds": [
            [-32.91777801513672, 855.8577270507812],
            [2.6014558124693748e-08, 1.0],
        ],
        "sample_bounds99": [
            [-32.91777801513672, 55.965354919433594],
            [0.06870501075297711, 0.994945902873534],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState3)
    # normalization
    linear_norm_parameter = (
        0.0474383914162283,
        0.3904581813825524,
    )  # error of 1.52E-01 with sample range (-1.08E+02,2.11E+02) resulting in fit range (-4.72E+00,1.04E+01)
    linear_norm_parameter_normdata = {"error": 0.15172572679493615}
    min_max_norm_parameter = (
        -3.8927946674623195,
        6.642361773646212,
    )  # error of 5.95E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.059454347223914183}
    sigmoidal_norm_parameter = (
        1.3162739635967782,
        0.43946395350992634,
    )  # error of 4.76E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (1.53E-21,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.047632258724711336}
    dual_sigmoidal_norm_parameter = (
        0.1908557460343406,
        0.8498742319003528,
        0.2639530557494804,
    )  # error of 1.52E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (1.44E-40,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.015179296379034103}
    genlog_norm_parameter = (
        0.3064933552795617,
        -13.292268939060781,
        0.0026299741474724203,
        4.6276819659564575e-05,
    )  # error of 3.11E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.031135421877569416}
    autogen_normdata = {
        "sample_bounds": [[-107.75056457519531, 210.5833282470703], [0.0, 1.0]],
        "sample_bounds99": [
            [-107.75056457519531, 17.214420318603516],
            [9.642779557744174e-05, 0.998644970232999],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class VSA_EState4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState4)
    # normalization
    linear_norm_parameter = (
        0.03989308824193185,
        0.6890187770089706,
    )  # error of 1.54E-01 with sample range (-3.80E+02,5.80E+01) resulting in fit range (-1.45E+01,3.00E+00)
    linear_norm_parameter_normdata = {"error": 0.15440463154178571}
    min_max_norm_parameter = (
        -9.819105425993582,
        0.9849828760699441,
    )  # error of 4.39E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0438783039685803}
    sigmoidal_norm_parameter = (
        -4.310085634921858,
        0.4319322795706566,
    )  # error of 2.94E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (3.08E-71,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.02944084738299047}
    dual_sigmoidal_norm_parameter = (
        -3.871053962031284,
        0.3319910911268734,
        0.5895341877288691,
    )  # error of 1.27E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (5.50E-55,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01274697956275428}
    genlog_norm_parameter = (
        0.949270551674369,
        -1.0168044804805934,
        0.9182029254180515,
        4.09300564578089,
    )  # error of 5.69E-03 with sample range (-3.80E+02,5.80E+01) resulting in fit range (6.56E-39,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005686255457372991}
    autogen_normdata = {
        "sample_bounds": [
            [-380.19622802734375, 57.99348831176758],
            [6.556521009969108e-39, 1.0],
        ],
        "sample_bounds99": [
            [-380.19622802734375, -0.29044482111930847],
            [0.0019035899139851725, 0.9993615111318908],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState5)
    # normalization
    linear_norm_parameter = (
        0.028069497462787085,
        0.6384948010105149,
    )  # error of 1.67E-01 with sample range (-3.60E+02,3.76E+02) resulting in fit range (-9.46E+00,1.12E+01)
    linear_norm_parameter_normdata = {"error": 0.1667247752235381}
    min_max_norm_parameter = (
        -11.200860573507633,
        1.4339724783749572,
    )  # error of 4.64E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0463623690973436}
    sigmoidal_norm_parameter = (
        -4.751658309551578,
        0.3690396610731733,
    )  # error of 3.12E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (1.23E-57,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.031164459411500357}
    dual_sigmoidal_norm_parameter = (
        -4.2245920363416305,
        0.2819947363703679,
        0.5221323236667128,
    )  # error of 1.32E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (2.81E-44,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01320814049024261}
    genlog_norm_parameter = (
        0.9230351559971732,
        0.5454865248379119,
        0.2568576583837676,
        4.705755620603767,
    )  # error of 5.93E-03 with sample range (-3.60E+02,3.76E+02) resulting in fit range (2.66E-31,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0059255855760834775}
    autogen_normdata = {
        "sample_bounds": [
            [-359.8406982421875, 376.4988708496094],
            [2.662169422859124e-31, 1.0],
        ],
        "sample_bounds99": [
            [-359.8406982421875, -0.7399227619171143],
            [0.00089463256320823, 0.9232862917269623],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState6)
    # normalization
    linear_norm_parameter = (
        0.026302870373594356,
        0.7190453852045258,
    )  # error of 1.52E-01 with sample range (-4.76E+02,1.38E+01) resulting in fit range (-1.18E+01,1.08E+00)
    linear_norm_parameter_normdata = {"error": 0.1523882335209083}
    min_max_norm_parameter = (
        -16.488126017392656,
        0.8283736118358731,
    )  # error of 3.98E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.039758916527822444}
    sigmoidal_norm_parameter = (
        -7.68869429717635,
        0.2724099515171271,
    )  # error of 2.50E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (4.47E-56,9.97E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.02499056515706096}
    dual_sigmoidal_norm_parameter = (
        -7.077584581907704,
        0.2169266620611263,
        0.35434306084612827,
    )  # error of 1.11E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (7.35E-45,9.99E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.011059469335263918}
    genlog_norm_parameter = (
        0.48753372514488386,
        -2.855074233910114,
        0.8709552154820239,
        3.1241419144322835,
    )  # error of 6.47E-03 with sample range (-4.76E+02,1.38E+01) resulting in fit range (9.63E-33,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0064664100810243146}
    autogen_normdata = {
        "sample_bounds": [
            [-475.5412902832031, 13.835240364074707],
            [9.632349688444906e-33, 0.9999184736071256],
        ],
        "sample_bounds99": [
            [-475.5412902832031, -0.9532303214073181],
            [0.001946738541420645, 0.9336977315016716],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState7)
    # normalization
    linear_norm_parameter = (
        0.006935598460676706,
        0.5277845083933819,
    )  # error of 1.85E-01 with sample range (-7.84E+02,1.08E+01) resulting in fit range (-4.91E+00,6.03E-01)
    linear_norm_parameter_normdata = {"error": 0.18504452563995974}
    min_max_norm_parameter = (
        -24.625854914081717,
        6.979695541871978,
    )  # error of 5.05E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.050548050588268466}
    sigmoidal_norm_parameter = (
        -8.479594411787089,
        0.14779915592252638,
    )  # error of 3.33E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.74E-50,9.46E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.0332615626590597}
    dual_sigmoidal_norm_parameter = (
        -7.046444035217077,
        0.11243405953450752,
        0.23343936380826846,
    )  # error of 1.51E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.19E-38,9.85E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.01509911032347625}
    genlog_norm_parameter = (
        0.5395427331098591,
        1.4029444653400476,
        0.8418305127213014,
        6.8177347470553835,
    )  # error of 9.45E-03 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.07E-27,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 0.009454416642238289}
    autogen_normdata = {
        "sample_bounds": [
            [-783.6892700195312, 10.83029842376709],
            [1.0664413360617732e-27, 0.9992392001625648],
        ],
        "sample_bounds99": [
            [-783.6892700195312, -0.7631514072418213],
            [0.00013629362412859736, 0.858665851894529],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState8)
    # normalization
    linear_norm_parameter = (
        0.018301201576055484,
        0.754648223799328,
    )  # error of 1.28E-01 with sample range (-8.99E+02,8.55E+01) resulting in fit range (-1.57E+01,2.32E+00)
    linear_norm_parameter_normdata = {"error": 0.12826016457756095}
    min_max_norm_parameter = (
        -28.897386873619986,
        2.4177080960805726,
    )  # error of 4.02E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04016237086456924}
    sigmoidal_norm_parameter = (
        -12.927059175873397,
        0.15008821168411976,
    )  # error of 2.62E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (1.64E-58,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.026179081163693326}
    dual_sigmoidal_norm_parameter = (
        -11.891432276406338,
        0.12203527371312223,
        0.19311608819477752,
    )  # error of 1.38E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (9.15E-48,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.013832477495664464}
    genlog_norm_parameter = (
        0.27145786900096325,
        -4.8394603149083,
        1.0302556085906796,
        3.1194594587463604,
    )  # error of 1.17E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (1.54E-34,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.011677538740104524}
    autogen_normdata = {
        "sample_bounds": [
            [-899.426513671875, 85.5],
            [1.5382627552043814e-34, 0.9999999999926124],
        ],
        "sample_bounds99": [
            [-899.426513671875, -2.882235050201416],
            [0.007988057655357239, 0.915943347199824],
        ],
    }
    preferred_normalization = "genlog"
    # functions


class VSA_EState9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(VSA_EState9)
    # normalization
    linear_norm_parameter = (
        0.022314103447673483,
        0.2903417115454465,
    )  # error of 6.95E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (-4.69E+00,5.00E+00)
    linear_norm_parameter_normdata = {"error": 0.06950927933174747}
    min_max_norm_parameter = (
        -8.357071428462406,
        11.006415888282014,
    )  # error of 4.34E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.04337066908320259}
    sigmoidal_norm_parameter = (
        -0.6709557801163907,
        0.3954700807696381,
    )  # error of 3.36E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (6.55E-39,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.03361980028717245}
    dual_sigmoidal_norm_parameter = (
        0.07077977321665872,
        0.3284370957099312,
        77.41637441191082,
    )  # error of 1.85E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (1.52E-32,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.018511962098522235}
    genlog_norm_parameter = (
        24.853269149189806,
        1.4321524383308715,
        0.016275273315106524,
        82.95029730946912,
    )  # error of 2.68E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.026838107213956172}
    autogen_normdata = {
        "sample_bounds": [[-222.9911346435547, 211.07403564453125], [0.0, 1.0]],
        "sample_bounds99": [
            [-222.9911346435547, -3.1380553245544434],
            [0.008610767204223012, 0.6635547212863377],
        ],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_Al_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Al_COO)
    # normalization
    linear_norm_parameter = (
        0.019808217260446237,
        0.9450166151713014,
    )  # error of 7.81E-03 with sample range (0.00E+00,2.20E+01) resulting in fit range (9.45E-01,1.38E+00)
    linear_norm_parameter_normdata = {"error": 0.007808113156719396}
    min_max_norm_parameter = (
        3.2382696256541155e-09,
        1.0424227893434397,
    )  # error of 1.20E-03 with sample range (0.00E+00,2.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.001199156731541787}
    sigmoidal_norm_parameter = (
        -0.4050322619852695,
        2.2492421348283718,
    )  # error of 3.59E-04 with sample range (0.00E+00,2.20E+01) resulting in fit range (7.13E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00035903577919620415}
    dual_sigmoidal_norm_parameter = (
        -0.405032320679578,
        1.0,
        2.249242042225464,
    )  # error of 3.59E-04 with sample range (0.00E+00,2.20E+01) resulting in fit range (7.13E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00035903577919687565}
    genlog_norm_parameter = (
        2.2309327767102376,
        -2.186974349548142,
        0.913642737133329,
        0.01796162303038676,
    )  # error of 3.55E-04 with sample range (0.00E+00,2.20E+01) resulting in fit range (6.80E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00035465690854523504}
    autogen_normdata = {
        "sample_bounds": [[0.0, 22.0], [0.6801210994937429, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.959310881344214, 0.959310881344214]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Al_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH)
    # normalization
    linear_norm_parameter = (
        0.006501081569325162,
        0.9535852884351536,
    )  # error of 1.73E-02 with sample range (0.00E+00,3.70E+01) resulting in fit range (9.54E-01,1.19E+00)
    linear_norm_parameter_normdata = {"error": 0.01730130746711582}
    min_max_norm_parameter = (
        5.758669341509425e-09,
        1.0871784274707261,
    )  # error of 4.26E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.004259759322713747}
    sigmoidal_norm_parameter = (
        -0.8719669615585766,
        1.3070605975374598,
    )  # error of 2.41E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (7.58E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0024129586002097983}
    dual_sigmoidal_norm_parameter = (
        -0.8719669615585766,
        1.0,
        1.3070605975374598,
    )  # error of 2.41E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (7.58E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0024129586002097983}
    genlog_norm_parameter = (
        1.277905074545129,
        -4.53120227256585,
        0.31773004334648924,
        0.0032579128409757074,
    )  # error of 2.37E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (7.42E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.002371618384224448}
    autogen_normdata = {
        "sample_bounds": [[0.0, 37.0], [0.7423355438742085, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.9203096359364267, 0.9771253672892224]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Al_OH_noTert_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH_noTert)
    # normalization
    linear_norm_parameter = (
        0.49733523982841565,
        0.49733523982841543,
    )  # error of 1.11E-16 with sample range (0.00E+00,2.00E+01) resulting in fit range (4.97E-01,1.04E+01)
    linear_norm_parameter_normdata = {"error": 1.1102230246251565e-16}
    min_max_norm_parameter = (
        1.5560199363023413e-09,
        1.005358076311579,
    )  # error of 1.54E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00015377172553700648}
    sigmoidal_norm_parameter = (
        -1.286728147327964,
        2.286728011845654,
    )  # error of 1.32E-07 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.50E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.3228508632945335e-07}
    dual_sigmoidal_norm_parameter = (
        -1.286728147327964,
        1.0,
        2.286728011845654,
    )  # error of 1.32E-07 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.3228508632945335e-07}
    genlog_norm_parameter = (
        1.1068750304866115,
        0.2851479099498231,
        0.021789171814512582,
        1.839175343427483,
    )  # error of 3.23E-08 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.84E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.230545064791812e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 20.0], [0.984121411052956, 0.9999999999960509]]
    }
    preferred_normalization = "unity"
    # functions


class fr_ArN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_ArN)
    # normalization
    linear_norm_parameter = (
        0.02460278574928243,
        0.9493045625893672,
    )  # error of 0.00E+00 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.49E-01,1.07E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        6.546180810957143e-09,
        1.026791718467936,
    )  # error of 6.13E-04 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0006125213989057668}
    sigmoidal_norm_parameter = (
        -0.2533822935500519,
        2.8879157330454324,
    )  # error of 7.85E-17 with sample range (0.00E+00,5.00E+00) resulting in fit range (6.75E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.253382336540342,
        1.0,
        2.88791558684182,
    )  # error of 1.08E-09 with sample range (0.00E+00,5.00E+00) resulting in fit range (6.75E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.083638555222544e-09}
    genlog_norm_parameter = (
        2.8987811051095203,
        0.43359979525360404,
        0.26109331686606746,
        1.8652669580891061,
    )  # error of 1.90E-12 with sample range (0.00E+00,5.00E+00) resulting in fit range (7.05E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.9042672072510272e-12}
    autogen_normdata = {
        "sample_bounds": [[0.0, 5.0], [0.7053518142607835, 0.9999997503676429]],
        "sample_bounds99": [[0.0, 0.0], [0.9739073483385209, 0.9739073483385209]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Ar_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ar_COO)
    # normalization
    linear_norm_parameter = (
        0.49411802937735627,
        0.49411802937735616,
    )  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.94E-01,3.46E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        8.646173628068578e-09,
        1.0119039788513677,
    )  # error of 2.64E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00026408637339231465}
    sigmoidal_norm_parameter = (
        -1.1049662105257694,
        2.10496612246408,
    )  # error of 7.86E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.11E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.855738282103175e-11}
    dual_sigmoidal_norm_parameter = (
        -1.1049662105257694,
        1.0,
        2.10496612246408,
    )  # error of 7.86E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.11E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 7.855738282103175e-11}
    genlog_norm_parameter = (
        1.103674412267105,
        0.2955961254852563,
        0.047591571300420826,
        1.8283911311064251,
    )  # error of 1.29E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.2929000869910112e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9656726097354408, 0.9999520047492106]],
        "sample_bounds99": [[0.0, 0.0], [0.988236071683713, 0.988236071683713]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Ar_N_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ar_N)
    # normalization
    linear_norm_parameter = (
        0.0505043831748716,
        0.706005745196576,
    )  # error of 5.36E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (7.06E-01,2.32E+00)
    linear_norm_parameter_normdata = {"error": 0.05356129012946231}
    min_max_norm_parameter = (
        0.0,
        2.1715264674397554,
    )  # error of 5.56E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.055585914693548524}
    sigmoidal_norm_parameter = (
        0.20224173292892467,
        0.8623990343842892,
    )  # error of 4.21E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (4.57E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0042142892089573275}
    dual_sigmoidal_norm_parameter = (
        0.20233370179919816,
        73.80950237221694,
        0.8624656268051756,
    )  # error of 2.50E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (3.27E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0024953307543293148}
    genlog_norm_parameter = (
        1.0634866717004823,
        1.9492600219770395,
        0.9437460656417345,
        3.1756444239315287,
    )  # error of 1.02E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (5.10E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0010237559220581838}
    autogen_normdata = {
        "sample_bounds": [[0.0, 32.0], [0.5096870332548603, 0.999999999999996]],
        "sample_bounds99": [[0.0, 4.0], [0.66866040083357, 0.9886851262489849]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_Ar_NH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ar_NH)
    # normalization
    linear_norm_parameter = (
        0.025767680908718393,
        0.9464298213160812,
    )  # error of 0.00E+00 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.46E-01,1.18E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        6.578269936941624e-09,
        1.0285975817965838,
    )  # error of 8.35E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0008346254288172433}
    sigmoidal_norm_parameter = (
        -0.3459278133476376,
        2.64087945539897,
    )  # error of 7.85E-17 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.14E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.3459278137373095,
        1.0,
        2.6408794542523295,
    )  # error of 1.01E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.14E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.0113388713633369e-11}
    genlog_norm_parameter = (
        2.652253980652054,
        0.4259417618290051,
        0.24728192595347642,
        1.8634030063302984,
    )  # error of 1.01E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.37E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.0054252287160277e-12}
    autogen_normdata = {
        "sample_bounds": [[0.0, 9.0], [0.7371358812409098, 0.9999999999823488]],
        "sample_bounds99": [[0.0, 0.0], [0.9721975022260839, 0.9721975022260839]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Ar_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ar_OH)
    # normalization
    linear_norm_parameter = (
        0.010934015936408747,
        0.9686094918123338,
    )  # error of 3.71E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.19E+00)
    linear_norm_parameter_normdata = {"error": 0.003714333333833346}
    min_max_norm_parameter = (
        6.848999398227221e-09,
        1.0236283333676808,
    )  # error of 1.24E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.001242212325573605}
    sigmoidal_norm_parameter = (
        -1.239714164093668,
        1.6725059613391224,
    )  # error of 2.37E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (8.88E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00023654081247867183}
    dual_sigmoidal_norm_parameter = (
        -1.239714164093668,
        1.0,
        1.6725059613391224,
    )  # error of 2.37E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (8.88E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00023654081247867183}
    genlog_norm_parameter = (
        1.663588039697213,
        -3.505514906167138,
        0.958080837822211,
        0.022804549619115978,
    )  # error of 2.33E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (8.84E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00023316073022877239}
    autogen_normdata = {
        "sample_bounds": [[0.0, 20.0], [0.8842193908579391, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9769309116398801, 0.9769309116398801]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_COO)
    # normalization
    linear_norm_parameter = (
        0.025487706106450125,
        0.9292380352434951,
    )  # error of 1.01E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (9.29E-01,1.74E+00)
    linear_norm_parameter_normdata = {"error": 0.01008479209103597}
    min_max_norm_parameter = (
        6.888269928388699e-09,
        1.0553034775863013,
    )  # error of 1.49E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.001487057589393272}
    sigmoidal_norm_parameter = (
        -0.27734172080976294,
        2.2665201420240852,
    )  # error of 4.93E-04 with sample range (0.00E+00,3.20E+01) resulting in fit range (6.52E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0004931422119301587}
    dual_sigmoidal_norm_parameter = (
        -0.27734172080976294,
        1.0,
        2.2665201420240852,
    )  # error of 4.93E-04 with sample range (0.00E+00,3.20E+01) resulting in fit range (6.52E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0004931422119301587}
    genlog_norm_parameter = (
        2.2425975591285736,
        -2.135907809055062,
        0.859243171425322,
        0.014086824742109993,
    )  # error of 4.86E-04 with sample range (0.00E+00,3.20E+01) resulting in fit range (6.03E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0004859199938165967}
    autogen_normdata = {
        "sample_bounds": [[0.0, 32.0], [0.6033633713450424, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9476045021152308, 0.9476045021152308]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_COO2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_COO2)
    # normalization
    linear_norm_parameter = (
        0.026660100588824065,
        0.9260616544509377,
    )  # error of 1.06E-02 with sample range (0.00E+00,2.70E+01) resulting in fit range (9.26E-01,1.65E+00)
    linear_norm_parameter_normdata = {"error": 0.010552618973501668}
    min_max_norm_parameter = (
        9.034811582204987e-09,
        1.0579100746345402,
    )  # error of 1.58E-03 with sample range (0.00E+00,2.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0015805509753703646}
    sigmoidal_norm_parameter = (
        -0.24732225870510682,
        2.284131120056377,
    )  # error of 4.83E-04 with sample range (0.00E+00,2.70E+01) resulting in fit range (6.38E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0004830756003664581}
    dual_sigmoidal_norm_parameter = (
        -0.24732225870510682,
        1.0,
        2.284131120056377,
    )  # error of 4.83E-04 with sample range (0.00E+00,2.70E+01) resulting in fit range (6.38E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0004830756003664581}
    genlog_norm_parameter = (
        4.896149542519247,
        -1.0300201145295764,
        1.0939135224824221,
        0.0009359282773528107,
    )  # error of 1.48E-03 with sample range (0.00E+00,2.70E+01) resulting in fit range (5.44E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.001479532445702464}
    autogen_normdata = {
        "sample_bounds": [[0.0, 27.0], [0.0005441118665063532, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9451763971317502, 0.9451763971317502]],
    }
    preferred_normalization = "genlog"
    # functions


class fr_C_O_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_C_O)
    # normalization
    linear_norm_parameter = (
        0.022515720837872344,
        0.7822215231398375,
    )  # error of 9.81E-02 with sample range (0.00E+00,7.50E+01) resulting in fit range (7.82E-01,2.47E+00)
    linear_norm_parameter_normdata = {"error": 0.09811116967489561}
    min_max_norm_parameter = (
        5e-324,
        2.3300769555158443,
    )  # error of 1.98E-02 with sample range (0.00E+00,7.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019821647980900747}
    sigmoidal_norm_parameter = (
        0.9129037319576498,
        1.3107513550393526,
    )  # error of 3.49E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (2.32E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0034888483447128495}
    dual_sigmoidal_norm_parameter = (
        0.9129037321984369,
        15.763177112455574,
        1.3107513626098963,
    )  # error of 1.96E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (5.63E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0019622645738111757}
    genlog_norm_parameter = (
        1.1392120273230875,
        -0.7675838506842031,
        0.5243526424444765,
        0.1056154749952747,
    )  # error of 2.72E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (1.54E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0027203369800100825}
    autogen_normdata = {
        "sample_bounds": [[0.0, 75.0], [0.15370415869829762, 1.0]],
        "sample_bounds99": [[0.0, 4.0], [0.5269639235438991, 0.9930706087897672]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_C_O_noCOO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_C_O_noCOO)
    # normalization
    linear_norm_parameter = (
        0.027301361059322793,
        0.7733988486490774,
    )  # error of 9.21E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (7.73E-01,2.08E+00)
    linear_norm_parameter_normdata = {"error": 0.09213881043143622}
    min_max_norm_parameter = (
        1.5777218104420236e-30,
        2.236082199928005,
    )  # error of 2.39E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02394498617473435}
    sigmoidal_norm_parameter = (
        0.7988894132889973,
        1.3355456372556587,
    )  # error of 2.67E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (2.56E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0026704515149997532}
    dual_sigmoidal_norm_parameter = (
        0.7988893624690527,
        19.049004921212244,
        1.3355454934130946,
    )  # error of 1.50E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (2.46E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0015013825507430403}
    genlog_norm_parameter = (
        1.185357553283848,
        -0.992112954843278,
        0.9287088404364799,
        0.1473716452396698,
    )  # error of 2.01E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (1.81E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0020084514180996244}
    autogen_normdata = {
        "sample_bounds": [[0.0, 48.0], [0.1809510298695975, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.5657432137139153, 0.9831988822756829]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_C_S_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_C_S)
    # normalization
    linear_norm_parameter = (
        0.4948954594086534,
        0.4948954594086533,
    )  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.95E-01,2.47E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        5.535786530479266e-12,
        1.010314379200814,
    )  # error of 1.64E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00016399743490266223}
    sigmoidal_norm_parameter = (
        -1.1387416964001449,
        2.138741698930851,
    )  # error of 4.26E-10 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.19E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.2592951388087386e-10}
    dual_sigmoidal_norm_parameter = (
        -1.1387416964001449,
        1.0,
        2.138741698930851,
    )  # error of 4.26E-10 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.19E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.2592951388087386e-10}
    genlog_norm_parameter = (
        1.104455720849779,
        0.2930520771726282,
        0.04140788756908054,
        1.8310088431718654,
    )  # error of 1.63E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.70E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.6287707405027163e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9700613252912462, 0.9996232181863494]],
        "sample_bounds99": [[0.0, 0.0], [0.9897909351050138, 0.9897909351050138]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_HOCCN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_HOCCN)
    # normalization
    linear_norm_parameter = (
        0.4977502024817768,
        0.4977502024817767,
    )  # error of 2.22E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.98E-01,1.99E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        1.26267034441651e-08,
        1.0045199328470353,
    )  # error of 8.00E-05 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 7.999280064796555e-05}
    sigmoidal_norm_parameter = (
        -1.3236118939770356,
        2.3236118577424274,
    )  # error of 3.85E-07 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 3.851500728346835e-07}
    dual_sigmoidal_norm_parameter = (
        -1.3236118939770356,
        1.0,
        2.3236118577424274,
    )  # error of 3.85E-07 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 3.851500728346835e-07}
    genlog_norm_parameter = (
        1.1072811009248538,
        0.28381613355093094,
        0.018420633690387628,
        1.8405567878608184,
    )  # error of 3.61E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.87E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.611681420601087e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 3.0], [0.9865574153755745, 0.9995058310482284]]
    }
    preferred_normalization = "unity"
    # functions


class fr_Imine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Imine)
    # normalization
    linear_norm_parameter = (
        0.03889649931506156,
        0.9186823185913269,
    )  # error of 0.00E+00 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.19E-01,1.93E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        5.062601797035858e-09,
        1.0443004597486796,
    )  # error of 1.04E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0010400646353493797}
    sigmoidal_norm_parameter = (
        -0.23305480673382584,
        2.52767371071209,
    )  # error of 0.00E+00 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.43E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.23305480786468896,
        1.0,
        2.5276737069674944,
    )  # error of 5.23E-11 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.43E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.233675389940342e-11}
    genlog_norm_parameter = (
        2.54376914192436,
        0.4607687831175127,
        0.31984182613502987,
        1.7997417746060027,
    )  # error of 2.76E-10 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.74E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.7617798578943506e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 26.0], [0.6742571815636278, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9575788176868263, 0.9575788176868263]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_NH0_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_NH0)
    # normalization
    linear_norm_parameter = (
        0.06607380335566115,
        0.5226975405536398,
    )  # error of 1.02E-01 with sample range (0.00E+00,6.40E+01) resulting in fit range (5.23E-01,4.75E+00)
    linear_norm_parameter_normdata = {"error": 0.10225050519595472}
    min_max_norm_parameter = (
        1.5777218104420236e-30,
        3.4322660362420145,
    )  # error of 3.03E-02 with sample range (0.00E+00,6.40E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0302812306912358}
    sigmoidal_norm_parameter = (
        1.446568823597894,
        0.9242400863243287,
    )  # error of 3.76E-03 with sample range (0.00E+00,6.40E+01) resulting in fit range (2.08E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.003755856242728876}
    dual_sigmoidal_norm_parameter = (
        1.382038072209329,
        1.138228128038037,
        0.8833018513448258,
    )  # error of 5.61E-04 with sample range (0.00E+00,6.40E+01) resulting in fit range (1.72E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0005606209326244682}
    genlog_norm_parameter = (
        0.843886503445007,
        -0.2510983118680596,
        1.9039682062511323,
        0.54503379561084,
    )  # error of 1.54E-03 with sample range (0.00E+00,6.40E+01) resulting in fit range (1.81E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0015440249128680094}
    autogen_normdata = {
        "sample_bounds": [[0.0, 64.0], [0.18076177666005358, 1.0]],
        "sample_bounds99": [[0.0, 5.0], [0.39354354700385996, 0.9823690990865395]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_NH1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_NH1)
    # normalization
    linear_norm_parameter = (
        0.031218773641592312,
        0.7849136633253863,
    )  # error of 8.37E-02 with sample range (0.00E+00,5.00E+01) resulting in fit range (7.85E-01,2.35E+00)
    linear_norm_parameter_normdata = {"error": 0.08372539667667236}
    min_max_norm_parameter = (
        7.888609052210118e-31,
        2.0805517126600583,
    )  # error of 2.92E-02 with sample range (0.00E+00,5.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.029194943362195133}
    sigmoidal_norm_parameter = (
        0.6671905805486065,
        1.553108631785838,
    )  # error of 1.81E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (2.62E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0018081396924453249}
    dual_sigmoidal_norm_parameter = (
        0.6671904374105158,
        23.867847813963294,
        1.5531081749737823,
    )  # error of 1.01E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (1.21E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0010052680333371812}
    genlog_norm_parameter = (
        1.464107554460365,
        0.08281058295864306,
        0.9546834345970568,
        0.47551676499525086,
    )  # error of 1.70E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (2.15E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0017011420839810802}
    autogen_normdata = {
        "sample_bounds": [[0.0, 50.0], [0.2148387771799452, 1.0]],
        "sample_bounds99": [[0.0, 2.0], [0.6262358959136987, 0.9725290685181707]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_NH2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_NH2)
    # normalization
    linear_norm_parameter = (
        0.027626013199217845,
        0.904411104115867,
    )  # error of 2.10E-02 with sample range (0.00E+00,2.40E+01) resulting in fit range (9.04E-01,1.57E+00)
    linear_norm_parameter_normdata = {"error": 0.02103458239739804}
    min_max_norm_parameter = (
        6.3934981194901046e-09,
        1.1000637973187075,
    )  # error of 2.52E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0025229274601359106}
    sigmoidal_norm_parameter = (
        -0.0551854803177655,
        2.1819036489792745,
    )  # error of 1.09E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (5.30E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0010876537643603292}
    dual_sigmoidal_norm_parameter = (
        9.005842740580992e-07,
        14977566.434938159,
        2.298853419847412,
    )  # error of 6.20E-04 with sample range (0.00E+00,2.40E+01) resulting in fit range (1.39E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0006204932750760248}
    genlog_norm_parameter = (
        2.139324187816158,
        -2.3168116602762083,
        0.66553945183396,
        0.0057830792267490275,
    )  # error of 1.07E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (4.46E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0010656341274527135}
    autogen_normdata = {
        "sample_bounds": [[0.0, 24.0], [0.4457104538542366, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.9090675810093001, 0.9888357513437641]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_N_O_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_N_O)
    # normalization
    linear_norm_parameter = (
        0.004979551840334274,
        0.9874611284984353,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.87E-01,1.05E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        4.290831526052814e-09,
        1.0076168981970937,
    )  # error of 8.76E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0008755606484271415}
    sigmoidal_norm_parameter = (
        -3.51573557027191,
        1.0800867262295535,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -3.515735715368847,
        1.0,
        1.0800866688067339,
    )  # error of 6.17E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.171981699943612e-10}
    genlog_norm_parameter = (
        1.0824467317964082,
        0.37674586640501356,
        0.02917406356466021,
        1.9438696719700521,
    )  # error of 2.57E-08 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.565840226900424e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.9781578042953372, 0.9999999484497815]]
    }
    preferred_normalization = "unity"
    # functions


class fr_Ndealkylation1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation1)
    # normalization
    linear_norm_parameter = (
        0.03791658750712423,
        0.9210171084602388,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (9.21E-01,1.22E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        6.156084770016071e-09,
        1.0428249668903355,
    )  # error of 1.06E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0010585700309633564}
    sigmoidal_norm_parameter = (
        -0.20868893262353097,
        2.6066541330168578,
    )  # error of 7.85E-17 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.33E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.20868916209862504,
        1.0,
        2.6066534426782026,
    )  # error of 6.89E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.33E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.892514102037599e-09}
    genlog_norm_parameter = (
        2.622377607773997,
        0.4623310330395769,
        0.3215828761048411,
        1.8024962252178414,
    )  # error of 6.74E-10 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.66E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 6.736178679816114e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 8.0], [0.665923807677635, 0.9999999995356055]],
        "sample_bounds99": [[0.0, 0.0], [0.9589336953741524, 0.9589336953741524]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Ndealkylation2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation2)
    # normalization
    linear_norm_parameter = (
        0.05745482906538413,
        0.8807107360337569,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.81E-01,1.34E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        2.739177310521382e-09,
        1.0659099385351551,
    )  # error of 1.79E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0017907231602327296}
    sigmoidal_norm_parameter = (
        -0.004626898029145944,
        2.706941292405111,
    )  # error of 5.03E-16 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.03E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 5.026748538604307e-16}
    dual_sigmoidal_norm_parameter = (
        -0.004626898216699545,
        1.0,
        2.7069412916287168,
    )  # error of 1.16E-11 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.03E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.1628282992413492e-11}
    genlog_norm_parameter = (
        2.736808549607176,
        0.0980462017750495,
        1.5972216205812708,
        1.9881777298003702,
    )  # error of 4.97E-13 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.67E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.969976457233551e-13}
    autogen_normdata = {
        "sample_bounds": [[0.0, 8.0], [0.5670841072915257, 0.9999999996743141]],
        "sample_bounds99": [[0.0, 0.0], [0.9381655650996661, 0.9381655650996661]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_Nhpyrrole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_Nhpyrrole)
    # normalization
    linear_norm_parameter = (
        0.025012748852603184,
        0.9478546930776232,
    )  # error of 0.00E+00 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.48E-01,1.10E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        3.340124193009499e-09,
        1.0278892650835505,
    )  # error of 8.08E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0008076423432666436}
    sigmoidal_norm_parameter = (
        -0.3902136788160999,
        2.5747937021696656,
    )  # error of 7.85E-17 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.32E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.3902136790305557,
        1.0,
        2.5747937015890643,
    )  # error of 4.92E-12 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.32E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.920647983189302e-12}
    genlog_norm_parameter = (
        2.5858938366189523,
        0.42074563442961355,
        0.235856987956551,
        1.868379564253036,
    )  # error of 1.50E-13 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.53E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.5006209497467547e-13}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.7527385442966315, 0.999999931536202]],
        "sample_bounds99": [[0.0, 0.0], [0.9728674419300142, 0.9728674419300142]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_SH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_SH)
    # normalization
    linear_norm_parameter = (
        0.49842014218720343,
        0.4984201421872032,
    )  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.98E-01,3.49E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        3.704484506370359e-09,
        1.0031697310626564,
    )  # error of 1.67E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00016685569142196825}
    sigmoidal_norm_parameter = (
        -1.3987722757661272,
        2.3987722116196286,
    )  # error of 7.94E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.942203561484007e-10}
    dual_sigmoidal_norm_parameter = (
        -1.3987722757661272,
        1.0,
        2.3987722116196286,
    )  # error of 7.94E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 7.942203561484007e-10}
    genlog_norm_parameter = (
        1.107933174630707,
        0.2816740776854764,
        0.012963205986552264,
        1.8427823615757748,
    )  # error of 2.55E-15 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.91E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.55351295663786e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9905182970262847, 0.9999875334006051]]
    }
    preferred_normalization = "unity"
    # functions


class fr_aldehyde_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_aldehyde)
    # normalization
    linear_norm_parameter = (
        0.49738773510384093,
        0.4973877351038408,
    )  # error of 4.44E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.97E-01,2.98E+00)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        7.478980473497863e-09,
        1.0052519688208172,
    )  # error of 9.75E-05 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 9.750190800361425e-05}
    sigmoidal_norm_parameter = (
        -1.291096523056102,
        2.291096475694497,
    )  # error of 1.51E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.51E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.5114949225569774e-07}
    dual_sigmoidal_norm_parameter = (
        -1.291096523056102,
        1.0,
        2.291096475694497,
    )  # error of 1.51E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.51E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.5114949225569774e-07}
    genlog_norm_parameter = (
        1.7845920940078177,
        -0.8738102194409251,
        0.4267160082745265,
        2.8538370580658303,
    )  # error of 4.00E-09 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.70E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.004206299867974e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 5.0], [0.9703407508830474, 0.9999958093904291]]
    }
    preferred_normalization = "unity"
    # functions


class fr_alkyl_carbamate_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_carbamate)
    # normalization
    linear_norm_parameter = (
        0.4965278124968755,
        0.4965278124968754,
    )  # error of 4.44E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.97E-01,2.98E+00)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        6.897442569832465e-09,
        1.0069929365320245,
    )  # error of 2.31E-04 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00023129786668582564}
    sigmoidal_norm_parameter = (
        -1.227745991027422,
        2.22774593441806,
    )  # error of 1.81E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.39E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.8101772525369597e-08}
    dual_sigmoidal_norm_parameter = (
        -1.227745991027422,
        1.0,
        2.22774593441806,
    )  # error of 1.81E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.8101772525369597e-08}
    genlog_norm_parameter = (
        1.1060801962429578,
        0.2877499472134811,
        0.028317304479567262,
        1.836481002761403,
    )  # error of 2.59E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.79E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.590639214261614e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 5.0], [0.9794191613597176, 0.9999159735624574]]
    }
    preferred_normalization = "unity"
    # functions


class fr_alkyl_halide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_halide)
    # normalization
    linear_norm_parameter = (
        0.008601680392056554,
        0.9308092271694038,
    )  # error of 1.58E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (9.31E-01,1.81E+00)
    linear_norm_parameter_normdata = {"error": 0.01584164183857822}
    min_max_norm_parameter = (
        9.154473872975373e-09,
        1.0948072482932691,
    )  # error of 1.38E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.013802463919201114}
    sigmoidal_norm_parameter = (
        -3.20022604341129,
        0.5453458753101628,
    )  # error of 5.09E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.51E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.005094571577186526}
    dual_sigmoidal_norm_parameter = (
        -3.200226429591724,
        1.0,
        0.5453458328225158,
    )  # error of 5.09E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.51E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00509457157718233}
    genlog_norm_parameter = (
        1.5706106658334722,
        4.1340803219645945,
        0.9824881072292492,
        52.63440780193744,
    )  # error of 3.04E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.84E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0030437137139028183}
    autogen_normdata = {
        "sample_bounds": [[0.0, 102.0], [0.8842154274094988, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.9108968778684846, 0.98502329187865]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_allylic_oxid_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_allylic_oxid)
    # normalization
    linear_norm_parameter = (
        0.004305773194698159,
        0.9499732166962105,
    )  # error of 1.75E-02 with sample range (0.00E+00,5.60E+01) resulting in fit range (9.50E-01,1.19E+00)
    linear_norm_parameter_normdata = {"error": 0.017498006246362188}
    min_max_norm_parameter = (
        6.2907096383782455e-09,
        1.107924268433823,
    )  # error of 1.06E-02 with sample range (0.00E+00,5.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.010608305695273114}
    sigmoidal_norm_parameter = (
        -2.624628166420796,
        0.6192650652321926,
    )  # error of 1.97E-03 with sample range (0.00E+00,5.60E+01) resulting in fit range (8.36E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.001967501562276078}
    dual_sigmoidal_norm_parameter = (
        -2.624628166420796,
        1.0,
        0.6192650652321926,
    )  # error of 1.97E-03 with sample range (0.00E+00,5.60E+01) resulting in fit range (8.36E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.001967501562276078}
    genlog_norm_parameter = (
        0.6028122437221227,
        -8.805076472793315,
        0.03394072114759804,
        0.0009117951456768067,
    )  # error of 1.88E-03 with sample range (0.00E+00,5.60E+01) resulting in fit range (8.32E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0018752154920356793}
    autogen_normdata = {
        "sample_bounds": [[0.0, 56.0], [0.8316473375927989, 1.0]],
        "sample_bounds99": [[0.0, 4.0], [0.9040312157742787, 0.9909901758344529]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_amide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_amide)
    # normalization
    linear_norm_parameter = (
        0.018247396195880828,
        0.8361339146143514,
    )  # error of 7.21E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (8.36E-01,1.71E+00)
    linear_norm_parameter_normdata = {"error": 0.07213720490497141}
    min_max_norm_parameter = (
        0.04651569330182743,
        1.4956993928811566,
    )  # error of 2.28E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.02277832158142506}
    sigmoidal_norm_parameter = (
        0.4809600464417933,
        1.266170913827053,
    )  # error of 2.07E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (3.52E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0020724848340902702}
    dual_sigmoidal_norm_parameter = (
        0.480960020641994,
        32.957619361505,
        1.2661708722360183,
    )  # error of 1.27E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (1.31E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0012658433021765115}
    genlog_norm_parameter = (
        1.140508340356683,
        -4.446633619292976,
        0.40847854442704473,
        0.001956413380845521,
    )  # error of 1.57E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (2.70E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00156697306779949}
    autogen_normdata = {
        "sample_bounds": [[0.0, 48.0], [0.270316177771659, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.658015398063566, 0.9864174984934097]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_amidine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_amidine)
    # normalization
    linear_norm_parameter = (
        0.02087812096911279,
        0.956658900698937,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.57E-01,1.21E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        1.0142986812940386e-08,
        1.0229791584422843,
    )  # error of 5.67E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0005674296578757846}
    sigmoidal_norm_parameter = (
        -0.4118456671316019,
        2.672507134625743,
    )  # error of 7.85E-17 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.50E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.41184569107524827,
        1.0,
        2.6725070662621717,
    )  # error of 5.18E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.175410926761861e-10}
    genlog_norm_parameter = (
        2.6819460011796368,
        0.4112879398405415,
        0.21255300531166724,
        1.8880910539352165,
    )  # error of 5.29E-11 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.69E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 5.2899745976809554e-11}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.7693810233598843, 0.9999999999999964]],
        "sample_bounds99": [[0.0, 0.0], [0.9775370216309092, 0.9775370216309092]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_aniline_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_aniline)
    # normalization
    linear_norm_parameter = (
        0.038572385626053585,
        0.8111639952400167,
    )  # error of 4.99E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (8.11E-01,1.39E+00)
    linear_norm_parameter_normdata = {"error": 0.049870489913451714}
    min_max_norm_parameter = (
        4.067547786674466e-09,
        1.3022513887988345,
    )  # error of 1.73E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.01727774099591378}
    sigmoidal_norm_parameter = (
        0.18131870166733124,
        1.4627551170239292,
    )  # error of 1.27E-03 with sample range (0.00E+00,1.50E+01) resulting in fit range (4.34E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0012660951604188918}
    dual_sigmoidal_norm_parameter = (
        0.18129261187647874,
        91.42925286295274,
        1.4627135733922834,
    )  # error of 8.26E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (6.33E-08,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.000825866491681948}
    genlog_norm_parameter = (
        2.7413001530719976,
        -2.1327613126838796,
        0.20953445963513556,
        0.00014215931700537761,
    )  # error of 1.35E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (1.41E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.013540806495829305}
    autogen_normdata = {
        "sample_bounds": [[0.0, 15.0], [0.014147325367881727, 1.0]],
        "sample_bounds99": [[0.0, 2.0], [0.7598183524704656, 0.9988583896434308]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_aryl_methyl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_aryl_methyl)
    # normalization
    linear_norm_parameter = (
        0.03813656770680851,
        0.8922580301103881,
    )  # error of 1.20E-02 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.92E-01,1.20E+00)
    linear_norm_parameter_normdata = {"error": 0.011960813241880672}
    min_max_norm_parameter = (
        4.251871042900079e-09,
        1.0846727833106184,
    )  # error of 5.19E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.005185547545784227}
    sigmoidal_norm_parameter = (
        -0.39967703372113306,
        1.763714100024283,
    )  # error of 4.24E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.69E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0004239793559654811}
    dual_sigmoidal_norm_parameter = (
        -0.39967703372113306,
        1.0,
        1.763714100024283,
    )  # error of 4.24E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0004239793559654811}
    genlog_norm_parameter = (
        2.186681836335556,
        1.2644126832361133,
        1.0408028177857813,
        12.909498676718341,
    )  # error of 2.13E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.01E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.13149756003618e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 8.0], [0.8010576531702694, 0.9999999676469545]],
        "sample_bounds99": [[0.0, 1.0], [0.9219370223967605, 0.9854463081542463]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_azide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_azide)
    # normalization
    linear_norm_parameter = (
        0.4959878610925019,
        0.4959878610925017,
    )  # error of 3.33E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.96E-01,1.98E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        2.1583046536306194e-11,
        1.0080891853213716,
    )  # error of 3.25E-05 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 3.249707528446419e-05}
    sigmoidal_norm_parameter = (
        -1.1948180919512132,
        2.1948180318881927,
    )  # error of 5.06E-09 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.32E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 5.0576094423604445e-09}
    dual_sigmoidal_norm_parameter = (
        -1.1948180919512132,
        1.0,
        2.1948180318881927,
    )  # error of 5.06E-09 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.32E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.0576094423604445e-09}
    genlog_norm_parameter = (
        1.939351454905407,
        -0.9724652889797012,
        1.054356287604753,
        2.8221381728147823,
    )  # error of 1.17E-09 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.49E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.1671676958258104e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 3.0], [0.9487880676907243, 0.9998315535518929]]
    }
    preferred_normalization = "unity"
    # functions


class fr_azo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_azo)
    # normalization
    linear_norm_parameter = (
        0.4990500854923059,
        0.4990500854923058,
    )  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        5.424965999671179e-12,
        1.0019034438689793,
    )  # error of 1.50E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0001502196323679034}
    sigmoidal_norm_parameter = (
        -1.502811180689176,
        2.502811017959295,
    )  # error of 5.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.77E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.997340119494709e-08}
    dual_sigmoidal_norm_parameter = (
        -1.502811180689176,
        1.0,
        2.502811017959295,
    )  # error of 5.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.77E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.997340119494709e-08}
    genlog_norm_parameter = (
        1.1085421390501606,
        0.27966911234213954,
        0.007809922138213044,
        1.844869688796806,
    )  # error of 3.00E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.94E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.9976021664879227e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9942750076401095, 0.9999315228125863]]
    }
    preferred_normalization = "unity"
    # functions


class fr_barbitur_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_barbitur)
    # normalization
    linear_norm_parameter = (
        0.49999999999890943,
        0.16647835027998992,
    )  # error of 2.35E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.66E-01,1.17E+00)
    linear_norm_parameter_normdata = {"error": 0.23543594081003158}
    min_max_norm_parameter = (
        5.002826379765727e-11,
        1.0005652758530887,
    )  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.239064645900985e-09}
    sigmoidal_norm_parameter = (
        0.6589560462180946,
        21.924533557093195,
    )  # error of 4.43E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (5.32E-07,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.4316509547777085e-07}
    dual_sigmoidal_norm_parameter = (
        0.4999997695881287,
        8.487133810156745,
        13.55104020684501,
    )  # error of 7.37E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.14E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0007372349977297952}
    genlog_norm_parameter = (
        10.256024960560502,
        -0.4376011212502097,
        0.662217173848721,
        0.00046301337926178443,
    )  # error of 6.48E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.10E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 6.48331592568311e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2.0], [1.1031405783778934e-07, 0.9999999801388522]]
    }
    preferred_normalization = "unity"
    # functions


class fr_benzene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_benzene)
    # normalization
    linear_norm_parameter = (
        0.02947953215467447,
        0.7331435625333342,
    )  # error of 1.11E-01 with sample range (0.00E+00,3.30E+01) resulting in fit range (7.33E-01,1.71E+00)
    linear_norm_parameter_normdata = {"error": 0.11069055201027347}
    min_max_norm_parameter = (
        7.888609052210118e-31,
        2.4698336963662135,
    )  # error of 1.96E-02 with sample range (0.00E+00,3.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.019574944078354105}
    sigmoidal_norm_parameter = (
        1.0920776300962571,
        1.3627096584489056,
    )  # error of 6.24E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (1.84E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.006237893249292523}
    dual_sigmoidal_norm_parameter = (
        1.092077640400224,
        13.381103648310065,
        1.3627097252491245,
    )  # error of 4.09E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (4.50E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.004088611068194654}
    genlog_norm_parameter = (
        1.1451758721637044,
        -0.9527455570274224,
        0.45000332745348404,
        0.061581474686799115,
    )  # error of 5.40E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (1.02E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.005404749651414472}
    autogen_normdata = {
        "sample_bounds": [[0.0, 33.0], [0.10171226331845197, 1.0]],
        "sample_bounds99": [[0.0, 5.0], [0.4664154075430777, 0.9974574139870213]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_benzodiazepine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_benzodiazepine)
    # normalization
    linear_norm_parameter = (
        1.0000000000000002,
        2.220446049250313e-16,
    )  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    linear_norm_parameter_normdata = {"error": 3.510833468576701e-16}
    min_max_norm_parameter = (
        1e-10,
        0.9999999999,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.5027041798086037,
        46.14043898879842,
    )  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 9.714558513783135e-11}
    dual_sigmoidal_norm_parameter = (
        0.49954642862905374,
        18.83817716761341,
        18.88166255189391,
    )  # error of 8.03E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.18E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.030546482298942e-05}
    genlog_norm_parameter = (
        12.011496984688954,
        -0.47958199521284717,
        1.1726250998531205,
        0.0003411801343887883,
    )  # error of 4.87E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.03E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.865612171061197e-05}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [2.0298985067945082e-05, 0.9999342520907933]]
    }
    preferred_normalization = "unity"
    # functions


class fr_bicyclic_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_bicyclic)
    # normalization
    linear_norm_parameter = (
        0.008955979676114945,
        0.8971616364051024,
    )  # error of 4.19E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (8.97E-01,1.27E+00)
    linear_norm_parameter_normdata = {"error": 0.041947333895311203}
    min_max_norm_parameter = (
        6.544180000802119e-09,
        1.2950088357093383,
    )  # error of 1.82E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.018167373832982142}
    sigmoidal_norm_parameter = (
        -0.398308593847868,
        0.8997093072499293,
    )  # error of 8.22E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (5.89E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.008222687795751952}
    dual_sigmoidal_norm_parameter = (
        2.0586534953672115e-07,
        60298839.84264371,
        1.1562634894824617,
    )  # error of 6.87E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (4.06E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.006868868569753383}
    genlog_norm_parameter = (
        0.8402817825613296,
        -5.369868660573329,
        0.010441194865324268,
        0.00019698453118541793,
    )  # error of 7.72E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (5.59E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.007721710562775739}
    autogen_normdata = {
        "sample_bounds": [[0.0, 42.0], [0.5589702926339357, 1.0]],
        "sample_bounds99": [[0.0, 6.0], [0.77798680354723, 0.9983788143046096]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_diazo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_diazo)
    # normalization
    linear_norm_parameter = (
        1.0000000000000002,
        2.220446049250313e-16,
    )  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    linear_norm_parameter_normdata = {"error": 3.510833468576701e-16}
    min_max_norm_parameter = (
        1e-10,
        0.9999999999,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.5027041798086037,
        46.14043898879842,
    )  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 9.714558513783135e-11}
    dual_sigmoidal_norm_parameter = (
        0.49954642862905374,
        18.83817716761341,
        18.88166255189391,
    )  # error of 8.03E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.18E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.030546482298942e-05}
    genlog_norm_parameter = (
        12.011496984688954,
        -0.47958199521284717,
        1.1726250998531205,
        0.0003411801343887883,
    )  # error of 4.87E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.03E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.865612171061197e-05}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [2.0298985067945082e-05, 0.9999342520907933]]
    }
    preferred_normalization = "unity"
    # functions


class fr_dihydropyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_dihydropyridine)
    # normalization
    linear_norm_parameter = (
        0.4987476127148559,
        0.4987476127148558,
    )  # error of 3.33E-16 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.99E-01,1.50E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        5.0125553583537566e-11,
        1.002511071570499,
    )  # error of 4.22E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.222599406096338e-09}
    sigmoidal_norm_parameter = (
        -1.4468440944675611,
        2.4468441050172953,
    )  # error of 6.43E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.72E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 6.434109689479328e-09}
    dual_sigmoidal_norm_parameter = (
        -1.4468440944675611,
        1.0,
        2.4468441050172953,
    )  # error of 6.43E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.72E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.434109689479328e-09}
    genlog_norm_parameter = (
        1.1082502597288388,
        0.2806306784339659,
        0.010286910761785893,
        1.8438680940135062,
    )  # error of 5.44E-15 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.92E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 5.440092820663267e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2.0], [0.9924672633973569, 0.9991711068222644]]
    }
    preferred_normalization = "unity"
    # functions


class fr_epoxide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_epoxide)
    # normalization
    linear_norm_parameter = (
        0.4991075803177717,
        0.49910758031777147,
    )  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        6.5085154339493485e-09,
        1.0017880307004197,
    )  # error of 1.50E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00014966952810904812}
    sigmoidal_norm_parameter = (
        -1.5152730763769928,
        2.5152729680451307,
    )  # error of 7.49E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.78E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.491895004374527e-08}
    dual_sigmoidal_norm_parameter = (
        -1.5152730763769928,
        1.0,
        2.5152729680451307,
    )  # error of 7.49E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 7.491895004374527e-08}
    genlog_norm_parameter = (
        1.1085975343386154,
        0.2794865726029763,
        0.007338549278490562,
        1.8450599552188773,
    )  # error of 4.55E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.95E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.551914400963142e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9946194534443819, 0.9999356883252044]]
    }
    preferred_normalization = "unity"
    # functions


class fr_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_ester)
    # normalization
    linear_norm_parameter = (
        0.020566648999453596,
        0.9270090691836141,
    )  # error of 1.36E-02 with sample range (0.00E+00,2.10E+01) resulting in fit range (9.27E-01,1.36E+00)
    linear_norm_parameter_normdata = {"error": 0.01361970029222347}
    min_max_norm_parameter = (
        5.795842863633832e-09,
        1.0721261560339788,
    )  # error of 3.48E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0034809530965885086}
    sigmoidal_norm_parameter = (
        -0.6092514604520473,
        1.6347138056465544,
    )  # error of 1.00E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.30E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0010009572103719824}
    dual_sigmoidal_norm_parameter = (
        -0.6092514604520473,
        1.0,
        1.6347138056465544,
    )  # error of 1.00E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.30E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0010009572103719824}
    genlog_norm_parameter = (
        1.6077740109680747,
        -3.484648053607161,
        0.5761249755706301,
        0.006119019305967105,
    )  # error of 9.71E-04 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.07E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0009709002051102233}
    autogen_normdata = {
        "sample_bounds": [[0.0, 21.0], [0.706862935289873, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.9328057889613466, 0.9861594829937798]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_ether_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_ether)
    # normalization
    linear_norm_parameter = (
        0.020806809203217647,
        0.8260344750786837,
    )  # error of 6.86E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (8.26E-01,1.80E+00)
    linear_norm_parameter_normdata = {"error": 0.0685786265638677}
    min_max_norm_parameter = (
        0.007090101184776188,
        1.486936690120486,
    )  # error of 2.42E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.024243201796160598}
    sigmoidal_norm_parameter = (
        0.40287571370199426,
        1.2035721351882562,
    )  # error of 3.32E-03 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.81E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.003322859194140991}
    dual_sigmoidal_norm_parameter = (
        0.40287522460673725,
        36.70635733642584,
        1.2035714245488187,
    )  # error of 2.03E-03 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.78E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.002031969396120821}
    genlog_norm_parameter = (
        2.0151422105709527,
        -2.5438938409575833,
        0.03747044394451335,
        6.690003457482172e-05,
    )  # error of 1.66E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.59E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.016566496719397007}
    autogen_normdata = {
        "sample_bounds": [[0.0, 47.0], [0.03594627425770625, 1.0]],
        "sample_bounds99": [[0.0, 3.0], [0.641871379027177, 0.9989503447871823]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_furan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_furan)
    # normalization
    linear_norm_parameter = (
        0.4908433241008311,
        0.490843324100831,
    )  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.91E-01,3.44E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        8.521463502068547e-09,
        1.0186549869816706,
    )  # error of 2.70E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00026986766560018903}
    sigmoidal_norm_parameter = (
        -0.9954042665388447,
        1.9954042397477896,
    )  # error of 6.52E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.79E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 6.522729245617143e-08}
    dual_sigmoidal_norm_parameter = (
        -0.9954042665388447,
        1.0,
        1.9954042397477896,
    )  # error of 6.52E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.79E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.522729245617143e-08}
    genlog_norm_parameter = (
        1.100336773268079,
        0.3064361213863763,
        0.07327336213609142,
        1.817286143721912,
    )  # error of 4.54E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.48E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.5396250092366586e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9476469065663239, 0.9999233183507434]],
        "sample_bounds99": [[0.0, 0.0], [0.9816866527412867, 0.9816866527412867]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_guanido_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_guanido)
    # normalization
    linear_norm_parameter = (
        0.4928506434420904,
        0.4928506434420903,
    )  # error of 3.33E-16 with sample range (0.00E+00,9.00E+00) resulting in fit range (4.93E-01,4.93E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        5.606358540599248e-09,
        1.01450613205635,
    )  # error of 2.91E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0002911524356072649}
    sigmoidal_norm_parameter = (
        -1.057470294719679,
        2.0574701969564333,
    )  # error of 5.63E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (8.98E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 5.632161403923419e-12}
    dual_sigmoidal_norm_parameter = (
        -1.057470294719679,
        1.0,
        2.0574701969564333,
    )  # error of 5.63E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (8.98E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.632161403923419e-12}
    genlog_norm_parameter = (
        1.1023910814729312,
        0.2997683808379268,
        0.057601324145092565,
        1.8241080691072127,
    )  # error of 8.76E-09 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.59E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 8.759107550382339e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 9.0], [0.958609416716312, 0.9999978420711286]],
        "sample_bounds99": [[0.0, 0.0], [0.9857012956432879, 0.9857012956432879]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_halogen_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_halogen)
    # normalization
    linear_norm_parameter = (
        0.017119543157199657,
        0.8437328519221161,
    )  # error of 5.64E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (8.44E-01,1.79E+00)
    linear_norm_parameter_normdata = {"error": 0.05635971910576326}
    min_max_norm_parameter = (
        0.041860184464005364,
        1.3840920614215728,
    )  # error of 2.65E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.026476261435344033}
    sigmoidal_norm_parameter = (
        -0.005684348962987795,
        0.9165937462942823,
    )  # error of 2.31E-03 with sample range (0.00E+00,5.50E+01) resulting in fit range (5.01E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0023069777496149556}
    dual_sigmoidal_norm_parameter = (
        1.9418851037243214e-06,
        6722143.368251634,
        0.919708897919921,
    )  # error of 1.44E-03 with sample range (0.00E+00,5.50E+01) resulting in fit range (2.14E-06,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0014404916697602788}
    genlog_norm_parameter = (
        2.152583587994702,
        -2.286713178546249,
        0.038304048670679505,
        8.437179632223308e-05,
    )  # error of 2.13E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (3.67E-02,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.021282417511587336}
    autogen_normdata = {
        "sample_bounds": [[0.0, 55.0], [0.03667974605009358, 1.0]],
        "sample_bounds99": [[0.0, 4.0], [0.6810661966634881, 0.9999300140798705]],
    }
    preferred_normalization = "dual_sig"
    # functions


class fr_hdrzine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_hdrzine)
    # normalization
    linear_norm_parameter = (
        0.49404553590176903,
        0.4940455359017689,
    )  # error of 4.44E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.94E-01,3.46E+00)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        7.616024548469903e-09,
        1.0120524599863319,
    )  # error of 2.81E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00028093289798092997}
    sigmoidal_norm_parameter = (
        -1.1020196341906385,
        2.1020196036271543,
    )  # error of 6.76E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.10E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 6.758626991398842e-11}
    dual_sigmoidal_norm_parameter = (
        -1.1020196341906385,
        1.0,
        2.1020196036271543,
    )  # error of 6.76E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.758626991398842e-11}
    genlog_norm_parameter = (
        1.1036013204302224,
        0.2958339487391666,
        0.04816650024562624,
        1.828146645901781,
    )  # error of 1.27E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.65E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.2657287884465518e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9652655667534271, 0.999951385471899]],
        "sample_bounds99": [[0.0, 0.0], [0.9880910844608254, 0.9880910844608254]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_hdrzone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_hdrzone)
    # normalization
    linear_norm_parameter = (
        0.4927256546910781,
        0.492725654691078,
    )  # error of 4.44E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.93E-01,2.46E+00)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        6.061851638299241e-09,
        1.014763479830144,
    )  # error of 2.35E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0002349362986882245}
    sigmoidal_norm_parameter = (
        -1.0531923508789955,
        2.053192297635956,
    )  # error of 4.24E-12 with sample range (0.00E+00,4.00E+00) resulting in fit range (8.97E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.23672208427206e-12}
    dual_sigmoidal_norm_parameter = (
        -1.0531923508789955,
        1.0,
        2.053192297635956,
    )  # error of 4.24E-12 with sample range (0.00E+00,4.00E+00) resulting in fit range (8.97E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.23672208427206e-12}
    genlog_norm_parameter = (
        1.1022639148991156,
        0.300181457521981,
        0.05858368643902903,
        1.8236846385965941,
    )  # error of 8.42E-09 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.58E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 8.423530095669207e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9579188798501342, 0.9994562942325274]],
        "sample_bounds99": [[0.0, 0.0], [0.9854513178056857, 0.9854513178056857]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_imidazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_imidazole)
    # normalization
    linear_norm_parameter = (
        0.02594266516013577,
        0.9467297943185111,
    )  # error of 0.00E+00 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.47E-01,1.10E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        3.74982058400203e-09,
        1.0280953163139084,
    )  # error of 5.25E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0005253063448314081}
    sigmoidal_norm_parameter = (
        -0.18731110481329363,
        3.0086069081003104,
    )  # error of 7.85E-17 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.37E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.1873112027321337,
        1.0,
        3.0086065444034595,
    )  # error of 2.63E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.37E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.6252523568022206e-09}
    genlog_norm_parameter = (
        3.019948159442087,
        0.44503442275095295,
        0.2819771401895589,
        1.855810108947968,
    )  # error of 1.13E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.74E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.130972288248859e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.6737202233458747, 0.9999999921280395]],
        "sample_bounds99": [[0.0, 0.0], [0.9726724593556906, 0.9726724593556906]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_imide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_imide)
    # normalization
    linear_norm_parameter = (
        0.016493515583597462,
        0.9652131308182264,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (9.65E-01,1.13E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        9.653585940122535e-09,
        1.0186342360914322,
    )  # error of 6.11E-04 with sample range (0.00E+00,1.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.000611148272533849}
    sigmoidal_norm_parameter = (
        -0.7053096141126804,
        2.335502419908216,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.39E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.7053096194358988,
        1.0,
        2.3355024086473413,
    )  # error of 8.90E-11 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.898073441986849e-11}
    genlog_norm_parameter = (
        2.3431472292765747,
        0.3898435515065067,
        0.1501931764331937,
        1.913181693506498,
    )  # error of 6.90E-12 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.47E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 6.9021347528757674e-12}
    autogen_normdata = {
        "sample_bounds": [[0.0, 10.0], [0.8468492277243684, 0.9999999999869549]],
        "sample_bounds99": [[0.0, 0.0], [0.981706646406259, 0.981706646406259]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_isocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_isocyan)
    # normalization
    linear_norm_parameter = (
        0.49999999999890943,
        0.1665333453304355,
    )  # error of 2.36E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.67E-01,1.17E+00)
    linear_norm_parameter_normdata = {"error": 0.23551371555623501}
    min_max_norm_parameter = (
        5.002000657445256e-11,
        1.000400131389011,
    )  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.240464819131697e-09}
    sigmoidal_norm_parameter = (
        0.6485353213537604,
        22.257526471575645,
    )  # error of 3.87E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (5.38E-07,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 3.8743061663972797e-07}
    dual_sigmoidal_norm_parameter = (
        0.49999969269758465,
        8.490066047780406,
        13.665791744120064,
    )  # error of 7.34E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.08E-03,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.000734123180439748}
    genlog_norm_parameter = (
        10.358577618423046,
        -0.41663923409965714,
        1.768796247246243,
        0.0018778063775081928,
    )  # error of 2.37E-06 with sample range (0.00E+00,2.00E+00) resulting in fit range (3.98E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.3677675511578168e-06}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2.0], [3.9795068243984265e-06, 0.9999999873422708]]
    }
    preferred_normalization = "unity"
    # functions


class fr_isothiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_isothiocyan)
    # normalization
    linear_norm_parameter = (
        0.5,
        0.16661833768294187,
    )  # error of 2.36E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.67E-01,1.17E+00)
    linear_norm_parameter_normdata = {"error": 0.2356339128912765}
    min_max_norm_parameter = (
        5.0007250771301266e-11,
        1.000145015326011,
    )  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.242628536746415e-09}
    sigmoidal_norm_parameter = (
        0.5352491266658849,
        18.830908818920925,
    )  # error of 2.54E-05 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.19E-05,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 2.5382165242040136e-05}
    dual_sigmoidal_norm_parameter = (
        0.49999999625955927,
        8.494601049886345,
        15.667407499578225,
    )  # error of 2.71E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (3.96E-04,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0002706915674567058}
    genlog_norm_parameter = (
        11.574461233779338,
        -0.548170562093171,
        1.5089352111336622,
        0.00017181019470497623,
    )  # error of 1.18E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (2.05E-07,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.1847282522188223e-07}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2.0], [2.0487972708490966e-07, 0.9999999986365358]]
    }
    preferred_normalization = "unity"
    # functions


class fr_ketone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_ketone)
    # normalization
    linear_norm_parameter = (
        0.020725634690741956,
        0.9421702046814519,
    )  # error of 7.64E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.42E-01,1.19E+00)
    linear_norm_parameter_normdata = {"error": 0.007639601206612041}
    min_max_norm_parameter = (
        5.742986740306006e-09,
        1.0443931574968837,
    )  # error of 1.71E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.001714795972439738}
    sigmoidal_norm_parameter = (
        -0.5164294479177101,
        2.0540442859352286,
    )  # error of 1.94E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.43E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00019430876116018327}
    dual_sigmoidal_norm_parameter = (
        -0.5164295209542769,
        1.0,
        2.054044188695136,
    )  # error of 1.94E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.43E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0001943087611602488}
    genlog_norm_parameter = (
        5.194735931222892,
        -0.24084766718631456,
        0.03316065225308425,
        0.0012097853826869174,
    )  # error of 1.65E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (4.07E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0016481694193924618}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.0004067690096152893, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9574320088510203, 0.9574320088510203]],
    }
    preferred_normalization = "genlog"
    # functions


class fr_ketone_Topliss_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_ketone_Topliss)
    # normalization
    linear_norm_parameter = (
        0.03132218100370976,
        0.9335059844613983,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (9.34E-01,1.25E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        7.195279033997925e-09,
        1.0364539879128276,
    )  # error of 1.24E-03 with sample range (0.00E+00,1.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0012426894292735132}
    sigmoidal_norm_parameter = (
        -0.47566559195294444,
        2.2442106325557347,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.44E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.4756655979317486,
        1.0,
        2.2442106182230197,
    )  # error of 1.95E-10 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.44E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.94956259116528e-10}
    genlog_norm_parameter = (
        2.257798855867578,
        0.4264529938645671,
        0.24900286908378794,
        1.8427238699610455,
    )  # error of 9.16E-14 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.61E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 9.156022140422263e-14}
    autogen_normdata = {
        "sample_bounds": [[0.0, 10.0], [0.7614945072038156, 0.9999999999446132]],
        "sample_bounds99": [[0.0, 0.0], [0.9648281654650128, 0.9648281654650128]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_lactam_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_lactam)
    # normalization
    linear_norm_parameter = (
        0.49918507334339934,
        0.49918507334339923,
    )  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        4.33658612363641e-09,
        1.0016325140575846,
    )  # error of 1.00E-05 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 9.999100080982425e-06}
    sigmoidal_norm_parameter = (
        -1.5332891839492102,
        2.533289107423889,
    )  # error of 1.30E-07 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.80E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.3047628721896132e-07}
    dual_sigmoidal_norm_parameter = (
        -1.5332891839492102,
        1.0,
        2.533289107423889,
    )  # error of 1.30E-07 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.80E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.3047628721896132e-07}
    genlog_norm_parameter = (
        1.1086721416460528,
        0.2792406765761083,
        0.006702947768401276,
        1.8453163220846005,
    )  # error of 3.89E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.95E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.885780586188048e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9950841250543423, 0.99994129844364]]
    }
    preferred_normalization = "unity"
    # functions


class fr_lactone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_lactone)
    # normalization
    linear_norm_parameter = (
        0.4971402573768363,
        0.4971402573768361,
    )  # error of 2.22E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.97E-01,2.49E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        5.474006801001615e-12,
        1.0057523827434252,
    )  # error of 2.72E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0002723814327463555}
    sigmoidal_norm_parameter = (
        -1.2711508768212254,
        2.271150883092475,
    )  # error of 8.10E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.47E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 8.101362303225557e-08}
    dual_sigmoidal_norm_parameter = (
        -1.2711508768212254,
        1.0,
        2.271150883092475,
    )  # error of 8.10E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.47E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.101362303225557e-08}
    genlog_norm_parameter = (
        2.0120913766259294,
        -1.0448109514095034,
        1.0050790005675916,
        2.83933807870229,
    )  # error of 2.52E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.60E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.5234728862066902e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9600273398682682, 0.9999861766581559]]
    }
    preferred_normalization = "unity"
    # functions


class fr_methoxy_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_methoxy)
    # normalization
    linear_norm_parameter = (
        0.03515383615264178,
        0.8758211760938799,
    )  # error of 2.13E-02 with sample range (0.00E+00,1.90E+01) resulting in fit range (8.76E-01,1.54E+00)
    linear_norm_parameter_normdata = {"error": 0.02130042367832541}
    min_max_norm_parameter = (
        8.17446329145094e-09,
        1.126246500879634,
    )  # error of 7.63E-03 with sample range (0.00E+00,1.90E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.007626039554010268}
    sigmoidal_norm_parameter = (
        -0.3282281758044606,
        1.5582781044378737,
    )  # error of 2.40E-04 with sample range (0.00E+00,1.90E+01) resulting in fit range (6.25E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0002403311888046684}
    dual_sigmoidal_norm_parameter = (
        -0.32822817276403427,
        1.0,
        1.5582781078352885,
    )  # error of 2.40E-04 with sample range (0.00E+00,1.90E+01) resulting in fit range (6.25E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0002403311888047416}
    genlog_norm_parameter = (
        3.8691459989155064,
        -1.4662921984098698,
        0.774476931409458,
        0.0004625108726966454,
    )  # error of 6.99E-03 with sample range (0.00E+00,1.90E+01) resulting in fit range (3.19E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006988743115591665}
    autogen_normdata = {
        "sample_bounds": [[0.0, 19.0], [0.003192944550798608, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.8868071148106738, 0.9974952635501159]],
    }
    preferred_normalization = "genlog"
    # functions


class fr_morpholine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_morpholine)
    # normalization
    linear_norm_parameter = (
        0.4932781049705528,
        0.4932781049705526,
    )  # error of 1.11E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.93E-01,2.47E+00)
    linear_norm_parameter_normdata = {"error": 1.1102230246251565e-16}
    min_max_norm_parameter = (
        9.48941571641272e-09,
        1.0136269883011773,
    )  # error of 1.28E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00012752283205449128}
    sigmoidal_norm_parameter = (
        -1.0726077972576291,
        2.0726077620218297,
    )  # error of 1.33E-11 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.02E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.3335443860285068e-11}
    dual_sigmoidal_norm_parameter = (
        -1.0726077972576291,
        1.0,
        2.0726077620218297,
    )  # error of 1.33E-11 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.3335443860285068e-11}
    genlog_norm_parameter = (
        1.1028251890758516,
        0.29835779886578007,
        0.05423513100224196,
        1.8255547824582754,
    )  # error of 1.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.61E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 1.0018537111022852e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9609791801697393, 0.9994991892808237]],
        "sample_bounds99": [[0.0, 0.0], [0.9865562199596424, 0.9865562199596424]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_nitrile_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_nitrile)
    # normalization
    linear_norm_parameter = (
        0.023442890139887318,
        0.9508294253517185,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.51E-01,1.33E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        8.341666743159058e-09,
        1.0264070772459553,
    )  # error of 6.79E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0006788319993059706}
    sigmoidal_norm_parameter = (
        -0.4863072489946901,
        2.445068623176201,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.67E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.48631181197205164,
        1.0,
        2.445057157239258,
    )  # error of 1.08E-07 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.67E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.0799450065921581e-07}
    genlog_norm_parameter = (
        2.4555807917671646,
        0.4112630726784054,
        0.2129770827599907,
        1.8782570208675453,
    )  # error of 1.40E-12 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.83E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.3958578782664832e-12}
    autogen_normdata = {
        "sample_bounds": [[0.0, 16.0], [0.7826157577810503, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9742723154910267, 0.9742723154910267]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_nitro_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_nitro)
    # normalization
    linear_norm_parameter = (
        0.018333349998500004,
        0.9621034106930377,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.62E-01,1.18E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        5.1424742898646085e-09,
        1.0199535961850923,
    )  # error of 4.39E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00043932434758145423}
    sigmoidal_norm_parameter = (
        -0.40538172903464703,
        2.78525457245914,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.56E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.4053823801522207,
        1.0,
        2.7852526496619983,
    )  # error of 1.23E-08 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.2298553132559951e-08}
    genlog_norm_parameter = (
        2.7936331792469957,
        0.40662040579396574,
        0.20071352442685395,
        1.899994928085164,
    )  # error of 4.77E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.74E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.770332690141064e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.7744932425572798, 0.9999999999999991]],
        "sample_bounds99": [[0.0, 0.0], [0.9804367606878238, 0.9804367606878238]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_nitro_arom_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom)
    # normalization
    linear_norm_parameter = (
        0.01529862312391872,
        0.9682928536431723,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.10E+01) resulting in fit range (9.68E-01,1.14E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        5.509706096657517e-09,
        1.0166822542997578,
    )  # error of 3.95E-04 with sample range (0.00E+00,1.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0003952530875705006}
    sigmoidal_norm_parameter = (
        -0.5110603956802732,
        2.708965007295647,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.00E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.5110628956662855,
        1.0,
        2.7089582891344866,
    )  # error of 3.94E-08 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 3.936807506395467e-08}
    genlog_norm_parameter = (
        2.716065577065721,
        0.3943409669145902,
        0.1668948198857728,
        1.916303793087377,
    )  # error of 9.44E-10 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.13E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 9.437717111354742e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 11.0], [0.8129616367045318, 0.9999999999999731]],
        "sample_bounds99": [[0.0, 0.0], [0.9835914776743994, 0.9835914776743994]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_nitro_arom_nonortho_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom_nonortho)
    # normalization
    linear_norm_parameter = (
        1.0000000000000002,
        2.220446049250313e-16,
    )  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    linear_norm_parameter_normdata = {"error": 3.510833468576701e-16}
    min_max_norm_parameter = (
        1e-10,
        0.9999999999,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.5027041798086037,
        46.14043898879842,
    )  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 9.714558513783135e-11}
    dual_sigmoidal_norm_parameter = (
        0.49954642862905374,
        18.83817716761341,
        18.88166255189391,
    )  # error of 8.03E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.18E-05,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.030546482298942e-05}
    genlog_norm_parameter = (
        12.011496984688954,
        -0.47958199521284717,
        1.1726250998531205,
        0.0003411801343887883,
    )  # error of 4.87E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.03E-05,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.865612171061197e-05}
    autogen_normdata = {
        "sample_bounds": [[0.0, 1.0], [2.0298985067945082e-05, 0.9999342520907933]]
    }
    preferred_normalization = "unity"
    # functions


class fr_nitroso_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_nitroso)
    # normalization
    linear_norm_parameter = (
        0.11434935141358371,
        0.5282342446312391,
    )  # error of 2.97E-01 with sample range (0.00E+00,6.00E+00) resulting in fit range (5.28E-01,1.21E+00)
    linear_norm_parameter_normdata = {"error": 0.2966191601424376}
    min_max_norm_parameter = (
        9.249617077400622e-09,
        1.0008656708869366,
    )  # error of 4.28E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.281358870586223e-05}
    sigmoidal_norm_parameter = (
        0.7036397732099205,
        23.795370886031456,
    )  # error of 4.28E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (5.35E-08,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.281359428986932e-05}
    dual_sigmoidal_norm_parameter = (
        0.0007713291885705107,
        20282.21901414945,
        7.057394345605325,
    )  # error of 4.25E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.61E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.252352092836749e-05}
    genlog_norm_parameter = (
        7.296365931941997,
        0.2058906214339969,
        -0.22262794047612086,
        -0.7836891501859671,
    )  # error of 4.26E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.93E-11,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.2585319518027426e-05}
    autogen_normdata = {"sample_bounds": [[0.0, 6.0], [8.927270947305488e-11, 1.0]]}
    preferred_normalization = "unity"
    # functions


class fr_oxazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_oxazole)
    # normalization
    linear_norm_parameter = (
        0.49769270765631113,
        0.497692707656311,
    )  # error of 2.22E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.98E-01,3.48E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        2.9758329081720466e-09,
        1.004635977785747,
    )  # error of 5.42E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 5.4193831026915936e-05}
    sigmoidal_norm_parameter = (
        -1.3181535623480032,
        2.318153583438047,
    )  # error of 3.31E-07 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.55E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 3.314393710596164e-07}
    dual_sigmoidal_norm_parameter = (
        -1.3181535623480032,
        1.0,
        2.318153583438047,
    )  # error of 3.31E-07 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.55E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 3.314393710596164e-07}
    genlog_norm_parameter = (
        1.1072249352443757,
        0.28400042247764307,
        0.01888790451945942,
        1.8403655068101625,
    )  # error of 3.56E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.86E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.556999172982245e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9862191045127617, 0.9999816910582199]]
    }
    preferred_normalization = "unity"
    # functions


class fr_oxime_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_oxime)
    # normalization
    linear_norm_parameter = (
        0.49776520113189837,
        0.49776520113189815,
    )  # error of 2.22E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.98E-01,2.99E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        6.196547898069593e-09,
        1.0044896647037036,
    )  # error of 4.74E-05 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.742989621187462e-05}
    sigmoidal_norm_parameter = (
        -1.3250564360659973,
        2.3250563474726307,
    )  # error of 4.01E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.0059560724170495e-07}
    dual_sigmoidal_norm_parameter = (
        -1.3250564360659973,
        1.0,
        2.3250563474726307,
    )  # error of 4.01E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.0059560724170495e-07}
    genlog_norm_parameter = (
        1.1072957676948878,
        0.28376809509102074,
        0.01829870805476491,
        1.8406066972036101,
    )  # error of 3.63E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.87E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.626234912346149e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 5.0], [0.9866457121848611, 0.9999463677433602]]
    }
    preferred_normalization = "unity"
    # functions


class fr_para_hydroxylation_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_para_hydroxylation)
    # normalization
    linear_norm_parameter = (
        0.029445349918507113,
        0.8952769250767446,
    )  # error of 1.78E-02 with sample range (0.00E+00,1.20E+01) resulting in fit range (8.95E-01,1.25E+00)
    linear_norm_parameter_normdata = {"error": 0.017803868054254247}
    min_max_norm_parameter = (
        8.207457396571228e-09,
        1.1044006396378632,
    )  # error of 6.70E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.006701050463847139}
    sigmoidal_norm_parameter = (
        -0.5046217800819417,
        1.5020028251556437,
    )  # error of 4.95E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.81E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0004952106894175548}
    dual_sigmoidal_norm_parameter = (
        -0.5046217800819417,
        1.0,
        1.5020028251556437,
    )  # error of 4.95E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.81E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0004952106894175548}
    genlog_norm_parameter = (
        4.099759832683954,
        -1.373084715141175,
        0.8258110942269167,
        0.0004908400444425549,
    )  # error of 6.28E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (2.40E-03,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.006278759629932389}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.0023988650916005197, 1.0]],
        "sample_bounds99": [[0.0, 1.0], [0.9047028559371252, 0.9983412062182745]],
    }
    preferred_normalization = "genlog"
    # functions


class fr_phenol_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_phenol)
    # normalization
    linear_norm_parameter = (
        0.010549050583289588,
        0.9694394171190582,
    )  # error of 3.47E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.69E-01,1.24E+00)
    linear_norm_parameter_normdata = {"error": 0.0034668682322772154}
    min_max_norm_parameter = (
        6.645986623350427e-09,
        1.0229791585226422,
    )  # error of 1.21E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0012052383406743704}
    sigmoidal_norm_parameter = (
        -1.3928295501175463,
        1.5771919998858934,
    )  # error of 2.42E-04 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.00E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0002417235281706862}
    dual_sigmoidal_norm_parameter = (
        -1.3928295501175463,
        1.0,
        1.5771919998858934,
    )  # error of 2.42E-04 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.0002417235281706862}
    genlog_norm_parameter = (
        1.5687636793788549,
        -3.39456367618222,
        0.44255217703145555,
        0.019759032360037464,
    )  # error of 2.38E-04 with sample range (0.00E+00,2.60E+01) resulting in fit range (8.97E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.00023807264525171516}
    autogen_normdata = {
        "sample_bounds": [[0.0, 26.0], [0.8968270581061235, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9775538905542561, 0.9775538905542561]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_phenol_noOrthoHbond_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_phenol_noOrthoHbond)
    # normalization
    linear_norm_parameter = (
        0.01009659130462326,
        0.9708109603468388,
    )  # error of 3.45E-03 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.71E-01,1.20E+00)
    linear_norm_parameter_normdata = {"error": 0.0034515489640855493}
    min_max_norm_parameter = (
        8.618314307387322e-09,
        1.0220069386129595,
    )  # error of 1.17E-03 with sample range (0.00E+00,2.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0011749398472344148}
    sigmoidal_norm_parameter = (
        -1.3483707918108945,
        1.6255249140159453,
    )  # error of 3.04E-04 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.00E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00030390390964187615}
    dual_sigmoidal_norm_parameter = (
        -1.3483707918108945,
        1.0,
        1.6255249140159453,
    )  # error of 3.04E-04 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00030390390964187615}
    genlog_norm_parameter = (
        1.6173076657774035,
        -3.59125476532644,
        0.872659059736088,
        0.023903571655077328,
    )  # error of 3.01E-04 with sample range (0.00E+00,2.30E+01) resulting in fit range (8.96E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0003007777931998195}
    autogen_normdata = {
        "sample_bounds": [[0.0, 23.0], [0.8962944239829443, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9784863770505957, 0.9784863770505957]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_phos_acid_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_phos_acid)
    # normalization
    linear_norm_parameter = (
        0.4985551300382969,
        0.49855513003829677,
    )  # error of 4.44E-16 with sample range (0.00E+00,2.00E+01) resulting in fit range (4.99E-01,1.05E+01)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        2.8684430641636124e-09,
        1.0028981147129061,
    )  # error of 3.15E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00031495578056400016}
    sigmoidal_norm_parameter = (
        -1.4173734339435282,
        2.417373387385202,
    )  # error of 1.85E-09 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.8525553313608611e-09}
    dual_sigmoidal_norm_parameter = (
        -1.4173734339435282,
        1.0,
        2.417373387385202,
    )  # error of 1.85E-09 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.8525553313608611e-09}
    genlog_norm_parameter = (
        1.1080640181083898,
        0.2812436657611854,
        0.011860682469887762,
        1.8432300901445153,
    )  # error of 1.33E-15 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.91E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.3322676295501878e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 20.0], [0.9913206618189959, 0.9999999999979139]]
    }
    preferred_normalization = "unity"
    # functions


class fr_phos_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_phos_ester)
    # normalization
    linear_norm_parameter = (
        0.49879260866522035,
        0.49879260866522024,
    )  # error of 3.33E-16 with sample range (0.00E+00,1.60E+01) resulting in fit range (4.99E-01,8.48E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        5.639896586482622e-09,
        1.002420627946355,
    )  # error of 1.68E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0001678464439141783}
    sigmoidal_norm_parameter = (
        -1.4543278081193836,
        2.4543276857095164,
    )  # error of 8.66E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.73E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 8.661627526329596e-09}
    dual_sigmoidal_norm_parameter = (
        -1.4543278081193836,
        1.0,
        2.4543276857095164,
    )  # error of 8.66E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.73E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 8.661627526329596e-09}
    genlog_norm_parameter = (
        1.1082937472193772,
        0.28048751803136773,
        0.009918736012409151,
        1.8440171768977176,
    )  # error of 4.66E-15 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.93E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 4.6629367034256575e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 16.0], [0.9927357232830881, 0.9999999998539562]]
    }
    preferred_normalization = "unity"
    # functions


class fr_piperdine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_piperdine)
    # normalization
    linear_norm_parameter = (
        0.04919057284844364,
        0.8975592196702297,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.98E-01,1.29E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        8.858196589138318e-09,
        1.0562452798304425,
    )  # error of 1.46E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0014582722061967624}
    sigmoidal_norm_parameter = (
        -0.09657686162446193,
        2.624561257313392,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.63E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.09657686264538319,
        1.0,
        2.6245612535395852,
    )  # error of 5.41E-11 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.63E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 5.4112992974806415e-11}
    genlog_norm_parameter = (
        2.6437258687250775,
        0.49541489641902126,
        0.38073895500116234,
        1.7466884523666442,
    )  # error of 3.09E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.04E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 3.086135765649738e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 8.0], [0.6042497199120308, 0.9999999994727935]],
        "sample_bounds99": [[0.0, 0.0], [0.9467497892826053, 0.9467497892826053]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_piperzine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_piperzine)
    # normalization
    linear_norm_parameter = (
        0.4872161505464513,
        0.48721615054645107,
    )  # error of 5.55E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.87E-01,3.41E+00)
    linear_norm_parameter_normdata = {"error": 5.551115123125783e-16}
    min_max_norm_parameter = (
        7.2423565017085415e-09,
        1.0262385582797817,
    )  # error of 2.26E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00022630903814168187}
    sigmoidal_norm_parameter = (
        -0.9080160452625068,
        1.9080160046689731,
    )  # error of 2.52E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.50E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 2.5226206679107577e-09}
    dual_sigmoidal_norm_parameter = (
        -0.9080160452625068,
        1.0,
        1.9080160046689731,
    )  # error of 2.52E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.5226206679107577e-09}
    genlog_norm_parameter = (
        1.0965711718419204,
        0.31865196956800257,
        0.10102198516471307,
        1.8048466433552084,
    )  # error of 1.16E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.28E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.158754425745201e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9284978422903257, 0.999889781939511]],
        "sample_bounds99": [[0.0, 0.0], [0.9744323022516562, 0.9744323022516562]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_priamide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_priamide)
    # normalization
    linear_norm_parameter = (
        0.49340809327160573,
        0.4934080932716056,
    )  # error of 3.33E-16 with sample range (0.00E+00,9.00E+00) resulting in fit range (4.93E-01,4.93E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        3.082917443384843e-09,
        1.0133599484847597,
    )  # error of 3.39E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0003390491222427521}
    sigmoidal_norm_parameter = (
        -1.0773767211105616,
        2.0773766830617735,
    )  # error of 1.77E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.04E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.7679302466433455e-11}
    dual_sigmoidal_norm_parameter = (
        -1.0773767211105616,
        1.0,
        2.0773766830617735,
    )  # error of 1.77E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.04E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.7679302466433455e-11}
    genlog_norm_parameter = (
        1.1029569530446681,
        0.2979295365034292,
        0.05320949599596312,
        1.8259942787635228,
    )  # error of 1.04E-08 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.62E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.0422452678682248e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 9.0], [0.9617023174235103, 0.9999980224561622]],
        "sample_bounds99": [[0.0, 0.0], [0.9868161969656637, 0.9868161969656637]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_prisulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_prisulfonamd)
    # normalization
    linear_norm_parameter = (
        1.0,
        1.0,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    sigmoidal_norm_parameter = (
        0.0,
        0.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.5}
    dual_sigmoidal_norm_parameter = (
        0.0,
        0.0,
        1.0,
    )  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.5}
    genlog_norm_parameter = (
        0.3561956366606025,
        0.3561956366606025,
        0.0,
        2.120992123914219,
    )  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0}
    autogen_normdata = {"sample_bounds": [[0.0, 0.0], [1.0, 1.0]]}
    preferred_normalization = "unity"
    # functions


class fr_pyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_pyridine)
    # normalization
    linear_norm_parameter = (
        0.03403943645071905,
        0.906233438990491,
    )  # error of 1.33E-02 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.06E-01,1.31E+00)
    linear_norm_parameter_normdata = {"error": 0.013310087247984373}
    min_max_norm_parameter = (
        8.84293141493641e-09,
        1.0742739897507971,
    )  # error of 2.32E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0023173398327729308}
    sigmoidal_norm_parameter = (
        -0.1001953260731105,
        2.363269723603649,
    )  # error of 2.37E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (5.59E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.00023659675130990028}
    dual_sigmoidal_norm_parameter = (
        -0.10019533355161231,
        1.0,
        2.36326970776348,
    )  # error of 2.37E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (5.59E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.00023659675131022326}
    genlog_norm_parameter = (
        4.603386911286616,
        -1.1197835832086922,
        0.9103975728959118,
        0.0007332403073216993,
    )  # error of 2.10E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.87E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0020975947322573758}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.0007869302847571399, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9307288029874364, 0.9307288029874364]],
    }
    preferred_normalization = "genlog"
    # functions


class fr_quatN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_quatN)
    # normalization
    linear_norm_parameter = (
        0.4970377666010063,
        0.49703776660100607,
    )  # error of 5.55E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.97E-01,2.49E+00)
    linear_norm_parameter_normdata = {"error": 5.551115123125783e-16}
    min_max_norm_parameter = (
        8.281984578257868e-09,
        1.0059597752394516,
    )  # error of 2.67E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.000267230509551931}
    sigmoidal_norm_parameter = (
        -1.2633408656906582,
        2.2633408527519925,
    )  # error of 6.28E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.46E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 6.27622285254148e-08}
    dual_sigmoidal_norm_parameter = (
        -1.2633408656906582,
        1.0,
        2.2633408527519925,
    )  # error of 6.28E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.46E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 6.27622285254148e-08}
    genlog_norm_parameter = (
        1.106582889794562,
        0.2861049257661485,
        0.024198327190510873,
        1.8381836695475,
    )  # error of 2.98E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.82E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.9799722134349338e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 4.0], [0.9823832449723512, 0.9997840122809857]]
    }
    preferred_normalization = "unity"
    # functions


class fr_sulfide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_sulfide)
    # normalization
    linear_norm_parameter = (
        0.041736243738063505,
        0.9134677878990892,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.13E-01,1.58E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        9.293070357320419e-09,
        1.0468967534295035,
    )  # error of 8.64E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0008639156592464572}
    sigmoidal_norm_parameter = (
        -0.12222280200277533,
        2.7265591239312834,
    )  # error of 1.11E-16 with sample range (0.00E+00,1.60E+01) resulting in fit range (5.83E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.1102230246251565e-16}
    dual_sigmoidal_norm_parameter = (
        -0.12222320614972192,
        1.0,
        2.726557767129511,
    )  # error of 1.33E-08 with sample range (0.00E+00,1.60E+01) resulting in fit range (5.83E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.3293927270921125e-08}
    genlog_norm_parameter = (
        2.7434265313363664,
        0.48093627268280686,
        0.35307402210030514,
        1.7800666193979122,
    )  # error of 5.90E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (6.23E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 5.899675922231449e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 16.0], [0.6231313621029366, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9552040253467974, 0.9552040253467974]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_sulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_sulfonamd)
    # normalization
    linear_norm_parameter = (
        0.03596176344129032,
        0.9269115779579837,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (9.27E-01,1.21E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        9.80315962195014e-09,
        1.0385581951856868,
    )  # error of 4.80E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00047967028581241423}
    sigmoidal_norm_parameter = (
        0.06939953825020206,
        3.4983720078550626,
    )  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (4.40E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        0.06939885794095012,
        1.0,
        3.4983684786379823,
    )  # error of 2.31E-08 with sample range (0.00E+00,8.00E+00) resulting in fit range (4.83E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.3148437541250093e-08}
    genlog_norm_parameter = (
        3.5129553874443054,
        0.5148439794982723,
        0.3848335971694203,
        1.7883039782489942,
    )  # error of 7.73E-10 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.09E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 7.73204466130559e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 8.0], [0.5087752835965452, 0.9999999999991813]],
        "sample_bounds99": [[0.0, 0.0], [0.9628733403656261, 0.9628733403656261]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_sulfone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_sulfone)
    # normalization
    linear_norm_parameter = (
        0.49420552150306496,
        0.49420552150306485,
    )  # error of 4.44E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.94E-01,1.98E+00)
    linear_norm_parameter_normdata = {"error": 4.440892098500626e-16}
    min_max_norm_parameter = (
        7.860937357634984e-09,
        1.0117248355173414,
    )  # error of 1.67E-04 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00016748492635659806}
    sigmoidal_norm_parameter = (
        -1.1085649360924517,
        2.108564882621699,
    )  # error of 9.49E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.12E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 9.49246237169632e-11}
    dual_sigmoidal_norm_parameter = (
        -1.1085649360924517,
        1.0,
        2.108564882621699,
    )  # error of 9.49E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.12E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 9.49246237169632e-11}
    genlog_norm_parameter = (
        1.1037625750768887,
        0.2953092372234201,
        0.04689730725246237,
        1.828686085845639,
    )  # error of 1.33E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.66E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 1.3272141496933898e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 3.0], [0.9661643641515827, 0.998706701954208]],
        "sample_bounds99": [[0.0, 0.0], [0.9884110562782709, 0.9884110562782709]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_term_acetylene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_term_acetylene)
    # normalization
    linear_norm_parameter = (
        0.4985401313881752,
        0.4985401313881751,
    )  # error of 2.22E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.99E-01,3.49E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        1.033924484011466e-08,
        1.0029282870220821,
    )  # error of 7.59E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 7.587685400206961e-05}
    sigmoidal_norm_parameter = (
        -1.4152302622952124,
        2.415230200746865,
    )  # error of 1.69E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.68E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 1.6868373364786748e-09}
    dual_sigmoidal_norm_parameter = (
        -1.4152302622952124,
        1.0,
        2.415230200746865,
    )  # error of 1.69E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.68E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.6868373364786748e-09}
    genlog_norm_parameter = (
        1.1080494768192577,
        0.28129148579972757,
        0.011983232040056609,
        1.8431803563568918,
    )  # error of 1.78E-15 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.91E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.7763568394002505e-15}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9912314393448513, 0.9999884908456532]]
    }
    preferred_normalization = "unity"
    # functions


class fr_tetrazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_tetrazole)
    # normalization
    linear_norm_parameter = (
        0.4981451669349761,
        0.4981451669349759,
    )  # error of 2.22E-16 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.98E-01,1.49E+00)
    linear_norm_parameter_normdata = {"error": 2.220446049250313e-16}
    min_max_norm_parameter = (
        5.01861743222311e-11,
        1.0037234863442497,
    )  # error of 4.21E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.212373026431394e-09}
    sigmoidal_norm_parameter = (
        0.7145014458224443,
        19.590579896967135,
    )  # error of 4.82E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (8.34E-07,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.822927872747481e-07}
    dual_sigmoidal_norm_parameter = (
        -1.3649097870387985,
        1.0,
        2.3649097938746766,
    )  # error of 1.10E-06 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.1034549377564318e-06}
    genlog_norm_parameter = (
        1.107666043299223,
        0.2825520504167154,
        0.015205956802819295,
        1.8418695786439092,
    )  # error of 4.01E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.89E-01,9.99E-01)
    genlog_norm_parameter_normdata = {"error": 4.010937981746565e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 2.0], [0.9888883984831989, 0.9987702649490324]]
    }
    preferred_normalization = "unity"
    # functions


class fr_thiazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_thiazole)
    # normalization
    linear_norm_parameter = (
        0.4896259336659703,
        0.4896259336659702,
    )  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.90E-01,3.43E+00)
    linear_norm_parameter_normdata = {"error": 3.3306690738754696e-16}
    min_max_norm_parameter = (
        4.718948044240454e-09,
        1.021187738581169,
    )  # error of 1.92E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.00019217401527640087}
    sigmoidal_norm_parameter = (
        -0.9632451839557359,
        1.963245141758428,
    )  # error of 2.15E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.69E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 2.1521195714413466e-08}
    dual_sigmoidal_norm_parameter = (
        -0.9632451839557359,
        1.0,
        1.963245141758428,
    )  # error of 2.15E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.1521195714413466e-08}
    genlog_norm_parameter = (
        1.0990796811027814,
        0.310513314155494,
        0.08266879735779394,
        1.8131267742888182,
    )  # error of 2.96E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.41E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 2.956390465769232e-09}
    autogen_normdata = {
        "sample_bounds": [[0.0, 6.0], [0.9411276215862902, 0.9999122736732728]],
        "sample_bounds99": [[0.0, 0.0], [0.9792518702883306, 0.9792518702883306]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_thiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_thiocyan)
    # normalization
    linear_norm_parameter = (
        0.11905202341120957,
        0.45235119315332584,
    )  # error of 3.36E-01 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.52E-01,1.17E+00)
    linear_norm_parameter_normdata = {"error": 0.33627423200378687}
    min_max_norm_parameter = (
        9.608678732145031e-12,
        1.000064998295486,
    )  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.999550040676859e-06}
    sigmoidal_norm_parameter = (
        0.6659376309621101,
        28.86030957889578,
    )  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.50E-09,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.999550549370849e-06}
    dual_sigmoidal_norm_parameter = (
        0.5520702086106327,
        34.65927490403444,
        21.52378723022447,
    )  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.90E-09,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.999550641242235e-06}
    genlog_norm_parameter = (
        12.074937536631607,
        -0.5226172823898794,
        0.6225857666003577,
        9.308019970300272e-05,
    )  # error of 6.05E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (5.31E-06,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 6.052008829492293e-06}
    autogen_normdata = {"sample_bounds": [[0.0, 6.0], [5.3070601644324225e-06, 1.0]]}
    preferred_normalization = "unity"
    # functions


class fr_thiophene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_thiophene)
    # normalization
    linear_norm_parameter = (
        0.027482526572608745,
        0.9433700966912973,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.43E-01,1.27E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        9.055969684787418e-09,
        1.0300224521968595,
    )  # error of 5.45E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0005450174581319078}
    sigmoidal_norm_parameter = (
        -0.2128545965336391,
        2.8905441523149107,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.49E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 0.0}
    dual_sigmoidal_norm_parameter = (
        -0.2128546047774725,
        1.0,
        2.8905441234721354,
    )  # error of 2.28E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.49E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 2.280620878532085e-10}
    genlog_norm_parameter = (
        2.902492127526775,
        0.4442046138253706,
        0.28226346472172587,
        1.8497442683925474,
    )  # error of 1.02E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.83E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.0239414842301784e-10}
    autogen_normdata = {
        "sample_bounds": [[0.0, 12.0], [0.682933448002194, 0.9999999999999996]],
        "sample_bounds99": [[0.0, 0.0], [0.9708526231714555, 0.9708526231714555]],
    }
    preferred_normalization = "min_max"
    # functions


class fr_unbrch_alkane_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_unbrch_alkane)
    # normalization
    linear_norm_parameter = (
        0.03889834636895251,
        0.7073099439562576,
    )  # error of 2.80E-01 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.07E-01,1.17E+00)
    linear_norm_parameter_normdata = {"error": 0.28031976013823723}
    min_max_norm_parameter = (
        7.511636441371496e-09,
        1.000225030377411,
    )  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 4.284601096626167e-05}
    sigmoidal_norm_parameter = (
        0.6677048142926155,
        25.276494721014547,
    )  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (4.68E-08,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 4.284601382677171e-05}
    dual_sigmoidal_norm_parameter = (
        0.021693028028538862,
        702.8923745160023,
        8.585406408136345,
    )  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (2.39E-07,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 4.283463844153024e-05}
    genlog_norm_parameter = (
        8.941625299297684,
        -1.2379652302544373,
        1.5318520290060416,
        2.973209020648289e-06,
    )  # error of 2.99E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (3.26E-04,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 0.0002987279257447884}
    autogen_normdata = {"sample_bounds": [[0.0, 12.0], [0.0003262586150529504, 1.0]]}
    preferred_normalization = "unity"
    # functions


class fr_urea_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    featurize = staticmethod(fr_urea)
    # normalization
    linear_norm_parameter = (
        0.027632513073823306,
        0.9435400813926746,
    )  # error of 0.00E+00 with sample range (0.00E+00,1.70E+01) resulting in fit range (9.44E-01,1.41E+00)
    linear_norm_parameter_normdata = {"error": 0.0}
    min_max_norm_parameter = (
        5.263068649567547e-09,
        1.0296830919097522,
    )  # error of 4.06E-04 with sample range (0.00E+00,1.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0004064968634073849}
    sigmoidal_norm_parameter = (
        -0.0952416256429324,
        3.2113257966470843,
    )  # error of 7.85E-17 with sample range (0.00E+00,1.70E+01) resulting in fit range (5.76E-01,1.00E+00)
    sigmoidal_norm_parameter_normdata = {"error": 7.850462293418876e-17}
    dual_sigmoidal_norm_parameter = (
        -0.09524163020540921,
        1.0,
        3.2113257769175463,
    )  # error of 1.40E-10 with sample range (0.00E+00,1.70E+01) resulting in fit range (5.76E-01,1.00E+00)
    dual_sigmoidal_norm_parameter_normdata = {"error": 1.3955317260171513e-10}
    genlog_norm_parameter = (
        3.223218467507025,
        0.46375875616833745,
        0.31181586995067834,
        1.8422869457271946,
    )  # error of 1.42E-08 with sample range (0.00E+00,1.70E+01) resulting in fit range (6.23E-01,1.00E+00)
    genlog_norm_parameter_normdata = {"error": 1.4235523363238444e-08}
    autogen_normdata = {
        "sample_bounds": [[0.0, 17.0], [0.6231372900227422, 1.0]],
        "sample_bounds99": [[0.0, 0.0], [0.9711725771016495, 0.9711725771016495]],
    }
    preferred_normalization = "min_max"
    # functions


class qed_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.float32
    featurize = staticmethod(qed)
    # normalization
    linear_norm_parameter = (
        1.2001504087852979,
        -0.19399001508562716,
    )  # error of 5.24E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (-1.90E-01,9.44E-01)
    linear_norm_parameter_normdata = {"error": 0.05240810587458282}
    min_max_norm_parameter = (
        0.22901735830941922,
        0.9481621980667114,
    )  # error of 3.90E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (0.00E+00,1.00E+00)
    min_max_norm_parameter_normdata = {"error": 0.0390031761307919}
    sigmoidal_norm_parameter = (
        0.6000031067059596,
        6.458412537955855,
    )  # error of 3.69E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (2.08E-02,9.05E-01)
    sigmoidal_norm_parameter_normdata = {"error": 0.03691416955865315}
    dual_sigmoidal_norm_parameter = (
        0.6370146711380908,
        4.945613751120506,
        8.80372765079648,
    )  # error of 2.33E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (4.18E-02,9.39E-01)
    dual_sigmoidal_norm_parameter_normdata = {"error": 0.023326241318225343}
    genlog_norm_parameter = (
        14.258762047454422,
        0.8482492065265593,
        0.734824211454585,
        4.18750548663114,
    )  # error of 2.20E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (6.06E-02,9.62E-01)
    genlog_norm_parameter_normdata = {"error": 0.02195012269809026}
    autogen_normdata = {
        "sample_bounds": [
            [0.0034539541229605675, 0.9481621980667114],
            [0.060627317776454574, 0.961869535236163],
        ],
        "sample_bounds99": [
            [0.0034539541229605675, 0.9136525988578796],
            [0.06745345327110297, 0.9484667915518582],
        ],
    }
    preferred_normalization = "genlog"
    # functions


molecule_Asphericity = Asphericity_Featurizer()
molecule_BCUT2D_CHGHI = BCUT2D_CHGHI_Featurizer()
molecule_BCUT2D_CHGLO = BCUT2D_CHGLO_Featurizer()
molecule_BCUT2D_LOGPHI = BCUT2D_LOGPHI_Featurizer()
molecule_BCUT2D_LOGPLOW = BCUT2D_LOGPLOW_Featurizer()
molecule_BCUT2D_MRHI = BCUT2D_MRHI_Featurizer()
molecule_BCUT2D_MRLOW = BCUT2D_MRLOW_Featurizer()
molecule_BCUT2D_MWHI = BCUT2D_MWHI_Featurizer()
molecule_BCUT2D_MWLOW = BCUT2D_MWLOW_Featurizer()
molecule_BalabanJ = BalabanJ_Featurizer()
molecule_BertzCT = BertzCT_Featurizer()
molecule_Chi0 = Chi0_Featurizer()
molecule_Chi0n = Chi0n_Featurizer()
molecule_Chi0v = Chi0v_Featurizer()
molecule_Chi1 = Chi1_Featurizer()
molecule_Chi1n = Chi1n_Featurizer()
molecule_Chi1v = Chi1v_Featurizer()
molecule_Chi2n = Chi2n_Featurizer()
molecule_Chi2v = Chi2v_Featurizer()
molecule_Chi3n = Chi3n_Featurizer()
molecule_Chi3v = Chi3v_Featurizer()
molecule_Chi4n = Chi4n_Featurizer()
molecule_Chi4v = Chi4v_Featurizer()
molecule_EState_VSA1 = EState_VSA1_Featurizer()
molecule_EState_VSA10 = EState_VSA10_Featurizer()
molecule_EState_VSA11 = EState_VSA11_Featurizer()
molecule_EState_VSA2 = EState_VSA2_Featurizer()
molecule_EState_VSA3 = EState_VSA3_Featurizer()
molecule_EState_VSA4 = EState_VSA4_Featurizer()
molecule_EState_VSA5 = EState_VSA5_Featurizer()
molecule_EState_VSA6 = EState_VSA6_Featurizer()
molecule_EState_VSA7 = EState_VSA7_Featurizer()
molecule_EState_VSA8 = EState_VSA8_Featurizer()
molecule_EState_VSA9 = EState_VSA9_Featurizer()
molecule_Eccentricity = Eccentricity_Featurizer()
molecule_ExactMolWt = ExactMolWt_Featurizer()
molecule_FormalCharge = FormalCharge_Featurizer()
molecule_FpDensityMorgan1 = FpDensityMorgan1_Featurizer()
molecule_FpDensityMorgan2 = FpDensityMorgan2_Featurizer()
molecule_FpDensityMorgan3 = FpDensityMorgan3_Featurizer()
molecule_FractionCSP3 = FractionCSP3_Featurizer()
molecule_HallKierAlpha = HallKierAlpha_Featurizer()
molecule_HeavyAtomCount = HeavyAtomCount_Featurizer()
molecule_HeavyAtomMolWt = HeavyAtomMolWt_Featurizer()
molecule_InertialShapeFactor = InertialShapeFactor_Featurizer()
molecule_Ipc = Ipc_Featurizer()
molecule_Kappa1 = Kappa1_Featurizer()
molecule_Kappa2 = Kappa2_Featurizer()
molecule_Kappa3 = Kappa3_Featurizer()
molecule_LabuteASA = LabuteASA_Featurizer()
molecule_MaxAbsEStateIndex = MaxAbsEStateIndex_Featurizer()
molecule_MaxAbsPartialCharge = MaxAbsPartialCharge_Featurizer()
molecule_MaxEStateIndex = MaxEStateIndex_Featurizer()
molecule_MaxPartialCharge = MaxPartialCharge_Featurizer()
molecule_MinAbsEStateIndex = MinAbsEStateIndex_Featurizer()
molecule_MinAbsPartialCharge = MinAbsPartialCharge_Featurizer()
molecule_MinEStateIndex = MinEStateIndex_Featurizer()
molecule_MinPartialCharge = MinPartialCharge_Featurizer()
molecule_MolLogP = MolLogP_Featurizer()
molecule_MolMR = MolMR_Featurizer()
molecule_MolWt = MolWt_Featurizer()
molecule_NHOHCount = NHOHCount_Featurizer()
molecule_NOCount = NOCount_Featurizer()
molecule_NPR1 = NPR1_Featurizer()
molecule_NPR2 = NPR2_Featurizer()
molecule_NumAliphaticCarbocycles = NumAliphaticCarbocycles_Featurizer()
molecule_NumAliphaticHeterocycles = NumAliphaticHeterocycles_Featurizer()
molecule_NumAliphaticRings = NumAliphaticRings_Featurizer()
molecule_NumAmideBonds = NumAmideBonds_Featurizer()
molecule_NumAromaticCarbocycles = NumAromaticCarbocycles_Featurizer()
molecule_NumAromaticHeterocycles = NumAromaticHeterocycles_Featurizer()
molecule_NumAromaticRings = NumAromaticRings_Featurizer()
molecule_NumAtoms = NumAtoms_Featurizer()
molecule_NumBonds = NumBonds_Featurizer()
molecule_NumBridgeheadAtoms = NumBridgeheadAtoms_Featurizer()
molecule_NumHAcceptors = NumHAcceptors_Featurizer()
molecule_NumHBA = NumHBA_Featurizer()
molecule_NumHBD = NumHBD_Featurizer()
molecule_NumHDonors = NumHDonors_Featurizer()
molecule_NumHeavyAtoms = NumHeavyAtoms_Featurizer()
molecule_NumHeteroatoms = NumHeteroatoms_Featurizer()
molecule_NumHeterocycles = NumHeterocycles_Featurizer()
molecule_NumLipinskiHBA = NumLipinskiHBA_Featurizer()
molecule_NumLipinskiHBD = NumLipinskiHBD_Featurizer()
molecule_NumRadicalElectrons = NumRadicalElectrons_Featurizer()
molecule_NumRings = NumRings_Featurizer()
molecule_NumRotatableBonds = NumRotatableBonds_Featurizer()
molecule_NumSaturatedCarbocycles = NumSaturatedCarbocycles_Featurizer()
molecule_NumSaturatedHeterocycles = NumSaturatedHeterocycles_Featurizer()
molecule_NumSaturatedRings = NumSaturatedRings_Featurizer()
molecule_NumSpiroAtoms = NumSpiroAtoms_Featurizer()
molecule_NumValenceElectrons = NumValenceElectrons_Featurizer()
molecule_NumberAtomsAc = NumberAtomsAc_Featurizer()
molecule_NumberAtomsAg = NumberAtomsAg_Featurizer()
molecule_NumberAtomsAl = NumberAtomsAl_Featurizer()
molecule_NumberAtomsAm = NumberAtomsAm_Featurizer()
molecule_NumberAtomsAr = NumberAtomsAr_Featurizer()
molecule_NumberAtomsAs = NumberAtomsAs_Featurizer()
molecule_NumberAtomsAt = NumberAtomsAt_Featurizer()
molecule_NumberAtomsAu = NumberAtomsAu_Featurizer()
molecule_NumberAtomsB = NumberAtomsB_Featurizer()
molecule_NumberAtomsBa = NumberAtomsBa_Featurizer()
molecule_NumberAtomsBe = NumberAtomsBe_Featurizer()
molecule_NumberAtomsBh = NumberAtomsBh_Featurizer()
molecule_NumberAtomsBi = NumberAtomsBi_Featurizer()
molecule_NumberAtomsBk = NumberAtomsBk_Featurizer()
molecule_NumberAtomsBr = NumberAtomsBr_Featurizer()
molecule_NumberAtomsC = NumberAtomsC_Featurizer()
molecule_NumberAtomsCa = NumberAtomsCa_Featurizer()
molecule_NumberAtomsCd = NumberAtomsCd_Featurizer()
molecule_NumberAtomsCe = NumberAtomsCe_Featurizer()
molecule_NumberAtomsCf = NumberAtomsCf_Featurizer()
molecule_NumberAtomsCl = NumberAtomsCl_Featurizer()
molecule_NumberAtomsCm = NumberAtomsCm_Featurizer()
molecule_NumberAtomsCn = NumberAtomsCn_Featurizer()
molecule_NumberAtomsCo = NumberAtomsCo_Featurizer()
molecule_NumberAtomsCr = NumberAtomsCr_Featurizer()
molecule_NumberAtomsCs = NumberAtomsCs_Featurizer()
molecule_NumberAtomsCu = NumberAtomsCu_Featurizer()
molecule_NumberAtomsDb = NumberAtomsDb_Featurizer()
molecule_NumberAtomsDs = NumberAtomsDs_Featurizer()
molecule_NumberAtomsDy = NumberAtomsDy_Featurizer()
molecule_NumberAtomsEr = NumberAtomsEr_Featurizer()
molecule_NumberAtomsEs = NumberAtomsEs_Featurizer()
molecule_NumberAtomsEu = NumberAtomsEu_Featurizer()
molecule_NumberAtomsF = NumberAtomsF_Featurizer()
molecule_NumberAtomsFe = NumberAtomsFe_Featurizer()
molecule_NumberAtomsFl = NumberAtomsFl_Featurizer()
molecule_NumberAtomsFm = NumberAtomsFm_Featurizer()
molecule_NumberAtomsFr = NumberAtomsFr_Featurizer()
molecule_NumberAtomsGa = NumberAtomsGa_Featurizer()
molecule_NumberAtomsGd = NumberAtomsGd_Featurizer()
molecule_NumberAtomsGe = NumberAtomsGe_Featurizer()
molecule_NumberAtomsH = NumberAtomsH_Featurizer()
molecule_NumberAtomsHe = NumberAtomsHe_Featurizer()
molecule_NumberAtomsHf = NumberAtomsHf_Featurizer()
molecule_NumberAtomsHg = NumberAtomsHg_Featurizer()
molecule_NumberAtomsHo = NumberAtomsHo_Featurizer()
molecule_NumberAtomsHs = NumberAtomsHs_Featurizer()
molecule_NumberAtomsI = NumberAtomsI_Featurizer()
molecule_NumberAtomsIn = NumberAtomsIn_Featurizer()
molecule_NumberAtomsIr = NumberAtomsIr_Featurizer()
molecule_NumberAtomsK = NumberAtomsK_Featurizer()
molecule_NumberAtomsKr = NumberAtomsKr_Featurizer()
molecule_NumberAtomsLa = NumberAtomsLa_Featurizer()
molecule_NumberAtomsLi = NumberAtomsLi_Featurizer()
molecule_NumberAtomsLr = NumberAtomsLr_Featurizer()
molecule_NumberAtomsLu = NumberAtomsLu_Featurizer()
molecule_NumberAtomsLv = NumberAtomsLv_Featurizer()
molecule_NumberAtomsMc = NumberAtomsMc_Featurizer()
molecule_NumberAtomsMd = NumberAtomsMd_Featurizer()
molecule_NumberAtomsMg = NumberAtomsMg_Featurizer()
molecule_NumberAtomsMn = NumberAtomsMn_Featurizer()
molecule_NumberAtomsMo = NumberAtomsMo_Featurizer()
molecule_NumberAtomsMt = NumberAtomsMt_Featurizer()
molecule_NumberAtomsN = NumberAtomsN_Featurizer()
molecule_NumberAtomsNa = NumberAtomsNa_Featurizer()
molecule_NumberAtomsNb = NumberAtomsNb_Featurizer()
molecule_NumberAtomsNd = NumberAtomsNd_Featurizer()
molecule_NumberAtomsNe = NumberAtomsNe_Featurizer()
molecule_NumberAtomsNh = NumberAtomsNh_Featurizer()
molecule_NumberAtomsNi = NumberAtomsNi_Featurizer()
molecule_NumberAtomsNo = NumberAtomsNo_Featurizer()
molecule_NumberAtomsNp = NumberAtomsNp_Featurizer()
molecule_NumberAtomsO = NumberAtomsO_Featurizer()
molecule_NumberAtomsOg = NumberAtomsOg_Featurizer()
molecule_NumberAtomsOs = NumberAtomsOs_Featurizer()
molecule_NumberAtomsP = NumberAtomsP_Featurizer()
molecule_NumberAtomsPa = NumberAtomsPa_Featurizer()
molecule_NumberAtomsPb = NumberAtomsPb_Featurizer()
molecule_NumberAtomsPd = NumberAtomsPd_Featurizer()
molecule_NumberAtomsPm = NumberAtomsPm_Featurizer()
molecule_NumberAtomsPo = NumberAtomsPo_Featurizer()
molecule_NumberAtomsPr = NumberAtomsPr_Featurizer()
molecule_NumberAtomsPt = NumberAtomsPt_Featurizer()
molecule_NumberAtomsPu = NumberAtomsPu_Featurizer()
molecule_NumberAtomsRa = NumberAtomsRa_Featurizer()
molecule_NumberAtomsRb = NumberAtomsRb_Featurizer()
molecule_NumberAtomsRe = NumberAtomsRe_Featurizer()
molecule_NumberAtomsRf = NumberAtomsRf_Featurizer()
molecule_NumberAtomsRg = NumberAtomsRg_Featurizer()
molecule_NumberAtomsRh = NumberAtomsRh_Featurizer()
molecule_NumberAtomsRn = NumberAtomsRn_Featurizer()
molecule_NumberAtomsRu = NumberAtomsRu_Featurizer()
molecule_NumberAtomsS = NumberAtomsS_Featurizer()
molecule_NumberAtomsSb = NumberAtomsSb_Featurizer()
molecule_NumberAtomsSc = NumberAtomsSc_Featurizer()
molecule_NumberAtomsSe = NumberAtomsSe_Featurizer()
molecule_NumberAtomsSg = NumberAtomsSg_Featurizer()
molecule_NumberAtomsSi = NumberAtomsSi_Featurizer()
molecule_NumberAtomsSm = NumberAtomsSm_Featurizer()
molecule_NumberAtomsSn = NumberAtomsSn_Featurizer()
molecule_NumberAtomsSr = NumberAtomsSr_Featurizer()
molecule_NumberAtomsTa = NumberAtomsTa_Featurizer()
molecule_NumberAtomsTb = NumberAtomsTb_Featurizer()
molecule_NumberAtomsTc = NumberAtomsTc_Featurizer()
molecule_NumberAtomsTe = NumberAtomsTe_Featurizer()
molecule_NumberAtomsTh = NumberAtomsTh_Featurizer()
molecule_NumberAtomsTi = NumberAtomsTi_Featurizer()
molecule_NumberAtomsTl = NumberAtomsTl_Featurizer()
molecule_NumberAtomsTm = NumberAtomsTm_Featurizer()
molecule_NumberAtomsTs = NumberAtomsTs_Featurizer()
molecule_NumberAtomsU = NumberAtomsU_Featurizer()
molecule_NumberAtomsV = NumberAtomsV_Featurizer()
molecule_NumberAtomsW = NumberAtomsW_Featurizer()
molecule_NumberAtomsXe = NumberAtomsXe_Featurizer()
molecule_NumberAtomsY = NumberAtomsY_Featurizer()
molecule_NumberAtomsYb = NumberAtomsYb_Featurizer()
molecule_NumberAtomsZn = NumberAtomsZn_Featurizer()
molecule_NumberAtomsZr = NumberAtomsZr_Featurizer()
molecule_PBF = PBF_Featurizer()
molecule_PEOE_VSA1 = PEOE_VSA1_Featurizer()
molecule_PEOE_VSA10 = PEOE_VSA10_Featurizer()
molecule_PEOE_VSA11 = PEOE_VSA11_Featurizer()
molecule_PEOE_VSA12 = PEOE_VSA12_Featurizer()
molecule_PEOE_VSA13 = PEOE_VSA13_Featurizer()
molecule_PEOE_VSA14 = PEOE_VSA14_Featurizer()
molecule_PEOE_VSA2 = PEOE_VSA2_Featurizer()
molecule_PEOE_VSA3 = PEOE_VSA3_Featurizer()
molecule_PEOE_VSA4 = PEOE_VSA4_Featurizer()
molecule_PEOE_VSA5 = PEOE_VSA5_Featurizer()
molecule_PEOE_VSA6 = PEOE_VSA6_Featurizer()
molecule_PEOE_VSA7 = PEOE_VSA7_Featurizer()
molecule_PEOE_VSA8 = PEOE_VSA8_Featurizer()
molecule_PEOE_VSA9 = PEOE_VSA9_Featurizer()
molecule_PMI1 = PMI1_Featurizer()
molecule_PMI2 = PMI2_Featurizer()
molecule_PMI3 = PMI3_Featurizer()
molecule_Phi = Phi_Featurizer()
molecule_RadiusOfGyration = RadiusOfGyration_Featurizer()
molecule_RelativeContentAc = RelativeContentAc_Featurizer()
molecule_RelativeContentAg = RelativeContentAg_Featurizer()
molecule_RelativeContentAl = RelativeContentAl_Featurizer()
molecule_RelativeContentAm = RelativeContentAm_Featurizer()
molecule_RelativeContentAr = RelativeContentAr_Featurizer()
molecule_RelativeContentAs = RelativeContentAs_Featurizer()
molecule_RelativeContentAt = RelativeContentAt_Featurizer()
molecule_RelativeContentAu = RelativeContentAu_Featurizer()
molecule_RelativeContentB = RelativeContentB_Featurizer()
molecule_RelativeContentBa = RelativeContentBa_Featurizer()
molecule_RelativeContentBe = RelativeContentBe_Featurizer()
molecule_RelativeContentBh = RelativeContentBh_Featurizer()
molecule_RelativeContentBi = RelativeContentBi_Featurizer()
molecule_RelativeContentBk = RelativeContentBk_Featurizer()
molecule_RelativeContentBr = RelativeContentBr_Featurizer()
molecule_RelativeContentC = RelativeContentC_Featurizer()
molecule_RelativeContentCa = RelativeContentCa_Featurizer()
molecule_RelativeContentCd = RelativeContentCd_Featurizer()
molecule_RelativeContentCe = RelativeContentCe_Featurizer()
molecule_RelativeContentCf = RelativeContentCf_Featurizer()
molecule_RelativeContentCl = RelativeContentCl_Featurizer()
molecule_RelativeContentCm = RelativeContentCm_Featurizer()
molecule_RelativeContentCn = RelativeContentCn_Featurizer()
molecule_RelativeContentCo = RelativeContentCo_Featurizer()
molecule_RelativeContentCr = RelativeContentCr_Featurizer()
molecule_RelativeContentCs = RelativeContentCs_Featurizer()
molecule_RelativeContentCu = RelativeContentCu_Featurizer()
molecule_RelativeContentDb = RelativeContentDb_Featurizer()
molecule_RelativeContentDs = RelativeContentDs_Featurizer()
molecule_RelativeContentDy = RelativeContentDy_Featurizer()
molecule_RelativeContentEr = RelativeContentEr_Featurizer()
molecule_RelativeContentEs = RelativeContentEs_Featurizer()
molecule_RelativeContentEu = RelativeContentEu_Featurizer()
molecule_RelativeContentF = RelativeContentF_Featurizer()
molecule_RelativeContentFe = RelativeContentFe_Featurizer()
molecule_RelativeContentFl = RelativeContentFl_Featurizer()
molecule_RelativeContentFm = RelativeContentFm_Featurizer()
molecule_RelativeContentFr = RelativeContentFr_Featurizer()
molecule_RelativeContentGa = RelativeContentGa_Featurizer()
molecule_RelativeContentGd = RelativeContentGd_Featurizer()
molecule_RelativeContentGe = RelativeContentGe_Featurizer()
molecule_RelativeContentH = RelativeContentH_Featurizer()
molecule_RelativeContentHe = RelativeContentHe_Featurizer()
molecule_RelativeContentHf = RelativeContentHf_Featurizer()
molecule_RelativeContentHg = RelativeContentHg_Featurizer()
molecule_RelativeContentHo = RelativeContentHo_Featurizer()
molecule_RelativeContentHs = RelativeContentHs_Featurizer()
molecule_RelativeContentI = RelativeContentI_Featurizer()
molecule_RelativeContentIn = RelativeContentIn_Featurizer()
molecule_RelativeContentIr = RelativeContentIr_Featurizer()
molecule_RelativeContentK = RelativeContentK_Featurizer()
molecule_RelativeContentKr = RelativeContentKr_Featurizer()
molecule_RelativeContentLa = RelativeContentLa_Featurizer()
molecule_RelativeContentLi = RelativeContentLi_Featurizer()
molecule_RelativeContentLr = RelativeContentLr_Featurizer()
molecule_RelativeContentLu = RelativeContentLu_Featurizer()
molecule_RelativeContentLv = RelativeContentLv_Featurizer()
molecule_RelativeContentMc = RelativeContentMc_Featurizer()
molecule_RelativeContentMd = RelativeContentMd_Featurizer()
molecule_RelativeContentMg = RelativeContentMg_Featurizer()
molecule_RelativeContentMn = RelativeContentMn_Featurizer()
molecule_RelativeContentMo = RelativeContentMo_Featurizer()
molecule_RelativeContentMt = RelativeContentMt_Featurizer()
molecule_RelativeContentN = RelativeContentN_Featurizer()
molecule_RelativeContentNa = RelativeContentNa_Featurizer()
molecule_RelativeContentNb = RelativeContentNb_Featurizer()
molecule_RelativeContentNd = RelativeContentNd_Featurizer()
molecule_RelativeContentNe = RelativeContentNe_Featurizer()
molecule_RelativeContentNh = RelativeContentNh_Featurizer()
molecule_RelativeContentNi = RelativeContentNi_Featurizer()
molecule_RelativeContentNo = RelativeContentNo_Featurizer()
molecule_RelativeContentNp = RelativeContentNp_Featurizer()
molecule_RelativeContentO = RelativeContentO_Featurizer()
molecule_RelativeContentOg = RelativeContentOg_Featurizer()
molecule_RelativeContentOs = RelativeContentOs_Featurizer()
molecule_RelativeContentP = RelativeContentP_Featurizer()
molecule_RelativeContentPa = RelativeContentPa_Featurizer()
molecule_RelativeContentPb = RelativeContentPb_Featurizer()
molecule_RelativeContentPd = RelativeContentPd_Featurizer()
molecule_RelativeContentPm = RelativeContentPm_Featurizer()
molecule_RelativeContentPo = RelativeContentPo_Featurizer()
molecule_RelativeContentPr = RelativeContentPr_Featurizer()
molecule_RelativeContentPt = RelativeContentPt_Featurizer()
molecule_RelativeContentPu = RelativeContentPu_Featurizer()
molecule_RelativeContentRa = RelativeContentRa_Featurizer()
molecule_RelativeContentRb = RelativeContentRb_Featurizer()
molecule_RelativeContentRe = RelativeContentRe_Featurizer()
molecule_RelativeContentRf = RelativeContentRf_Featurizer()
molecule_RelativeContentRg = RelativeContentRg_Featurizer()
molecule_RelativeContentRh = RelativeContentRh_Featurizer()
molecule_RelativeContentRn = RelativeContentRn_Featurizer()
molecule_RelativeContentRu = RelativeContentRu_Featurizer()
molecule_RelativeContentS = RelativeContentS_Featurizer()
molecule_RelativeContentSb = RelativeContentSb_Featurizer()
molecule_RelativeContentSc = RelativeContentSc_Featurizer()
molecule_RelativeContentSe = RelativeContentSe_Featurizer()
molecule_RelativeContentSg = RelativeContentSg_Featurizer()
molecule_RelativeContentSi = RelativeContentSi_Featurizer()
molecule_RelativeContentSm = RelativeContentSm_Featurizer()
molecule_RelativeContentSn = RelativeContentSn_Featurizer()
molecule_RelativeContentSr = RelativeContentSr_Featurizer()
molecule_RelativeContentTa = RelativeContentTa_Featurizer()
molecule_RelativeContentTb = RelativeContentTb_Featurizer()
molecule_RelativeContentTc = RelativeContentTc_Featurizer()
molecule_RelativeContentTe = RelativeContentTe_Featurizer()
molecule_RelativeContentTh = RelativeContentTh_Featurizer()
molecule_RelativeContentTi = RelativeContentTi_Featurizer()
molecule_RelativeContentTl = RelativeContentTl_Featurizer()
molecule_RelativeContentTm = RelativeContentTm_Featurizer()
molecule_RelativeContentTs = RelativeContentTs_Featurizer()
molecule_RelativeContentU = RelativeContentU_Featurizer()
molecule_RelativeContentV = RelativeContentV_Featurizer()
molecule_RelativeContentW = RelativeContentW_Featurizer()
molecule_RelativeContentXe = RelativeContentXe_Featurizer()
molecule_RelativeContentY = RelativeContentY_Featurizer()
molecule_RelativeContentYb = RelativeContentYb_Featurizer()
molecule_RelativeContentZn = RelativeContentZn_Featurizer()
molecule_RelativeContentZr = RelativeContentZr_Featurizer()
molecule_RingCount = RingCount_Featurizer()
molecule_SMR_VSA1 = SMR_VSA1_Featurizer()
molecule_SMR_VSA10 = SMR_VSA10_Featurizer()
molecule_SMR_VSA2 = SMR_VSA2_Featurizer()
molecule_SMR_VSA3 = SMR_VSA3_Featurizer()
molecule_SMR_VSA4 = SMR_VSA4_Featurizer()
molecule_SMR_VSA5 = SMR_VSA5_Featurizer()
molecule_SMR_VSA6 = SMR_VSA6_Featurizer()
molecule_SMR_VSA7 = SMR_VSA7_Featurizer()
molecule_SMR_VSA8 = SMR_VSA8_Featurizer()
molecule_SMR_VSA9 = SMR_VSA9_Featurizer()
molecule_SSSR = SSSR_Featurizer()
molecule_SlogP_VSA1 = SlogP_VSA1_Featurizer()
molecule_SlogP_VSA10 = SlogP_VSA10_Featurizer()
molecule_SlogP_VSA11 = SlogP_VSA11_Featurizer()
molecule_SlogP_VSA12 = SlogP_VSA12_Featurizer()
molecule_SlogP_VSA2 = SlogP_VSA2_Featurizer()
molecule_SlogP_VSA3 = SlogP_VSA3_Featurizer()
molecule_SlogP_VSA4 = SlogP_VSA4_Featurizer()
molecule_SlogP_VSA5 = SlogP_VSA5_Featurizer()
molecule_SlogP_VSA6 = SlogP_VSA6_Featurizer()
molecule_SlogP_VSA7 = SlogP_VSA7_Featurizer()
molecule_SlogP_VSA8 = SlogP_VSA8_Featurizer()
molecule_SlogP_VSA9 = SlogP_VSA9_Featurizer()
molecule_SpherocityIndex = SpherocityIndex_Featurizer()
molecule_TPSA = TPSA_Featurizer()
molecule_VSA_EState1 = VSA_EState1_Featurizer()
molecule_VSA_EState10 = VSA_EState10_Featurizer()
molecule_VSA_EState2 = VSA_EState2_Featurizer()
molecule_VSA_EState3 = VSA_EState3_Featurizer()
molecule_VSA_EState4 = VSA_EState4_Featurizer()
molecule_VSA_EState5 = VSA_EState5_Featurizer()
molecule_VSA_EState6 = VSA_EState6_Featurizer()
molecule_VSA_EState7 = VSA_EState7_Featurizer()
molecule_VSA_EState8 = VSA_EState8_Featurizer()
molecule_VSA_EState9 = VSA_EState9_Featurizer()
molecule_fr_Al_COO = fr_Al_COO_Featurizer()
molecule_fr_Al_OH = fr_Al_OH_Featurizer()
molecule_fr_Al_OH_noTert = fr_Al_OH_noTert_Featurizer()
molecule_fr_ArN = fr_ArN_Featurizer()
molecule_fr_Ar_COO = fr_Ar_COO_Featurizer()
molecule_fr_Ar_N = fr_Ar_N_Featurizer()
molecule_fr_Ar_NH = fr_Ar_NH_Featurizer()
molecule_fr_Ar_OH = fr_Ar_OH_Featurizer()
molecule_fr_COO = fr_COO_Featurizer()
molecule_fr_COO2 = fr_COO2_Featurizer()
molecule_fr_C_O = fr_C_O_Featurizer()
molecule_fr_C_O_noCOO = fr_C_O_noCOO_Featurizer()
molecule_fr_C_S = fr_C_S_Featurizer()
molecule_fr_HOCCN = fr_HOCCN_Featurizer()
molecule_fr_Imine = fr_Imine_Featurizer()
molecule_fr_NH0 = fr_NH0_Featurizer()
molecule_fr_NH1 = fr_NH1_Featurizer()
molecule_fr_NH2 = fr_NH2_Featurizer()
molecule_fr_N_O = fr_N_O_Featurizer()
molecule_fr_Ndealkylation1 = fr_Ndealkylation1_Featurizer()
molecule_fr_Ndealkylation2 = fr_Ndealkylation2_Featurizer()
molecule_fr_Nhpyrrole = fr_Nhpyrrole_Featurizer()
molecule_fr_SH = fr_SH_Featurizer()
molecule_fr_aldehyde = fr_aldehyde_Featurizer()
molecule_fr_alkyl_carbamate = fr_alkyl_carbamate_Featurizer()
molecule_fr_alkyl_halide = fr_alkyl_halide_Featurizer()
molecule_fr_allylic_oxid = fr_allylic_oxid_Featurizer()
molecule_fr_amide = fr_amide_Featurizer()
molecule_fr_amidine = fr_amidine_Featurizer()
molecule_fr_aniline = fr_aniline_Featurizer()
molecule_fr_aryl_methyl = fr_aryl_methyl_Featurizer()
molecule_fr_azide = fr_azide_Featurizer()
molecule_fr_azo = fr_azo_Featurizer()
molecule_fr_barbitur = fr_barbitur_Featurizer()
molecule_fr_benzene = fr_benzene_Featurizer()
molecule_fr_benzodiazepine = fr_benzodiazepine_Featurizer()
molecule_fr_bicyclic = fr_bicyclic_Featurizer()
molecule_fr_diazo = fr_diazo_Featurizer()
molecule_fr_dihydropyridine = fr_dihydropyridine_Featurizer()
molecule_fr_epoxide = fr_epoxide_Featurizer()
molecule_fr_ester = fr_ester_Featurizer()
molecule_fr_ether = fr_ether_Featurizer()
molecule_fr_furan = fr_furan_Featurizer()
molecule_fr_guanido = fr_guanido_Featurizer()
molecule_fr_halogen = fr_halogen_Featurizer()
molecule_fr_hdrzine = fr_hdrzine_Featurizer()
molecule_fr_hdrzone = fr_hdrzone_Featurizer()
molecule_fr_imidazole = fr_imidazole_Featurizer()
molecule_fr_imide = fr_imide_Featurizer()
molecule_fr_isocyan = fr_isocyan_Featurizer()
molecule_fr_isothiocyan = fr_isothiocyan_Featurizer()
molecule_fr_ketone = fr_ketone_Featurizer()
molecule_fr_ketone_Topliss = fr_ketone_Topliss_Featurizer()
molecule_fr_lactam = fr_lactam_Featurizer()
molecule_fr_lactone = fr_lactone_Featurizer()
molecule_fr_methoxy = fr_methoxy_Featurizer()
molecule_fr_morpholine = fr_morpholine_Featurizer()
molecule_fr_nitrile = fr_nitrile_Featurizer()
molecule_fr_nitro = fr_nitro_Featurizer()
molecule_fr_nitro_arom = fr_nitro_arom_Featurizer()
molecule_fr_nitro_arom_nonortho = fr_nitro_arom_nonortho_Featurizer()
molecule_fr_nitroso = fr_nitroso_Featurizer()
molecule_fr_oxazole = fr_oxazole_Featurizer()
molecule_fr_oxime = fr_oxime_Featurizer()
molecule_fr_para_hydroxylation = fr_para_hydroxylation_Featurizer()
molecule_fr_phenol = fr_phenol_Featurizer()
molecule_fr_phenol_noOrthoHbond = fr_phenol_noOrthoHbond_Featurizer()
molecule_fr_phos_acid = fr_phos_acid_Featurizer()
molecule_fr_phos_ester = fr_phos_ester_Featurizer()
molecule_fr_piperdine = fr_piperdine_Featurizer()
molecule_fr_piperzine = fr_piperzine_Featurizer()
molecule_fr_priamide = fr_priamide_Featurizer()
molecule_fr_prisulfonamd = fr_prisulfonamd_Featurizer()
molecule_fr_pyridine = fr_pyridine_Featurizer()
molecule_fr_quatN = fr_quatN_Featurizer()
molecule_fr_sulfide = fr_sulfide_Featurizer()
molecule_fr_sulfonamd = fr_sulfonamd_Featurizer()
molecule_fr_sulfone = fr_sulfone_Featurizer()
molecule_fr_term_acetylene = fr_term_acetylene_Featurizer()
molecule_fr_tetrazole = fr_tetrazole_Featurizer()
molecule_fr_thiazole = fr_thiazole_Featurizer()
molecule_fr_thiocyan = fr_thiocyan_Featurizer()
molecule_fr_thiophene = fr_thiophene_Featurizer()
molecule_fr_unbrch_alkane = fr_unbrch_alkane_Featurizer()
molecule_fr_urea = fr_urea_Featurizer()
molecule_qed = qed_Featurizer()

_available_featurizer = {
    "molecule_Asphericity": molecule_Asphericity,
    "molecule_BCUT2D_CHGHI": molecule_BCUT2D_CHGHI,
    "molecule_BCUT2D_CHGLO": molecule_BCUT2D_CHGLO,
    "molecule_BCUT2D_LOGPHI": molecule_BCUT2D_LOGPHI,
    "molecule_BCUT2D_LOGPLOW": molecule_BCUT2D_LOGPLOW,
    "molecule_BCUT2D_MRHI": molecule_BCUT2D_MRHI,
    "molecule_BCUT2D_MRLOW": molecule_BCUT2D_MRLOW,
    "molecule_BCUT2D_MWHI": molecule_BCUT2D_MWHI,
    "molecule_BCUT2D_MWLOW": molecule_BCUT2D_MWLOW,
    "molecule_BalabanJ": molecule_BalabanJ,
    "molecule_BertzCT": molecule_BertzCT,
    "molecule_Chi0": molecule_Chi0,
    "molecule_Chi0n": molecule_Chi0n,
    "molecule_Chi0v": molecule_Chi0v,
    "molecule_Chi1": molecule_Chi1,
    "molecule_Chi1n": molecule_Chi1n,
    "molecule_Chi1v": molecule_Chi1v,
    "molecule_Chi2n": molecule_Chi2n,
    "molecule_Chi2v": molecule_Chi2v,
    "molecule_Chi3n": molecule_Chi3n,
    "molecule_Chi3v": molecule_Chi3v,
    "molecule_Chi4n": molecule_Chi4n,
    "molecule_Chi4v": molecule_Chi4v,
    "molecule_EState_VSA1": molecule_EState_VSA1,
    "molecule_EState_VSA10": molecule_EState_VSA10,
    "molecule_EState_VSA11": molecule_EState_VSA11,
    "molecule_EState_VSA2": molecule_EState_VSA2,
    "molecule_EState_VSA3": molecule_EState_VSA3,
    "molecule_EState_VSA4": molecule_EState_VSA4,
    "molecule_EState_VSA5": molecule_EState_VSA5,
    "molecule_EState_VSA6": molecule_EState_VSA6,
    "molecule_EState_VSA7": molecule_EState_VSA7,
    "molecule_EState_VSA8": molecule_EState_VSA8,
    "molecule_EState_VSA9": molecule_EState_VSA9,
    "molecule_Eccentricity": molecule_Eccentricity,
    "molecule_ExactMolWt": molecule_ExactMolWt,
    "molecule_FormalCharge": molecule_FormalCharge,
    "molecule_FpDensityMorgan1": molecule_FpDensityMorgan1,
    "molecule_FpDensityMorgan2": molecule_FpDensityMorgan2,
    "molecule_FpDensityMorgan3": molecule_FpDensityMorgan3,
    "molecule_FractionCSP3": molecule_FractionCSP3,
    "molecule_HallKierAlpha": molecule_HallKierAlpha,
    "molecule_HeavyAtomCount": molecule_HeavyAtomCount,
    "molecule_HeavyAtomMolWt": molecule_HeavyAtomMolWt,
    "molecule_InertialShapeFactor": molecule_InertialShapeFactor,
    "molecule_Ipc": molecule_Ipc,
    "molecule_Kappa1": molecule_Kappa1,
    "molecule_Kappa2": molecule_Kappa2,
    "molecule_Kappa3": molecule_Kappa3,
    "molecule_LabuteASA": molecule_LabuteASA,
    "molecule_MaxAbsEStateIndex": molecule_MaxAbsEStateIndex,
    "molecule_MaxAbsPartialCharge": molecule_MaxAbsPartialCharge,
    "molecule_MaxEStateIndex": molecule_MaxEStateIndex,
    "molecule_MaxPartialCharge": molecule_MaxPartialCharge,
    "molecule_MinAbsEStateIndex": molecule_MinAbsEStateIndex,
    "molecule_MinAbsPartialCharge": molecule_MinAbsPartialCharge,
    "molecule_MinEStateIndex": molecule_MinEStateIndex,
    "molecule_MinPartialCharge": molecule_MinPartialCharge,
    "molecule_MolLogP": molecule_MolLogP,
    "molecule_MolMR": molecule_MolMR,
    "molecule_MolWt": molecule_MolWt,
    "molecule_NHOHCount": molecule_NHOHCount,
    "molecule_NOCount": molecule_NOCount,
    "molecule_NPR1": molecule_NPR1,
    "molecule_NPR2": molecule_NPR2,
    "molecule_NumAliphaticCarbocycles": molecule_NumAliphaticCarbocycles,
    "molecule_NumAliphaticHeterocycles": molecule_NumAliphaticHeterocycles,
    "molecule_NumAliphaticRings": molecule_NumAliphaticRings,
    "molecule_NumAmideBonds": molecule_NumAmideBonds,
    "molecule_NumAromaticCarbocycles": molecule_NumAromaticCarbocycles,
    "molecule_NumAromaticHeterocycles": molecule_NumAromaticHeterocycles,
    "molecule_NumAromaticRings": molecule_NumAromaticRings,
    "molecule_NumAtoms": molecule_NumAtoms,
    "molecule_NumBonds": molecule_NumBonds,
    "molecule_NumBridgeheadAtoms": molecule_NumBridgeheadAtoms,
    "molecule_NumHAcceptors": molecule_NumHAcceptors,
    "molecule_NumHBA": molecule_NumHBA,
    "molecule_NumHBD": molecule_NumHBD,
    "molecule_NumHDonors": molecule_NumHDonors,
    "molecule_NumHeavyAtoms": molecule_NumHeavyAtoms,
    "molecule_NumHeteroatoms": molecule_NumHeteroatoms,
    "molecule_NumHeterocycles": molecule_NumHeterocycles,
    "molecule_NumLipinskiHBA": molecule_NumLipinskiHBA,
    "molecule_NumLipinskiHBD": molecule_NumLipinskiHBD,
    "molecule_NumRadicalElectrons": molecule_NumRadicalElectrons,
    "molecule_NumRings": molecule_NumRings,
    "molecule_NumRotatableBonds": molecule_NumRotatableBonds,
    "molecule_NumSaturatedCarbocycles": molecule_NumSaturatedCarbocycles,
    "molecule_NumSaturatedHeterocycles": molecule_NumSaturatedHeterocycles,
    "molecule_NumSaturatedRings": molecule_NumSaturatedRings,
    "molecule_NumSpiroAtoms": molecule_NumSpiroAtoms,
    "molecule_NumValenceElectrons": molecule_NumValenceElectrons,
    "molecule_NumberAtomsAc": molecule_NumberAtomsAc,
    "molecule_NumberAtomsAg": molecule_NumberAtomsAg,
    "molecule_NumberAtomsAl": molecule_NumberAtomsAl,
    "molecule_NumberAtomsAm": molecule_NumberAtomsAm,
    "molecule_NumberAtomsAr": molecule_NumberAtomsAr,
    "molecule_NumberAtomsAs": molecule_NumberAtomsAs,
    "molecule_NumberAtomsAt": molecule_NumberAtomsAt,
    "molecule_NumberAtomsAu": molecule_NumberAtomsAu,
    "molecule_NumberAtomsB": molecule_NumberAtomsB,
    "molecule_NumberAtomsBa": molecule_NumberAtomsBa,
    "molecule_NumberAtomsBe": molecule_NumberAtomsBe,
    "molecule_NumberAtomsBh": molecule_NumberAtomsBh,
    "molecule_NumberAtomsBi": molecule_NumberAtomsBi,
    "molecule_NumberAtomsBk": molecule_NumberAtomsBk,
    "molecule_NumberAtomsBr": molecule_NumberAtomsBr,
    "molecule_NumberAtomsC": molecule_NumberAtomsC,
    "molecule_NumberAtomsCa": molecule_NumberAtomsCa,
    "molecule_NumberAtomsCd": molecule_NumberAtomsCd,
    "molecule_NumberAtomsCe": molecule_NumberAtomsCe,
    "molecule_NumberAtomsCf": molecule_NumberAtomsCf,
    "molecule_NumberAtomsCl": molecule_NumberAtomsCl,
    "molecule_NumberAtomsCm": molecule_NumberAtomsCm,
    "molecule_NumberAtomsCn": molecule_NumberAtomsCn,
    "molecule_NumberAtomsCo": molecule_NumberAtomsCo,
    "molecule_NumberAtomsCr": molecule_NumberAtomsCr,
    "molecule_NumberAtomsCs": molecule_NumberAtomsCs,
    "molecule_NumberAtomsCu": molecule_NumberAtomsCu,
    "molecule_NumberAtomsDb": molecule_NumberAtomsDb,
    "molecule_NumberAtomsDs": molecule_NumberAtomsDs,
    "molecule_NumberAtomsDy": molecule_NumberAtomsDy,
    "molecule_NumberAtomsEr": molecule_NumberAtomsEr,
    "molecule_NumberAtomsEs": molecule_NumberAtomsEs,
    "molecule_NumberAtomsEu": molecule_NumberAtomsEu,
    "molecule_NumberAtomsF": molecule_NumberAtomsF,
    "molecule_NumberAtomsFe": molecule_NumberAtomsFe,
    "molecule_NumberAtomsFl": molecule_NumberAtomsFl,
    "molecule_NumberAtomsFm": molecule_NumberAtomsFm,
    "molecule_NumberAtomsFr": molecule_NumberAtomsFr,
    "molecule_NumberAtomsGa": molecule_NumberAtomsGa,
    "molecule_NumberAtomsGd": molecule_NumberAtomsGd,
    "molecule_NumberAtomsGe": molecule_NumberAtomsGe,
    "molecule_NumberAtomsH": molecule_NumberAtomsH,
    "molecule_NumberAtomsHe": molecule_NumberAtomsHe,
    "molecule_NumberAtomsHf": molecule_NumberAtomsHf,
    "molecule_NumberAtomsHg": molecule_NumberAtomsHg,
    "molecule_NumberAtomsHo": molecule_NumberAtomsHo,
    "molecule_NumberAtomsHs": molecule_NumberAtomsHs,
    "molecule_NumberAtomsI": molecule_NumberAtomsI,
    "molecule_NumberAtomsIn": molecule_NumberAtomsIn,
    "molecule_NumberAtomsIr": molecule_NumberAtomsIr,
    "molecule_NumberAtomsK": molecule_NumberAtomsK,
    "molecule_NumberAtomsKr": molecule_NumberAtomsKr,
    "molecule_NumberAtomsLa": molecule_NumberAtomsLa,
    "molecule_NumberAtomsLi": molecule_NumberAtomsLi,
    "molecule_NumberAtomsLr": molecule_NumberAtomsLr,
    "molecule_NumberAtomsLu": molecule_NumberAtomsLu,
    "molecule_NumberAtomsLv": molecule_NumberAtomsLv,
    "molecule_NumberAtomsMc": molecule_NumberAtomsMc,
    "molecule_NumberAtomsMd": molecule_NumberAtomsMd,
    "molecule_NumberAtomsMg": molecule_NumberAtomsMg,
    "molecule_NumberAtomsMn": molecule_NumberAtomsMn,
    "molecule_NumberAtomsMo": molecule_NumberAtomsMo,
    "molecule_NumberAtomsMt": molecule_NumberAtomsMt,
    "molecule_NumberAtomsN": molecule_NumberAtomsN,
    "molecule_NumberAtomsNa": molecule_NumberAtomsNa,
    "molecule_NumberAtomsNb": molecule_NumberAtomsNb,
    "molecule_NumberAtomsNd": molecule_NumberAtomsNd,
    "molecule_NumberAtomsNe": molecule_NumberAtomsNe,
    "molecule_NumberAtomsNh": molecule_NumberAtomsNh,
    "molecule_NumberAtomsNi": molecule_NumberAtomsNi,
    "molecule_NumberAtomsNo": molecule_NumberAtomsNo,
    "molecule_NumberAtomsNp": molecule_NumberAtomsNp,
    "molecule_NumberAtomsO": molecule_NumberAtomsO,
    "molecule_NumberAtomsOg": molecule_NumberAtomsOg,
    "molecule_NumberAtomsOs": molecule_NumberAtomsOs,
    "molecule_NumberAtomsP": molecule_NumberAtomsP,
    "molecule_NumberAtomsPa": molecule_NumberAtomsPa,
    "molecule_NumberAtomsPb": molecule_NumberAtomsPb,
    "molecule_NumberAtomsPd": molecule_NumberAtomsPd,
    "molecule_NumberAtomsPm": molecule_NumberAtomsPm,
    "molecule_NumberAtomsPo": molecule_NumberAtomsPo,
    "molecule_NumberAtomsPr": molecule_NumberAtomsPr,
    "molecule_NumberAtomsPt": molecule_NumberAtomsPt,
    "molecule_NumberAtomsPu": molecule_NumberAtomsPu,
    "molecule_NumberAtomsRa": molecule_NumberAtomsRa,
    "molecule_NumberAtomsRb": molecule_NumberAtomsRb,
    "molecule_NumberAtomsRe": molecule_NumberAtomsRe,
    "molecule_NumberAtomsRf": molecule_NumberAtomsRf,
    "molecule_NumberAtomsRg": molecule_NumberAtomsRg,
    "molecule_NumberAtomsRh": molecule_NumberAtomsRh,
    "molecule_NumberAtomsRn": molecule_NumberAtomsRn,
    "molecule_NumberAtomsRu": molecule_NumberAtomsRu,
    "molecule_NumberAtomsS": molecule_NumberAtomsS,
    "molecule_NumberAtomsSb": molecule_NumberAtomsSb,
    "molecule_NumberAtomsSc": molecule_NumberAtomsSc,
    "molecule_NumberAtomsSe": molecule_NumberAtomsSe,
    "molecule_NumberAtomsSg": molecule_NumberAtomsSg,
    "molecule_NumberAtomsSi": molecule_NumberAtomsSi,
    "molecule_NumberAtomsSm": molecule_NumberAtomsSm,
    "molecule_NumberAtomsSn": molecule_NumberAtomsSn,
    "molecule_NumberAtomsSr": molecule_NumberAtomsSr,
    "molecule_NumberAtomsTa": molecule_NumberAtomsTa,
    "molecule_NumberAtomsTb": molecule_NumberAtomsTb,
    "molecule_NumberAtomsTc": molecule_NumberAtomsTc,
    "molecule_NumberAtomsTe": molecule_NumberAtomsTe,
    "molecule_NumberAtomsTh": molecule_NumberAtomsTh,
    "molecule_NumberAtomsTi": molecule_NumberAtomsTi,
    "molecule_NumberAtomsTl": molecule_NumberAtomsTl,
    "molecule_NumberAtomsTm": molecule_NumberAtomsTm,
    "molecule_NumberAtomsTs": molecule_NumberAtomsTs,
    "molecule_NumberAtomsU": molecule_NumberAtomsU,
    "molecule_NumberAtomsV": molecule_NumberAtomsV,
    "molecule_NumberAtomsW": molecule_NumberAtomsW,
    "molecule_NumberAtomsXe": molecule_NumberAtomsXe,
    "molecule_NumberAtomsY": molecule_NumberAtomsY,
    "molecule_NumberAtomsYb": molecule_NumberAtomsYb,
    "molecule_NumberAtomsZn": molecule_NumberAtomsZn,
    "molecule_NumberAtomsZr": molecule_NumberAtomsZr,
    "molecule_PBF": molecule_PBF,
    "molecule_PEOE_VSA1": molecule_PEOE_VSA1,
    "molecule_PEOE_VSA10": molecule_PEOE_VSA10,
    "molecule_PEOE_VSA11": molecule_PEOE_VSA11,
    "molecule_PEOE_VSA12": molecule_PEOE_VSA12,
    "molecule_PEOE_VSA13": molecule_PEOE_VSA13,
    "molecule_PEOE_VSA14": molecule_PEOE_VSA14,
    "molecule_PEOE_VSA2": molecule_PEOE_VSA2,
    "molecule_PEOE_VSA3": molecule_PEOE_VSA3,
    "molecule_PEOE_VSA4": molecule_PEOE_VSA4,
    "molecule_PEOE_VSA5": molecule_PEOE_VSA5,
    "molecule_PEOE_VSA6": molecule_PEOE_VSA6,
    "molecule_PEOE_VSA7": molecule_PEOE_VSA7,
    "molecule_PEOE_VSA8": molecule_PEOE_VSA8,
    "molecule_PEOE_VSA9": molecule_PEOE_VSA9,
    "molecule_PMI1": molecule_PMI1,
    "molecule_PMI2": molecule_PMI2,
    "molecule_PMI3": molecule_PMI3,
    "molecule_Phi": molecule_Phi,
    "molecule_RadiusOfGyration": molecule_RadiusOfGyration,
    "molecule_RelativeContentAc": molecule_RelativeContentAc,
    "molecule_RelativeContentAg": molecule_RelativeContentAg,
    "molecule_RelativeContentAl": molecule_RelativeContentAl,
    "molecule_RelativeContentAm": molecule_RelativeContentAm,
    "molecule_RelativeContentAr": molecule_RelativeContentAr,
    "molecule_RelativeContentAs": molecule_RelativeContentAs,
    "molecule_RelativeContentAt": molecule_RelativeContentAt,
    "molecule_RelativeContentAu": molecule_RelativeContentAu,
    "molecule_RelativeContentB": molecule_RelativeContentB,
    "molecule_RelativeContentBa": molecule_RelativeContentBa,
    "molecule_RelativeContentBe": molecule_RelativeContentBe,
    "molecule_RelativeContentBh": molecule_RelativeContentBh,
    "molecule_RelativeContentBi": molecule_RelativeContentBi,
    "molecule_RelativeContentBk": molecule_RelativeContentBk,
    "molecule_RelativeContentBr": molecule_RelativeContentBr,
    "molecule_RelativeContentC": molecule_RelativeContentC,
    "molecule_RelativeContentCa": molecule_RelativeContentCa,
    "molecule_RelativeContentCd": molecule_RelativeContentCd,
    "molecule_RelativeContentCe": molecule_RelativeContentCe,
    "molecule_RelativeContentCf": molecule_RelativeContentCf,
    "molecule_RelativeContentCl": molecule_RelativeContentCl,
    "molecule_RelativeContentCm": molecule_RelativeContentCm,
    "molecule_RelativeContentCn": molecule_RelativeContentCn,
    "molecule_RelativeContentCo": molecule_RelativeContentCo,
    "molecule_RelativeContentCr": molecule_RelativeContentCr,
    "molecule_RelativeContentCs": molecule_RelativeContentCs,
    "molecule_RelativeContentCu": molecule_RelativeContentCu,
    "molecule_RelativeContentDb": molecule_RelativeContentDb,
    "molecule_RelativeContentDs": molecule_RelativeContentDs,
    "molecule_RelativeContentDy": molecule_RelativeContentDy,
    "molecule_RelativeContentEr": molecule_RelativeContentEr,
    "molecule_RelativeContentEs": molecule_RelativeContentEs,
    "molecule_RelativeContentEu": molecule_RelativeContentEu,
    "molecule_RelativeContentF": molecule_RelativeContentF,
    "molecule_RelativeContentFe": molecule_RelativeContentFe,
    "molecule_RelativeContentFl": molecule_RelativeContentFl,
    "molecule_RelativeContentFm": molecule_RelativeContentFm,
    "molecule_RelativeContentFr": molecule_RelativeContentFr,
    "molecule_RelativeContentGa": molecule_RelativeContentGa,
    "molecule_RelativeContentGd": molecule_RelativeContentGd,
    "molecule_RelativeContentGe": molecule_RelativeContentGe,
    "molecule_RelativeContentH": molecule_RelativeContentH,
    "molecule_RelativeContentHe": molecule_RelativeContentHe,
    "molecule_RelativeContentHf": molecule_RelativeContentHf,
    "molecule_RelativeContentHg": molecule_RelativeContentHg,
    "molecule_RelativeContentHo": molecule_RelativeContentHo,
    "molecule_RelativeContentHs": molecule_RelativeContentHs,
    "molecule_RelativeContentI": molecule_RelativeContentI,
    "molecule_RelativeContentIn": molecule_RelativeContentIn,
    "molecule_RelativeContentIr": molecule_RelativeContentIr,
    "molecule_RelativeContentK": molecule_RelativeContentK,
    "molecule_RelativeContentKr": molecule_RelativeContentKr,
    "molecule_RelativeContentLa": molecule_RelativeContentLa,
    "molecule_RelativeContentLi": molecule_RelativeContentLi,
    "molecule_RelativeContentLr": molecule_RelativeContentLr,
    "molecule_RelativeContentLu": molecule_RelativeContentLu,
    "molecule_RelativeContentLv": molecule_RelativeContentLv,
    "molecule_RelativeContentMc": molecule_RelativeContentMc,
    "molecule_RelativeContentMd": molecule_RelativeContentMd,
    "molecule_RelativeContentMg": molecule_RelativeContentMg,
    "molecule_RelativeContentMn": molecule_RelativeContentMn,
    "molecule_RelativeContentMo": molecule_RelativeContentMo,
    "molecule_RelativeContentMt": molecule_RelativeContentMt,
    "molecule_RelativeContentN": molecule_RelativeContentN,
    "molecule_RelativeContentNa": molecule_RelativeContentNa,
    "molecule_RelativeContentNb": molecule_RelativeContentNb,
    "molecule_RelativeContentNd": molecule_RelativeContentNd,
    "molecule_RelativeContentNe": molecule_RelativeContentNe,
    "molecule_RelativeContentNh": molecule_RelativeContentNh,
    "molecule_RelativeContentNi": molecule_RelativeContentNi,
    "molecule_RelativeContentNo": molecule_RelativeContentNo,
    "molecule_RelativeContentNp": molecule_RelativeContentNp,
    "molecule_RelativeContentO": molecule_RelativeContentO,
    "molecule_RelativeContentOg": molecule_RelativeContentOg,
    "molecule_RelativeContentOs": molecule_RelativeContentOs,
    "molecule_RelativeContentP": molecule_RelativeContentP,
    "molecule_RelativeContentPa": molecule_RelativeContentPa,
    "molecule_RelativeContentPb": molecule_RelativeContentPb,
    "molecule_RelativeContentPd": molecule_RelativeContentPd,
    "molecule_RelativeContentPm": molecule_RelativeContentPm,
    "molecule_RelativeContentPo": molecule_RelativeContentPo,
    "molecule_RelativeContentPr": molecule_RelativeContentPr,
    "molecule_RelativeContentPt": molecule_RelativeContentPt,
    "molecule_RelativeContentPu": molecule_RelativeContentPu,
    "molecule_RelativeContentRa": molecule_RelativeContentRa,
    "molecule_RelativeContentRb": molecule_RelativeContentRb,
    "molecule_RelativeContentRe": molecule_RelativeContentRe,
    "molecule_RelativeContentRf": molecule_RelativeContentRf,
    "molecule_RelativeContentRg": molecule_RelativeContentRg,
    "molecule_RelativeContentRh": molecule_RelativeContentRh,
    "molecule_RelativeContentRn": molecule_RelativeContentRn,
    "molecule_RelativeContentRu": molecule_RelativeContentRu,
    "molecule_RelativeContentS": molecule_RelativeContentS,
    "molecule_RelativeContentSb": molecule_RelativeContentSb,
    "molecule_RelativeContentSc": molecule_RelativeContentSc,
    "molecule_RelativeContentSe": molecule_RelativeContentSe,
    "molecule_RelativeContentSg": molecule_RelativeContentSg,
    "molecule_RelativeContentSi": molecule_RelativeContentSi,
    "molecule_RelativeContentSm": molecule_RelativeContentSm,
    "molecule_RelativeContentSn": molecule_RelativeContentSn,
    "molecule_RelativeContentSr": molecule_RelativeContentSr,
    "molecule_RelativeContentTa": molecule_RelativeContentTa,
    "molecule_RelativeContentTb": molecule_RelativeContentTb,
    "molecule_RelativeContentTc": molecule_RelativeContentTc,
    "molecule_RelativeContentTe": molecule_RelativeContentTe,
    "molecule_RelativeContentTh": molecule_RelativeContentTh,
    "molecule_RelativeContentTi": molecule_RelativeContentTi,
    "molecule_RelativeContentTl": molecule_RelativeContentTl,
    "molecule_RelativeContentTm": molecule_RelativeContentTm,
    "molecule_RelativeContentTs": molecule_RelativeContentTs,
    "molecule_RelativeContentU": molecule_RelativeContentU,
    "molecule_RelativeContentV": molecule_RelativeContentV,
    "molecule_RelativeContentW": molecule_RelativeContentW,
    "molecule_RelativeContentXe": molecule_RelativeContentXe,
    "molecule_RelativeContentY": molecule_RelativeContentY,
    "molecule_RelativeContentYb": molecule_RelativeContentYb,
    "molecule_RelativeContentZn": molecule_RelativeContentZn,
    "molecule_RelativeContentZr": molecule_RelativeContentZr,
    "molecule_RingCount": molecule_RingCount,
    "molecule_SMR_VSA1": molecule_SMR_VSA1,
    "molecule_SMR_VSA10": molecule_SMR_VSA10,
    "molecule_SMR_VSA2": molecule_SMR_VSA2,
    "molecule_SMR_VSA3": molecule_SMR_VSA3,
    "molecule_SMR_VSA4": molecule_SMR_VSA4,
    "molecule_SMR_VSA5": molecule_SMR_VSA5,
    "molecule_SMR_VSA6": molecule_SMR_VSA6,
    "molecule_SMR_VSA7": molecule_SMR_VSA7,
    "molecule_SMR_VSA8": molecule_SMR_VSA8,
    "molecule_SMR_VSA9": molecule_SMR_VSA9,
    "molecule_SSSR": molecule_SSSR,
    "molecule_SlogP_VSA1": molecule_SlogP_VSA1,
    "molecule_SlogP_VSA10": molecule_SlogP_VSA10,
    "molecule_SlogP_VSA11": molecule_SlogP_VSA11,
    "molecule_SlogP_VSA12": molecule_SlogP_VSA12,
    "molecule_SlogP_VSA2": molecule_SlogP_VSA2,
    "molecule_SlogP_VSA3": molecule_SlogP_VSA3,
    "molecule_SlogP_VSA4": molecule_SlogP_VSA4,
    "molecule_SlogP_VSA5": molecule_SlogP_VSA5,
    "molecule_SlogP_VSA6": molecule_SlogP_VSA6,
    "molecule_SlogP_VSA7": molecule_SlogP_VSA7,
    "molecule_SlogP_VSA8": molecule_SlogP_VSA8,
    "molecule_SlogP_VSA9": molecule_SlogP_VSA9,
    "molecule_SpherocityIndex": molecule_SpherocityIndex,
    "molecule_TPSA": molecule_TPSA,
    "molecule_VSA_EState1": molecule_VSA_EState1,
    "molecule_VSA_EState10": molecule_VSA_EState10,
    "molecule_VSA_EState2": molecule_VSA_EState2,
    "molecule_VSA_EState3": molecule_VSA_EState3,
    "molecule_VSA_EState4": molecule_VSA_EState4,
    "molecule_VSA_EState5": molecule_VSA_EState5,
    "molecule_VSA_EState6": molecule_VSA_EState6,
    "molecule_VSA_EState7": molecule_VSA_EState7,
    "molecule_VSA_EState8": molecule_VSA_EState8,
    "molecule_VSA_EState9": molecule_VSA_EState9,
    "molecule_fr_Al_COO": molecule_fr_Al_COO,
    "molecule_fr_Al_OH": molecule_fr_Al_OH,
    "molecule_fr_Al_OH_noTert": molecule_fr_Al_OH_noTert,
    "molecule_fr_ArN": molecule_fr_ArN,
    "molecule_fr_Ar_COO": molecule_fr_Ar_COO,
    "molecule_fr_Ar_N": molecule_fr_Ar_N,
    "molecule_fr_Ar_NH": molecule_fr_Ar_NH,
    "molecule_fr_Ar_OH": molecule_fr_Ar_OH,
    "molecule_fr_COO": molecule_fr_COO,
    "molecule_fr_COO2": molecule_fr_COO2,
    "molecule_fr_C_O": molecule_fr_C_O,
    "molecule_fr_C_O_noCOO": molecule_fr_C_O_noCOO,
    "molecule_fr_C_S": molecule_fr_C_S,
    "molecule_fr_HOCCN": molecule_fr_HOCCN,
    "molecule_fr_Imine": molecule_fr_Imine,
    "molecule_fr_NH0": molecule_fr_NH0,
    "molecule_fr_NH1": molecule_fr_NH1,
    "molecule_fr_NH2": molecule_fr_NH2,
    "molecule_fr_N_O": molecule_fr_N_O,
    "molecule_fr_Ndealkylation1": molecule_fr_Ndealkylation1,
    "molecule_fr_Ndealkylation2": molecule_fr_Ndealkylation2,
    "molecule_fr_Nhpyrrole": molecule_fr_Nhpyrrole,
    "molecule_fr_SH": molecule_fr_SH,
    "molecule_fr_aldehyde": molecule_fr_aldehyde,
    "molecule_fr_alkyl_carbamate": molecule_fr_alkyl_carbamate,
    "molecule_fr_alkyl_halide": molecule_fr_alkyl_halide,
    "molecule_fr_allylic_oxid": molecule_fr_allylic_oxid,
    "molecule_fr_amide": molecule_fr_amide,
    "molecule_fr_amidine": molecule_fr_amidine,
    "molecule_fr_aniline": molecule_fr_aniline,
    "molecule_fr_aryl_methyl": molecule_fr_aryl_methyl,
    "molecule_fr_azide": molecule_fr_azide,
    "molecule_fr_azo": molecule_fr_azo,
    "molecule_fr_barbitur": molecule_fr_barbitur,
    "molecule_fr_benzene": molecule_fr_benzene,
    "molecule_fr_benzodiazepine": molecule_fr_benzodiazepine,
    "molecule_fr_bicyclic": molecule_fr_bicyclic,
    "molecule_fr_diazo": molecule_fr_diazo,
    "molecule_fr_dihydropyridine": molecule_fr_dihydropyridine,
    "molecule_fr_epoxide": molecule_fr_epoxide,
    "molecule_fr_ester": molecule_fr_ester,
    "molecule_fr_ether": molecule_fr_ether,
    "molecule_fr_furan": molecule_fr_furan,
    "molecule_fr_guanido": molecule_fr_guanido,
    "molecule_fr_halogen": molecule_fr_halogen,
    "molecule_fr_hdrzine": molecule_fr_hdrzine,
    "molecule_fr_hdrzone": molecule_fr_hdrzone,
    "molecule_fr_imidazole": molecule_fr_imidazole,
    "molecule_fr_imide": molecule_fr_imide,
    "molecule_fr_isocyan": molecule_fr_isocyan,
    "molecule_fr_isothiocyan": molecule_fr_isothiocyan,
    "molecule_fr_ketone": molecule_fr_ketone,
    "molecule_fr_ketone_Topliss": molecule_fr_ketone_Topliss,
    "molecule_fr_lactam": molecule_fr_lactam,
    "molecule_fr_lactone": molecule_fr_lactone,
    "molecule_fr_methoxy": molecule_fr_methoxy,
    "molecule_fr_morpholine": molecule_fr_morpholine,
    "molecule_fr_nitrile": molecule_fr_nitrile,
    "molecule_fr_nitro": molecule_fr_nitro,
    "molecule_fr_nitro_arom": molecule_fr_nitro_arom,
    "molecule_fr_nitro_arom_nonortho": molecule_fr_nitro_arom_nonortho,
    "molecule_fr_nitroso": molecule_fr_nitroso,
    "molecule_fr_oxazole": molecule_fr_oxazole,
    "molecule_fr_oxime": molecule_fr_oxime,
    "molecule_fr_para_hydroxylation": molecule_fr_para_hydroxylation,
    "molecule_fr_phenol": molecule_fr_phenol,
    "molecule_fr_phenol_noOrthoHbond": molecule_fr_phenol_noOrthoHbond,
    "molecule_fr_phos_acid": molecule_fr_phos_acid,
    "molecule_fr_phos_ester": molecule_fr_phos_ester,
    "molecule_fr_piperdine": molecule_fr_piperdine,
    "molecule_fr_piperzine": molecule_fr_piperzine,
    "molecule_fr_priamide": molecule_fr_priamide,
    "molecule_fr_prisulfonamd": molecule_fr_prisulfonamd,
    "molecule_fr_pyridine": molecule_fr_pyridine,
    "molecule_fr_quatN": molecule_fr_quatN,
    "molecule_fr_sulfide": molecule_fr_sulfide,
    "molecule_fr_sulfonamd": molecule_fr_sulfonamd,
    "molecule_fr_sulfone": molecule_fr_sulfone,
    "molecule_fr_term_acetylene": molecule_fr_term_acetylene,
    "molecule_fr_tetrazole": molecule_fr_tetrazole,
    "molecule_fr_thiazole": molecule_fr_thiazole,
    "molecule_fr_thiocyan": molecule_fr_thiocyan,
    "molecule_fr_thiophene": molecule_fr_thiophene,
    "molecule_fr_unbrch_alkane": molecule_fr_unbrch_alkane,
    "molecule_fr_urea": molecule_fr_urea,
    "molecule_qed": molecule_qed,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "Asphericity_Featurizer",
    "BCUT2D_CHGHI_Featurizer",
    "BCUT2D_CHGLO_Featurizer",
    "BCUT2D_LOGPHI_Featurizer",
    "BCUT2D_LOGPLOW_Featurizer",
    "BCUT2D_MRHI_Featurizer",
    "BCUT2D_MRLOW_Featurizer",
    "BCUT2D_MWHI_Featurizer",
    "BCUT2D_MWLOW_Featurizer",
    "BalabanJ_Featurizer",
    "BertzCT_Featurizer",
    "Chi0_Featurizer",
    "Chi0n_Featurizer",
    "Chi0v_Featurizer",
    "Chi1_Featurizer",
    "Chi1n_Featurizer",
    "Chi1v_Featurizer",
    "Chi2n_Featurizer",
    "Chi2v_Featurizer",
    "Chi3n_Featurizer",
    "Chi3v_Featurizer",
    "Chi4n_Featurizer",
    "Chi4v_Featurizer",
    "EState_VSA1_Featurizer",
    "EState_VSA10_Featurizer",
    "EState_VSA11_Featurizer",
    "EState_VSA2_Featurizer",
    "EState_VSA3_Featurizer",
    "EState_VSA4_Featurizer",
    "EState_VSA5_Featurizer",
    "EState_VSA6_Featurizer",
    "EState_VSA7_Featurizer",
    "EState_VSA8_Featurizer",
    "EState_VSA9_Featurizer",
    "Eccentricity_Featurizer",
    "ExactMolWt_Featurizer",
    "FormalCharge_Featurizer",
    "FpDensityMorgan1_Featurizer",
    "FpDensityMorgan2_Featurizer",
    "FpDensityMorgan3_Featurizer",
    "FractionCSP3_Featurizer",
    "HallKierAlpha_Featurizer",
    "HeavyAtomCount_Featurizer",
    "HeavyAtomMolWt_Featurizer",
    "InertialShapeFactor_Featurizer",
    "Ipc_Featurizer",
    "Kappa1_Featurizer",
    "Kappa2_Featurizer",
    "Kappa3_Featurizer",
    "LabuteASA_Featurizer",
    "MaxAbsEStateIndex_Featurizer",
    "MaxAbsPartialCharge_Featurizer",
    "MaxEStateIndex_Featurizer",
    "MaxPartialCharge_Featurizer",
    "MinAbsEStateIndex_Featurizer",
    "MinAbsPartialCharge_Featurizer",
    "MinEStateIndex_Featurizer",
    "MinPartialCharge_Featurizer",
    "MolLogP_Featurizer",
    "MolMR_Featurizer",
    "MolWt_Featurizer",
    "NHOHCount_Featurizer",
    "NOCount_Featurizer",
    "NPR1_Featurizer",
    "NPR2_Featurizer",
    "NumAliphaticCarbocycles_Featurizer",
    "NumAliphaticHeterocycles_Featurizer",
    "NumAliphaticRings_Featurizer",
    "NumAmideBonds_Featurizer",
    "NumAromaticCarbocycles_Featurizer",
    "NumAromaticHeterocycles_Featurizer",
    "NumAromaticRings_Featurizer",
    "NumAtoms_Featurizer",
    "NumBonds_Featurizer",
    "NumBridgeheadAtoms_Featurizer",
    "NumHAcceptors_Featurizer",
    "NumHBA_Featurizer",
    "NumHBD_Featurizer",
    "NumHDonors_Featurizer",
    "NumHeavyAtoms_Featurizer",
    "NumHeteroatoms_Featurizer",
    "NumHeterocycles_Featurizer",
    "NumLipinskiHBA_Featurizer",
    "NumLipinskiHBD_Featurizer",
    "NumRadicalElectrons_Featurizer",
    "NumRings_Featurizer",
    "NumRotatableBonds_Featurizer",
    "NumSaturatedCarbocycles_Featurizer",
    "NumSaturatedHeterocycles_Featurizer",
    "NumSaturatedRings_Featurizer",
    "NumSpiroAtoms_Featurizer",
    "NumValenceElectrons_Featurizer",
    "NumberAtomsAc_Featurizer",
    "NumberAtomsAg_Featurizer",
    "NumberAtomsAl_Featurizer",
    "NumberAtomsAm_Featurizer",
    "NumberAtomsAr_Featurizer",
    "NumberAtomsAs_Featurizer",
    "NumberAtomsAt_Featurizer",
    "NumberAtomsAu_Featurizer",
    "NumberAtomsB_Featurizer",
    "NumberAtomsBa_Featurizer",
    "NumberAtomsBe_Featurizer",
    "NumberAtomsBh_Featurizer",
    "NumberAtomsBi_Featurizer",
    "NumberAtomsBk_Featurizer",
    "NumberAtomsBr_Featurizer",
    "NumberAtomsC_Featurizer",
    "NumberAtomsCa_Featurizer",
    "NumberAtomsCd_Featurizer",
    "NumberAtomsCe_Featurizer",
    "NumberAtomsCf_Featurizer",
    "NumberAtomsCl_Featurizer",
    "NumberAtomsCm_Featurizer",
    "NumberAtomsCn_Featurizer",
    "NumberAtomsCo_Featurizer",
    "NumberAtomsCr_Featurizer",
    "NumberAtomsCs_Featurizer",
    "NumberAtomsCu_Featurizer",
    "NumberAtomsDb_Featurizer",
    "NumberAtomsDs_Featurizer",
    "NumberAtomsDy_Featurizer",
    "NumberAtomsEr_Featurizer",
    "NumberAtomsEs_Featurizer",
    "NumberAtomsEu_Featurizer",
    "NumberAtomsF_Featurizer",
    "NumberAtomsFe_Featurizer",
    "NumberAtomsFl_Featurizer",
    "NumberAtomsFm_Featurizer",
    "NumberAtomsFr_Featurizer",
    "NumberAtomsGa_Featurizer",
    "NumberAtomsGd_Featurizer",
    "NumberAtomsGe_Featurizer",
    "NumberAtomsH_Featurizer",
    "NumberAtomsHe_Featurizer",
    "NumberAtomsHf_Featurizer",
    "NumberAtomsHg_Featurizer",
    "NumberAtomsHo_Featurizer",
    "NumberAtomsHs_Featurizer",
    "NumberAtomsI_Featurizer",
    "NumberAtomsIn_Featurizer",
    "NumberAtomsIr_Featurizer",
    "NumberAtomsK_Featurizer",
    "NumberAtomsKr_Featurizer",
    "NumberAtomsLa_Featurizer",
    "NumberAtomsLi_Featurizer",
    "NumberAtomsLr_Featurizer",
    "NumberAtomsLu_Featurizer",
    "NumberAtomsLv_Featurizer",
    "NumberAtomsMc_Featurizer",
    "NumberAtomsMd_Featurizer",
    "NumberAtomsMg_Featurizer",
    "NumberAtomsMn_Featurizer",
    "NumberAtomsMo_Featurizer",
    "NumberAtomsMt_Featurizer",
    "NumberAtomsN_Featurizer",
    "NumberAtomsNa_Featurizer",
    "NumberAtomsNb_Featurizer",
    "NumberAtomsNd_Featurizer",
    "NumberAtomsNe_Featurizer",
    "NumberAtomsNh_Featurizer",
    "NumberAtomsNi_Featurizer",
    "NumberAtomsNo_Featurizer",
    "NumberAtomsNp_Featurizer",
    "NumberAtomsO_Featurizer",
    "NumberAtomsOg_Featurizer",
    "NumberAtomsOs_Featurizer",
    "NumberAtomsP_Featurizer",
    "NumberAtomsPa_Featurizer",
    "NumberAtomsPb_Featurizer",
    "NumberAtomsPd_Featurizer",
    "NumberAtomsPm_Featurizer",
    "NumberAtomsPo_Featurizer",
    "NumberAtomsPr_Featurizer",
    "NumberAtomsPt_Featurizer",
    "NumberAtomsPu_Featurizer",
    "NumberAtomsRa_Featurizer",
    "NumberAtomsRb_Featurizer",
    "NumberAtomsRe_Featurizer",
    "NumberAtomsRf_Featurizer",
    "NumberAtomsRg_Featurizer",
    "NumberAtomsRh_Featurizer",
    "NumberAtomsRn_Featurizer",
    "NumberAtomsRu_Featurizer",
    "NumberAtomsS_Featurizer",
    "NumberAtomsSb_Featurizer",
    "NumberAtomsSc_Featurizer",
    "NumberAtomsSe_Featurizer",
    "NumberAtomsSg_Featurizer",
    "NumberAtomsSi_Featurizer",
    "NumberAtomsSm_Featurizer",
    "NumberAtomsSn_Featurizer",
    "NumberAtomsSr_Featurizer",
    "NumberAtomsTa_Featurizer",
    "NumberAtomsTb_Featurizer",
    "NumberAtomsTc_Featurizer",
    "NumberAtomsTe_Featurizer",
    "NumberAtomsTh_Featurizer",
    "NumberAtomsTi_Featurizer",
    "NumberAtomsTl_Featurizer",
    "NumberAtomsTm_Featurizer",
    "NumberAtomsTs_Featurizer",
    "NumberAtomsU_Featurizer",
    "NumberAtomsV_Featurizer",
    "NumberAtomsW_Featurizer",
    "NumberAtomsXe_Featurizer",
    "NumberAtomsY_Featurizer",
    "NumberAtomsYb_Featurizer",
    "NumberAtomsZn_Featurizer",
    "NumberAtomsZr_Featurizer",
    "PBF_Featurizer",
    "PEOE_VSA1_Featurizer",
    "PEOE_VSA10_Featurizer",
    "PEOE_VSA11_Featurizer",
    "PEOE_VSA12_Featurizer",
    "PEOE_VSA13_Featurizer",
    "PEOE_VSA14_Featurizer",
    "PEOE_VSA2_Featurizer",
    "PEOE_VSA3_Featurizer",
    "PEOE_VSA4_Featurizer",
    "PEOE_VSA5_Featurizer",
    "PEOE_VSA6_Featurizer",
    "PEOE_VSA7_Featurizer",
    "PEOE_VSA8_Featurizer",
    "PEOE_VSA9_Featurizer",
    "PMI1_Featurizer",
    "PMI2_Featurizer",
    "PMI3_Featurizer",
    "Phi_Featurizer",
    "RadiusOfGyration_Featurizer",
    "RelativeContentAc_Featurizer",
    "RelativeContentAg_Featurizer",
    "RelativeContentAl_Featurizer",
    "RelativeContentAm_Featurizer",
    "RelativeContentAr_Featurizer",
    "RelativeContentAs_Featurizer",
    "RelativeContentAt_Featurizer",
    "RelativeContentAu_Featurizer",
    "RelativeContentB_Featurizer",
    "RelativeContentBa_Featurizer",
    "RelativeContentBe_Featurizer",
    "RelativeContentBh_Featurizer",
    "RelativeContentBi_Featurizer",
    "RelativeContentBk_Featurizer",
    "RelativeContentBr_Featurizer",
    "RelativeContentC_Featurizer",
    "RelativeContentCa_Featurizer",
    "RelativeContentCd_Featurizer",
    "RelativeContentCe_Featurizer",
    "RelativeContentCf_Featurizer",
    "RelativeContentCl_Featurizer",
    "RelativeContentCm_Featurizer",
    "RelativeContentCn_Featurizer",
    "RelativeContentCo_Featurizer",
    "RelativeContentCr_Featurizer",
    "RelativeContentCs_Featurizer",
    "RelativeContentCu_Featurizer",
    "RelativeContentDb_Featurizer",
    "RelativeContentDs_Featurizer",
    "RelativeContentDy_Featurizer",
    "RelativeContentEr_Featurizer",
    "RelativeContentEs_Featurizer",
    "RelativeContentEu_Featurizer",
    "RelativeContentF_Featurizer",
    "RelativeContentFe_Featurizer",
    "RelativeContentFl_Featurizer",
    "RelativeContentFm_Featurizer",
    "RelativeContentFr_Featurizer",
    "RelativeContentGa_Featurizer",
    "RelativeContentGd_Featurizer",
    "RelativeContentGe_Featurizer",
    "RelativeContentH_Featurizer",
    "RelativeContentHe_Featurizer",
    "RelativeContentHf_Featurizer",
    "RelativeContentHg_Featurizer",
    "RelativeContentHo_Featurizer",
    "RelativeContentHs_Featurizer",
    "RelativeContentI_Featurizer",
    "RelativeContentIn_Featurizer",
    "RelativeContentIr_Featurizer",
    "RelativeContentK_Featurizer",
    "RelativeContentKr_Featurizer",
    "RelativeContentLa_Featurizer",
    "RelativeContentLi_Featurizer",
    "RelativeContentLr_Featurizer",
    "RelativeContentLu_Featurizer",
    "RelativeContentLv_Featurizer",
    "RelativeContentMc_Featurizer",
    "RelativeContentMd_Featurizer",
    "RelativeContentMg_Featurizer",
    "RelativeContentMn_Featurizer",
    "RelativeContentMo_Featurizer",
    "RelativeContentMt_Featurizer",
    "RelativeContentN_Featurizer",
    "RelativeContentNa_Featurizer",
    "RelativeContentNb_Featurizer",
    "RelativeContentNd_Featurizer",
    "RelativeContentNe_Featurizer",
    "RelativeContentNh_Featurizer",
    "RelativeContentNi_Featurizer",
    "RelativeContentNo_Featurizer",
    "RelativeContentNp_Featurizer",
    "RelativeContentO_Featurizer",
    "RelativeContentOg_Featurizer",
    "RelativeContentOs_Featurizer",
    "RelativeContentP_Featurizer",
    "RelativeContentPa_Featurizer",
    "RelativeContentPb_Featurizer",
    "RelativeContentPd_Featurizer",
    "RelativeContentPm_Featurizer",
    "RelativeContentPo_Featurizer",
    "RelativeContentPr_Featurizer",
    "RelativeContentPt_Featurizer",
    "RelativeContentPu_Featurizer",
    "RelativeContentRa_Featurizer",
    "RelativeContentRb_Featurizer",
    "RelativeContentRe_Featurizer",
    "RelativeContentRf_Featurizer",
    "RelativeContentRg_Featurizer",
    "RelativeContentRh_Featurizer",
    "RelativeContentRn_Featurizer",
    "RelativeContentRu_Featurizer",
    "RelativeContentS_Featurizer",
    "RelativeContentSb_Featurizer",
    "RelativeContentSc_Featurizer",
    "RelativeContentSe_Featurizer",
    "RelativeContentSg_Featurizer",
    "RelativeContentSi_Featurizer",
    "RelativeContentSm_Featurizer",
    "RelativeContentSn_Featurizer",
    "RelativeContentSr_Featurizer",
    "RelativeContentTa_Featurizer",
    "RelativeContentTb_Featurizer",
    "RelativeContentTc_Featurizer",
    "RelativeContentTe_Featurizer",
    "RelativeContentTh_Featurizer",
    "RelativeContentTi_Featurizer",
    "RelativeContentTl_Featurizer",
    "RelativeContentTm_Featurizer",
    "RelativeContentTs_Featurizer",
    "RelativeContentU_Featurizer",
    "RelativeContentV_Featurizer",
    "RelativeContentW_Featurizer",
    "RelativeContentXe_Featurizer",
    "RelativeContentY_Featurizer",
    "RelativeContentYb_Featurizer",
    "RelativeContentZn_Featurizer",
    "RelativeContentZr_Featurizer",
    "RingCount_Featurizer",
    "SMR_VSA1_Featurizer",
    "SMR_VSA10_Featurizer",
    "SMR_VSA2_Featurizer",
    "SMR_VSA3_Featurizer",
    "SMR_VSA4_Featurizer",
    "SMR_VSA5_Featurizer",
    "SMR_VSA6_Featurizer",
    "SMR_VSA7_Featurizer",
    "SMR_VSA8_Featurizer",
    "SMR_VSA9_Featurizer",
    "SSSR_Featurizer",
    "SlogP_VSA1_Featurizer",
    "SlogP_VSA10_Featurizer",
    "SlogP_VSA11_Featurizer",
    "SlogP_VSA12_Featurizer",
    "SlogP_VSA2_Featurizer",
    "SlogP_VSA3_Featurizer",
    "SlogP_VSA4_Featurizer",
    "SlogP_VSA5_Featurizer",
    "SlogP_VSA6_Featurizer",
    "SlogP_VSA7_Featurizer",
    "SlogP_VSA8_Featurizer",
    "SlogP_VSA9_Featurizer",
    "SpherocityIndex_Featurizer",
    "TPSA_Featurizer",
    "VSA_EState1_Featurizer",
    "VSA_EState10_Featurizer",
    "VSA_EState2_Featurizer",
    "VSA_EState3_Featurizer",
    "VSA_EState4_Featurizer",
    "VSA_EState5_Featurizer",
    "VSA_EState6_Featurizer",
    "VSA_EState7_Featurizer",
    "VSA_EState8_Featurizer",
    "VSA_EState9_Featurizer",
    "fr_Al_COO_Featurizer",
    "fr_Al_OH_Featurizer",
    "fr_Al_OH_noTert_Featurizer",
    "fr_ArN_Featurizer",
    "fr_Ar_COO_Featurizer",
    "fr_Ar_N_Featurizer",
    "fr_Ar_NH_Featurizer",
    "fr_Ar_OH_Featurizer",
    "fr_COO_Featurizer",
    "fr_COO2_Featurizer",
    "fr_C_O_Featurizer",
    "fr_C_O_noCOO_Featurizer",
    "fr_C_S_Featurizer",
    "fr_HOCCN_Featurizer",
    "fr_Imine_Featurizer",
    "fr_NH0_Featurizer",
    "fr_NH1_Featurizer",
    "fr_NH2_Featurizer",
    "fr_N_O_Featurizer",
    "fr_Ndealkylation1_Featurizer",
    "fr_Ndealkylation2_Featurizer",
    "fr_Nhpyrrole_Featurizer",
    "fr_SH_Featurizer",
    "fr_aldehyde_Featurizer",
    "fr_alkyl_carbamate_Featurizer",
    "fr_alkyl_halide_Featurizer",
    "fr_allylic_oxid_Featurizer",
    "fr_amide_Featurizer",
    "fr_amidine_Featurizer",
    "fr_aniline_Featurizer",
    "fr_aryl_methyl_Featurizer",
    "fr_azide_Featurizer",
    "fr_azo_Featurizer",
    "fr_barbitur_Featurizer",
    "fr_benzene_Featurizer",
    "fr_benzodiazepine_Featurizer",
    "fr_bicyclic_Featurizer",
    "fr_diazo_Featurizer",
    "fr_dihydropyridine_Featurizer",
    "fr_epoxide_Featurizer",
    "fr_ester_Featurizer",
    "fr_ether_Featurizer",
    "fr_furan_Featurizer",
    "fr_guanido_Featurizer",
    "fr_halogen_Featurizer",
    "fr_hdrzine_Featurizer",
    "fr_hdrzone_Featurizer",
    "fr_imidazole_Featurizer",
    "fr_imide_Featurizer",
    "fr_isocyan_Featurizer",
    "fr_isothiocyan_Featurizer",
    "fr_ketone_Featurizer",
    "fr_ketone_Topliss_Featurizer",
    "fr_lactam_Featurizer",
    "fr_lactone_Featurizer",
    "fr_methoxy_Featurizer",
    "fr_morpholine_Featurizer",
    "fr_nitrile_Featurizer",
    "fr_nitro_Featurizer",
    "fr_nitro_arom_Featurizer",
    "fr_nitro_arom_nonortho_Featurizer",
    "fr_nitroso_Featurizer",
    "fr_oxazole_Featurizer",
    "fr_oxime_Featurizer",
    "fr_para_hydroxylation_Featurizer",
    "fr_phenol_Featurizer",
    "fr_phenol_noOrthoHbond_Featurizer",
    "fr_phos_acid_Featurizer",
    "fr_phos_ester_Featurizer",
    "fr_piperdine_Featurizer",
    "fr_piperzine_Featurizer",
    "fr_priamide_Featurizer",
    "fr_prisulfonamd_Featurizer",
    "fr_pyridine_Featurizer",
    "fr_quatN_Featurizer",
    "fr_sulfide_Featurizer",
    "fr_sulfonamd_Featurizer",
    "fr_sulfone_Featurizer",
    "fr_term_acetylene_Featurizer",
    "fr_tetrazole_Featurizer",
    "fr_thiazole_Featurizer",
    "fr_thiocyan_Featurizer",
    "fr_thiophene_Featurizer",
    "fr_unbrch_alkane_Featurizer",
    "fr_urea_Featurizer",
    "qed_Featurizer",
    "molecule_Asphericity",
    "molecule_BCUT2D_CHGHI",
    "molecule_BCUT2D_CHGLO",
    "molecule_BCUT2D_LOGPHI",
    "molecule_BCUT2D_LOGPLOW",
    "molecule_BCUT2D_MRHI",
    "molecule_BCUT2D_MRLOW",
    "molecule_BCUT2D_MWHI",
    "molecule_BCUT2D_MWLOW",
    "molecule_BalabanJ",
    "molecule_BertzCT",
    "molecule_Chi0",
    "molecule_Chi0n",
    "molecule_Chi0v",
    "molecule_Chi1",
    "molecule_Chi1n",
    "molecule_Chi1v",
    "molecule_Chi2n",
    "molecule_Chi2v",
    "molecule_Chi3n",
    "molecule_Chi3v",
    "molecule_Chi4n",
    "molecule_Chi4v",
    "molecule_EState_VSA1",
    "molecule_EState_VSA10",
    "molecule_EState_VSA11",
    "molecule_EState_VSA2",
    "molecule_EState_VSA3",
    "molecule_EState_VSA4",
    "molecule_EState_VSA5",
    "molecule_EState_VSA6",
    "molecule_EState_VSA7",
    "molecule_EState_VSA8",
    "molecule_EState_VSA9",
    "molecule_Eccentricity",
    "molecule_ExactMolWt",
    "molecule_FormalCharge",
    "molecule_FpDensityMorgan1",
    "molecule_FpDensityMorgan2",
    "molecule_FpDensityMorgan3",
    "molecule_FractionCSP3",
    "molecule_HallKierAlpha",
    "molecule_HeavyAtomCount",
    "molecule_HeavyAtomMolWt",
    "molecule_InertialShapeFactor",
    "molecule_Ipc",
    "molecule_Kappa1",
    "molecule_Kappa2",
    "molecule_Kappa3",
    "molecule_LabuteASA",
    "molecule_MaxAbsEStateIndex",
    "molecule_MaxAbsPartialCharge",
    "molecule_MaxEStateIndex",
    "molecule_MaxPartialCharge",
    "molecule_MinAbsEStateIndex",
    "molecule_MinAbsPartialCharge",
    "molecule_MinEStateIndex",
    "molecule_MinPartialCharge",
    "molecule_MolLogP",
    "molecule_MolMR",
    "molecule_MolWt",
    "molecule_NHOHCount",
    "molecule_NOCount",
    "molecule_NPR1",
    "molecule_NPR2",
    "molecule_NumAliphaticCarbocycles",
    "molecule_NumAliphaticHeterocycles",
    "molecule_NumAliphaticRings",
    "molecule_NumAmideBonds",
    "molecule_NumAromaticCarbocycles",
    "molecule_NumAromaticHeterocycles",
    "molecule_NumAromaticRings",
    "molecule_NumAtoms",
    "molecule_NumBonds",
    "molecule_NumBridgeheadAtoms",
    "molecule_NumHAcceptors",
    "molecule_NumHBA",
    "molecule_NumHBD",
    "molecule_NumHDonors",
    "molecule_NumHeavyAtoms",
    "molecule_NumHeteroatoms",
    "molecule_NumHeterocycles",
    "molecule_NumLipinskiHBA",
    "molecule_NumLipinskiHBD",
    "molecule_NumRadicalElectrons",
    "molecule_NumRings",
    "molecule_NumRotatableBonds",
    "molecule_NumSaturatedCarbocycles",
    "molecule_NumSaturatedHeterocycles",
    "molecule_NumSaturatedRings",
    "molecule_NumSpiroAtoms",
    "molecule_NumValenceElectrons",
    "molecule_NumberAtomsAc",
    "molecule_NumberAtomsAg",
    "molecule_NumberAtomsAl",
    "molecule_NumberAtomsAm",
    "molecule_NumberAtomsAr",
    "molecule_NumberAtomsAs",
    "molecule_NumberAtomsAt",
    "molecule_NumberAtomsAu",
    "molecule_NumberAtomsB",
    "molecule_NumberAtomsBa",
    "molecule_NumberAtomsBe",
    "molecule_NumberAtomsBh",
    "molecule_NumberAtomsBi",
    "molecule_NumberAtomsBk",
    "molecule_NumberAtomsBr",
    "molecule_NumberAtomsC",
    "molecule_NumberAtomsCa",
    "molecule_NumberAtomsCd",
    "molecule_NumberAtomsCe",
    "molecule_NumberAtomsCf",
    "molecule_NumberAtomsCl",
    "molecule_NumberAtomsCm",
    "molecule_NumberAtomsCn",
    "molecule_NumberAtomsCo",
    "molecule_NumberAtomsCr",
    "molecule_NumberAtomsCs",
    "molecule_NumberAtomsCu",
    "molecule_NumberAtomsDb",
    "molecule_NumberAtomsDs",
    "molecule_NumberAtomsDy",
    "molecule_NumberAtomsEr",
    "molecule_NumberAtomsEs",
    "molecule_NumberAtomsEu",
    "molecule_NumberAtomsF",
    "molecule_NumberAtomsFe",
    "molecule_NumberAtomsFl",
    "molecule_NumberAtomsFm",
    "molecule_NumberAtomsFr",
    "molecule_NumberAtomsGa",
    "molecule_NumberAtomsGd",
    "molecule_NumberAtomsGe",
    "molecule_NumberAtomsH",
    "molecule_NumberAtomsHe",
    "molecule_NumberAtomsHf",
    "molecule_NumberAtomsHg",
    "molecule_NumberAtomsHo",
    "molecule_NumberAtomsHs",
    "molecule_NumberAtomsI",
    "molecule_NumberAtomsIn",
    "molecule_NumberAtomsIr",
    "molecule_NumberAtomsK",
    "molecule_NumberAtomsKr",
    "molecule_NumberAtomsLa",
    "molecule_NumberAtomsLi",
    "molecule_NumberAtomsLr",
    "molecule_NumberAtomsLu",
    "molecule_NumberAtomsLv",
    "molecule_NumberAtomsMc",
    "molecule_NumberAtomsMd",
    "molecule_NumberAtomsMg",
    "molecule_NumberAtomsMn",
    "molecule_NumberAtomsMo",
    "molecule_NumberAtomsMt",
    "molecule_NumberAtomsN",
    "molecule_NumberAtomsNa",
    "molecule_NumberAtomsNb",
    "molecule_NumberAtomsNd",
    "molecule_NumberAtomsNe",
    "molecule_NumberAtomsNh",
    "molecule_NumberAtomsNi",
    "molecule_NumberAtomsNo",
    "molecule_NumberAtomsNp",
    "molecule_NumberAtomsO",
    "molecule_NumberAtomsOg",
    "molecule_NumberAtomsOs",
    "molecule_NumberAtomsP",
    "molecule_NumberAtomsPa",
    "molecule_NumberAtomsPb",
    "molecule_NumberAtomsPd",
    "molecule_NumberAtomsPm",
    "molecule_NumberAtomsPo",
    "molecule_NumberAtomsPr",
    "molecule_NumberAtomsPt",
    "molecule_NumberAtomsPu",
    "molecule_NumberAtomsRa",
    "molecule_NumberAtomsRb",
    "molecule_NumberAtomsRe",
    "molecule_NumberAtomsRf",
    "molecule_NumberAtomsRg",
    "molecule_NumberAtomsRh",
    "molecule_NumberAtomsRn",
    "molecule_NumberAtomsRu",
    "molecule_NumberAtomsS",
    "molecule_NumberAtomsSb",
    "molecule_NumberAtomsSc",
    "molecule_NumberAtomsSe",
    "molecule_NumberAtomsSg",
    "molecule_NumberAtomsSi",
    "molecule_NumberAtomsSm",
    "molecule_NumberAtomsSn",
    "molecule_NumberAtomsSr",
    "molecule_NumberAtomsTa",
    "molecule_NumberAtomsTb",
    "molecule_NumberAtomsTc",
    "molecule_NumberAtomsTe",
    "molecule_NumberAtomsTh",
    "molecule_NumberAtomsTi",
    "molecule_NumberAtomsTl",
    "molecule_NumberAtomsTm",
    "molecule_NumberAtomsTs",
    "molecule_NumberAtomsU",
    "molecule_NumberAtomsV",
    "molecule_NumberAtomsW",
    "molecule_NumberAtomsXe",
    "molecule_NumberAtomsY",
    "molecule_NumberAtomsYb",
    "molecule_NumberAtomsZn",
    "molecule_NumberAtomsZr",
    "molecule_PBF",
    "molecule_PEOE_VSA1",
    "molecule_PEOE_VSA10",
    "molecule_PEOE_VSA11",
    "molecule_PEOE_VSA12",
    "molecule_PEOE_VSA13",
    "molecule_PEOE_VSA14",
    "molecule_PEOE_VSA2",
    "molecule_PEOE_VSA3",
    "molecule_PEOE_VSA4",
    "molecule_PEOE_VSA5",
    "molecule_PEOE_VSA6",
    "molecule_PEOE_VSA7",
    "molecule_PEOE_VSA8",
    "molecule_PEOE_VSA9",
    "molecule_PMI1",
    "molecule_PMI2",
    "molecule_PMI3",
    "molecule_Phi",
    "molecule_RadiusOfGyration",
    "molecule_RelativeContentAc",
    "molecule_RelativeContentAg",
    "molecule_RelativeContentAl",
    "molecule_RelativeContentAm",
    "molecule_RelativeContentAr",
    "molecule_RelativeContentAs",
    "molecule_RelativeContentAt",
    "molecule_RelativeContentAu",
    "molecule_RelativeContentB",
    "molecule_RelativeContentBa",
    "molecule_RelativeContentBe",
    "molecule_RelativeContentBh",
    "molecule_RelativeContentBi",
    "molecule_RelativeContentBk",
    "molecule_RelativeContentBr",
    "molecule_RelativeContentC",
    "molecule_RelativeContentCa",
    "molecule_RelativeContentCd",
    "molecule_RelativeContentCe",
    "molecule_RelativeContentCf",
    "molecule_RelativeContentCl",
    "molecule_RelativeContentCm",
    "molecule_RelativeContentCn",
    "molecule_RelativeContentCo",
    "molecule_RelativeContentCr",
    "molecule_RelativeContentCs",
    "molecule_RelativeContentCu",
    "molecule_RelativeContentDb",
    "molecule_RelativeContentDs",
    "molecule_RelativeContentDy",
    "molecule_RelativeContentEr",
    "molecule_RelativeContentEs",
    "molecule_RelativeContentEu",
    "molecule_RelativeContentF",
    "molecule_RelativeContentFe",
    "molecule_RelativeContentFl",
    "molecule_RelativeContentFm",
    "molecule_RelativeContentFr",
    "molecule_RelativeContentGa",
    "molecule_RelativeContentGd",
    "molecule_RelativeContentGe",
    "molecule_RelativeContentH",
    "molecule_RelativeContentHe",
    "molecule_RelativeContentHf",
    "molecule_RelativeContentHg",
    "molecule_RelativeContentHo",
    "molecule_RelativeContentHs",
    "molecule_RelativeContentI",
    "molecule_RelativeContentIn",
    "molecule_RelativeContentIr",
    "molecule_RelativeContentK",
    "molecule_RelativeContentKr",
    "molecule_RelativeContentLa",
    "molecule_RelativeContentLi",
    "molecule_RelativeContentLr",
    "molecule_RelativeContentLu",
    "molecule_RelativeContentLv",
    "molecule_RelativeContentMc",
    "molecule_RelativeContentMd",
    "molecule_RelativeContentMg",
    "molecule_RelativeContentMn",
    "molecule_RelativeContentMo",
    "molecule_RelativeContentMt",
    "molecule_RelativeContentN",
    "molecule_RelativeContentNa",
    "molecule_RelativeContentNb",
    "molecule_RelativeContentNd",
    "molecule_RelativeContentNe",
    "molecule_RelativeContentNh",
    "molecule_RelativeContentNi",
    "molecule_RelativeContentNo",
    "molecule_RelativeContentNp",
    "molecule_RelativeContentO",
    "molecule_RelativeContentOg",
    "molecule_RelativeContentOs",
    "molecule_RelativeContentP",
    "molecule_RelativeContentPa",
    "molecule_RelativeContentPb",
    "molecule_RelativeContentPd",
    "molecule_RelativeContentPm",
    "molecule_RelativeContentPo",
    "molecule_RelativeContentPr",
    "molecule_RelativeContentPt",
    "molecule_RelativeContentPu",
    "molecule_RelativeContentRa",
    "molecule_RelativeContentRb",
    "molecule_RelativeContentRe",
    "molecule_RelativeContentRf",
    "molecule_RelativeContentRg",
    "molecule_RelativeContentRh",
    "molecule_RelativeContentRn",
    "molecule_RelativeContentRu",
    "molecule_RelativeContentS",
    "molecule_RelativeContentSb",
    "molecule_RelativeContentSc",
    "molecule_RelativeContentSe",
    "molecule_RelativeContentSg",
    "molecule_RelativeContentSi",
    "molecule_RelativeContentSm",
    "molecule_RelativeContentSn",
    "molecule_RelativeContentSr",
    "molecule_RelativeContentTa",
    "molecule_RelativeContentTb",
    "molecule_RelativeContentTc",
    "molecule_RelativeContentTe",
    "molecule_RelativeContentTh",
    "molecule_RelativeContentTi",
    "molecule_RelativeContentTl",
    "molecule_RelativeContentTm",
    "molecule_RelativeContentTs",
    "molecule_RelativeContentU",
    "molecule_RelativeContentV",
    "molecule_RelativeContentW",
    "molecule_RelativeContentXe",
    "molecule_RelativeContentY",
    "molecule_RelativeContentYb",
    "molecule_RelativeContentZn",
    "molecule_RelativeContentZr",
    "molecule_RingCount",
    "molecule_SMR_VSA1",
    "molecule_SMR_VSA10",
    "molecule_SMR_VSA2",
    "molecule_SMR_VSA3",
    "molecule_SMR_VSA4",
    "molecule_SMR_VSA5",
    "molecule_SMR_VSA6",
    "molecule_SMR_VSA7",
    "molecule_SMR_VSA8",
    "molecule_SMR_VSA9",
    "molecule_SSSR",
    "molecule_SlogP_VSA1",
    "molecule_SlogP_VSA10",
    "molecule_SlogP_VSA11",
    "molecule_SlogP_VSA12",
    "molecule_SlogP_VSA2",
    "molecule_SlogP_VSA3",
    "molecule_SlogP_VSA4",
    "molecule_SlogP_VSA5",
    "molecule_SlogP_VSA6",
    "molecule_SlogP_VSA7",
    "molecule_SlogP_VSA8",
    "molecule_SlogP_VSA9",
    "molecule_SpherocityIndex",
    "molecule_TPSA",
    "molecule_VSA_EState1",
    "molecule_VSA_EState10",
    "molecule_VSA_EState2",
    "molecule_VSA_EState3",
    "molecule_VSA_EState4",
    "molecule_VSA_EState5",
    "molecule_VSA_EState6",
    "molecule_VSA_EState7",
    "molecule_VSA_EState8",
    "molecule_VSA_EState9",
    "molecule_fr_Al_COO",
    "molecule_fr_Al_OH",
    "molecule_fr_Al_OH_noTert",
    "molecule_fr_ArN",
    "molecule_fr_Ar_COO",
    "molecule_fr_Ar_N",
    "molecule_fr_Ar_NH",
    "molecule_fr_Ar_OH",
    "molecule_fr_COO",
    "molecule_fr_COO2",
    "molecule_fr_C_O",
    "molecule_fr_C_O_noCOO",
    "molecule_fr_C_S",
    "molecule_fr_HOCCN",
    "molecule_fr_Imine",
    "molecule_fr_NH0",
    "molecule_fr_NH1",
    "molecule_fr_NH2",
    "molecule_fr_N_O",
    "molecule_fr_Ndealkylation1",
    "molecule_fr_Ndealkylation2",
    "molecule_fr_Nhpyrrole",
    "molecule_fr_SH",
    "molecule_fr_aldehyde",
    "molecule_fr_alkyl_carbamate",
    "molecule_fr_alkyl_halide",
    "molecule_fr_allylic_oxid",
    "molecule_fr_amide",
    "molecule_fr_amidine",
    "molecule_fr_aniline",
    "molecule_fr_aryl_methyl",
    "molecule_fr_azide",
    "molecule_fr_azo",
    "molecule_fr_barbitur",
    "molecule_fr_benzene",
    "molecule_fr_benzodiazepine",
    "molecule_fr_bicyclic",
    "molecule_fr_diazo",
    "molecule_fr_dihydropyridine",
    "molecule_fr_epoxide",
    "molecule_fr_ester",
    "molecule_fr_ether",
    "molecule_fr_furan",
    "molecule_fr_guanido",
    "molecule_fr_halogen",
    "molecule_fr_hdrzine",
    "molecule_fr_hdrzone",
    "molecule_fr_imidazole",
    "molecule_fr_imide",
    "molecule_fr_isocyan",
    "molecule_fr_isothiocyan",
    "molecule_fr_ketone",
    "molecule_fr_ketone_Topliss",
    "molecule_fr_lactam",
    "molecule_fr_lactone",
    "molecule_fr_methoxy",
    "molecule_fr_morpholine",
    "molecule_fr_nitrile",
    "molecule_fr_nitro",
    "molecule_fr_nitro_arom",
    "molecule_fr_nitro_arom_nonortho",
    "molecule_fr_nitroso",
    "molecule_fr_oxazole",
    "molecule_fr_oxime",
    "molecule_fr_para_hydroxylation",
    "molecule_fr_phenol",
    "molecule_fr_phenol_noOrthoHbond",
    "molecule_fr_phos_acid",
    "molecule_fr_phos_ester",
    "molecule_fr_piperdine",
    "molecule_fr_piperzine",
    "molecule_fr_priamide",
    "molecule_fr_prisulfonamd",
    "molecule_fr_pyridine",
    "molecule_fr_quatN",
    "molecule_fr_sulfide",
    "molecule_fr_sulfonamd",
    "molecule_fr_sulfone",
    "molecule_fr_term_acetylene",
    "molecule_fr_tetrazole",
    "molecule_fr_thiazole",
    "molecule_fr_thiocyan",
    "molecule_fr_thiophene",
    "molecule_fr_unbrch_alkane",
    "molecule_fr_urea",
    "molecule_qed",
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for n, f in get_available_featurizer().items():
        print(n, f(testmol))


if __name__ == "__main__":
    main()
