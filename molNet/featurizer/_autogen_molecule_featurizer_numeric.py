from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,SingleValueMoleculeFeaturizer
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import (PEOE_VSA3,fr_bicyclic,fr_tetrazole,fr_SH,fr_oxime,PEOE_VSA2,fr_benzene,SlogP_VSA2,Chi3n,SlogP_VSA7,NumSaturatedHeterocycles,HeavyAtomMolWt,VSA_EState7,fr_thiophene,MinAbsEStateIndex,NumHAcceptors,SMR_VSA4,VSA_EState9,EState_VSA3,fr_isothiocyan,fr_piperzine,PEOE_VSA1,Chi2v,fr_nitro,fr_C_O_noCOO,VSA_EState2,fr_quatN,SlogP_VSA4,PEOE_VSA13,fr_hdrzine,fr_phenol,RingCount,PEOE_VSA10,fr_Imine,fr_Al_COO,fr_ester,fr_sulfide,fr_Ar_COO,fr_amide,BCUT2D_CHGHI,fr_sulfonamd,BCUT2D_LOGPLOW,SMR_VSA8,BCUT2D_LOGPHI,Chi0v,fr_alkyl_carbamate,SlogP_VSA11,Chi4v,fr_thiocyan,fr_Ndealkylation1,EState_VSA10,EState_VSA11,FpDensityMorgan1,NumAliphaticHeterocycles,fr_ketone_Topliss,NumAliphaticRings,fr_benzodiazepine,fr_C_S,fr_epoxide,Chi0n,SlogP_VSA1,fr_nitroso,PEOE_VSA4,SMR_VSA10,SlogP_VSA10,MinEStateIndex,VSA_EState4,fr_aniline,fr_furan,Kappa1,TPSA,MaxAbsEStateIndex,EState_VSA6,MinAbsPartialCharge,Chi2n,fr_morpholine,SlogP_VSA6,NumAromaticCarbocycles,fr_diazo,fr_hdrzone,NumAromaticRings,fr_nitro_arom,NumAromaticHeterocycles,NHOHCount,fr_guanido,EState_VSA9,NumSaturatedRings,PEOE_VSA11,SMR_VSA7,PEOE_VSA7,SMR_VSA1,fr_phenol_noOrthoHbond,Ipc,fr_prisulfonamd,Chi3v,fr_term_acetylene,fr_halogen,fr_dihydropyridine,PEOE_VSA5,fr_piperdine,SlogP_VSA5,LabuteASA,Chi1v,VSA_EState6,EState_VSA8,SMR_VSA9,fr_oxazole,fr_alkyl_halide,fr_Nhpyrrole,Chi0,NumSaturatedCarbocycles,HallKierAlpha,fr_Ndealkylation2,NumHeteroatoms,fr_para_hydroxylation,SMR_VSA5,VSA_EState5,fr_ArN,fr_ether,fr_pyridine,fr_HOCCN,fr_sulfone,MaxAbsPartialCharge,NumAliphaticCarbocycles,Chi4n,fr_isocyan,fr_Al_OH,MaxPartialCharge,fr_NH2,VSA_EState3,fr_lactone,Kappa2,SMR_VSA6,Chi1n,BCUT2D_MWLOW,fr_nitrile,BCUT2D_MWHI,EState_VSA5,ExactMolWt,PEOE_VSA8,MolMR,fr_amidine,VSA_EState1,SlogP_VSA8,fr_azo,FractionCSP3,NumRotatableBonds,fr_NH0,PEOE_VSA6,SlogP_VSA12,BalabanJ,SlogP_VSA9,fr_aryl_methyl,qed,BCUT2D_MRHI,Chi1,FpDensityMorgan3,fr_lactam,fr_urea,FpDensityMorgan2,fr_priamide,NumValenceElectrons,PEOE_VSA9,NumHDonors,fr_methoxy,EState_VSA2,HeavyAtomCount,fr_azide,VSA_EState8,BCUT2D_CHGLO,NumRadicalElectrons,fr_nitro_arom_nonortho,fr_Ar_NH,SlogP_VSA3,EState_VSA4,VSA_EState10,MinPartialCharge,fr_aldehyde,BertzCT,Kappa3,fr_thiazole,fr_imidazole,fr_imide,fr_N_O,fr_allylic_oxid,fr_phos_ester,BCUT2D_MRLOW,EState_VSA7,fr_barbitur,fr_COO2,PEOE_VSA14,NOCount,MaxEStateIndex,fr_NH1,SMR_VSA3,fr_C_O,EState_VSA1,fr_ketone,fr_unbrch_alkane,fr_Ar_N,SMR_VSA2,fr_Ar_OH,fr_Al_OH_noTert,fr_phos_acid,fr_COO,PEOE_VSA12,MolLogP,MolWt,)
from rdkit.Chem.rdMolDescriptors import (CalcNumAromaticHeterocycles,CalcExactMolWt,CalcNumLipinskiHBD,CalcNPR2,CalcChi4v,CalcNumSaturatedHeterocycles,CalcChi2v,CalcRadiusOfGyration,CalcKappa1,CalcNumSpiroAtoms,CalcChi3v,CalcPBF,CalcNumAliphaticRings,CalcNumAromaticCarbocycles,CalcChi0n,CalcFractionCSP3,CalcChi1n,CalcKappa2,CalcNumSaturatedCarbocycles,CalcNumLipinskiHBA,CalcChi0v,CalcPMI2,CalcHallKierAlpha,CalcNumSaturatedRings,CalcNumRotatableBonds,CalcPMI3,CalcNumHeteroatoms,CalcLabuteASA,CalcNumAromaticRings,CalcChi3n,CalcPMI1,CalcEccentricity,CalcPhi,CalcNumRings,CalcNumAmideBonds,CalcNumAliphaticCarbocycles,CalcChi4n,CalcKappa3,CalcNumHeterocycles,CalcInertialShapeFactor,CalcTPSA,CalcChi1v,CalcNumBridgeheadAtoms,CalcAsphericity,CalcSpherocityIndex,CalcNumHBD,CalcChi2n,CalcNumHBA,CalcNPR1,CalcNumAliphaticHeterocycles,)
from rdkit.Chem.GraphDescriptors import (Ipc,Chi3n,Chi3v,Chi0n,Chi1v,BalabanJ,Chi1,Chi2v,Chi0,HallKierAlpha,Kappa1,Chi2n,BertzCT,Kappa3,Chi4n,Kappa2,Chi0v,Chi1n,Chi4v,)
from rdkit.Chem.Descriptors3D import (NPR1,NPR2,PMI2,PMI1,Eccentricity,PMI3,InertialShapeFactor,RadiusOfGyration,Asphericity,SpherocityIndex,)
from rdkit.Chem.rdmolops import (GetFormalCharge,GetSSSR,)


class  fr_nitro_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_nitro)
    # normalization
    linear_norm_parameter = (0.018333349998500004, 0.9621034106930377)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.62E-01,1.18E+00)
    min_max_norm_parameter = (5.1424742898646085e-09, 1.0199535961850923)  # error of 4.39E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.40538172903464703, 2.78525457245914)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.4053823801522207, 1.0, 2.7852526496619983)  # error of 1.23E-08 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.56E-01,1.00E+00)
    genlog_norm_parameter = (2.793633195929456, 0.4066263250703562, 0.2007103648054374, 1.8999963907274144)  # error of 4.74E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.74E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PEOE_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA8)
    # normalization
    linear_norm_parameter = (0.009235349219404814, 0.2024712092074698)  # error of 1.20E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.02E-01,1.69E+01)
    min_max_norm_parameter = (4.887388416091266, 65.78861410255793)  # error of 3.62E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (35.028275267622924, 0.07929972923744978)  # error of 1.39E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (5.85E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (33.9224654778114, 0.09599116002460914, 0.0717701519861949)  # error of 5.71E-03 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.71E-02,1.00E+00)
    genlog_norm_parameter = (0.06293662866979713, -3.9046608885740355, 1.5491722381461936, 0.18793636153672155)  # error of 1.81E-03 with sample range (0.00E+00,1.80E+03) resulting in fit range (1.46E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_Imine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Imine)
    # normalization
    linear_norm_parameter = (0.03889649931506156, 0.9186823185913269)  # error of 0.00E+00 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.19E-01,1.93E+00)
    min_max_norm_parameter = (5.062601797035858e-09, 1.0443004597486796)  # error of 1.04E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.23305480673382584, 2.52767371071209)  # error of 0.00E+00 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.43E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.23305480786468896, 1.0, 2.5276737069674944)  # error of 5.23E-11 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.43E-01,1.00E+00)
    genlog_norm_parameter = (2.5437691503221864, 0.46076932982623947, 0.3198414553822113, 1.7997421675799443)  # error of 2.76E-10 with sample range (0.00E+00,2.60E+01) resulting in fit range (6.74E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcPMI3)
    # normalization
    linear_norm_parameter = (2.8952313203676817e-05, 0.2866572166015228)  # error of 1.92E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.87E-01,2.52E+01)
    min_max_norm_parameter = (147.52756421954018, 11758.510463576746)  # error of 5.52E-02 with sample range (0.00E+00,8.59E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumRotatableBonds)
    # normalization
    linear_norm_parameter = (0.0016857886272905187, 0.8125456140318361)  # error of 1.82E-01 with sample range (0.00E+00,2.48E+02) resulting in fit range (8.13E-01,1.23E+00)
    min_max_norm_parameter = (1.6292528997825728, 13.174098358632333)  # error of 1.89E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.304462270693269, 0.4542440447080067)  # error of 1.34E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (3.50E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.7090015245725985, 0.6333818384913666, 0.3400706997276338)  # error of 8.05E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (1.41E-02,1.00E+00)
    genlog_norm_parameter = (0.3119530376866779, -5.246761061121691, 0.016007203239645374, 0.0005020797342553604)  # error of 6.90E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (2.04E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi0v)
    # normalization
    linear_norm_parameter = (0.027431175492895643, 0.1959231987976101)  # error of 1.84E-01 with sample range (1.00E+00,2.77E+02) resulting in fit range (2.23E-01,7.80E+00)
    min_max_norm_parameter = (5.833583645428592, 19.149899507438242)  # error of 4.23E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (12.391158701361956, 0.36836959542101233)  # error of 2.62E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (1.48E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (11.842303238148796, 0.5140666350188914, 0.29912163479666287)  # error of 1.16E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (3.78E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_quatN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_quatN)
    # normalization
    linear_norm_parameter = (0.4970377666010063, 0.49703776660100607)  # error of 5.55E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.97E-01,2.49E+00)
    min_max_norm_parameter = (8.281984578257868e-09, 1.0059597752394516)  # error of 2.67E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.2633408656906582, 2.2633408527519925)  # error of 6.28E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.46E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.2633408656906582, 1.0, 2.2633408527519925)  # error of 6.28E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.46E-01,1.00E+00)
    genlog_norm_parameter = (1.1065829196789287, 0.2861049426965705, 0.02419832738834138, 1.8381836800910687)  # error of 2.98E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.82E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_Ndealkylation2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ndealkylation2)
    # normalization
    linear_norm_parameter = (0.05745482906538413, 0.8807107360337569)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.81E-01,1.34E+00)
    min_max_norm_parameter = (2.739177310521382e-09, 1.0659099385351551)  # error of 1.79E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.004626898029145944, 2.706941292405111)  # error of 5.03E-16 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.03E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.004626898216699545, 1.0, 2.7069412916287168)  # error of 1.16E-11 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.03E-01,1.00E+00)
    genlog_norm_parameter = (2.7281878207400423, 0.5221851192482195, 0.42313946052940726, 1.7041201197919922)  # error of 7.98E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.51E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  EState_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA8)
    # normalization
    linear_norm_parameter = (0.007671915193011093, 0.672048083837521)  # error of 1.07E-01 with sample range (0.00E+00,3.49E+02) resulting in fit range (6.72E-01,3.35E+00)
    min_max_norm_parameter = (1.8002683601870467e-16, 21.232186746380208)  # error of 5.81E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (8.641715400832636, 0.17831995658486827)  # error of 2.66E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (1.76E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (8.12772381561056, 0.281669304272584, 0.16711726634028284)  # error of 2.50E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (9.20E-02,1.00E+00)
    genlog_norm_parameter = (0.14956610363091913, -19.654798331950023, 0.010062387245818023, 0.00021025562951752848)  # error of 2.40E-02 with sample range (0.00E+00,3.49E+02) resulting in fit range (7.96E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_phos_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_phos_ester)
    # normalization
    linear_norm_parameter = (0.49879260866522035, 0.49879260866522024)  # error of 3.33E-16 with sample range (0.00E+00,1.60E+01) resulting in fit range (4.99E-01,8.48E+00)
    min_max_norm_parameter = (5.639896586482622e-09, 1.002420627946355)  # error of 1.68E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.4543278081193836, 2.4543276857095164)  # error of 8.66E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.73E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.4543278081193836, 1.0, 2.4543276857095164)  # error of 8.66E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.73E-01,1.00E+00)
    genlog_norm_parameter = (1.1082937287141854, 0.28048751109971, 0.009918735917814574, 1.8440171697133898)  # error of 4.22E-15 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.93E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  RingCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(RingCount)
    # normalization
    linear_norm_parameter = (0.03860180770005206, 0.5113290906759573)  # error of 1.73E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (5.11E-01,2.87E+00)
    min_max_norm_parameter = (0.334517739633367, 5.208998938983079)  # error of 1.60E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.7238161788458215, 1.1018114352614388)  # error of 1.08E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.74E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.5715742346316732, 1.2832261760477313, 0.9174927781864283)  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (3.56E-02,1.00E+00)
    genlog_norm_parameter = (0.813320209949437, 1.988272685010793, 0.21462517997652011, 0.17389824103982)  # error of 4.52E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.48E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PEOE_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA5)
    # normalization
    linear_norm_parameter = (0.006570381366511557, 0.7983008331386301)  # error of 4.66E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (7.98E-01,1.71E+00)
    min_max_norm_parameter = (5e-324, 6.468041922682696)  # error of 8.78E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-3.9038317058155574, 0.13911276842288417)  # error of 2.41E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.33E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-3.903831899586245, 1.0, 0.1391127663161195)  # error of 2.41E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.33E-01,1.00E+00)
    genlog_norm_parameter = (0.36314293084192834, 13.090294898491566, 3.740737255729277, 16.894010370817867)  # error of 2.06E-02 with sample range (0.00E+00,1.39E+02) resulting in fit range (6.98E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  EState_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA11)
    # normalization
    linear_norm_parameter = (0.0013586330466129892, 0.9547790258899351)  # error of 1.61E-02 with sample range (0.00E+00,2.06E+02) resulting in fit range (9.55E-01,1.23E+00)
    min_max_norm_parameter = (1.272927585246758e-08, 1.5189995462591486)  # error of 2.36E-02 with sample range (0.00E+00,2.06E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-10.572305720334622, 0.17246009308288568)  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.61E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-10.572306217753033, 1.0, 0.17246008806155144)  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.61E-01,1.00E+00)
    genlog_norm_parameter = (0.1683364064162695, -0.41339243657734037, 0.004095510687534889, 0.025021145295607363)  # error of 6.05E-03 with sample range (0.00E+00,2.06E+02) resulting in fit range (8.59E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  VSA_EState1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState1)
    # normalization
    linear_norm_parameter = (0.0021288093550339005, 0.07050165770753636)  # error of 1.68E-01 with sample range (-7.93E+00,6.18E+03) resulting in fit range (5.36E-02,1.32E+01)
    min_max_norm_parameter = (79.6151847284827, 292.20287575940097)  # error of 3.66E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (184.75792508436405, 0.022940041643693208)  # error of 2.38E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (1.19E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (178.29194517948878, 0.02827752243312229, 0.018358023909216228)  # error of 1.08E-02 with sample range (-7.93E+00,6.18E+03) resulting in fit range (5.14E-03,1.00E+00)
    genlog_norm_parameter = (0.01604348673593412, -26.595443742121414, 0.9470080387441423, 0.04774078059095727)  # error of 7.31E-03 with sample range (-7.93E+00,6.18E+03) resulting in fit range (1.45E-05,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_diazo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_diazo)
    # normalization
    linear_norm_parameter = (1.0000000000000002, 2.220446049250313e-16)  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    min_max_norm_parameter = (1e-10, 0.9999999999)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5027041798086037, 46.14043898879842)  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.49874072864947394, 3.68328813585096, 20.104860546890446)  # error of 4.31E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.42E-05,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa1)
    # normalization
    linear_norm_parameter = (0.07531844296316315, 0.10272005612925916)  # error of 1.58E-01 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.06E-01,3.83E+01)
    min_max_norm_parameter = (1.58553325314983, 8.60938851018364)  # error of 3.82E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.045698563226252, 0.6990022831903797)  # error of 2.71E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (2.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.767359124032586, 0.9160893907811511, 0.5589277261645443)  # error of 1.25E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.30E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi4v)
    # normalization
    linear_norm_parameter = (0.1846796920584458, 0.12308022240634398)  # error of 1.49E-01 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.23E-01,4.41E+01)
    min_max_norm_parameter = (0.4365031459643245, 3.319224643809378)  # error of 3.97E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.8494300166801143, 1.6807292424962799)  # error of 3.13E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (4.28E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.7216523305680784, 2.287102552977168, 1.2587726126085808)  # error of 1.11E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.91E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1)
    # normalization
    linear_norm_parameter = (0.021402601422805922, 0.057358382906918415)  # error of 1.75E-01 with sample range (0.00E+00,4.28E+02) resulting in fit range (5.74E-02,9.23E+00)
    min_max_norm_parameter = (10.35770293280183, 30.318616722974067)  # error of 4.02E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (20.218648442527503, 0.24321989118039597)  # error of 2.60E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (7.26E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (19.497180383047468, 0.31860651031309306, 0.19451567750637339)  # error of 1.08E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (2.00E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_ketone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_ketone)
    # normalization
    linear_norm_parameter = (0.020725634690741956, 0.9421702046814519)  # error of 7.64E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.42E-01,1.19E+00)
    min_max_norm_parameter = (5.742986740306006e-09, 1.0443931574968837)  # error of 1.71E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.5164294479177101, 2.0540442859352286)  # error of 1.94E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.43E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.5164295209542769, 1.0, 2.054044188695136)  # error of 1.94E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.43E-01,1.00E+00)
    genlog_norm_parameter = (1.6087286152330658, 0.7570384536825558, -1.002172697672123, -26.085121171481475)  # error of 1.05E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_lactone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_lactone)
    # normalization
    linear_norm_parameter = (0.4971402573768363, 0.4971402573768361)  # error of 2.22E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.97E-01,2.49E+00)
    min_max_norm_parameter = (5.474006801001615e-12, 1.0057523827434252)  # error of 2.72E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.2711508768212254, 2.271150883092475)  # error of 8.10E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.47E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.2711508768212254, 1.0, 2.271150883092475)  # error of 8.10E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.47E-01,1.00E+00)
    genlog_norm_parameter = (1.106683635479975, 0.28577500733800526, 0.023368814288469427, 1.838525465283933)  # error of 3.06E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.83E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_thiazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_thiazole)
    # normalization
    linear_norm_parameter = (0.4896259336659703, 0.4896259336659702)  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.90E-01,3.43E+00)
    min_max_norm_parameter = (4.718948044240454e-09, 1.021187738581169)  # error of 1.92E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.9632451839557359, 1.963245141758428)  # error of 2.15E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.9632451839557359, 1.0, 1.963245141758428)  # error of 2.15E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.69E-01,1.00E+00)
    genlog_norm_parameter = (1.0990797102104737, 0.3105133091898195, 0.08266879961025278, 1.8131267775109896)  # error of 2.96E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.41E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  MinAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MinAbsEStateIndex)
    # normalization
    linear_norm_parameter = (0.43546020000054764, 0.22371772226149234)  # error of 1.33E-01 with sample range (0.00E+00,6.27E+00) resulting in fit range (2.24E-01,2.95E+00)
    min_max_norm_parameter = (1.5407439555097887e-33, 1.18171530209488)  # error of 6.48E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5580567270616906, 3.2920268348402595)  # error of 3.85E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (1.37E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4804909365206003, 4.528448995771037, 2.473174565611486)  # error of 2.70E-02 with sample range (0.00E+00,6.27E+00) resulting in fit range (1.02E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_COO2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_COO2)
    # normalization
    linear_norm_parameter = (0.026660100588824065, 0.9260616544509377)  # error of 1.06E-02 with sample range (0.00E+00,2.70E+01) resulting in fit range (9.26E-01,1.65E+00)
    min_max_norm_parameter = (9.034811582204987e-09, 1.0579100746345402)  # error of 1.58E-03 with sample range (0.00E+00,2.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.24732225870510682, 2.284131120056377)  # error of 4.83E-04 with sample range (0.00E+00,2.70E+01) resulting in fit range (6.38E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.24732225870510682, 1.0, 2.284131120056377)  # error of 4.83E-04 with sample range (0.00E+00,2.70E+01) resulting in fit range (6.38E-01,1.00E+00)
    genlog_norm_parameter = (1.279206110765303, 0.8498205579510066, -1.1534376846046783, -53.878719461549125)  # error of 3.39E-13 with sample range (0.00E+00,2.70E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SlogP_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA9)
    # normalization
    linear_norm_parameter = (1.0, 1.0)  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (0.0, 1.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (0.33705420273837067, 0.3370542027383706, -9.284058023162583e-10, 2.1473819766010673)  # error of 4.84E-10 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  VSA_EState9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState9)
    # normalization
    linear_norm_parameter = (0.022314103447673483, 0.2903417115454465)  # error of 6.95E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (-4.69E+00,5.00E+00)
    min_max_norm_parameter = (-8.357071428462406, 11.006415888282014)  # error of 4.34E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.6709557801163907, 0.3954700807696381)  # error of 3.36E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (6.55E-39,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.06317360890340902, 0.3383220043747596, 21.649602529681054)  # error of 1.76E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (1.76E-33,1.00E+00)
    genlog_norm_parameter = (26.51339376649436, 1.2703582868532193, 0.9007127336035414, 88.49043249715089)  # error of 2.68E-02 with sample range (-2.23E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_isothiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_isothiocyan)
    # normalization
    linear_norm_parameter = (0.5, 0.16661833768294187)  # error of 2.36E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.67E-01,1.17E+00)
    min_max_norm_parameter = (5.0007250771301266e-11, 1.000145015326011)  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5352491266658849, 18.830908818920925)  # error of 2.54E-05 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.19E-05,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.49999999625955927, 8.494601049886345, 15.667407499578225)  # error of 2.71E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (3.96E-04,1.00E+00)
    genlog_norm_parameter = (1.6812681874777036, 0.1084604721306692, 0.00012118782821307256, 8.882819150479717e-05)  # error of 1.91E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.95E-01,9.45E-01)
    preferred_normalization = 'unity'
    # functions
    
    

class  Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi2n)
    # normalization
    linear_norm_parameter = (0.11014910768339037, 0.02748881833733474)  # error of 1.49E-01 with sample range (0.00E+00,1.02E+02) resulting in fit range (2.75E-02,1.13E+01)
    min_max_norm_parameter = (1.6850371430659572, 6.553874507410937)  # error of 3.80E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.081549676171799, 0.9904095492145635)  # error of 2.83E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (1.73E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.87238559628832, 1.3409369944950944, 0.767555104248775)  # error of 1.08E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (5.53E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_Al_OH_noTert_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Al_OH_noTert)
    # normalization
    linear_norm_parameter = (0.49733523982841565, 0.49733523982841543)  # error of 1.11E-16 with sample range (0.00E+00,2.00E+01) resulting in fit range (4.97E-01,1.04E+01)
    min_max_norm_parameter = (1.5560199363023413e-09, 1.005358076311579)  # error of 1.54E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.286728147327964, 2.286728011845654)  # error of 1.32E-07 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.286728147327964, 1.0, 2.286728011845654)  # error of 1.32E-07 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.50E-01,1.00E+00)
    genlog_norm_parameter = (1.1068750184240401, 0.2851479096584255, 0.021789171679651578, 1.8391753470073229)  # error of 3.23E-08 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.84E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcFractionCSP3)
    # normalization
    linear_norm_parameter = (1.1210114366616568, 0.024238385885020208)  # error of 6.63E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.42E-02,1.15E+00)
    min_max_norm_parameter = (0.030077967569291073, 0.7816451738469055)  # error of 3.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4036392556168974, 6.810271411900117)  # error of 2.19E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (6.01E-02,9.83E-01)
    dual_sigmoidal_norm_parameter = (0.37933375807687925, 8.369078319667627, 5.735309352343304)  # error of 1.04E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.01E-02,9.72E-01)
    genlog_norm_parameter = (5.121341713037929, 0.3541805503951657, 0.1654208707795316, 0.18747997008450548)  # error of 4.98E-03 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.38E-02,9.68E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAliphaticHeterocycles)
    # normalization
    linear_norm_parameter = (0.07182403555851259, 0.6979616844087828)  # error of 6.50E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (6.98E-01,3.71E+00)
    min_max_norm_parameter = (6.816327279061287e-09, 1.4697587576089106)  # error of 1.83E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5242081959438346, 1.5883082643053523)  # error of 6.06E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (3.03E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.5241697890415581, 31.347564520476126, 1.5882035074486005)  # error of 3.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (7.31E-08,1.00E+00)
    genlog_norm_parameter = (1.5563961013766956, 0.6667571953724875, 0.5870663542985489, 0.7783004040766286)  # error of 5.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.85E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0)
    # normalization
    linear_norm_parameter = (0.007513574566740955, 0.2957974579215942)  # error of 2.01E-01 with sample range (0.00E+00,7.19E+02) resulting in fit range (2.96E-01,5.70E+00)
    min_max_norm_parameter = (17.447753846956868, 55.71754607510952)  # error of 4.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (36.35211468883877, 0.1295884036148426)  # error of 2.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (8.92E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (34.78188468223621, 0.18372914466871643, 0.10529824432103296)  # error of 1.38E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (1.67E-03,1.00E+00)
    genlog_norm_parameter = (0.09192921396182452, 3.197386465432143, 4.054083564087513e-07, 2.977582294894024e-08)  # error of 1.17E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (1.17E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcKappa1)
    # normalization
    linear_norm_parameter = (0.07531844296316315, 0.10272005612925916)  # error of 1.58E-01 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.06E-01,3.83E+01)
    min_max_norm_parameter = (1.58553325314983, 8.60938851018364)  # error of 3.82E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.045698563226252, 0.6990022831903797)  # error of 2.71E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (2.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.767359124032586, 0.9160893907811511, 0.5589277261645443)  # error of 1.25E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.30E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MaxAbsEStateIndex)
    # normalization
    linear_norm_parameter = (0.11609490354923989, -0.965852149695141)  # error of 9.45E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (-9.66E-01,2.80E+00)
    min_max_norm_parameter = (10.794835917382866, 15.53613807930635)  # error of 7.37E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (13.16028708089213, 0.9139468145600386)  # error of 6.32E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (5.98E-06,1.00E+00)
    dual_sigmoidal_norm_parameter = (13.444763362908514, 0.4909275403082198, 1.4564963345323978)  # error of 3.13E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (1.36E-03,1.00E+00)
    genlog_norm_parameter = (2.972191234687917, 14.149085374516742, 11.048956911098314, 7.283386625774641)  # error of 3.39E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (2.23E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA14_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA14)
    # normalization
    linear_norm_parameter = (0.0006563772490514073, 0.9227826193892867)  # error of 7.32E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (9.23E-01,1.26E+00)
    min_max_norm_parameter = (2.415003722890725e-18, 8.09692280816151)  # error of 5.41E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.8016925269391764, 0.23235287767416216)  # error of 3.04E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (4.54E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.8017088240819955, 13.645839498944362, 0.23235348768870404)  # error of 2.97E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (1.77E-05,1.00E+00)
    genlog_norm_parameter = (0.22225301681800952, -2.3695067664597853, 1.0446302749239949, 0.5795681123293127)  # error of 3.04E-02 with sample range (0.00E+00,5.10E+02) resulting in fit range (4.36E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA7)
    # normalization
    linear_norm_parameter = (0.006731394986636996, 0.31295466008048334)  # error of 1.50E-01 with sample range (0.00E+00,1.25E+03) resulting in fit range (3.13E-01,8.73E+00)
    min_max_norm_parameter = (7.51983930256869, 68.45113819370312)  # error of 4.09E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (37.76437056260887, 0.08135882094622694)  # error of 1.76E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (4.43E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (36.482128078042514, 0.1009691384003825, 0.07316927556894248)  # error of 1.14E-02 with sample range (0.00E+00,1.25E+03) resulting in fit range (2.45E-02,1.00E+00)
    genlog_norm_parameter = (0.06460929764312966, -3.2413499720442633, 1.840439341992175, 0.18492989797809087)  # error of 9.38E-03 with sample range (0.00E+00,1.25E+03) resulting in fit range (7.16E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(NPR1)
    # normalization
    linear_norm_parameter = (1.993532814687622, -0.07291478792771389)  # error of 6.17E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (-7.29E-02,1.83E+00)
    min_max_norm_parameter = (0.06225581266603365, 0.49572101730415397)  # error of 2.98E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.27514214399673687, 11.301895844887198)  # error of 2.74E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (4.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.25986760502174205, 14.138759638283469, 9.152837968958812)  # error of 1.65E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (2.47E-02,9.98E-01)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi3v)
    # normalization
    linear_norm_parameter = (0.13641142111605653, 0.06481842365937796)  # error of 1.43E-01 with sample range (0.00E+00,1.56E+02) resulting in fit range (6.48E-02,2.13E+01)
    min_max_norm_parameter = (0.9571111913222607, 5.030476311519345)  # error of 3.74E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.9588780643186574, 1.1889256254049316)  # error of 2.82E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (2.88E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.798222121514068, 1.5668854376121493, 0.9198752649381355)  # error of 1.06E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (1.23E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi4n)
    # normalization
    linear_norm_parameter = (0.2235474556767324, 0.12057837229443968)  # error of 1.52E-01 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.21E-01,8.63E+00)
    min_max_norm_parameter = (0.40708839430881494, 2.71836618230583)  # error of 3.94E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.5414132407381294, 2.0927265327645506)  # error of 3.03E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (3.82E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.441026134320466, 2.8235640784391123, 1.5793253223087178)  # error of 1.12E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.68E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumSaturatedHeterocycles)
    # normalization
    linear_norm_parameter = (0.06338179563839252, 0.7789048985591301)  # error of 4.13E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.79E-01,2.68E+00)
    min_max_norm_parameter = (5.3308083108781935e-09, 1.2539605902115512)  # error of 8.89E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.24911412739948294, 1.825541426846478)  # error of 5.15E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.88E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.24911412739948294, 1.0, 1.825541426846478)  # error of 5.15E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.88E-01,1.00E+00)
    genlog_norm_parameter = (3.0410569853018057, -0.5365635014003982, 0.010320409148970734, 0.00041830035148444656)  # error of 6.76E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (8.06E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi2v)
    # normalization
    linear_norm_parameter = (0.09656674011706068, 0.026475463133179833)  # error of 1.43E-01 with sample range (0.00E+00,1.68E+02) resulting in fit range (2.65E-02,1.62E+01)
    min_max_norm_parameter = (1.85040714882094, 7.55574015613779)  # error of 3.72E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.659367062804354, 0.84727788625193)  # error of 2.67E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (1.89E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.449991335141199, 1.1069417196730784, 0.6703056005048558)  # error of 1.05E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (7.20E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi3n)
    # normalization
    linear_norm_parameter = (0.16045617502782894, 0.06275167455712582)  # error of 1.47E-01 with sample range (0.00E+00,6.23E+01) resulting in fit range (6.28E-02,1.01E+01)
    min_max_norm_parameter = (0.8698290958911558, 4.249166143207709)  # error of 3.75E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.5307853903255304, 1.4301006733612187)  # error of 2.83E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (2.61E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.390012863113382, 1.9077779294117436, 1.1008446640044367)  # error of 1.03E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (1.04E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_amidine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_amidine)
    # normalization
    linear_norm_parameter = (0.02087812096911279, 0.956658900698937)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.57E-01,1.21E+00)
    min_max_norm_parameter = (1.0142986812940386e-08, 1.0229791584422843)  # error of 5.67E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.4118456671316019, 2.672507134625743)  # error of 7.85E-17 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.41184569107524827, 1.0, 2.6725070662621717)  # error of 5.18E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.50E-01,1.00E+00)
    genlog_norm_parameter = (2.6819460184508435, 0.4112882522699607, 0.21255301611208302, 1.8880926775866433)  # error of 5.33E-11 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.69E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcAsphericity)
    # normalization
    linear_norm_parameter = (1.4721378049698663, -0.11228706292547166)  # error of 2.90E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-1.12E-01,1.36E+00)
    min_max_norm_parameter = (0.09005620590041596, 0.7355219615267257)  # error of 1.68E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.40970204866580884, 7.643961121304289)  # error of 2.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.18E-02,9.89E-01)
    dual_sigmoidal_norm_parameter = (0.3993757142716246, 8.419588600488629, 6.968980500840633)  # error of 1.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (3.35E-02,9.85E-01)
    genlog_norm_parameter = (6.206340781992686, 0.43464496231710853, 0.25779424922808303, 0.39659201683500284)  # error of 1.81E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.89E-02,9.81E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa2)
    # normalization
    linear_norm_parameter = (0.030418143185359847, 0.2871593951766924)  # error of 2.34E-01 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.87E-01,1.08E+03)
    min_max_norm_parameter = (3.446066777200142, 12.00058941506709)  # error of 4.58E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.66772112289306, 0.5795843116540953)  # error of 3.08E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (1.16E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (7.271319784936047, 0.8080128800230181, 0.4459683689150529)  # error of 1.32E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.80E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BertzCT)
    # normalization
    linear_norm_parameter = (0.0002435689316791905, 0.08402337833593164)  # error of 1.62E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (8.40E-02,1.15E+01)
    min_max_norm_parameter = (589.3484428971769, 2551.6280739737376)  # error of 3.81E-02 with sample range (0.00E+00,4.67E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  MaxPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MaxPartialCharge)
    # normalization
    linear_norm_parameter = (3.0906307507635584, -0.334505311086143)  # error of 7.24E-02 with sample range (-4.12E-01,INF) resulting in fit range (-1.61E+00,INF)
    min_max_norm_parameter = (0.1437213632361377, 0.3915749702129172)  # error of 3.92E-02 with sample range (-4.12E-01,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.26834153843781333, 19.306170677985925)  # error of 1.77E-02 with sample range (-4.12E-01,INF) resulting in fit range (1.99E-06,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.26692142305427585, 20.140051277382142, 18.554459487006113)  # error of 1.72E-02 with sample range (-4.12E-01,INF) resulting in fit range (1.16E-06,1.00E+00)
    genlog_norm_parameter = (19.09091940309053, 0.35102481862378276, 0.1963118558023432, 0.9662656391252168)  # error of 1.77E-02 with sample range (-4.12E-01,INF) resulting in fit range (1.54E-06,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  SlogP_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA12)
    # normalization
    linear_norm_parameter = (0.006154770006633981, 0.6983876686482481)  # error of 8.20E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (6.98E-01,2.37E+00)
    min_max_norm_parameter = (5e-324, 16.317969218729832)  # error of 7.75E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.978333217945518, 0.12756642613388391)  # error of 2.43E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (3.46E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.978320474929455, 2.185653842795465, 0.1275661877112387)  # error of 2.34E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (1.88E-05,1.00E+00)
    genlog_norm_parameter = (0.12015565174919032, 6.735768505141773, 0.3722611574944573, 0.5327919499893109)  # error of 2.42E-02 with sample range (0.00E+00,2.71E+02) resulting in fit range (3.20E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  VSA_EState2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState2)
    # normalization
    linear_norm_parameter = (0.01528826607315403, 0.2670303270450643)  # error of 1.35E-01 with sample range (-3.29E+01,8.56E+02) resulting in fit range (-2.36E-01,1.34E+01)
    min_max_norm_parameter = (-4.878450507072114, 34.03022413712193)  # error of 4.20E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (14.37762800683558, 0.12195442376720393)  # error of 4.00E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (3.12E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (13.675781179464368, 0.14469304482094406, 0.10641180812795803)  # error of 3.56E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (1.18E-03,1.00E+00)
    genlog_norm_parameter = (0.08922402753158253, -3.1375517024170803, 0.3191874567853099, 0.0981416430000738)  # error of 3.45E-02 with sample range (-3.29E+01,8.56E+02) resulting in fit range (2.61E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_HOCCN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_HOCCN)
    # normalization
    linear_norm_parameter = (0.4977502024817768, 0.4977502024817767)  # error of 2.22E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.98E-01,1.99E+00)
    min_max_norm_parameter = (1.26267034441651e-08, 1.0045199328470353)  # error of 8.00E-05 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3236118939770356, 2.3236118577424274)  # error of 3.85E-07 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3236118939770356, 1.0, 2.3236118577424274)  # error of 3.85E-07 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    genlog_norm_parameter = (1.1072811204116666, 0.283816136518937, 0.018420634004107303, 1.8405567986012576)  # error of 3.61E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.87E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  MinPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MinPartialCharge)
    # normalization
    linear_norm_parameter = (2.948628224298744, 1.7394347203560183)  # error of 1.05E-01 with sample range (-2.00E+00,INF) resulting in fit range (-4.16E+00,INF)
    min_max_norm_parameter = (-0.5479342566156457, -0.28873718774420354)  # error of 4.48E-02 with sample range (-2.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.420139769859882, 17.629128066695653)  # error of 5.34E-02 with sample range (-2.00E+00,INF) resulting in fit range (8.02E-13,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.43972856902364077, 26.868401369842132, 13.463159347885448)  # error of 5.23E-02 with sample range (-2.00E+00,INF) resulting in fit range (6.22E-19,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1n)
    # normalization
    linear_norm_parameter = (0.02460656481380441, 0.05696382112467857)  # error of 1.83E-01 with sample range (0.00E+00,3.32E+02) resulting in fit range (5.70E-02,8.22E+00)
    min_max_norm_parameter = (8.215243805842091, 25.568763302868422)  # error of 3.90E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (16.800375876069985, 0.2846180730802336)  # error of 2.44E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (8.31E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (16.22787766733102, 0.3557663030596946, 0.2286121746689847)  # error of 1.13E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (3.10E-03,1.00E+00)
    genlog_norm_parameter = (0.20081911303774566, -0.3618619378577607, 1.199011524398242, 0.05743559815299407)  # error of 7.92E-03 with sample range (0.00E+00,3.32E+02) resulting in fit range (2.17E-06,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PEOE_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA6)
    # normalization
    linear_norm_parameter = (0.0023302771680984535, 0.6632167817066577)  # error of 1.34E-01 with sample range (0.00E+00,1.31E+03) resulting in fit range (6.63E-01,3.70E+00)
    min_max_norm_parameter = (1.94951131475506e-20, 53.568112487136325)  # error of 6.28E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (22.943177031905776, 0.06530117675670048)  # error of 2.26E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (1.83E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (20.77325824422444, 0.11533425472212071, 0.05805165987942096)  # error of 1.56E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (8.35E-02,1.00E+00)
    genlog_norm_parameter = (0.0529677336429405, -35.34156868005128, 0.0017394739065960728, 0.00011634621863982968)  # error of 1.43E-02 with sample range (0.00E+00,1.31E+03) resulting in fit range (1.00E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  EState_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA10)
    # normalization
    linear_norm_parameter = (0.004005936989037917, 0.7781446305553839)  # error of 1.04E-01 with sample range (0.00E+00,5.61E+02) resulting in fit range (7.78E-01,3.02E+00)
    min_max_norm_parameter = (2.7254874994650347e-24, 19.011338996425444)  # error of 5.00E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.65981607322896, 0.1802261811598023)  # error of 1.90E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (2.01E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.86219794446307, 0.35061192985231965, 0.16362403625943647)  # error of 1.63E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (8.27E-02,1.00E+00)
    genlog_norm_parameter = (0.15082974766082996, -18.484638192736522, 0.007439276999279547, 0.00020806426143220736)  # error of 1.54E-02 with sample range (0.00E+00,5.61E+02) resulting in fit range (1.11E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  EState_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA7)
    # normalization
    linear_norm_parameter = (0.0009047456761177575, 0.988557120440686)  # error of 2.13E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (9.89E-01,1.03E+00)
    min_max_norm_parameter = (2.1257783917477867e-14, 4.312791487510295)  # error of 4.52E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.0761099576023025, 0.6793696501735206)  # error of 1.30E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (8.04E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.0761099576023025, 1.0, 0.6793696501735206)  # error of 1.30E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (8.04E-01,1.00E+00)
    genlog_norm_parameter = (0.9278558605436115, 0.9750697326238789, -0.4046534848332459, -1.2077899416726605)  # error of 1.24E-03 with sample range (0.00E+00,4.78E+01) resulting in fit range (1.40E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_pyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_pyridine)
    # normalization
    linear_norm_parameter = (0.03403943645071905, 0.906233438990491)  # error of 1.33E-02 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.06E-01,1.31E+00)
    min_max_norm_parameter = (8.84293141493641e-09, 1.0742739897507971)  # error of 2.32E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.1001953260731105, 2.363269723603649)  # error of 2.37E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (5.59E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.10019533355161231, 1.0, 2.36326970776348)  # error of 2.37E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (5.59E-01,1.00E+00)
    genlog_norm_parameter = (1.8182773439194169, 0.85963139939007, -0.9439682575712153, -18.344148087280377)  # error of 2.28E-11 with sample range (0.00E+00,1.20E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  BCUT2D_CHGHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_CHGHI)
    # normalization
    linear_norm_parameter = (1.0601079754933302, 0.40787344122731317)  # error of 1.66E-01 with sample range (-1.56E+00,3.00E+00) resulting in fit range (-1.24E+00,3.59E+00)
    min_max_norm_parameter = (-0.09795833235878999, 0.13496394474519627)  # error of 8.61E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.015414812733986287, 19.410301428357236)  # error of 8.02E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (5.54E-14,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.008123422957527535, 43.42891499503567, 6.984539197542873)  # error of 3.60E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (6.07E-30,1.00E+00)
    genlog_norm_parameter = (13.467113000799394, -0.21996298838853404, 0.0011345067054706437, 7.109293287179587e-05)  # error of 6.72E-02 with sample range (-1.56E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA2)
    # normalization
    linear_norm_parameter = (0.009229128833342298, 0.6965819780408643)  # error of 1.14E-01 with sample range (0.00E+00,4.96E+02) resulting in fit range (6.97E-01,5.27E+00)
    min_max_norm_parameter = (0.0, 15.384470706540956)  # error of 6.59E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.006780263970044, 0.24361243133391997)  # error of 3.95E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (1.88E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (5.108545135970047, 0.9767373573037742, 0.2124051105161742)  # error of 3.59E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (6.76E-03,1.00E+00)
    genlog_norm_parameter = (0.2026692161636364, -10.116783288314046, 0.017985027417691864, 0.0009958173671104694)  # error of 3.87E-02 with sample range (0.00E+00,4.96E+02) resulting in fit range (9.81E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  qed_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(qed)
    # normalization
    linear_norm_parameter = (1.2001504087852979, -0.19399001508562716)  # error of 5.24E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (-1.90E-01,9.44E-01)
    min_max_norm_parameter = (0.22901735830941922, 0.9481621980667114)  # error of 3.90E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6000031067059596, 6.458412537955855)  # error of 3.69E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (2.08E-02,9.05E-01)
    dual_sigmoidal_norm_parameter = (0.6370145592271749, 4.945616362932774, 8.803721507227248)  # error of 2.33E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (4.18E-02,9.39E-01)
    genlog_norm_parameter = (14.258861720160791, 0.8275500529167287, 0.9871179281649253, 4.187543624653561)  # error of 2.20E-02 with sample range (3.45E-03,9.48E-01) resulting in fit range (6.06E-02,9.62E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SMR_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA1)
    # normalization
    linear_norm_parameter = (0.0060832031355171345, 0.29372966007484347)  # error of 1.94E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.94E-01,1.13E+01)
    min_max_norm_parameter = (18.58970931537026, 67.49192460811703)  # error of 4.61E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (42.77007350322991, 0.09980774119785008)  # error of 2.58E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (1.38E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (41.1105092185979, 0.1350839878005177, 0.08255127492140925)  # error of 1.34E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.86E-03,1.00E+00)
    genlog_norm_parameter = (0.07260546024264776, -25.110752914170668, 0.016405910234250563, 0.0001819439798804859)  # error of 1.05E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.83E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_lactam_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_lactam)
    # normalization
    linear_norm_parameter = (0.49918507334339934, 0.49918507334339923)  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    min_max_norm_parameter = (4.33658612363641e-09, 1.0016325140575846)  # error of 1.00E-05 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.5332891839492102, 2.533289107423889)  # error of 1.30E-07 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.80E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.5332891839492102, 1.0, 2.533289107423889)  # error of 1.30E-07 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.80E-01,1.00E+00)
    genlog_norm_parameter = (1.1086720787101432, 0.2792406392035079, 0.0067029476374034, 1.8453162933127039)  # error of 4.33E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.95E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_imidazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_imidazole)
    # normalization
    linear_norm_parameter = (0.02594266516013577, 0.9467297943185111)  # error of 0.00E+00 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.47E-01,1.10E+00)
    min_max_norm_parameter = (3.74982058400203e-09, 1.0280953163139084)  # error of 5.25E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.18731110481329363, 3.0086069081003104)  # error of 7.85E-17 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.37E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.1873112027321337, 1.0, 3.0086065444034595)  # error of 2.63E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.37E-01,1.00E+00)
    genlog_norm_parameter = (3.0199481826741894, 0.44503661288807106, 0.28197554648621365, 1.8558118261834247)  # error of 1.13E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (6.74E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_Al_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Al_COO)
    # normalization
    linear_norm_parameter = (0.019808217260446237, 0.9450166151713014)  # error of 7.81E-03 with sample range (0.00E+00,2.20E+01) resulting in fit range (9.45E-01,1.38E+00)
    min_max_norm_parameter = (3.2382696256541155e-09, 1.0424227893434397)  # error of 1.20E-03 with sample range (0.00E+00,2.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.4050322619852695, 2.2492421348283718)  # error of 3.59E-04 with sample range (0.00E+00,2.20E+01) resulting in fit range (7.13E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.405032320679578, 1.0, 2.249242042225464)  # error of 3.59E-04 with sample range (0.00E+00,2.20E+01) resulting in fit range (7.13E-01,1.00E+00)
    genlog_norm_parameter = (1.277744885066904, 0.9069887992547633, -1.066482163726238, -70.69130845711288)  # error of 3.40E-10 with sample range (0.00E+00,2.20E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0v)
    # normalization
    linear_norm_parameter = (0.027431175492895643, 0.1959231987976101)  # error of 1.84E-01 with sample range (1.00E+00,2.77E+02) resulting in fit range (2.23E-01,7.80E+00)
    min_max_norm_parameter = (5.833583645428592, 19.149899507438242)  # error of 4.23E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (12.391158701361956, 0.36836959542101233)  # error of 2.62E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (1.48E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (11.842303238148796, 0.5140666350188914, 0.29912163479666287)  # error of 1.16E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (3.78E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumHBD)
    # normalization
    linear_norm_parameter = (0.012771534772396786, 0.8154539249362293)  # error of 1.08E-01 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.15E-01,1.62E+00)
    min_max_norm_parameter = (4.930380657631324e-32, 2.5821101687257144)  # error of 1.47E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.192951047473294, 1.324568319014574)  # error of 5.53E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (1.71E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.1929510837567654, 12.327419967877942, 1.324568589322017)  # error of 3.38E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.10E-07,1.00E+00)
    genlog_norm_parameter = (1.234223982974127, -2.5173462989289015, 0.009239861737341579, 0.0001326440589740326)  # error of 8.35E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.43E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAliphaticRings)
    # normalization
    linear_norm_parameter = (0.04716081590128, 0.7025071278256165)  # error of 8.86E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (7.03E-01,2.68E+00)
    min_max_norm_parameter = (1.5777218104420236e-29, 2.223682308863912)  # error of 2.88E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.7813916309499217, 1.3224930670990482)  # error of 4.20E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.7813895517185011, 19.442917588992998, 1.322487361093461)  # error of 2.34E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.52E-07,1.00E+00)
    genlog_norm_parameter = (1.5963506985004239, -1.5768456408816305, 0.0075354951062004665, 0.0001966513751535621)  # error of 1.55E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (4.55E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumRings)
    # normalization
    linear_norm_parameter = (0.03860180770005206, 0.5113290906759573)  # error of 1.73E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (5.11E-01,2.87E+00)
    min_max_norm_parameter = (0.334517739633367, 5.208998938983079)  # error of 1.60E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.7238161788458215, 1.1018114352614388)  # error of 1.08E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.74E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.5715742346316732, 1.2832261760477313, 0.9174927781864283)  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (3.56E-02,1.00E+00)
    genlog_norm_parameter = (0.813320209949437, 1.988272685010793, 0.21462517997652011, 0.17389824103982)  # error of 4.52E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.48E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_nitroso_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_nitroso)
    # normalization
    linear_norm_parameter = (0.11434935141358371, 0.5282342446312391)  # error of 2.97E-01 with sample range (0.00E+00,6.00E+00) resulting in fit range (5.28E-01,1.21E+00)
    min_max_norm_parameter = (9.249617077400622e-09, 1.0008656708869366)  # error of 4.28E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.7036397732099205, 23.795370886031456)  # error of 4.28E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (5.35E-08,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4999999564484462, 6.648382398811204, 12.36564692910073)  # error of 9.73E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (2.06E-03,1.00E+00)
    genlog_norm_parameter = (7.296365931941997, 0.2058906214339969, -0.22262794047612086, -0.7836891501859671)  # error of 4.26E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.93E-11,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_prisulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_prisulfonamd)
    # normalization
    linear_norm_parameter = (1.0, 1.0)  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (0.0, 1.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (0.33705420273837067, 0.3370542027383706, -9.284058023162583e-10, 2.1473819766010673)  # error of 4.84E-10 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SMR_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA2)
    # normalization
    linear_norm_parameter = (0.010815720584295, 0.8621985340234535)  # error of 5.64E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (8.62E-01,1.62E+00)
    min_max_norm_parameter = (1.833608967890303e-18, 1.5590020213009668)  # error of 5.87E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.7174803678465178, 0.5694423092539833)  # error of 1.98E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (6.01E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.7174798896423839, 0.21990603605382986, 0.5694424246725555)  # error of 1.98E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (6.01E-01,1.00E+00)
    genlog_norm_parameter = (0.5276885859408835, -1.9004372112033896, 0.0010520560915783057, 0.000706253590168276)  # error of 1.92E-02 with sample range (0.00E+00,7.01E+01) resulting in fit range (5.79E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi4n)
    # normalization
    linear_norm_parameter = (0.2235474556767324, 0.12057837229443968)  # error of 1.52E-01 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.21E-01,8.63E+00)
    min_max_norm_parameter = (0.40708839430881494, 2.71836618230583)  # error of 3.94E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.5414132407381294, 2.0927265327645506)  # error of 3.03E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (3.82E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.441026134320466, 2.8235640784391123, 1.5793253223087178)  # error of 1.12E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.68E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(InertialShapeFactor)
    # normalization
    linear_norm_parameter = (249.802812058766, 0.2626729794953536)  # error of 1.70E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (2.63E-01,2.41E+06)
    min_max_norm_parameter = (3.4403992924852414e-29, 0.0015516770560768962)  # error of 5.89E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0007537390827077178, 2772.3791520268146)  # error of 5.07E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (1.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.0006221969804982748, 4615.762401524865, 1723.753663227779)  # error of 2.12E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (5.36E-02,1.00E+00)
    genlog_norm_parameter = (0.6335439368983742, 1.1952465398370342, 0.06337814286503886, 0.1824990146140799)  # error of 2.88E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (4.99E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumAmideBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAmideBonds)
    # normalization
    linear_norm_parameter = (0.025379291619204047, 0.8062087745432029)  # error of 7.32E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.06E-01,2.41E+00)
    min_max_norm_parameter = (9.860761315262648e-31, 2.0799891850295853)  # error of 3.30E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4876800061173155, 1.2673698495352894)  # error of 2.03E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (3.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.5000000369459837, 10.198287769485447, 1.2823618205208733)  # error of 1.53E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (6.06E-03,1.00E+00)
    genlog_norm_parameter = (1.954963623739784, -1.2547927135979628, 0.009691030241488695, 0.00025254291606265794)  # error of 1.41E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (3.69E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumHDonors_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumHDonors)
    # normalization
    linear_norm_parameter = (0.012771534772396786, 0.8154539249362293)  # error of 1.08E-01 with sample range (0.00E+00,6.30E+01) resulting in fit range (8.15E-01,1.62E+00)
    min_max_norm_parameter = (4.930380657631324e-32, 2.5821101687257144)  # error of 1.47E-02 with sample range (0.00E+00,6.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.192951047473294, 1.324568319014574)  # error of 5.53E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (1.71E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.1929510837567654, 12.327419967877942, 1.324568589322017)  # error of 3.38E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.10E-07,1.00E+00)
    genlog_norm_parameter = (1.234223982974127, -2.5173462989289015, 0.009239861737341579, 0.0001326440589740326)  # error of 8.35E-03 with sample range (0.00E+00,6.30E+01) resulting in fit range (4.43E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  EState_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA5)
    # normalization
    linear_norm_parameter = (0.0017540134931710583, 0.9543908360985814)  # error of 8.35E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.54E-01,1.07E+00)
    min_max_norm_parameter = (5.383565720355907e-09, 1.3949325598805027)  # error of 2.48E-02 with sample range (0.00E+00,6.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-27.554098729991185, 0.09771212063268499)  # error of 7.09E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.37E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-27.55410005194067, 0.9999999999998883, 0.097712116945909)  # error of 7.09E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.37E-01,1.00E+00)
    genlog_norm_parameter = (0.0980914500590183, 10.732175091093676, 0.030637819553717135, 1.2864033105440875)  # error of 7.09E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.37E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_methoxy_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_methoxy)
    # normalization
    linear_norm_parameter = (0.03515383615264178, 0.8758211760938799)  # error of 2.13E-02 with sample range (0.00E+00,1.90E+01) resulting in fit range (8.76E-01,1.54E+00)
    min_max_norm_parameter = (8.17446329145094e-09, 1.126246500879634)  # error of 7.63E-03 with sample range (0.00E+00,1.90E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.3282281758044606, 1.5582781044378737)  # error of 2.40E-04 with sample range (0.00E+00,1.90E+01) resulting in fit range (6.25E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.32822817276403427, 1.0, 1.5582781078352885)  # error of 2.40E-04 with sample range (0.00E+00,1.90E+01) resulting in fit range (6.25E-01,1.00E+00)
    genlog_norm_parameter = (0.358542051528062, -3.477720453242525, -3.4795708967924757, -4.804876655086344)  # error of 5.40E-02 with sample range (0.00E+00,1.90E+01) resulting in fit range (2.96E-02,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumHBA)
    # normalization
    linear_norm_parameter = (0.018653528399554697, 0.5574729713710027)  # error of 1.93E-01 with sample range (0.00E+00,1.71E+02) resulting in fit range (5.57E-01,3.75E+00)
    min_max_norm_parameter = (1.2630439561525542, 7.515443068519551)  # error of 1.49E-02 with sample range (0.00E+00,1.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.367102956502307, 0.8007786530186061)  # error of 9.05E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (2.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.18861863906966, 0.9639214876904849, 0.6754396233905521)  # error of 5.33E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (1.73E-02,1.00E+00)
    genlog_norm_parameter = (0.5934193512469899, 0.22514360787120383, 1.466124180627735, 0.18363318748043794)  # error of 3.35E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (4.70E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcNPR1)
    # normalization
    linear_norm_parameter = (1.993532814687622, -0.07291478792771389)  # error of 6.17E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (-7.29E-02,1.83E+00)
    min_max_norm_parameter = (0.06225581266603365, 0.49572101730415397)  # error of 2.98E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.27514214399673687, 11.301895844887198)  # error of 2.74E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (4.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.25986760502174205, 14.138759638283469, 9.152837968958812)  # error of 1.65E-02 with sample range (0.00E+00,9.54E-01) resulting in fit range (2.47E-02,9.98E-01)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  MaxAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MaxAbsPartialCharge)
    # normalization
    linear_norm_parameter = (3.0434418303394315, -0.7918951311620821)  # error of 1.02E-01 with sample range (0.00E+00,INF) resulting in fit range (-7.92E-01,INF)
    min_max_norm_parameter = (0.2952082545313489, 0.5504603722073848)  # error of 4.52E-02 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.425391072059637, 18.052436340913527)  # error of 5.24E-02 with sample range (0.00E+00,INF) resulting in fit range (4.62E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4471126495528674, 13.358409659537145, 30.95400226537322)  # error of 4.77E-02 with sample range (0.00E+00,INF) resulting in fit range (2.54E-03,1.00E+00)
    genlog_norm_parameter = (662.0986334620413, 0.5172134029000983, 0.2733025946745107, 71.61636863748764)  # error of 4.71E-02 with sample range (0.00E+00,INF) resulting in fit range (8.54E-03,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi2n)
    # normalization
    linear_norm_parameter = (0.11014910768339037, 0.02748881833733474)  # error of 1.49E-01 with sample range (0.00E+00,1.02E+02) resulting in fit range (2.75E-02,1.13E+01)
    min_max_norm_parameter = (1.6850371430659572, 6.553874507410937)  # error of 3.80E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.081549676171799, 0.9904095492145635)  # error of 2.83E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (1.73E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.87238559628832, 1.3409369944950944, 0.767555104248775)  # error of 1.08E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (5.53E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1v)
    # normalization
    linear_norm_parameter = (0.0627452397151258, 0.08691967075790497)  # error of 1.60E-01 with sample range (0.00E+00,1.66E+02) resulting in fit range (8.69E-02,1.05E+01)
    min_max_norm_parameter = (2.910758257183072, 10.518729378759696)  # error of 3.99E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.661928594684052, 0.6389477840131871)  # error of 2.54E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (1.40E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.384313047563608, 0.8556587318530761, 0.5214673705729024)  # error of 1.05E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (4.22E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_sulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_sulfonamd)
    # normalization
    linear_norm_parameter = (0.03596176344129032, 0.9269115779579837)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (9.27E-01,1.21E+00)
    min_max_norm_parameter = (9.80315962195014e-09, 1.0385581951856868)  # error of 4.80E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.06939953825020206, 3.4983720078550626)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (4.40E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.06939885794095012, 1.0, 3.4983684786379823)  # error of 2.31E-08 with sample range (0.00E+00,8.00E+00) resulting in fit range (4.40E-01,1.00E+00)
    genlog_norm_parameter = (3.512955385541479, 0.514844672287699, 0.38483266354717705, 1.7883039930433544)  # error of 7.75E-10 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.09E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PEOE_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA9)
    # normalization
    linear_norm_parameter = (0.009582946298100165, 0.3577672573591545)  # error of 1.18E-01 with sample range (0.00E+00,9.81E+02) resulting in fit range (3.58E-01,9.76E+00)
    min_max_norm_parameter = (7.644079543108024e-15, 47.75350503201181)  # error of 3.98E-02 with sample range (0.00E+00,9.81E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (23.587460909634668, 0.10069901844268675)  # error of 1.27E-02 with sample range (0.00E+00,9.81E+02) resulting in fit range (8.51E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (22.77434626844181, 0.12981857665919092, 0.0929999907140519)  # error of 5.01E-03 with sample range (0.00E+00,9.81E+02) resulting in fit range (4.94E-02,1.00E+00)
    genlog_norm_parameter = (0.08096934092650773, -3.7749568013325585, 0.6998432950587882, 0.10931896552672962)  # error of 1.35E-03 with sample range (0.00E+00,9.81E+02) resulting in fit range (2.23E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_Ndealkylation1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ndealkylation1)
    # normalization
    linear_norm_parameter = (0.03791658750712423, 0.9210171084602388)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (9.21E-01,1.22E+00)
    min_max_norm_parameter = (6.156084770016071e-09, 1.0428249668903355)  # error of 1.06E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.20868893262353097, 2.6066541330168578)  # error of 7.85E-17 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.33E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.20868916209862504, 1.0, 2.6066534426782026)  # error of 6.89E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.33E-01,1.00E+00)
    genlog_norm_parameter = (2.622377594627108, 0.46233013533189127, 0.321583486548102, 1.8024954460958964)  # error of 6.69E-10 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.66E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(LabuteASA)
    # normalization
    linear_norm_parameter = (0.0029906658775537, -0.06099902360508458)  # error of 1.58E-01 with sample range (1.50E+01,3.79E+03) resulting in fit range (-1.62E-02,1.13E+01)
    min_max_norm_parameter = (93.68303193611298, 261.4054366254348)  # error of 3.76E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (176.2509725704667, 0.028811070902678346)  # error of 2.90E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (9.50E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (168.88110527265619, 0.038768813519735566, 0.021877794721835753)  # error of 1.06E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (2.55E-03,1.00E+00)
    genlog_norm_parameter = (0.6933180534129533, 5.290531782672155, 26.32875258879068, 0.003951618013326181)  # error of 5.70E-01 with sample range (1.50E+01,3.79E+03) resulting in fit range (3.31E-04,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcPMI1)
    # normalization
    linear_norm_parameter = (6.013420622833213e-05, 0.37438762357376315)  # error of 2.26E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (3.74E-01,3.55E+01)
    min_max_norm_parameter = (5.253076648348887e-17, 2920.8284746927293)  # error of 6.82E-02 with sample range (0.00E+00,5.85E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcLabuteASA)
    # normalization
    linear_norm_parameter = (0.0029906658775537, -0.06099902360508458)  # error of 1.58E-01 with sample range (1.50E+01,3.79E+03) resulting in fit range (-1.62E-02,1.13E+01)
    min_max_norm_parameter = (93.68303193611298, 261.4054366254348)  # error of 3.76E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (176.2509725704667, 0.028811070902678346)  # error of 2.90E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (9.50E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (168.88110527265619, 0.038768813519735566, 0.021877794721835753)  # error of 1.06E-02 with sample range (1.50E+01,3.79E+03) resulting in fit range (2.55E-03,1.00E+00)
    genlog_norm_parameter = (0.6933180534129533, 5.290531782672155, 26.32875258879068, 0.003951618013326181)  # error of 5.70E-01 with sample range (1.50E+01,3.79E+03) resulting in fit range (3.31E-04,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_imide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_imide)
    # normalization
    linear_norm_parameter = (0.016493515583597462, 0.9652131308182264)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (9.65E-01,1.13E+00)
    min_max_norm_parameter = (9.653585940122535e-09, 1.0186342360914322)  # error of 6.11E-04 with sample range (0.00E+00,1.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.7053096141126804, 2.335502419908216)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.7053096194358988, 1.0, 2.3355024086473413)  # error of 8.90E-11 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.39E-01,1.00E+00)
    genlog_norm_parameter = (2.3431472061413503, 0.38984158222989124, 0.1501936452544861, 1.9131789137956052)  # error of 6.71E-12 with sample range (0.00E+00,1.00E+01) resulting in fit range (8.47E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAromaticRings)
    # normalization
    linear_norm_parameter = (0.034866795939075525, 0.6345283877004492)  # error of 1.43E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (6.35E-01,2.76E+00)
    min_max_norm_parameter = (4.733165431326071e-30, 3.555582078939516)  # error of 1.41E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.7284808042419078, 1.2169190611775202)  # error of 6.86E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.09E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.686248103154103, 1.3303663811394981, 1.1588291961663748)  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (9.59E-02,1.00E+00)
    genlog_norm_parameter = (1.0818733953733572, 0.3821001451970787, 1.9211953644983686, 0.5468773865895864)  # error of 5.78E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (8.28E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Ipc)
    # normalization
    linear_norm_parameter = (1.5595572574111734e-38, 0.4932082097701287)  # error of 2.81E-01 with sample range (0.00E+00,INF) resulting in fit range (4.93E-01,INF)
    min_max_norm_parameter = (5e-324, 1794244951.795763)  # error of 2.42E-01 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcExactMolWt)
    # normalization
    linear_norm_parameter = (0.0012840291612334909, 0.046753757513299354)  # error of 1.65E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (8.92E-02,1.78E+01)
    min_max_norm_parameter = (175.3029880379195, 537.2110757327455)  # error of 3.98E-02 with sample range (3.31E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SMR_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA3)
    # normalization
    linear_norm_parameter = (0.011429994095032314, 0.595492552884596)  # error of 1.14E-01 with sample range (0.00E+00,2.58E+02) resulting in fit range (5.95E-01,3.55E+00)
    min_max_norm_parameter = (0.0, 18.19914049080423)  # error of 5.20E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (8.390853699132085, 0.20684364498704194)  # error of 2.62E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (1.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (8.015271352886963, 0.25797161577301486, 0.1954104655375538)  # error of 2.55E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (1.12E-01,1.00E+00)
    genlog_norm_parameter = (0.1764602483188016, -5.670846348806433, 2.0149145147528085, 0.2300791634713772)  # error of 2.52E-02 with sample range (0.00E+00,2.58E+02) resulting in fit range (8.99E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_C_S_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_C_S)
    # normalization
    linear_norm_parameter = (0.4948954594086534, 0.4948954594086533)  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.95E-01,2.47E+00)
    min_max_norm_parameter = (5.535786530479266e-12, 1.010314379200814)  # error of 1.64E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.1387416964001449, 2.138741698930851)  # error of 4.26E-10 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.19E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.1387416964001449, 1.0, 2.138741698930851)  # error of 4.26E-10 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.19E-01,1.00E+00)
    genlog_norm_parameter = (1.10445574196643, 0.2930520726655979, 0.04140788879731429, 1.8310088594492906)  # error of 1.63E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.70E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumSaturatedRings)
    # normalization
    linear_norm_parameter = (0.037486269092924296, 0.7906195585254473)  # error of 6.21E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.91E-01,1.92E+00)
    min_max_norm_parameter = (5.730855425109739e-09, 1.404946368678208)  # error of 1.95E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.36617461217911906, 1.4317359459562997)  # error of 2.83E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.72E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.36617461217911906, 1.0, 1.4317359459562997)  # error of 2.83E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.72E-01,1.00E+00)
    genlog_norm_parameter = (2.361126814701105, -0.9332372421941358, 0.010759488748759415, 0.0003109299935290176)  # error of 1.37E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (2.20E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAliphaticRings)
    # normalization
    linear_norm_parameter = (0.04716081590128, 0.7025071278256165)  # error of 8.86E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (7.03E-01,2.68E+00)
    min_max_norm_parameter = (1.5777218104420236e-29, 2.223682308863912)  # error of 2.88E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.7813916309499217, 1.3224930670990482)  # error of 4.20E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.7813895517185011, 19.442917588992998, 1.322487361093461)  # error of 2.34E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.52E-07,1.00E+00)
    genlog_norm_parameter = (1.5963506985004239, -1.5768456408816305, 0.0075354951062004665, 0.0001966513751535621)  # error of 1.55E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (4.55E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_NH0_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_NH0)
    # normalization
    linear_norm_parameter = (0.06607380335566115, 0.5226975405536398)  # error of 1.02E-01 with sample range (0.00E+00,6.40E+01) resulting in fit range (5.23E-01,4.75E+00)
    min_max_norm_parameter = (1.5777218104420236e-30, 3.4322660362420145)  # error of 3.03E-02 with sample range (0.00E+00,6.40E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.446568823597894, 0.9242400863243287)  # error of 3.76E-03 with sample range (0.00E+00,6.40E+01) resulting in fit range (2.08E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.4465608187161392, 10.288469526455993, 0.9242182221199369)  # error of 2.07E-03 with sample range (0.00E+00,6.40E+01) resulting in fit range (3.44E-07,1.00E+00)
    genlog_norm_parameter = (0.8438864537468908, 0.8863761551955617, 0.7290881665632817, 0.5450332276288733)  # error of 1.54E-03 with sample range (0.00E+00,6.40E+01) resulting in fit range (1.81E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_alkyl_halide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_alkyl_halide)
    # normalization
    linear_norm_parameter = (0.008601680392056554, 0.9308092271694038)  # error of 1.58E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (9.31E-01,1.81E+00)
    min_max_norm_parameter = (9.154473872975373e-09, 1.0948072482932691)  # error of 1.38E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-3.20022604341129, 0.5453458753101628)  # error of 5.09E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.51E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-3.200226429591724, 1.0, 0.5453458328225158)  # error of 5.09E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.51E-01,1.00E+00)
    genlog_norm_parameter = (1.5705429629004721, 3.467282907183939, 2.799707118499844, 52.63151277254762)  # error of 3.04E-03 with sample range (0.00E+00,1.02E+02) resulting in fit range (8.84E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_hdrzone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_hdrzone)
    # normalization
    linear_norm_parameter = (0.4927256546910781, 0.492725654691078)  # error of 4.44E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.93E-01,2.46E+00)
    min_max_norm_parameter = (6.061851638299241e-09, 1.014763479830144)  # error of 2.35E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.0531923508789955, 2.053192297635956)  # error of 4.24E-12 with sample range (0.00E+00,4.00E+00) resulting in fit range (8.97E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.0531923508789955, 1.0, 2.053192297635956)  # error of 4.24E-12 with sample range (0.00E+00,4.00E+00) resulting in fit range (8.97E-01,1.00E+00)
    genlog_norm_parameter = (1.1022639184961518, 0.30018146782599797, 0.05858368625557883, 1.8236846488672203)  # error of 8.42E-09 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.58E-01,9.99E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0n)
    # normalization
    linear_norm_parameter = (0.009618083281699108, 0.17347984942891348)  # error of 2.14E-01 with sample range (7.63E-01,6.43E+02) resulting in fit range (1.81E-01,6.36E+00)
    min_max_norm_parameter = (16.004022253403853, 51.11689241252734)  # error of 4.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (33.37586607594709, 0.1432287247282441)  # error of 2.67E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.28E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (31.98836545966142, 0.18812033507921852, 0.1136090082864722)  # error of 1.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (2.80E-03,1.00E+00)
    genlog_norm_parameter = (0.09991708558883805, -13.42358800932477, 0.271792016045397, 0.0039352878486176145)  # error of 1.01E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.15E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAromaticCarbocycles)
    # normalization
    linear_norm_parameter = (0.029344509343809166, 0.7342345249533615)  # error of 1.10E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (7.34E-01,2.52E+00)
    min_max_norm_parameter = (9.860761315262648e-32, 2.4631545244076647)  # error of 1.96E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0872508968285497, 1.3660589721714311)  # error of 6.47E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.85E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0872509124487726, 13.4357718685885, 1.366059068627403)  # error of 4.24E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.53E-07,1.00E+00)
    genlog_norm_parameter = (1.329356595623853, -2.22234483458619, 0.009613465218557701, 0.0001605349922075129)  # error of 1.21E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.42E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  BCUT2D_MWLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_MWLOW)
    # normalization
    linear_norm_parameter = (1.018429176169886, 2.7560478377646005)  # error of 6.41E-02 with sample range (-3.04E+00,3.04E+01) resulting in fit range (-3.35E-01,3.38E+01)
    min_max_norm_parameter = (-2.64231464861815, -1.799802847378942)  # error of 2.41E-02 with sample range (-3.04E+00,3.04E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.2221798051728805, 5.9554575147054996)  # error of 9.16E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (7.81E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.227857345416726, 6.237138211059447, 5.6956729965539425)  # error of 7.90E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (6.45E-03,1.00E+00)
    genlog_norm_parameter = (5.497690839708295, -1.5915933694558018, 0.02130998762071418, 0.7613978532016417)  # error of 7.69E-03 with sample range (-3.04E+00,3.04E+01) resulting in fit range (4.55E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PEOE_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA10)
    # normalization
    linear_norm_parameter = (0.008224721394031964, 0.6823272493410806)  # error of 9.26E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (6.82E-01,4.36E+00)
    min_max_norm_parameter = (3.4024230320080053e-25, 18.337088617478788)  # error of 7.24E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.490192767810458, 0.173177943098159)  # error of 1.90E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (2.45E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.256034233577422, 0.34399533607200733, 0.168166619388783)  # error of 1.70E-02 with sample range (0.00E+00,4.47E+02) resulting in fit range (1.04E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA1)
    # normalization
    linear_norm_parameter = (0.005573163277374748, 0.6734703650713099)  # error of 1.59E-01 with sample range (0.00E+00,4.11E+02) resulting in fit range (6.73E-01,2.97E+00)
    min_max_norm_parameter = (1.2124620489038405, 22.986430313946542)  # error of 4.38E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (12.034307131924646, 0.21632262910541178)  # error of 2.81E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (6.89E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (11.249123373426334, 0.3444808633220492, 0.1894116980918965)  # error of 2.51E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (2.03E-02,1.00E+00)
    genlog_norm_parameter = (0.17343236164840786, -2.160691929113601, 0.6350478546700745, 0.07906535726568056)  # error of 2.56E-02 with sample range (0.00E+00,4.11E+02) resulting in fit range (1.02E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  GetSSSR_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(GetSSSR)
    # normalization
    linear_norm_parameter = (0.03860180770005206, 0.5113290906759573)  # error of 1.73E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (5.11E-01,2.87E+00)
    min_max_norm_parameter = (0.334517739633367, 5.208998938983079)  # error of 1.60E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.7238161788458215, 1.1018114352614388)  # error of 1.08E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.74E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.5715742346316732, 1.2832261760477313, 0.9174927781864283)  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (3.56E-02,1.00E+00)
    genlog_norm_parameter = (0.813320209949437, 1.988272685010793, 0.21462517997652011, 0.17389824103982)  # error of 4.52E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.48E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi4v)
    # normalization
    linear_norm_parameter = (0.1846796920584458, 0.12308022240634398)  # error of 1.49E-01 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.23E-01,4.41E+01)
    min_max_norm_parameter = (0.4365031459643245, 3.319224643809378)  # error of 3.97E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.8494300166801143, 1.6807292424962799)  # error of 3.13E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (4.28E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.7216523305680784, 2.287102552977168, 1.2587726126085808)  # error of 1.11E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.91E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  EState_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA3)
    # normalization
    linear_norm_parameter = (0.003554939219269828, 0.9223002361310408)  # error of 1.14E-02 with sample range (0.00E+00,2.03E+02) resulting in fit range (9.22E-01,1.65E+00)
    min_max_norm_parameter = (1.1454288690296624e-20, 3.325963599839221)  # error of 3.62E-02 with sample range (0.00E+00,2.03E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-12.712321107593572, 0.14776745606907848)  # error of 7.76E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (8.67E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-12.712321897413275, 0.999999999999936, 0.14776744978227158)  # error of 7.76E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (8.67E-01,1.00E+00)
    genlog_norm_parameter = (0.37954395120713624, 12.868898798516755, 2.477387761175603, 55.419351229090374)  # error of 7.16E-03 with sample range (0.00E+00,2.03E+02) resulting in fit range (9.01E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Asphericity)
    # normalization
    linear_norm_parameter = (1.4721378049698663, -0.11228706292547166)  # error of 2.90E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-1.12E-01,1.36E+00)
    min_max_norm_parameter = (0.09005620590041596, 0.7355219615267257)  # error of 1.68E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.40970204866580884, 7.643961121304289)  # error of 2.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.18E-02,9.89E-01)
    dual_sigmoidal_norm_parameter = (0.3993757142716246, 8.419588600488629, 6.968980500840633)  # error of 1.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (3.35E-02,9.85E-01)
    genlog_norm_parameter = (6.206340781992686, 0.43464496231710853, 0.25779424922808303, 0.39659201683500284)  # error of 1.81E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.89E-02,9.81E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Phi_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcPhi)
    # normalization
    linear_norm_parameter = (0.22667929243030271, 0.11757079564019635)  # error of 1.70E-01 with sample range (0.00E+00,2.25E+04) resulting in fit range (1.18E-01,5.10E+03)
    min_max_norm_parameter = (0.5556463225317848, 2.6045119614877015)  # error of 3.59E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.5662194643242748, 2.3640636223858147)  # error of 2.70E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (2.41E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.4899535284636134, 3.0418615386053856, 1.8570405999560915)  # error of 1.21E-02 with sample range (0.00E+00,2.25E+04) resulting in fit range (1.06E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  VSA_EState4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState4)
    # normalization
    linear_norm_parameter = (0.03989308824193185, 0.6890187770089706)  # error of 1.54E-01 with sample range (-3.80E+02,5.80E+01) resulting in fit range (-1.45E+01,3.00E+00)
    min_max_norm_parameter = (-9.819105425993582, 0.9849828760699441)  # error of 4.39E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-4.310085634921858, 0.4319322795706566)  # error of 2.94E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (3.08E-71,1.00E+00)
    dual_sigmoidal_norm_parameter = (-3.8709253184243924, 0.33197642799656213, 0.5895773862269263)  # error of 1.27E-02 with sample range (-3.80E+02,5.80E+01) resulting in fit range (5.53E-55,1.00E+00)
    genlog_norm_parameter = (0.9492760171365159, -2.7790693678579554, 4.891735878037185, 4.093037272751542)  # error of 5.69E-03 with sample range (-3.80E+02,5.80E+01) resulting in fit range (6.56E-39,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_ketone_Topliss_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_ketone_Topliss)
    # normalization
    linear_norm_parameter = (0.03132218100370976, 0.9335059844613983)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (9.34E-01,1.25E+00)
    min_max_norm_parameter = (7.195279033997925e-09, 1.0364539879128276)  # error of 1.24E-03 with sample range (0.00E+00,1.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.47566559195294444, 2.2442106325557347)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.44E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.4756655979317486, 1.0, 2.2442106182230197)  # error of 1.95E-10 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.44E-01,1.00E+00)
    genlog_norm_parameter = (2.2577988570731184, 0.4264538172114882, 0.24900241673381004, 1.8427239441681684)  # error of 7.28E-14 with sample range (0.00E+00,1.00E+01) resulting in fit range (7.61E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumSaturatedHeterocycles)
    # normalization
    linear_norm_parameter = (0.06338179563839252, 0.7789048985591301)  # error of 4.13E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.79E-01,2.68E+00)
    min_max_norm_parameter = (5.3308083108781935e-09, 1.2539605902115512)  # error of 8.89E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.24911412739948294, 1.825541426846478)  # error of 5.15E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.88E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.24911412739948294, 1.0, 1.825541426846478)  # error of 5.15E-04 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.88E-01,1.00E+00)
    genlog_norm_parameter = (3.0410569853018057, -0.5365635014003982, 0.010320409148970734, 0.00041830035148444656)  # error of 6.76E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (8.06E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(HallKierAlpha)
    # normalization
    linear_norm_parameter = (0.10941027797520916, 0.8206630827667591)  # error of 1.86E-01 with sample range (-7.86E+01,6.10E+01) resulting in fit range (-7.78E+00,7.50E+00)
    min_max_norm_parameter = (-4.025047303628864, -0.27476699527518833)  # error of 2.37E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.136641082271498, 1.3958219191517867)  # error of 1.43E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (4.45E-47,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.0451007845368863, 1.1887380822440157, 1.5856961746935594)  # error of 8.43E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (3.00E-40,1.00E+00)
    genlog_norm_parameter = (1.7618968864206013, -2.0459113897478645, 2.4289416394683374, 1.8402620037112714)  # error of 7.26E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (9.11E-33,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  GetFormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(GetFormalCharge)
    # normalization
    linear_norm_parameter = (0.2979676829070069, 0.49840014398594723)  # error of 1.37E-01 with sample range (-1.20E+01,8.00E+00) resulting in fit range (-3.08E+00,2.88E+00)
    min_max_norm_parameter = (-0.7560934218119267, 0.768364201281212)  # error of 2.79E-03 with sample range (-1.20E+01,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0034213113586227585, 4.843231446590721)  # error of 7.75E-04 with sample range (-1.20E+01,8.00E+00) resulting in fit range (5.65E-26,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.0034772349678885662, 5.1333954284899646, 4.629536470490271)  # error of 4.18E-04 with sample range (-1.20E+01,8.00E+00) resulting in fit range (1.74E-27,1.00E+00)
    genlog_norm_parameter = (4.571552065705464, 0.15901769882795194, 0.39319359293010914, 0.8488246235209407)  # error of 4.19E-04 with sample range (-1.20E+01,8.00E+00) resulting in fit range (1.09E-28,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  VSA_EState5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState5)
    # normalization
    linear_norm_parameter = (0.028069497462787085, 0.6384948010105149)  # error of 1.67E-01 with sample range (-3.60E+02,3.76E+02) resulting in fit range (-9.46E+00,1.12E+01)
    min_max_norm_parameter = (-11.200860573507633, 1.4339724783749572)  # error of 4.64E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-4.751658309551578, 0.3690396610731733)  # error of 3.12E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (1.23E-57,1.00E+00)
    dual_sigmoidal_norm_parameter = (-4.224592012720491, 0.2819947410204209, 0.5221323374749486)  # error of 1.32E-02 with sample range (-3.60E+02,3.76E+02) resulting in fit range (2.81E-44,1.00E+00)
    genlog_norm_parameter = (0.9230351619235945, -3.3573920880339525, 9.424012632118597, 4.705755656931828)  # error of 5.93E-03 with sample range (-3.60E+02,3.76E+02) resulting in fit range (2.66E-31,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_guanido_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_guanido)
    # normalization
    linear_norm_parameter = (0.4928506434420904, 0.4928506434420903)  # error of 3.33E-16 with sample range (0.00E+00,9.00E+00) resulting in fit range (4.93E-01,4.93E+00)
    min_max_norm_parameter = (5.606358540599248e-09, 1.01450613205635)  # error of 2.91E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.057470294719679, 2.0574701969564333)  # error of 5.63E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (8.98E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.057470294719679, 1.0, 2.0574701969564333)  # error of 5.63E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (8.98E-01,1.00E+00)
    genlog_norm_parameter = (1.1023910990444217, 0.2997683891694997, 0.05760132412520002, 1.8241080628426662)  # error of 8.76E-09 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.59E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumRadicalElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumRadicalElectrons)
    # normalization
    linear_norm_parameter = (0.0019698227137791013, 0.9930256276934915)  # error of 3.18E-04 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.93E-01,1.24E+00)
    min_max_norm_parameter = (7.363905475264589e-09, 1.0052570210496823)  # error of 7.40E-04 with sample range (0.00E+00,1.26E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-6.451319539635213, 0.7043741012907843)  # error of 5.68E-06 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.89E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-6.45131932074973, 1.0, 0.7043741212164152)  # error of 5.68E-06 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.89E-01,1.00E+00)
    genlog_norm_parameter = (0.6858092699432776, -1.9972543632647004, -0.6048803487070232, -15.372406220895236)  # error of 8.48E-12 with sample range (0.00E+00,1.26E+02) resulting in fit range (9.89E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SMR_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA6)
    # normalization
    linear_norm_parameter = (0.007356009115044526, 0.5604258193276979)  # error of 1.01E-01 with sample range (0.00E+00,7.02E+02) resulting in fit range (5.60E-01,5.72E+00)
    min_max_norm_parameter = (4.58167181735802e-19, 33.13532929564598)  # error of 5.79E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (13.927321244136747, 0.10991207620783901)  # error of 1.58E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (1.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (13.138113263168435, 0.17400805877550418, 0.10256931868406938)  # error of 1.06E-02 with sample range (0.00E+00,7.02E+02) resulting in fit range (9.23E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BalabanJ)
    # normalization
    linear_norm_parameter = (0.17660976361946912, 0.011623897998747967)  # error of 1.60E-01 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.16E-02,8.95E+00)
    min_max_norm_parameter = (1.51069451962931, 3.7663482034024947)  # error of 4.43E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.6275405979312487, 2.067375370708902)  # error of 2.69E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (4.35E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.548536235030395, 2.66279982408227, 1.6191594985042048)  # error of 1.47E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.13E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(TPSA)
    # normalization
    linear_norm_parameter = (0.0021678074691265925, 0.4909310697599548)  # error of 1.90E-01 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.91E-01,6.20E+00)
    min_max_norm_parameter = (14.599527973028703, 118.60656324496115)  # error of 3.84E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (66.26559293182696, 0.047786843726724)  # error of 1.40E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.04E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (64.41269346904815, 0.05696911545435016, 0.043756733059762035)  # error of 9.63E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (2.49E-02,1.00E+00)
    genlog_norm_parameter = (0.039396470555296006, -1.565855746207205, 3.3127216535027557, 0.3094077325658369)  # error of 7.60E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (1.03E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi2n)
    # normalization
    linear_norm_parameter = (0.11014910768339037, 0.02748881833733474)  # error of 1.49E-01 with sample range (0.00E+00,1.02E+02) resulting in fit range (2.75E-02,1.13E+01)
    min_max_norm_parameter = (1.6850371430659572, 6.553874507410937)  # error of 3.80E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.081549676171799, 0.9904095492145635)  # error of 2.83E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (1.73E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.87238559628832, 1.3409369944950944, 0.767555104248775)  # error of 1.08E-02 with sample range (0.00E+00,1.02E+02) resulting in fit range (5.53E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_nitrile_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_nitrile)
    # normalization
    linear_norm_parameter = (0.023442890139887318, 0.9508294253517185)  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.51E-01,1.33E+00)
    min_max_norm_parameter = (8.341666743159058e-09, 1.0264070772459553)  # error of 6.79E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.4863072489946901, 2.445068623176201)  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.67E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.48631181197205164, 1.0, 2.445057157239258)  # error of 1.08E-07 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.67E-01,1.00E+00)
    genlog_norm_parameter = (2.455580784071338, 0.4112644239917774, 0.21297630106333174, 1.8782563837946038)  # error of 1.51E-12 with sample range (0.00E+00,1.60E+01) resulting in fit range (7.83E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAliphaticCarbocycles)
    # normalization
    linear_norm_parameter = (0.029386355676372537, 0.8723594863010827)  # error of 2.62E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (8.72E-01,1.40E+00)
    min_max_norm_parameter = (7.2035108451548135e-09, 1.1561266295897044)  # error of 1.01E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.4231823522456474, 1.309702732778145)  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.4231823921093463, 1.0, 1.3097026985565412)  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    genlog_norm_parameter = (3.5827272813887046, -0.1961244562254873, 0.008501771849148765, 0.0007936347601062923)  # error of 9.27E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (5.02E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  EState_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA6)
    # normalization
    linear_norm_parameter = (0.0007744620095372357, 0.9845100004329747)  # error of 3.11E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.85E-01,1.05E+00)
    min_max_norm_parameter = (2.427726210776916e-14, 4.450565380641523)  # error of 7.79E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-15.161431554355305, 0.2124627730460597)  # error of 2.28E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-15.161441858605182, 1.0, 0.2124626681786106)  # error of 2.28E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.62E-01,1.00E+00)
    genlog_norm_parameter = (0.21177775611437727, -0.5350000495135983, 0.0009499838705636394, 0.021464249004377132)  # error of 2.27E-03 with sample range (0.00E+00,9.04E+01) resulting in fit range (9.61E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  EState_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA1)
    # normalization
    linear_norm_parameter = (0.004092014874662153, -0.010698392745672747)  # error of 1.49E-01 with sample range (0.00E+00,3.00E+03) resulting in fit range (-1.07E-02,1.23E+01)
    min_max_norm_parameter = (52.7894991169631, 182.85287865533604)  # error of 3.50E-02 with sample range (0.00E+00,3.00E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (117.01950459885676, 0.03736764559445686)  # error of 2.35E-02 with sample range (0.00E+00,3.00E+03) resulting in fit range (1.25E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (112.5976748795547, 0.04708036335381056, 0.029919332045275523)  # error of 8.64E-03 with sample range (0.00E+00,3.00E+03) resulting in fit range (4.96E-03,1.00E+00)
    genlog_norm_parameter = (0.02620707258119035, -21.837253264138216, 0.6590028020453111, 0.026268439974705866)  # error of 5.55E-03 with sample range (0.00E+00,3.00E+03) resulting in fit range (5.93E-06,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAromaticCarbocycles)
    # normalization
    linear_norm_parameter = (0.029344509343809166, 0.7342345249533615)  # error of 1.10E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (7.34E-01,2.52E+00)
    min_max_norm_parameter = (9.860761315262648e-32, 2.4631545244076647)  # error of 1.96E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0872508968285497, 1.3660589721714311)  # error of 6.47E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.85E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0872509124487726, 13.4357718685885, 1.366059068627403)  # error of 4.24E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.53E-07,1.00E+00)
    genlog_norm_parameter = (1.329356595623853, -2.22234483458619, 0.009613465218557701, 0.0001605349922075129)  # error of 1.21E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (4.42E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAromaticHeterocycles)
    # normalization
    linear_norm_parameter = (0.06608705216326816, 0.7231759141671211)  # error of 6.12E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (7.23E-01,1.71E+00)
    min_max_norm_parameter = (2.425436375356735e-09, 1.4183360150961999)  # error of 2.28E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4595690682034204, 1.6119949186776596)  # error of 6.29E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.23E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4595690682034204, 1.0, 1.6119949186776596)  # error of 6.29E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.23E-01,1.00E+00)
    genlog_norm_parameter = (1.6718834720391067, 1.1087698014134977, 0.5469129170165173, 1.4432376478453104)  # error of 5.24E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.53E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SlogP_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA6)
    # normalization
    linear_norm_parameter = (0.0046552542351420145, 0.4184194987223895)  # error of 1.54E-01 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.18E-01,8.79E+00)
    min_max_norm_parameter = (5.914526578091133e-16, 72.98691414016453)  # error of 4.02E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (35.2283623788487, 0.06244680638294594)  # error of 1.72E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (9.98E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (33.5836002536413, 0.07806210477123002, 0.05645850654142751)  # error of 1.20E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (6.78E-02,1.00E+00)
    genlog_norm_parameter = (0.05016216584871228, -4.018204948640934, 0.9520860993587169, 0.18739663596243378)  # error of 1.17E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.63E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  MolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MolWt)
    # normalization
    linear_norm_parameter = (0.0012559144823710566, 0.056127425145544585)  # error of 1.68E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (9.77E-02,1.74E+01)
    min_max_norm_parameter = (175.4052910230421, 538.3280010265217)  # error of 4.00E-02 with sample range (3.31E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 5.02E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(ExactMolWt)
    # normalization
    linear_norm_parameter = (0.0012840291612334909, 0.046753757513299354)  # error of 1.65E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (8.92E-02,1.78E+01)
    min_max_norm_parameter = (175.3029880379195, 537.2110757327455)  # error of 3.98E-02 with sample range (3.31E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 5.01E-01 with sample range (3.31E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PMI1)
    # normalization
    linear_norm_parameter = (6.013420622833213e-05, 0.37438762357376315)  # error of 2.26E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (3.74E-01,3.55E+01)
    min_max_norm_parameter = (5.253076648348887e-17, 2920.8284746927293)  # error of 6.82E-02 with sample range (0.00E+00,5.85E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,5.85E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcRadiusOfGyration)
    # normalization
    linear_norm_parameter = (0.274743032958996, -0.6363152066427643)  # error of 8.52E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (-6.36E-01,2.85E+00)
    min_max_norm_parameter = (2.68976145372615, 5.459084333173108)  # error of 2.97E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.059081941807601, 1.764235395780906)  # error of 1.73E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (7.76E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.992187925992685, 2.0752401030308176, 1.5066909435317473)  # error of 7.09E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (2.52E-04,1.00E+00)
    genlog_norm_parameter = (1.347041014911159, 2.24246525276504, 1.9688953365934632, 0.23779698513445705)  # error of 2.95E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (1.59E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BCUT2D_MRLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_MRLOW)
    # normalization
    linear_norm_parameter = (1.1473576459333634, 3.176799142930221)  # error of 5.86E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (-3.35E-01,9.95E+00)
    min_max_norm_parameter = (-2.7210155512608267, -1.9615807430024825)  # error of 2.64E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.3468285836092604, 6.523147526382408)  # error of 1.96E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (9.42E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.3658211705530543, 7.709376024364005, 5.6043685967967205)  # error of 1.15E-02 with sample range (-3.06E+00,5.91E+00) resulting in fit range (4.70E-03,1.00E+00)
    genlog_norm_parameter = (5.026733571856869, -1.6651300778889573, 0.0057764626146007475, 0.2480965549443172)  # error of 9.54E-03 with sample range (-3.06E+00,5.91E+00) resulting in fit range (3.09E-04,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  VSA_EState3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState3)
    # normalization
    linear_norm_parameter = (0.0474383914162283, 0.3904581813825524)  # error of 1.52E-01 with sample range (-1.08E+02,2.11E+02) resulting in fit range (-4.72E+00,1.04E+01)
    min_max_norm_parameter = (-3.8927946674623195, 6.642361773646212)  # error of 5.95E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.3162739635967782, 0.43946395350992634)  # error of 4.76E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (1.53E-21,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.1940012505116652, 0.8480462897882002, 0.2641566203010557)  # error of 1.52E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (1.75E-40,1.00E+00)
    genlog_norm_parameter = (0.30648757235124613, -10.278556574658467, 0.0015813657329094423, 7.008328975788321e-05)  # error of 3.11E-02 with sample range (-1.08E+02,2.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_ether_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_ether)
    # normalization
    linear_norm_parameter = (0.020806809203217647, 0.8260344750786837)  # error of 6.86E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (8.26E-01,1.80E+00)
    min_max_norm_parameter = (0.007090101184776188, 1.486936690120486)  # error of 2.42E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.40287571370199426, 1.2035721351882562)  # error of 3.32E-03 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.81E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4028757153548411, 1.0, 1.2035721372607417)  # error of 3.32E-03 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.81E-01,1.00E+00)
    genlog_norm_parameter = (2.0153320810582254, -1.2747366222229382, 0.012774368189050011, 0.00029415961351168265)  # error of 1.66E-02 with sample range (0.00E+00,4.70E+01) resulting in fit range (3.60E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumValenceElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumValenceElectrons)
    # normalization
    linear_norm_parameter = (0.0012640047970202684, 0.37615455634302164)  # error of 2.30E-01 with sample range (9.00E+00,2.83E+03) resulting in fit range (3.88E-01,3.96E+00)
    min_max_norm_parameter = (65.29068287368149, 197.1901127519803)  # error of 2.92E-02 with sample range (9.00E+00,2.83E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (130.58040996825486, 0.038386270577942194)  # error of 2.24E-02 with sample range (9.00E+00,2.83E+03) resulting in fit range (9.31E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (122.38497312618034, 0.05621287995999412, 0.027604086661787704)  # error of 9.40E-03 with sample range (9.00E+00,2.83E+03) resulting in fit range (1.70E-03,1.00E+00)
    genlog_norm_parameter = (0.02628595735040041, -8.511479126521245, 0.0036924624761911205, 0.00015020436373755752)  # error of 8.31E-03 with sample range (9.00E+00,2.83E+03) resulting in fit range (1.86E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_N_O_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_N_O)
    # normalization
    linear_norm_parameter = (0.004979551840334274, 0.9874611284984353)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.87E-01,1.05E+00)
    min_max_norm_parameter = (4.290831526052814e-09, 1.0076168981970937)  # error of 8.76E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-3.51573557027191, 1.0800867262295535)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-3.515735715368847, 1.0, 1.0800866688067339)  # error of 6.17E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    genlog_norm_parameter = (1.0824922898653329, 0.36677431272929406, 0.029720727429199463, 1.9588719209073335)  # error of 1.45E-13 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.78E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcNPR2)
    # normalization
    linear_norm_parameter = (3.0363848124092283, -2.1050145083551772)  # error of 7.55E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-2.11E+00,9.31E-01)
    min_max_norm_parameter = (0.7328377706938303, 0.9989929606321605)  # error of 3.95E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.8690167433696966, 18.125099021117805)  # error of 3.57E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.44E-07,9.15E-01)
    dual_sigmoidal_norm_parameter = (0.8826119375023386, 13.441774499098555, 25.188219083401254)  # error of 1.76E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (7.04E-06,9.51E-01)
    genlog_norm_parameter = (41.229691541183385, 0.9463746291627356, 1.2160790825123706, 4.408357021765385)  # error of 1.41E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.37E-04,9.72E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0n)
    # normalization
    linear_norm_parameter = (0.009618083281699108, 0.17347984942891348)  # error of 2.14E-01 with sample range (7.63E-01,6.43E+02) resulting in fit range (1.81E-01,6.36E+00)
    min_max_norm_parameter = (16.004022253403853, 51.11689241252734)  # error of 4.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (33.37586607594709, 0.1432287247282441)  # error of 2.67E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.28E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (31.98836545966142, 0.18812033507921852, 0.1136090082864722)  # error of 1.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (2.80E-03,1.00E+00)
    genlog_norm_parameter = (0.09991708558883805, -13.42358800932477, 0.271792016045397, 0.0039352878486176145)  # error of 1.01E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.15E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  MolMR_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MolMR)
    # normalization
    linear_norm_parameter = (0.005557720699731816, -0.061030996122217296)  # error of 1.53E-01 with sample range (0.00E+00,2.20E+03) resulting in fit range (-6.10E-02,1.22E+01)
    min_max_norm_parameter = (48.121442195030106, 142.36977751407525)  # error of 3.62E-02 with sample range (0.00E+00,2.20E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (94.48683995909174, 0.05138768339127881)  # error of 2.83E-02 with sample range (0.00E+00,2.20E+03) resulting in fit range (7.73E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.74E-01 with sample range (0.00E+00,2.20E+03) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.74E-01 with sample range (0.00E+00,2.20E+03) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'sig'
    # functions
    
    

class  Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Eccentricity)
    # normalization
    linear_norm_parameter = (4.5706147462074735, -3.8232476425495223)  # error of 1.44E-01 with sample range (0.00E+00,1.00E+00) resulting in fit range (-3.82E+00,7.47E-01)
    min_max_norm_parameter = (0.9128972962156874, 1.0)  # error of 8.67E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.9575243366977794, 40.76860157160623)  # error of 6.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.11E-17,8.50E-01)
    dual_sigmoidal_norm_parameter = (0.9687366557382439, 23.541266354615367, 80.80169522056022)  # error of 2.49E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.25E-10,9.26E-01)
    genlog_norm_parameter = (2587.2961331618844, 0.9990899523225164, 1.6270117230482044, 136.40842975157108)  # error of 9.91E-03 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,9.99E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  FpDensityMorgan3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(FpDensityMorgan3)
    # normalization
    linear_norm_parameter = (0.30167675052364684, -0.3594212717972325)  # error of 1.33E-01 with sample range (1.72E-02,5.00E+00) resulting in fit range (-3.54E-01,1.15E+00)
    min_max_norm_parameter = (2.026192791712374, 3.2639324204504474)  # error of 3.99E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.6480344039052963, 3.9835088427839342)  # error of 2.03E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (2.81E-05,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.6898921449193867, 3.403156292639231, 5.419455652683462)  # error of 1.41E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (1.12E-04,1.00E+00)
    genlog_norm_parameter = (6.765878096344304, 2.8853651345440103, 1.2021929989946318, 2.6037279622396685)  # error of 8.55E-03 with sample range (1.72E-02,5.00E+00) resulting in fit range (5.40E-04,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NHOHCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NHOHCount)
    # normalization
    linear_norm_parameter = (0.011451805303745677, 0.8103893376864633)  # error of 1.09E-01 with sample range (0.00E+00,8.20E+01) resulting in fit range (8.10E-01,1.75E+00)
    min_max_norm_parameter = (9.860761315262648e-32, 3.187051718282465)  # error of 1.94E-02 with sample range (0.00E+00,8.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.3178308349589882, 1.1160540358868796)  # error of 6.16E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.87E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.3178308564763852, 10.470315521888269, 1.1160541755540871)  # error of 3.83E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.02E-06,1.00E+00)
    genlog_norm_parameter = (0.9056090949083335, -2.61356505138166, 0.03003179441465175, 0.0012602623451871441)  # error of 4.05E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.07E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA12)
    # normalization
    linear_norm_parameter = (0.006473147723920936, 0.807818201399682)  # error of 7.84E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (8.08E-01,2.45E+00)
    min_max_norm_parameter = (1.6316394719639418e-20, 8.845548026863177)  # error of 6.74E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.364978109571876, 0.25437535485941726)  # error of 2.74E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (3.54E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.407365640802542, -0.060543838757735835, 0.2561035154440695)  # error of 2.73E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (5.36E-01,1.00E+00)
    genlog_norm_parameter = (0.23768085743263392, 0.4643139902655747, 0.5007417190841468, 0.380066816602528)  # error of 2.73E-02 with sample range (0.00E+00,2.54E+02) resulting in fit range (3.11E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_azide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_azide)
    # normalization
    linear_norm_parameter = (0.4959878610925019, 0.4959878610925017)  # error of 3.33E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.96E-01,1.98E+00)
    min_max_norm_parameter = (2.1583046536306194e-11, 1.0080891853213716)  # error of 3.25E-05 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.1948180919512132, 2.1948180318881927)  # error of 5.06E-09 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.32E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.1948180919512132, 1.0, 2.1948180318881927)  # error of 5.06E-09 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.32E-01,1.00E+00)
    genlog_norm_parameter = (1.10554537439295, 0.28949771504214294, 0.032663358017739196, 1.834674601359803)  # error of 2.23E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.76E-01,9.99E-01)
    preferred_normalization = 'unity'
    # functions
    
    

class  Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi3n)
    # normalization
    linear_norm_parameter = (0.16045617502782894, 0.06275167455712582)  # error of 1.47E-01 with sample range (0.00E+00,6.23E+01) resulting in fit range (6.28E-02,1.01E+01)
    min_max_norm_parameter = (0.8698290958911558, 4.249166143207709)  # error of 3.75E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.5307853903255304, 1.4301006733612187)  # error of 2.83E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (2.61E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.390012863113382, 1.9077779294117436, 1.1008446640044367)  # error of 1.03E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (1.04E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcEccentricity)
    # normalization
    linear_norm_parameter = (4.5706147462074735, -3.8232476425495223)  # error of 1.44E-01 with sample range (0.00E+00,1.00E+00) resulting in fit range (-3.82E+00,7.47E-01)
    min_max_norm_parameter = (0.9128972962156874, 1.0)  # error of 8.67E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.9575243366977794, 40.76860157160623)  # error of 6.20E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.11E-17,8.50E-01)
    dual_sigmoidal_norm_parameter = (0.9687366557382439, 23.541266354615367, 80.80169522056022)  # error of 2.49E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.25E-10,9.26E-01)
    genlog_norm_parameter = (2587.2961331618844, 0.9990899523225164, 1.6270117230482044, 136.40842975157108)  # error of 9.91E-03 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,9.99E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NOCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NOCount)
    # normalization
    linear_norm_parameter = (0.0021474686509596053, 0.8235464950362099)  # error of 1.92E-01 with sample range (0.00E+00,1.96E+02) resulting in fit range (8.24E-01,1.24E+00)
    min_max_norm_parameter = (1.2880062839960937, 9.187763624570907)  # error of 1.38E-02 with sample range (0.00E+00,1.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.184102720727361, 0.6674369952484219)  # error of 7.25E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (3.05E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (5.000220272676166, 0.7699798791319306, 0.5818113932872683)  # error of 5.06E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (2.08E-02,1.00E+00)
    genlog_norm_parameter = (0.5268272560910332, 2.9870414098551117, 0.8114362377392241, 0.34748243027398057)  # error of 3.62E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (1.02E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PEOE_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA3)
    # normalization
    linear_norm_parameter = (0.011713317780700283, 0.7004807757872287)  # error of 7.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (7.00E-01,3.33E+00)
    min_max_norm_parameter = (8.166483837005377e-17, 11.680862968336768)  # error of 5.66E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (3.964877055534785, 0.24576276616405693)  # error of 2.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (2.74E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.975887144236942, 0.19925982360149636, 0.24623031071032647)  # error of 2.38E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (3.12E-01,1.00E+00)
    genlog_norm_parameter = (0.21940814583639343, -0.7497227201418301, 0.3658885023452447, 0.17057290982900755)  # error of 2.35E-02 with sample range (0.00E+00,2.25E+02) resulting in fit range (2.05E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_NH1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_NH1)
    # normalization
    linear_norm_parameter = (0.031218773641592312, 0.7849136633253863)  # error of 8.37E-02 with sample range (0.00E+00,5.00E+01) resulting in fit range (7.85E-01,2.35E+00)
    min_max_norm_parameter = (7.888609052210118e-31, 2.0805517126600583)  # error of 2.92E-02 with sample range (0.00E+00,5.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6671905805486065, 1.553108631785838)  # error of 1.81E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (2.62E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.6671904344116224, 23.348768785513762, 1.553108167548447)  # error of 1.01E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (1.72E-07,1.00E+00)
    genlog_norm_parameter = (1.464110970944641, 0.4816737490309036, 0.5324300087598685, 0.4755375718025074)  # error of 1.70E-03 with sample range (0.00E+00,5.00E+01) resulting in fit range (2.15E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA11)
    # normalization
    linear_norm_parameter = (0.00828695135755142, 0.7644503184456655)  # error of 5.96E-02 with sample range (0.00E+00,1.99E+02) resulting in fit range (7.64E-01,2.41E+00)
    min_max_norm_parameter = (7.634541570684105e-30, 11.947029795426264)  # error of 7.48E-02 with sample range (0.00E+00,1.99E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.445806162262592, 0.204409252101571)  # error of 7.55E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (3.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.4163166606458475, 0.37652639805143134, 0.20369708811500437)  # error of 7.44E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (2.87E-01,1.00E+00)
    genlog_norm_parameter = (0.1881054084139518, -0.69641763217766, 0.34033764507844283, 0.2340987322955706)  # error of 7.17E-03 with sample range (0.00E+00,1.99E+02) resulting in fit range (3.28E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumHeteroatoms)
    # normalization
    linear_norm_parameter = (0.016077495185775725, 0.5033528543882042)  # error of 2.02E-01 with sample range (0.00E+00,2.15E+02) resulting in fit range (5.03E-01,3.96E+00)
    min_max_norm_parameter = (1.9439247544979925, 10.63914346171618)  # error of 1.53E-02 with sample range (0.00E+00,2.15E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.309633287283373, 0.5961308359128846)  # error of 8.73E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (2.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.045320930283959, 0.7179354032850181, 0.5008061943406604)  # error of 5.10E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (1.29E-02,1.00E+00)
    genlog_norm_parameter = (0.45068040681824034, 1.1172662980244155, 1.7144405539074021, 0.23628999990547292)  # error of 3.24E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (3.38E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SlogP_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA10)
    # normalization
    linear_norm_parameter = (0.005233706932712693, 0.7982300755527156)  # error of 6.65E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (7.98E-01,2.58E+00)
    min_max_norm_parameter = (2.633367426076831e-17, 12.74842203329568)  # error of 4.94E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (3.6455899379715135, 0.1901649776805424)  # error of 1.29E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (3.33E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.615359539682259, 0.2741057268623292, 0.1895119926287925)  # error of 1.28E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (2.71E-01,1.00E+00)
    genlog_norm_parameter = (0.16769339036833567, -9.262927974226667, 0.002131470551772922, 0.00033340869591382363)  # error of 1.16E-02 with sample range (0.00E+00,3.41E+02) resulting in fit range (2.59E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0)
    # normalization
    linear_norm_parameter = (0.007513574566740955, 0.2957974579215942)  # error of 2.01E-01 with sample range (0.00E+00,7.19E+02) resulting in fit range (2.96E-01,5.70E+00)
    min_max_norm_parameter = (17.447753846956868, 55.71754607510952)  # error of 4.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (36.35211468883877, 0.1295884036148426)  # error of 2.70E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (8.92E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (34.78188468223621, 0.18372914466871643, 0.10529824432103296)  # error of 1.38E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (1.67E-03,1.00E+00)
    genlog_norm_parameter = (0.09192921396182452, 3.197386465432143, 4.054083564087513e-07, 2.977582294894024e-08)  # error of 1.17E-02 with sample range (0.00E+00,7.19E+02) resulting in fit range (1.17E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_morpholine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_morpholine)
    # normalization
    linear_norm_parameter = (0.4932781049705528, 0.4932781049705526)  # error of 1.11E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.93E-01,2.47E+00)
    min_max_norm_parameter = (9.48941571641272e-09, 1.0136269883011773)  # error of 1.28E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.0726077972576291, 2.0726077620218297)  # error of 1.33E-11 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.0726077972576291, 1.0, 2.0726077620218297)  # error of 1.33E-11 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.02E-01,1.00E+00)
    genlog_norm_parameter = (1.1028251873807355, 0.29835781455741844, 0.054235129725539614, 1.825554772999035)  # error of 1.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.61E-01,9.99E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  EState_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA9)
    # normalization
    linear_norm_parameter = (0.008676493919004956, 0.275505806742827)  # error of 1.93E-01 with sample range (0.00E+00,6.11E+02) resulting in fit range (2.76E-01,5.57E+00)
    min_max_norm_parameter = (14.505227188937377, 52.02218632660705)  # error of 4.07E-02 with sample range (0.00E+00,6.11E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (33.12486658378458, 0.13300161576607572)  # error of 1.91E-02 with sample range (0.00E+00,6.11E+02) resulting in fit range (1.21E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (32.05647210374936, 0.1688075153862539, 0.11490618912706498)  # error of 9.95E-03 with sample range (0.00E+00,6.11E+02) resulting in fit range (4.45E-03,1.00E+00)
    genlog_norm_parameter = (0.10208124641901982, 1.5100397068901317, 2.572466445901273, 0.14880989913945686)  # error of 7.12E-03 with sample range (0.00E+00,6.11E+02) resulting in fit range (8.98E-05,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PBF_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcPBF)
    # normalization
    linear_norm_parameter = (0.8002336508637942, -0.27832921776549496)  # error of 1.06E-01 with sample range (0.00E+00,4.59E+00) resulting in fit range (-2.78E-01,3.39E+00)
    min_max_norm_parameter = (0.5464675217564985, 1.3692618279142443)  # error of 3.39E-02 with sample range (0.00E+00,4.59E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.9571543856358279, 5.963418540555978)  # error of 8.11E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (3.31E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.9519185338635874, 6.24803308673302, 5.68213347864401)  # error of 6.60E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (2.61E-03,1.00E+00)
    genlog_norm_parameter = (5.524923571667459, 0.8859978971497328, 1.0389823352387015, 0.7751221827451574)  # error of 6.72E-03 with sample range (0.00E+00,4.59E+00) resulting in fit range (1.71E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  MinAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MinAbsPartialCharge)
    # normalization
    linear_norm_parameter = (3.1682344592574303, -0.3438979216493595)  # error of 8.02E-02 with sample range (0.00E+00,INF) resulting in fit range (-3.44E-01,INF)
    min_max_norm_parameter = (0.14478720365913814, 0.3864847550968621)  # error of 3.67E-02 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.266281069473945, 20.060338031364914)  # error of 1.73E-02 with sample range (0.00E+00,INF) resulting in fit range (4.76E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.2663093844074067, 20.04277183939622, 20.075906992777337)  # error of 1.73E-02 with sample range (0.00E+00,INF) resulting in fit range (4.78E-03,1.00E+00)
    genlog_norm_parameter = (21.295576149418388, 0.3642568591085718, 0.1608997466871537, 1.1882017693601543)  # error of 1.70E-02 with sample range (0.00E+00,INF) resulting in fit range (6.78E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SlogP_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA11)
    # normalization
    linear_norm_parameter = (0.0055772293963739505, 0.8513122937313707)  # error of 5.02E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (8.51E-01,1.88E+00)
    min_max_norm_parameter = (1.6665175605263984, 6.577923952521674)  # error of 2.87E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.411498413610341, 1.231868331024134)  # error of 2.88E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (4.34E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.531604389044145, 2.8773323747778226, 0.4480597493562967)  # error of 2.84E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (6.86E-04,1.00E+00)
    genlog_norm_parameter = (1.06463212487301, 0.27304809906625216, 0.3237383119230303, 0.005377730545659635)  # error of 2.88E-02 with sample range (0.00E+00,1.84E+02) resulting in fit range (8.88E-30,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_aniline_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_aniline)
    # normalization
    linear_norm_parameter = (0.038572385626053585, 0.8111639952400167)  # error of 4.99E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (8.11E-01,1.39E+00)
    min_max_norm_parameter = (4.067547786674466e-09, 1.3022513887988345)  # error of 1.73E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.18131870166733124, 1.4627551170239292)  # error of 1.27E-03 with sample range (0.00E+00,1.50E+01) resulting in fit range (4.34E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.18131870166733124, 1.0, 1.4627551170239292)  # error of 1.27E-03 with sample range (0.00E+00,1.50E+01) resulting in fit range (4.34E-01,1.00E+00)
    genlog_norm_parameter = (2.7416784853412803, -0.5918191727088815, 0.008059096289137093, 0.0003733096193239703)  # error of 1.35E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (1.42E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BCUT2D_MWHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_MWHI)
    # normalization
    linear_norm_parameter = (0.06382980925203052, 0.0914070582631521)  # error of 7.74E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (8.35E-02,8.19E+00)
    min_max_norm_parameter = (-0.12367738038301467, 13.619949109255183)  # error of 9.41E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.983255681107075, 0.32288774435232176)  # error of 8.34E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (1.22E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.7931432911967606, 1.1841140348165382, 0.20355402639270814)  # error of 6.08E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (3.07E-02,1.00E+00)
    genlog_norm_parameter = (0.24661874523046712, -2.2059114704545935, 0.0006588747523962266, 0.00015970079475727318)  # error of 7.49E-02 with sample range (-1.24E-01,1.27E+02) resulting in fit range (8.47E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi4v)
    # normalization
    linear_norm_parameter = (0.1846796920584458, 0.12308022240634398)  # error of 1.49E-01 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.23E-01,4.41E+01)
    min_max_norm_parameter = (0.4365031459643245, 3.319224643809378)  # error of 3.97E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.8494300166801143, 1.6807292424962799)  # error of 3.13E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (4.28E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.7216523305680784, 2.287102552977168, 1.2587726126085808)  # error of 1.11E-02 with sample range (0.00E+00,2.38E+02) resulting in fit range (1.91E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_piperdine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_piperdine)
    # normalization
    linear_norm_parameter = (0.04919057284844364, 0.8975592196702297)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.98E-01,1.29E+00)
    min_max_norm_parameter = (8.858196589138318e-09, 1.0562452798304425)  # error of 1.46E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.09657686162446193, 2.624561257313392)  # error of 0.00E+00 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.63E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.09657686264538319, 1.0, 2.6245612535395852)  # error of 5.41E-11 with sample range (0.00E+00,8.00E+00) resulting in fit range (5.63E-01,1.00E+00)
    genlog_norm_parameter = (2.643725874229428, 0.4954144189864354, 0.38073948233735966, 1.7466886525864032)  # error of 3.08E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.04E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi3v)
    # normalization
    linear_norm_parameter = (0.13641142111605653, 0.06481842365937796)  # error of 1.43E-01 with sample range (0.00E+00,1.56E+02) resulting in fit range (6.48E-02,2.13E+01)
    min_max_norm_parameter = (0.9571111913222607, 5.030476311519345)  # error of 3.74E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.9588780643186574, 1.1889256254049316)  # error of 2.82E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (2.88E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.798222121514068, 1.5668854376121493, 0.9198752649381355)  # error of 1.06E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (1.23E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_Nhpyrrole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Nhpyrrole)
    # normalization
    linear_norm_parameter = (0.025012748852603184, 0.9478546930776232)  # error of 0.00E+00 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.48E-01,1.10E+00)
    min_max_norm_parameter = (3.340124193009499e-09, 1.0278892650835505)  # error of 8.08E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.3902136788160999, 2.5747937021696656)  # error of 7.85E-17 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.32E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.3902136790305557, 1.0, 2.5747937015890643)  # error of 4.92E-12 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.32E-01,1.00E+00)
    genlog_norm_parameter = (2.585893826872336, 0.42074502179419004, 0.23585726133433205, 1.8683788003053245)  # error of 1.19E-13 with sample range (0.00E+00,6.00E+00) resulting in fit range (7.53E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1)
    # normalization
    linear_norm_parameter = (0.021402601422805922, 0.057358382906918415)  # error of 1.75E-01 with sample range (0.00E+00,4.28E+02) resulting in fit range (5.74E-02,9.23E+00)
    min_max_norm_parameter = (10.35770293280183, 30.318616722974067)  # error of 4.02E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (20.218648442527503, 0.24321989118039597)  # error of 2.60E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (7.26E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (19.497180383047468, 0.31860651031309306, 0.19451567750637339)  # error of 1.08E-02 with sample range (0.00E+00,4.28E+02) resulting in fit range (2.00E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  FpDensityMorgan1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(FpDensityMorgan1)
    # normalization
    linear_norm_parameter = (0.6114489442416012, -0.2500130662147515)  # error of 1.07E-01 with sample range (1.72E-02,4.50E+00) resulting in fit range (-2.39E-01,2.50E+00)
    min_max_norm_parameter = (0.7464667046271726, 1.5888733629158822)  # error of 3.23E-02 with sample range (1.72E-02,4.50E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.167586022382926, 6.06701731964724)  # error of 7.87E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (9.30E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.1745399912705168, 5.851158145308263, 6.523356769876131)  # error of 7.03E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (1.14E-03,1.00E+00)
    genlog_norm_parameter = (7.016933461459958, 0.9758849407148891, 6.252276609214549, 1.3640331931641156)  # error of 5.29E-03 with sample range (1.72E-02,4.50E+00) resulting in fit range (1.88E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_epoxide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_epoxide)
    # normalization
    linear_norm_parameter = (0.4991075803177717, 0.49910758031777147)  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    min_max_norm_parameter = (6.5085154339493485e-09, 1.0017880307004197)  # error of 1.50E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.5152730763769928, 2.5152729680451307)  # error of 7.49E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.78E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.5152730763769928, 1.0, 2.5152729680451307)  # error of 7.49E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.78E-01,1.00E+00)
    genlog_norm_parameter = (1.1085975163035535, 0.2794865643718473, 0.007338549198124006, 1.8450599421767278)  # error of 7.22E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.95E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SlogP_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA5)
    # normalization
    linear_norm_parameter = (0.0030832580459571788, 0.41517180790701097)  # error of 1.77E-01 with sample range (0.00E+00,2.18E+03) resulting in fit range (4.15E-01,7.12E+00)
    min_max_norm_parameter = (13.274337247983418, 99.35584702819898)  # error of 4.24E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (55.95034092538948, 0.05594705841606074)  # error of 2.08E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (4.19E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (53.71656336108127, 0.0743286674865532, 0.04858962687912679)  # error of 1.11E-02 with sample range (0.00E+00,2.18E+03) resulting in fit range (1.81E-02,1.00E+00)
    genlog_norm_parameter = (1.090165598041791, 0.6339658426861253, 0.8996167179643567, 5.664716239481225e-09)  # error of 3.94E-01 with sample range (0.00E+00,2.18E+03) resulting in fit range (0.00E+00,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_hdrzine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_hdrzine)
    # normalization
    linear_norm_parameter = (0.49404553590176903, 0.4940455359017689)  # error of 4.44E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.94E-01,3.46E+00)
    min_max_norm_parameter = (7.616024548469903e-09, 1.0120524599863319)  # error of 2.81E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.1020196341906385, 2.1020196036271543)  # error of 6.76E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.1020196341906385, 1.0, 2.1020196036271543)  # error of 6.76E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.10E-01,1.00E+00)
    genlog_norm_parameter = (1.103601350017191, 0.29583395743887125, 0.048166501340416334, 1.828146666503602)  # error of 1.27E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.65E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_Ar_N_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ar_N)
    # normalization
    linear_norm_parameter = (0.0505043831748716, 0.706005745196576)  # error of 5.36E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (7.06E-01,2.32E+00)
    min_max_norm_parameter = (0.0, 2.1715264674397554)  # error of 5.56E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.20224173292892467, 0.8623990343842892)  # error of 4.21E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (4.57E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.20224173108680912, 1.0, 0.8623990337225991)  # error of 4.21E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (4.57E-01,1.00E+00)
    genlog_norm_parameter = (1.0634893160980838, 1.442852440655972, 1.6171594102405107, 3.1756676466784426)  # error of 1.02E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (5.10E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SMR_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA5)
    # normalization
    linear_norm_parameter = (0.002902392328484771, 0.5504163136192717)  # error of 1.80E-01 with sample range (0.00E+00,1.49E+03) resulting in fit range (5.50E-01,4.89E+00)
    min_max_norm_parameter = (2.783727923618457e-16, 60.67206561974408)  # error of 4.76E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (29.231452891359663, 0.07160204674656424)  # error of 2.47E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (1.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (27.033473604340355, 0.10374643251952695, 0.06093935202142024)  # error of 1.53E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (5.71E-02,1.00E+00)
    genlog_norm_parameter = (0.05458732293792028, -36.99489788287852, 0.0030850944423266584, 0.0001255986870847963)  # error of 1.36E-02 with sample range (0.00E+00,1.49E+03) resulting in fit range (3.84E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SlogP_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA3)
    # normalization
    linear_norm_parameter = (0.005572864820903245, 0.7183501823787287)  # error of 1.09E-01 with sample range (0.00E+00,4.79E+02) resulting in fit range (7.18E-01,3.39E+00)
    min_max_norm_parameter = (2.408585330451039e-20, 22.439152351441383)  # error of 5.09E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (9.49158537571278, 0.16960656906498323)  # error of 1.48E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (1.67E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (8.952585532461732, 0.26147507132416203, 0.15907497123208453)  # error of 1.16E-02 with sample range (0.00E+00,4.79E+02) resulting in fit range (8.78E-02,1.00E+00)
    genlog_norm_parameter = (0.1413990191079397, -19.24491345309785, 0.006259613060538872, 0.00015608204799613998)  # error of 9.62E-03 with sample range (0.00E+00,4.79E+02) resulting in fit range (7.15E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BertzCT)
    # normalization
    linear_norm_parameter = (0.0002435689316791905, 0.08402337833593164)  # error of 1.62E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (8.40E-02,1.15E+01)
    min_max_norm_parameter = (589.3484428971769, 2551.6280739737376)  # error of 3.81E-02 with sample range (0.00E+00,4.67E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.74E-01 with sample range (0.00E+00,4.67E+04) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_C_O_noCOO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_C_O_noCOO)
    # normalization
    linear_norm_parameter = (0.027301361059322793, 0.7733988486490774)  # error of 9.21E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (7.73E-01,2.08E+00)
    min_max_norm_parameter = (1.5777218104420236e-30, 2.236082199928005)  # error of 2.39E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.7988894132889973, 1.3355456372556587)  # error of 2.67E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (2.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.7988893624690527, 19.049004921212244, 1.3355454934130946)  # error of 1.50E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (2.46E-07,1.00E+00)
    genlog_norm_parameter = (1.185376103692059, 0.019289566183798557, 0.28026262777407235, 0.14748302877922412)  # error of 2.01E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (1.81E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumSaturatedRings)
    # normalization
    linear_norm_parameter = (0.037486269092924296, 0.7906195585254473)  # error of 6.21E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (7.91E-01,1.92E+00)
    min_max_norm_parameter = (5.730855425109739e-09, 1.404946368678208)  # error of 1.95E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.36617461217911906, 1.4317359459562997)  # error of 2.83E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.72E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.36617461217911906, 1.0, 1.4317359459562997)  # error of 2.83E-03 with sample range (0.00E+00,3.00E+01) resulting in fit range (3.72E-01,1.00E+00)
    genlog_norm_parameter = (2.361126814701105, -0.9332372421941358, 0.010759488748759415, 0.0003109299935290176)  # error of 1.37E-02 with sample range (0.00E+00,3.00E+01) resulting in fit range (2.20E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SMR_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA10)
    # normalization
    linear_norm_parameter = (0.00805950974671309, 0.47073915323460086)  # error of 1.02E-01 with sample range (0.00E+00,9.17E+02) resulting in fit range (4.71E-01,7.86E+00)
    min_max_norm_parameter = (1.750898428174181e-15, 43.23543148663909)  # error of 4.79E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (20.019352651600183, 0.09701956716011305)  # error of 1.52E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (1.25E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (19.446748714614163, 0.12516006519982537, 0.09265838451052572)  # error of 1.34E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (8.06E-02,1.00E+00)
    genlog_norm_parameter = (0.08397536225621226, -9.67695202560633, 2.6257752347199226, 0.2851178401354698)  # error of 1.34E-02 with sample range (0.00E+00,9.17E+02) resulting in fit range (6.66E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumSpiroAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumSpiroAtoms)
    # normalization
    linear_norm_parameter = (0.4935155835974765, 0.49351558359747627)  # error of 2.22E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.94E-01,1.97E+00)
    min_max_norm_parameter = (1.3202332968823136e-08, 1.0131392331517604)  # error of 4.00E-04 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.0813823993563174, 2.081382361017821)  # error of 2.19E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.05E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.0813823993563174, 1.0, 2.081382361017821)  # error of 2.19E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.05E-01,1.00E+00)
    genlog_norm_parameter = (1.103065827005491, 0.2975756058113469, 0.052360674886894276, 1.8263575757464208)  # error of 1.08E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.62E-01,9.99E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi3v)
    # normalization
    linear_norm_parameter = (0.13641142111605653, 0.06481842365937796)  # error of 1.43E-01 with sample range (0.00E+00,1.56E+02) resulting in fit range (6.48E-02,2.13E+01)
    min_max_norm_parameter = (0.9571111913222607, 5.030476311519345)  # error of 3.74E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.9588780643186574, 1.1889256254049316)  # error of 2.82E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (2.88E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.798222121514068, 1.5668854376121493, 0.9198752649381355)  # error of 1.06E-02 with sample range (0.00E+00,1.56E+02) resulting in fit range (1.23E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumSaturatedCarbocycles)
    # normalization
    linear_norm_parameter = (0.020972112507738894, 0.9098571128596467)  # error of 1.95E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (9.10E-01,1.29E+00)
    min_max_norm_parameter = (8.1643240157463e-09, 1.1069248512275087)  # error of 6.94E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.6176260310959208, 1.385111849563034)  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.6176260310959208, 1.0, 1.385111849563034)  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    genlog_norm_parameter = (4.069531491711097, -0.26470302233247034, 0.01816997314515009, 0.0010314014664690044)  # error of 6.57E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (2.53E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi0n)
    # normalization
    linear_norm_parameter = (0.009618083281699108, 0.17347984942891348)  # error of 2.14E-01 with sample range (7.63E-01,6.43E+02) resulting in fit range (1.81E-01,6.36E+00)
    min_max_norm_parameter = (16.004022253403853, 51.11689241252734)  # error of 4.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (33.37586607594709, 0.1432287247282441)  # error of 2.67E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.28E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (31.98836545966142, 0.18812033507921852, 0.1136090082864722)  # error of 1.31E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (2.80E-03,1.00E+00)
    genlog_norm_parameter = (0.09991708558883805, -13.42358800932477, 0.271792016045397, 0.0039352878486176145)  # error of 1.01E-02 with sample range (7.63E-01,6.43E+02) resulting in fit range (9.15E-08,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_oxime_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_oxime)
    # normalization
    linear_norm_parameter = (0.49776520113189837, 0.49776520113189815)  # error of 2.22E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.98E-01,2.99E+00)
    min_max_norm_parameter = (6.196547898069593e-09, 1.0044896647037036)  # error of 4.74E-05 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3250564360659973, 2.3250563474726307)  # error of 4.01E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3250564360659973, 1.0, 2.3250563474726307)  # error of 4.01E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.56E-01,1.00E+00)
    genlog_norm_parameter = (1.107295730141592, 0.28376806666724697, 0.01829870777085616, 1.8406066637654226)  # error of 3.63E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.87E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi2v)
    # normalization
    linear_norm_parameter = (0.09656674011706068, 0.026475463133179833)  # error of 1.43E-01 with sample range (0.00E+00,1.68E+02) resulting in fit range (2.65E-02,1.62E+01)
    min_max_norm_parameter = (1.85040714882094, 7.55574015613779)  # error of 3.72E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.659367062804354, 0.84727788625193)  # error of 2.67E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (1.89E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.449991335141199, 1.1069417196730784, 0.6703056005048558)  # error of 1.05E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (7.20E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  BCUT2D_MRHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_MRHI)
    # normalization
    linear_norm_parameter = (0.2760716898709984, -0.05891671770132878)  # error of 3.51E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (-2.01E-01,3.86E+00)
    min_max_norm_parameter = (0.3604159078021756, 3.7266268387929946)  # error of 1.95E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.042877009089624, 1.4588708297468977)  # error of 2.02E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.34E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.0470577588560617, 1.4477594375506484, 1.4702672532434768)  # error of 2.01E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.39E-02,1.00E+00)
    genlog_norm_parameter = (1.472661851120768, 1.144889269421317, 3.912155034373345, 1.028603526979817)  # error of 2.02E-02 with sample range (-5.14E-01,1.42E+01) resulting in fit range (2.42E-02,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SMR_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA4)
    # normalization
    linear_norm_parameter = (0.004491810149551356, 0.8258291767126574)  # error of 7.18E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (8.26E-01,1.68E+00)
    min_max_norm_parameter = (7.03458954642263e-19, 11.924815721181979)  # error of 5.26E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (3.0253911905893234, 0.22133635579572117)  # error of 1.52E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (3.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.0253926833769804, 3.9225922666021376, 0.22133642902354464)  # error of 1.45E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (7.02E-06,1.00E+00)
    genlog_norm_parameter = (0.1962338488568962, -6.361459309546196, 0.0021288761629645763, 0.00045569961868591007)  # error of 1.38E-02 with sample range (0.00E+00,1.90E+02) resulting in fit range (2.62E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcPMI2)
    # normalization
    linear_norm_parameter = (3.364102817740711e-05, 0.2861109449057394)  # error of 1.90E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.86E-01,2.45E+01)
    min_max_norm_parameter = (44.665447243295176, 10171.614030140174)  # error of 5.66E-02 with sample range (0.00E+00,7.21E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_Ar_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ar_COO)
    # normalization
    linear_norm_parameter = (0.49411802937735627, 0.49411802937735616)  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.94E-01,3.46E+00)
    min_max_norm_parameter = (8.646173628068578e-09, 1.0119039788513677)  # error of 2.64E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.1049662105257694, 2.10496612246408)  # error of 7.86E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.11E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.1049662105257694, 1.0, 2.10496612246408)  # error of 7.86E-11 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.11E-01,1.00E+00)
    genlog_norm_parameter = (1.1036744213826977, 0.29559613615718794, 0.04759157050559711, 1.8283911113753144)  # error of 1.29E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcHallKierAlpha)
    # normalization
    linear_norm_parameter = (0.10941027797520916, 0.8206630827667591)  # error of 1.86E-01 with sample range (-7.86E+01,6.10E+01) resulting in fit range (-7.78E+00,7.50E+00)
    min_max_norm_parameter = (-4.025047303628864, -0.27476699527518833)  # error of 2.37E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.136641082271498, 1.3958219191517867)  # error of 1.43E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (4.45E-47,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.0451007845368863, 1.1887380822440157, 1.5856961746935594)  # error of 8.43E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (3.00E-40,1.00E+00)
    genlog_norm_parameter = (1.7618968864206013, -2.0459113897478645, 2.4289416394683374, 1.8402620037112714)  # error of 7.26E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (9.11E-33,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_nitro_arom_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_nitro_arom)
    # normalization
    linear_norm_parameter = (0.01529862312391872, 0.9682928536431723)  # error of 0.00E+00 with sample range (0.00E+00,1.10E+01) resulting in fit range (9.68E-01,1.14E+00)
    min_max_norm_parameter = (5.509706096657517e-09, 1.0166822542997578)  # error of 3.95E-04 with sample range (0.00E+00,1.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.5110603956802732, 2.708965007295647)  # error of 0.00E+00 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.5110628956662855, 1.0, 2.7089582891344866)  # error of 3.94E-08 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.00E-01,1.00E+00)
    genlog_norm_parameter = (2.716065564404975, 0.39434100416328627, 0.16689464990269173, 1.9163020761718925)  # error of 9.39E-10 with sample range (0.00E+00,1.10E+01) resulting in fit range (8.13E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_SH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_SH)
    # normalization
    linear_norm_parameter = (0.49842014218720343, 0.4984201421872032)  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.98E-01,3.49E+00)
    min_max_norm_parameter = (3.704484506370359e-09, 1.0031697310626564)  # error of 1.67E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3987722757661272, 2.3987722116196286)  # error of 7.94E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3987722757661272, 1.0, 2.3987722116196286)  # error of 7.94E-10 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.66E-01,1.00E+00)
    genlog_norm_parameter = (1.1079331743675414, 0.2816740739909799, 0.012963206035942691, 1.8427823614028687)  # error of 2.66E-15 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.91E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_piperzine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_piperzine)
    # normalization
    linear_norm_parameter = (0.4872161505464513, 0.48721615054645107)  # error of 5.55E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.87E-01,3.41E+00)
    min_max_norm_parameter = (7.2423565017085415e-09, 1.0262385582797817)  # error of 2.26E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.9080160452625068, 1.9080160046689731)  # error of 2.52E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.50E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.9080160452625068, 1.0, 1.9080160046689731)  # error of 2.52E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.50E-01,1.00E+00)
    genlog_norm_parameter = (1.0965711654531403, 0.3186519696314755, 0.10102198519304303, 1.804846651655294)  # error of 1.16E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.28E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1n)
    # normalization
    linear_norm_parameter = (0.02460656481380441, 0.05696382112467857)  # error of 1.83E-01 with sample range (0.00E+00,3.32E+02) resulting in fit range (5.70E-02,8.22E+00)
    min_max_norm_parameter = (8.215243805842091, 25.568763302868422)  # error of 3.90E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (16.800375876069985, 0.2846180730802336)  # error of 2.44E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (8.31E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (16.22787766733102, 0.3557663030596946, 0.2286121746689847)  # error of 1.13E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (3.10E-03,1.00E+00)
    genlog_norm_parameter = (0.20081911303774566, -0.3618619378577607, 1.199011524398242, 0.05743559815299407)  # error of 7.92E-03 with sample range (0.00E+00,3.32E+02) resulting in fit range (2.17E-06,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAromaticRings)
    # normalization
    linear_norm_parameter = (0.034866795939075525, 0.6345283877004492)  # error of 1.43E-01 with sample range (0.00E+00,6.10E+01) resulting in fit range (6.35E-01,2.76E+00)
    min_max_norm_parameter = (4.733165431326071e-30, 3.555582078939516)  # error of 1.41E-02 with sample range (0.00E+00,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.7284808042419078, 1.2169190611775202)  # error of 6.86E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (1.09E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.686248103154103, 1.3303663811394981, 1.1588291961663748)  # error of 6.13E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (9.59E-02,1.00E+00)
    genlog_norm_parameter = (1.0818733953733572, 0.3821001451970787, 1.9211953644983686, 0.5468773865895864)  # error of 5.78E-03 with sample range (0.00E+00,6.10E+01) resulting in fit range (8.28E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  EState_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA2)
    # normalization
    linear_norm_parameter = (0.004010356366189516, 0.8323097045695795)  # error of 5.49E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (8.32E-01,8.07E+00)
    min_max_norm_parameter = (1.0419551447906305e-18, 11.80200135547047)  # error of 7.44E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.1763739551036555, 0.14514432511172065)  # error of 1.12E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (5.06E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.17637470543076575, 1.0, 0.14514431482294865)  # error of 1.12E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (5.06E-01,1.00E+00)
    genlog_norm_parameter = (0.13267267604890298, -5.38995753218738, 0.008988954422818305, 0.00577772306001971)  # error of 1.11E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (4.68E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi2v)
    # normalization
    linear_norm_parameter = (0.09656674011706068, 0.026475463133179833)  # error of 1.43E-01 with sample range (0.00E+00,1.68E+02) resulting in fit range (2.65E-02,1.62E+01)
    min_max_norm_parameter = (1.85040714882094, 7.55574015613779)  # error of 3.72E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.659367062804354, 0.84727788625193)  # error of 2.67E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (1.89E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.449991335141199, 1.1069417196730784, 0.6703056005048558)  # error of 1.05E-02 with sample range (0.00E+00,1.68E+02) resulting in fit range (7.20E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_urea_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_urea)
    # normalization
    linear_norm_parameter = (0.027632513073823306, 0.9435400813926746)  # error of 0.00E+00 with sample range (0.00E+00,1.70E+01) resulting in fit range (9.44E-01,1.41E+00)
    min_max_norm_parameter = (5.263068649567547e-09, 1.0296830919097522)  # error of 4.06E-04 with sample range (0.00E+00,1.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.0952416256429324, 3.2113257966470843)  # error of 7.85E-17 with sample range (0.00E+00,1.70E+01) resulting in fit range (5.76E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.09524163020540921, 1.0, 3.2113257769175463)  # error of 1.40E-10 with sample range (0.00E+00,1.70E+01) resulting in fit range (5.76E-01,1.00E+00)
    genlog_norm_parameter = (3.223218484281986, 0.4637575859898466, 0.3118172029835349, 1.8422878319719684)  # error of 1.42E-08 with sample range (0.00E+00,1.70E+01) resulting in fit range (6.23E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_benzene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_benzene)
    # normalization
    linear_norm_parameter = (0.02947953215467447, 0.7331435625333342)  # error of 1.11E-01 with sample range (0.00E+00,3.30E+01) resulting in fit range (7.33E-01,1.71E+00)
    min_max_norm_parameter = (7.888609052210118e-31, 2.4698336963662135)  # error of 1.96E-02 with sample range (0.00E+00,3.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0920776300962571, 1.3627096584489056)  # error of 6.24E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (1.84E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.092077640400224, 13.381103648310065, 1.3627097252491245)  # error of 4.09E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (4.50E-07,1.00E+00)
    genlog_norm_parameter = (1.145226583492056, -0.28269912318539797, 0.20974044730823685, 0.06181499181149289)  # error of 5.40E-03 with sample range (0.00E+00,3.30E+01) resulting in fit range (1.02E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_bicyclic_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_bicyclic)
    # normalization
    linear_norm_parameter = (0.008955979676114945, 0.8971616364051024)  # error of 4.19E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (8.97E-01,1.27E+00)
    min_max_norm_parameter = (6.544180000802119e-09, 1.2950088357093383)  # error of 1.82E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.398308593847868, 0.8997093072499293)  # error of 8.22E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (5.89E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.398308537650087, 1.0, 0.8997093336250057)  # error of 8.22E-03 with sample range (0.00E+00,4.20E+01) resulting in fit range (5.89E-01,1.00E+00)
    genlog_norm_parameter = (2.6720031851941237, -0.6536880834459016, 0.010298516560499068, 0.00045059766538596655)  # error of 1.61E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (1.87E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_ArN_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_ArN)
    # normalization
    linear_norm_parameter = (0.02460278574928243, 0.9493045625893672)  # error of 0.00E+00 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.49E-01,1.07E+00)
    min_max_norm_parameter = (6.546180810957143e-09, 1.026791718467936)  # error of 6.13E-04 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.2533822935500519, 2.8879157330454324)  # error of 7.85E-17 with sample range (0.00E+00,5.00E+00) resulting in fit range (6.75E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.253382336540342, 1.0, 2.88791558684182)  # error of 1.08E-09 with sample range (0.00E+00,5.00E+00) resulting in fit range (6.75E-01,1.00E+00)
    genlog_norm_parameter = (2.898781083591291, 0.4335990121804178, 0.26109365413962266, 1.865265199884988)  # error of 1.53E-12 with sample range (0.00E+00,5.00E+00) resulting in fit range (7.05E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa2)
    # normalization
    linear_norm_parameter = (0.030418143185359847, 0.2871593951766924)  # error of 2.34E-01 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.87E-01,1.08E+03)
    min_max_norm_parameter = (3.446066777200142, 12.00058941506709)  # error of 4.58E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.66772112289306, 0.5795843116540953)  # error of 3.08E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (1.16E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (7.271319784936047, 0.8080128800230181, 0.4459683689150529)  # error of 1.32E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.80E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_halogen_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_halogen)
    # normalization
    linear_norm_parameter = (0.017119543157199657, 0.8437328519221161)  # error of 5.64E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (8.44E-01,1.79E+00)
    min_max_norm_parameter = (0.041860184464005364, 1.3840920614215728)  # error of 2.65E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.005684348962987795, 0.9165937462942823)  # error of 2.31E-03 with sample range (0.00E+00,5.50E+01) resulting in fit range (5.01E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.005684348962987795, 1.0, 0.9165937462942823)  # error of 2.31E-03 with sample range (0.00E+00,5.50E+01) resulting in fit range (5.01E-01,1.00E+00)
    genlog_norm_parameter = (2.1527177639001365, -1.1481370597209608, 0.009703126945728951, 0.0002478140151054729)  # error of 2.13E-02 with sample range (0.00E+00,5.50E+01) resulting in fit range (3.67E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  VSA_EState6_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState6)
    # normalization
    linear_norm_parameter = (0.026302870373594356, 0.7190453852045258)  # error of 1.52E-01 with sample range (-4.76E+02,1.38E+01) resulting in fit range (-1.18E+01,1.08E+00)
    min_max_norm_parameter = (-16.488126017392656, 0.8283736118358731)  # error of 3.98E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-7.68869429717635, 0.2724099515171271)  # error of 2.50E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (4.47E-56,9.97E-01)
    dual_sigmoidal_norm_parameter = (-7.078277106127244, 0.2169591096464027, 0.354261836499568)  # error of 1.11E-02 with sample range (-4.76E+02,1.38E+01) resulting in fit range (7.24E-45,9.99E-01)
    genlog_norm_parameter = (0.4875336273543425, -8.435737342184726, 13.23143621938788, 3.1241409093443937)  # error of 6.47E-03 with sample range (-4.76E+02,1.38E+01) resulting in fit range (9.63E-33,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_Ar_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ar_OH)
    # normalization
    linear_norm_parameter = (0.010934015936408747, 0.9686094918123338)  # error of 3.71E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.19E+00)
    min_max_norm_parameter = (6.848999398227221e-09, 1.0236283333676808)  # error of 1.24E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.239714164093668, 1.6725059613391224)  # error of 2.37E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (8.88E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.239714164093668, 1.0, 1.6725059613391224)  # error of 2.37E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (8.88E-01,1.00E+00)
    genlog_norm_parameter = (1.1600916957620944, 0.8202648940772637, -0.9692965833371077, -66.19331353686114)  # error of 9.46E-09 with sample range (0.00E+00,2.00E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PMI3)
    # normalization
    linear_norm_parameter = (2.8952313203676817e-05, 0.2866572166015228)  # error of 1.92E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.87E-01,2.52E+01)
    min_max_norm_parameter = (147.52756421954018, 11758.510463576746)  # error of 5.52E-02 with sample range (0.00E+00,8.59E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,8.59E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  VSA_EState8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState8)
    # normalization
    linear_norm_parameter = (0.018301201576055484, 0.754648223799328)  # error of 1.28E-01 with sample range (-8.99E+02,8.55E+01) resulting in fit range (-1.57E+01,2.32E+00)
    min_max_norm_parameter = (-28.897386873619986, 2.4177080960805726)  # error of 4.02E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-12.927059175873397, 0.15008821168411976)  # error of 2.62E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (1.64E-58,1.00E+00)
    dual_sigmoidal_norm_parameter = (-11.891432065245638, 0.12203526246250417, 0.19311609108878666)  # error of 1.38E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (9.15E-48,1.00E+00)
    genlog_norm_parameter = (0.2714572668516165, -14.400885749752701, 13.80898076286348, 3.1194498243217916)  # error of 1.17E-02 with sample range (-8.99E+02,8.55E+01) resulting in fit range (1.54E-34,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BCUT2D_CHGLO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_CHGLO)
    # normalization
    linear_norm_parameter = (1.7653570916472574, 5.065686201381148)  # error of 5.60E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (-4.74E-01,2.57E+00)
    min_max_norm_parameter = (-2.8372498211712145, -2.348799925638046)  # error of 3.18E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.597093268685726, 10.060474784720004)  # error of 2.27E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (4.32E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.6102535244872342, 12.029554668715791, 8.464996666963719)  # error of 1.37E-02 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (1.75E-03,1.00E+00)
    genlog_norm_parameter = (7.289649684268674, -2.0627835591883987, 0.0013325492833258581, 0.09718128045838645)  # error of 9.20E-03 with sample range (-3.14E+00,-1.41E+00) resulting in fit range (2.54E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_phos_acid_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_phos_acid)
    # normalization
    linear_norm_parameter = (0.4985551300382969, 0.49855513003829677)  # error of 4.44E-16 with sample range (0.00E+00,2.00E+01) resulting in fit range (4.99E-01,1.05E+01)
    min_max_norm_parameter = (2.8684430641636124e-09, 1.0028981147129061)  # error of 3.15E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.4173734339435282, 2.417373387385202)  # error of 1.85E-09 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.4173734339435282, 1.0, 2.417373387385202)  # error of 1.85E-09 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.69E-01,1.00E+00)
    genlog_norm_parameter = (1.1080640261873442, 0.28124367991935634, 0.01186068242567172, 1.8432301014572186)  # error of 2.66E-15 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.91E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SpherocityIndex)
    # normalization
    linear_norm_parameter = (2.014040859270808, -0.004922239185380645)  # error of 8.41E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (-4.92E-03,1.89E+00)
    min_max_norm_parameter = (0.04763405868523997, 0.42911985390392243)  # error of 3.56E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.2351671867619151, 12.82810781848393)  # error of 2.66E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (4.67E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.2212419123873155, 16.425058420900328, 10.08081100416669)  # error of 1.03E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (2.57E-02,9.99E-01)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumAliphaticHeterocycles)
    # normalization
    linear_norm_parameter = (0.07182403555851259, 0.6979616844087828)  # error of 6.50E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (6.98E-01,3.71E+00)
    min_max_norm_parameter = (6.816327279061287e-09, 1.4697587576089106)  # error of 1.83E-02 with sample range (0.00E+00,4.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5242081959438346, 1.5883082643053523)  # error of 6.06E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (3.03E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.5241697890415581, 31.347564520476126, 1.5882035074486005)  # error of 3.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (7.31E-08,1.00E+00)
    genlog_norm_parameter = (1.5563961013766956, 0.6667571953724875, 0.5870663542985489, 0.7783004040766286)  # error of 5.63E-04 with sample range (0.00E+00,4.20E+01) resulting in fit range (2.85E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_NH2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_NH2)
    # normalization
    linear_norm_parameter = (0.027626013199217845, 0.904411104115867)  # error of 2.10E-02 with sample range (0.00E+00,2.40E+01) resulting in fit range (9.04E-01,1.57E+00)
    min_max_norm_parameter = (6.3934981194901046e-09, 1.1000637973187075)  # error of 2.52E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.0551854803177655, 2.1819036489792745)  # error of 1.09E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (5.30E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.055185742977858755, 1.0, 2.181903125192423)  # error of 1.09E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (5.30E-01,1.00E+00)
    genlog_norm_parameter = (2.139757066021772, 0.08837800260788788, 0.010543799958122033, 0.015712998419264372)  # error of 1.07E-03 with sample range (0.00E+00,2.40E+01) resulting in fit range (4.47E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_furan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_furan)
    # normalization
    linear_norm_parameter = (0.4908433241008311, 0.490843324100831)  # error of 3.33E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.91E-01,3.44E+00)
    min_max_norm_parameter = (8.521463502068547e-09, 1.0186549869816706)  # error of 2.70E-04 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.9954042665388447, 1.9954042397477896)  # error of 6.52E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.79E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.9954042665388447, 1.0, 1.9954042397477896)  # error of 6.52E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.79E-01,1.00E+00)
    genlog_norm_parameter = (1.1003367750355502, 0.306436121082991, 0.07327336188983939, 1.8172861344615885)  # error of 4.53E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.48E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_nitro_arom_nonortho_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_nitro_arom_nonortho)
    # normalization
    linear_norm_parameter = (1.0000000000000002, 2.220446049250313e-16)  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    min_max_norm_parameter = (1e-10, 0.9999999999)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5027041798086037, 46.14043898879842)  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.49874072864947394, 3.68328813585096, 20.104860546890446)  # error of 4.31E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.42E-05,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SlogP_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA4)
    # normalization
    linear_norm_parameter = (0.0055106885127717, 0.760669380441174)  # error of 7.89E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (7.61E-01,1.07E+01)
    min_max_norm_parameter = (0.0, 14.831828070749904)  # error of 5.76E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.655549562955406, 0.16917248467895607)  # error of 1.66E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (3.13E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.61614010770599, 0.20613306417396923, 0.16841100863165057)  # error of 1.66E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.79E-01,1.00E+00)
    genlog_norm_parameter = (0.14666017073020396, -11.420609605665565, 0.00658359777337903, 0.0008638133605987126)  # error of 1.58E-02 with sample range (0.00E+00,1.80E+03) resulting in fit range (2.40E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumRotatableBonds)
    # normalization
    linear_norm_parameter = (0.0016857886272905187, 0.8125456140318361)  # error of 1.82E-01 with sample range (0.00E+00,2.48E+02) resulting in fit range (8.13E-01,1.23E+00)
    min_max_norm_parameter = (1.6292528997825728, 13.174098358632333)  # error of 1.89E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.304462270693269, 0.4542440447080067)  # error of 1.34E-02 with sample range (0.00E+00,2.48E+02) resulting in fit range (3.50E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.7090015245725985, 0.6333818384913666, 0.3400706997276338)  # error of 8.05E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (1.41E-02,1.00E+00)
    genlog_norm_parameter = (0.3119530376866779, -5.246761061121691, 0.016007203239645374, 0.0005020797342553604)  # error of 6.90E-03 with sample range (0.00E+00,2.48E+02) resulting in fit range (2.04E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi1n)
    # normalization
    linear_norm_parameter = (0.02460656481380441, 0.05696382112467857)  # error of 1.83E-01 with sample range (0.00E+00,3.32E+02) resulting in fit range (5.70E-02,8.22E+00)
    min_max_norm_parameter = (8.215243805842091, 25.568763302868422)  # error of 3.90E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (16.800375876069985, 0.2846180730802336)  # error of 2.44E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (8.31E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (16.22787766733102, 0.3557663030596946, 0.2286121746689847)  # error of 1.13E-02 with sample range (0.00E+00,3.32E+02) resulting in fit range (3.10E-03,1.00E+00)
    genlog_norm_parameter = (0.20081911303774566, -0.3618619378577607, 1.199011524398242, 0.05743559815299407)  # error of 7.92E-03 with sample range (0.00E+00,3.32E+02) resulting in fit range (2.17E-06,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_thiophene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_thiophene)
    # normalization
    linear_norm_parameter = (0.027482526572608745, 0.9433700966912973)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (9.43E-01,1.27E+00)
    min_max_norm_parameter = (9.055969684787418e-09, 1.0300224521968595)  # error of 5.45E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.2128545965336391, 2.8905441523149107)  # error of 0.00E+00 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.49E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.2128546047774725, 1.0, 2.8905441234721354)  # error of 2.28E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.49E-01,1.00E+00)
    genlog_norm_parameter = (2.9024921333781952, 0.4442033177604824, 0.2822646045620353, 1.8497447601273862)  # error of 1.03E-10 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.83E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  PEOE_VSA13_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA13)
    # normalization
    linear_norm_parameter = (0.014179141123108696, 0.7394291555972963)  # error of 6.57E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (7.39E-01,2.50E+00)
    min_max_norm_parameter = (3.944304526105059e-30, 7.6449284829185435)  # error of 6.49E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.5958556965144552, 0.30276265905030647)  # error of 3.71E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (3.82E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.07933651635307, -0.4986062273546034, 0.33289138645741445)  # error of 3.55E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (7.38E-01,1.00E+00)
    genlog_norm_parameter = (0.3872457370028876, 4.316346368590775, 2.291146781648952, 3.2508927309627635)  # error of 3.62E-02 with sample range (0.00E+00,1.24E+02) resulting in fit range (4.52E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcKappa2)
    # normalization
    linear_norm_parameter = (0.030418143185359847, 0.2871593951766924)  # error of 2.34E-01 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.87E-01,1.08E+03)
    min_max_norm_parameter = (3.446066777200142, 12.00058941506709)  # error of 4.58E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.66772112289306, 0.5795843116540953)  # error of 3.08E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (1.16E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (7.271319784936047, 0.8080128800230181, 0.4459683689150529)  # error of 1.32E-02 with sample range (0.00E+00,3.57E+04) resulting in fit range (2.80E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  HeavyAtomCount_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(HeavyAtomCount)
    # normalization
    linear_norm_parameter = (0.005752556775892259, 0.43633424647546626)  # error of 2.06E-01 with sample range (2.00E+00,5.72E+02) resulting in fit range (4.48E-01,3.73E+00)
    min_max_norm_parameter = (11.422941214588905, 38.12137726479023)  # error of 2.20E-02 with sample range (2.00E+00,5.72E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (24.563511621430862, 0.19297073304914714)  # error of 1.69E-02 with sample range (2.00E+00,5.72E+02) resulting in fit range (1.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (22.896239973647287, 0.28441190110771175, 0.1398321789344552)  # error of 7.75E-03 with sample range (2.00E+00,5.72E+02) resulting in fit range (2.62E-03,1.00E+00)
    genlog_norm_parameter = (0.1328591610228903, -10.662363837901518, 0.012852917815098123, 0.00018843531781919377)  # error of 6.51E-03 with sample range (2.00E+00,5.72E+02) resulting in fit range (3.15E-06,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_amide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_amide)
    # normalization
    linear_norm_parameter = (0.018247396195880828, 0.8361339146143514)  # error of 7.21E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (8.36E-01,1.71E+00)
    min_max_norm_parameter = (0.04651569330182743, 1.4956993928811566)  # error of 2.28E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4809600464417933, 1.266170913827053)  # error of 2.07E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (3.52E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.5000000055065682, 9.81151356748907, 1.2896670560945562)  # error of 1.88E-03 with sample range (0.00E+00,4.80E+01) resulting in fit range (7.35E-03,1.00E+00)
    genlog_norm_parameter = (1.9646058102099027, -1.2572857779178173, 0.009448860004671444, 0.00024144940936888026)  # error of 1.47E-02 with sample range (0.00E+00,4.80E+01) resulting in fit range (3.66E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_barbitur_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_barbitur)
    # normalization
    linear_norm_parameter = (0.49999999999890943, 0.16647835027998992)  # error of 2.35E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.66E-01,1.17E+00)
    min_max_norm_parameter = (5.002826379765727e-11, 1.0005652758530887)  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6589560462180946, 21.924533557093195)  # error of 4.43E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (5.32E-07,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4999997695881287, 8.487133810156745, 13.55104020684501)  # error of 7.37E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.14E-03,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SlogP_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA8)
    # normalization
    linear_norm_parameter = (0.0012320682276687166, 0.9018392839757955)  # error of 5.04E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (9.02E-01,1.34E+00)
    min_max_norm_parameter = (1.6753303243403e-23, 12.492939268423935)  # error of 5.71E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.9543059714466142, 0.13070084470487742)  # error of 1.44E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (5.31E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.9543065457922881, 1.0, 0.1307008395582034)  # error of 1.44E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (5.31E-01,1.00E+00)
    genlog_norm_parameter = (0.12040713160705811, -8.829473521301857, 0.001479650499031926, 0.0007251364390775469)  # error of 1.40E-02 with sample range (0.00E+00,3.57E+02) resulting in fit range (4.94E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(RadiusOfGyration)
    # normalization
    linear_norm_parameter = (0.274743032958996, -0.6363152066427643)  # error of 8.52E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (-6.36E-01,2.85E+00)
    min_max_norm_parameter = (2.68976145372615, 5.459084333173108)  # error of 2.97E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.059081941807601, 1.764235395780906)  # error of 1.73E-02 with sample range (0.00E+00,1.27E+01) resulting in fit range (7.76E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.992187925992685, 2.0752401030308176, 1.5066909435317473)  # error of 7.09E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (2.52E-04,1.00E+00)
    genlog_norm_parameter = (1.347041014911159, 2.24246525276504, 1.9688953365934632, 0.23779698513445705)  # error of 2.95E-03 with sample range (0.00E+00,1.27E+01) resulting in fit range (1.59E-07,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa3)
    # normalization
    linear_norm_parameter = (0.03561704845372893, 0.3431665835165363)  # error of 2.45E-01 with sample range (-4.32E+00,2.80E+04) resulting in fit range (1.89E-01,9.97E+02)
    min_max_norm_parameter = (1.6209280367339833, 6.869092211937303)  # error of 4.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.207644140181935, 0.935275527719581)  # error of 3.22E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (3.42E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.9792547209665945, 1.2655211736041962, 0.6994417885081644)  # error of 1.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (2.73E-05,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi0v)
    # normalization
    linear_norm_parameter = (0.027431175492895643, 0.1959231987976101)  # error of 1.84E-01 with sample range (1.00E+00,2.77E+02) resulting in fit range (2.23E-01,7.80E+00)
    min_max_norm_parameter = (5.833583645428592, 19.149899507438242)  # error of 4.23E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (12.391158701361956, 0.36836959542101233)  # error of 2.62E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (1.48E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (11.842303238148796, 0.5140666350188914, 0.29912163479666287)  # error of 1.16E-02 with sample range (1.00E+00,2.77E+02) resulting in fit range (3.78E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumBridgeheadAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumBridgeheadAtoms)
    # normalization
    linear_norm_parameter = (0.0022274780962232565, 0.983622366844093)  # error of 2.88E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.84E-01,1.03E+00)
    min_max_norm_parameter = (5.24640715369184e-09, 2.0351023563443684)  # error of 2.27E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-4.164793009240712, 0.6606716515917349)  # error of 8.24E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.40E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-4.164793009240712, 1.0, 0.6606716515917349)  # error of 8.24E-04 with sample range (0.00E+00,2.00E+01) resulting in fit range (9.40E-01,1.00E+00)
    genlog_norm_parameter = (3.008930969948126, 0.14273145549950425, 0.0037597630799410166, 0.0007848218524412082)  # error of 2.11E-03 with sample range (0.00E+00,2.00E+01) resulting in fit range (6.50E-04,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAromaticHeterocycles)
    # normalization
    linear_norm_parameter = (0.06608705216326816, 0.7231759141671211)  # error of 6.12E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (7.23E-01,1.71E+00)
    min_max_norm_parameter = (2.425436375356735e-09, 1.4183360150961999)  # error of 2.28E-02 with sample range (0.00E+00,1.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4595690682034204, 1.6119949186776596)  # error of 6.29E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.23E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4595690682034204, 1.0, 1.6119949186776596)  # error of 6.29E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.23E-01,1.00E+00)
    genlog_norm_parameter = (1.6718834720391067, 1.1087698014134977, 0.5469129170165173, 1.4432376478453104)  # error of 5.24E-04 with sample range (0.00E+00,1.50E+01) resulting in fit range (3.53E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Ipc)
    # normalization
    linear_norm_parameter = (1.5595572574111734e-38, 0.4932082097701287)  # error of 2.81E-01 with sample range (0.00E+00,INF) resulting in fit range (4.93E-01,INF)
    min_max_norm_parameter = (5e-324, 1794244951.795763)  # error of 2.42E-01 with sample range (0.00E+00,INF) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 5.79E-01 with sample range (0.00E+00,INF) resulting in fit range (2.69E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_C_O_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_C_O)
    # normalization
    linear_norm_parameter = (0.022515720837872344, 0.7822215231398375)  # error of 9.81E-02 with sample range (0.00E+00,7.50E+01) resulting in fit range (7.82E-01,2.47E+00)
    min_max_norm_parameter = (5e-324, 2.3300769555158443)  # error of 1.98E-02 with sample range (0.00E+00,7.50E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.9129037319576498, 1.3107513550393526)  # error of 3.49E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (2.32E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.9129037321984369, 15.763177112455574, 1.3107513626098963)  # error of 1.96E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (5.63E-07,1.00E+00)
    genlog_norm_parameter = (1.1392132675704918, -0.13824593331385832, 0.256026454542409, 0.1056220644977619)  # error of 2.72E-03 with sample range (0.00E+00,7.50E+01) resulting in fit range (1.54E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_aldehyde_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_aldehyde)
    # normalization
    linear_norm_parameter = (0.49738773510384093, 0.4973877351038408)  # error of 4.44E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.97E-01,2.98E+00)
    min_max_norm_parameter = (7.478980473497863e-09, 1.0052519688208172)  # error of 9.75E-05 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.291096523056102, 2.291096475694497)  # error of 1.51E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.51E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.291096523056102, 1.0, 2.291096475694497)  # error of 1.51E-07 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.51E-01,1.00E+00)
    genlog_norm_parameter = (1.1069264725345216, 0.28497922099754747, 0.02136353694670169, 1.8393502310616188)  # error of 3.28E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.84E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcKappa3)
    # normalization
    linear_norm_parameter = (0.03561704845372893, 0.3431665835165363)  # error of 2.45E-01 with sample range (-4.32E+00,2.80E+04) resulting in fit range (1.89E-01,9.97E+02)
    min_max_norm_parameter = (1.6209280367339833, 6.869092211937303)  # error of 4.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.207644140181935, 0.935275527719581)  # error of 3.22E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (3.42E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.9792547209665945, 1.2655211736041962, 0.6994417885081644)  # error of 1.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (2.73E-05,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  EState_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(EState_VSA4)
    # normalization
    linear_norm_parameter = (0.0016155123555318786, 0.9703629575801614)  # error of 5.51E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.70E-01,1.08E+00)
    min_max_norm_parameter = (4.536992969157554e-16, 2.4512528381627163)  # error of 1.42E-02 with sample range (0.00E+00,6.78E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-13.655427016167506, 0.20068528350641354)  # error of 3.15E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-13.65542695721612, 1.0, 0.20068528388954926)  # error of 3.15E-03 with sample range (0.00E+00,6.78E+01) resulting in fit range (9.39E-01,1.00E+00)
    genlog_norm_parameter = (1.584025593452512, -0.7274076027985914, 0.004645841965062258, 0.00039148840169634573)  # error of 1.27E-02 with sample range (0.00E+00,6.78E+01) resulting in fit range (2.36E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumAliphaticCarbocycles)
    # normalization
    linear_norm_parameter = (0.029386355676372537, 0.8723594863010827)  # error of 2.62E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (8.72E-01,1.40E+00)
    min_max_norm_parameter = (7.2035108451548135e-09, 1.1561266295897044)  # error of 1.01E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.4231823522456474, 1.309702732778145)  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.4231823921093463, 1.0, 1.3097026985565412)  # error of 3.86E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (6.35E-01,1.00E+00)
    genlog_norm_parameter = (3.5827272813887046, -0.1961244562254873, 0.008501771849148765, 0.0007936347601062923)  # error of 9.27E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (5.02E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_tetrazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_tetrazole)
    # normalization
    linear_norm_parameter = (0.4981451669349761, 0.4981451669349759)  # error of 2.22E-16 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.98E-01,1.49E+00)
    min_max_norm_parameter = (5.01861743222311e-11, 1.0037234863442497)  # error of 4.21E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.7145014458224443, 19.590579896967135)  # error of 4.82E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (8.34E-07,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3649097870387985, 1.0, 2.3649097938746766)  # error of 1.10E-06 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.62E-01,1.00E+00)
    genlog_norm_parameter = (1.1076660719575397, 0.2825520764849789, 0.015205956722974494, 1.8418695837413916)  # error of 4.01E-08 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.89E-01,9.99E-01)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_oxazole_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_oxazole)
    # normalization
    linear_norm_parameter = (0.49769270765631113, 0.497692707656311)  # error of 2.22E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.98E-01,3.48E+00)
    min_max_norm_parameter = (2.9758329081720466e-09, 1.004635977785747)  # error of 5.42E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3181535623480032, 2.318153583438047)  # error of 3.31E-07 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.55E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3181535623480032, 1.0, 2.318153583438047)  # error of 3.31E-07 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.55E-01,1.00E+00)
    genlog_norm_parameter = (1.1072249258768634, 0.2840004343701646, 0.0188879042419569, 1.840365515292451)  # error of 3.56E-08 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.86E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_benzodiazepine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_benzodiazepine)
    # normalization
    linear_norm_parameter = (1.0000000000000002, 2.220446049250313e-16)  # error of 3.51E-16 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.22E-16,1.00E+00)
    min_max_norm_parameter = (1e-10, 0.9999999999)  # error of 0.00E+00 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.5027041798086037, 46.14043898879842)  # error of 9.71E-11 with sample range (0.00E+00,1.00E+00) resulting in fit range (8.44E-11,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.49874072864947394, 3.68328813585096, 20.104860546890446)  # error of 4.31E-05 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.42E-05,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(NPR2)
    # normalization
    linear_norm_parameter = (3.0363848124092283, -2.1050145083551772)  # error of 7.55E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (-2.11E+00,9.31E-01)
    min_max_norm_parameter = (0.7328377706938303, 0.9989929606321605)  # error of 3.95E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.8690167433696966, 18.125099021117805)  # error of 3.57E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.44E-07,9.15E-01)
    dual_sigmoidal_norm_parameter = (0.8826119375023386, 13.441774499098555, 25.188219083401254)  # error of 1.76E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (7.04E-06,9.51E-01)
    genlog_norm_parameter = (41.229691541183385, 0.9463746291627356, 1.2160790825123706, 4.408357021765385)  # error of 1.41E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (1.37E-04,9.72E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  FpDensityMorgan2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(FpDensityMorgan2)
    # normalization
    linear_norm_parameter = (0.3914967754398223, -0.3064699812712066)  # error of 1.22E-01 with sample range (1.72E-02,5.00E+00) resulting in fit range (-3.00E-01,1.65E+00)
    min_max_norm_parameter = (1.3833635629107808, 2.497866940325602)  # error of 3.66E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.9426344460243463, 4.4864204391015186)  # error of 1.50E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (1.77E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.9697213353061016, 4.019684853988071, 5.57668286972073)  # error of 1.08E-02 with sample range (1.72E-02,5.00E+00) resulting in fit range (3.90E-04,1.00E+00)
    genlog_norm_parameter = (5.761552987636311, 158.08674397977347, -130.86735690333066, 17.70616615221605)  # error of 3.14E-01 with sample range (1.72E-02,5.00E+00) resulting in fit range (0.00E+00,0.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  NumHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumHeterocycles)
    # normalization
    linear_norm_parameter = (0.08010064808252315, 0.5457901645984795)  # error of 1.04E-01 with sample range (0.00E+00,5.10E+01) resulting in fit range (5.46E-01,4.63E+00)
    min_max_norm_parameter = (2.3665827156630354e-30, 2.5826903013211857)  # error of 2.27E-02 with sample range (0.00E+00,5.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.1686560826020225, 1.2821472679988912)  # error of 9.61E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (1.83E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.168651725696023, 14.277943710321482, 1.2821257124964283)  # error of 5.53E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (5.67E-08,1.00E+00)
    genlog_norm_parameter = (1.2460768793121881, 1.6487717290239703, 0.438514655086741, 0.8475385257399649)  # error of 4.68E-04 with sample range (0.00E+00,5.10E+01) resulting in fit range (1.73E-01,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  NumLipinskiHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumLipinskiHBD)
    # normalization
    linear_norm_parameter = (0.011451805303745677, 0.8103893376864633)  # error of 1.09E-01 with sample range (0.00E+00,8.20E+01) resulting in fit range (8.10E-01,1.75E+00)
    min_max_norm_parameter = (9.860761315262648e-32, 3.187051718282465)  # error of 1.94E-02 with sample range (0.00E+00,8.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.3178308349589882, 1.1160540358868796)  # error of 6.16E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.87E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.3178308564763852, 10.470315521888269, 1.1160541755540871)  # error of 3.83E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.02E-06,1.00E+00)
    genlog_norm_parameter = (0.9056090949083335, -2.61356505138166, 0.03003179441465175, 0.0012602623451871441)  # error of 4.05E-03 with sample range (0.00E+00,8.20E+01) resulting in fit range (1.07E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi1v)
    # normalization
    linear_norm_parameter = (0.0627452397151258, 0.08691967075790497)  # error of 1.60E-01 with sample range (0.00E+00,1.66E+02) resulting in fit range (8.69E-02,1.05E+01)
    min_max_norm_parameter = (2.910758257183072, 10.518729378759696)  # error of 3.99E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.661928594684052, 0.6389477840131871)  # error of 2.54E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (1.40E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.384313047563608, 0.8556587318530761, 0.5214673705729024)  # error of 1.05E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (4.22E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_Ar_NH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Ar_NH)
    # normalization
    linear_norm_parameter = (0.025767680908718393, 0.9464298213160812)  # error of 0.00E+00 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.46E-01,1.18E+00)
    min_max_norm_parameter = (6.578269936941624e-09, 1.0285975817965838)  # error of 8.35E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.3459278133476376, 2.64087945539897)  # error of 7.85E-17 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.14E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.3459278137373095, 1.0, 2.6408794542523295)  # error of 1.01E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.14E-01,1.00E+00)
    genlog_norm_parameter = (2.6522539776099108, 0.42594124972784025, 0.24728223139928077, 1.8634027861924218)  # error of 1.04E-12 with sample range (0.00E+00,9.00E+00) resulting in fit range (7.37E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_aryl_methyl_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_aryl_methyl)
    # normalization
    linear_norm_parameter = (0.03813656770680851, 0.8922580301103881)  # error of 1.20E-02 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.92E-01,1.20E+00)
    min_max_norm_parameter = (4.251871042900079e-09, 1.0846727833106184)  # error of 5.19E-03 with sample range (0.00E+00,8.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.39967703372113306, 1.763714100024283)  # error of 4.24E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.69E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.39967703372113306, 1.0, 1.763714100024283)  # error of 4.24E-04 with sample range (0.00E+00,8.00E+00) resulting in fit range (6.69E-01,1.00E+00)
    genlog_norm_parameter = (2.1866818357968, 1.264315201585735, 1.0410246979383717, 12.909498651731809)  # error of 2.15E-09 with sample range (0.00E+00,8.00E+00) resulting in fit range (8.01E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcChi1v)
    # normalization
    linear_norm_parameter = (0.0627452397151258, 0.08691967075790497)  # error of 1.60E-01 with sample range (0.00E+00,1.66E+02) resulting in fit range (8.69E-02,1.05E+01)
    min_max_norm_parameter = (2.910758257183072, 10.518729378759696)  # error of 3.99E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.661928594684052, 0.6389477840131871)  # error of 2.54E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (1.40E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.384313047563608, 0.8556587318530761, 0.5214673705729024)  # error of 1.05E-02 with sample range (0.00E+00,1.66E+02) resulting in fit range (4.22E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_Al_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_Al_OH)
    # normalization
    linear_norm_parameter = (0.006501081569325162, 0.9535852884351536)  # error of 1.73E-02 with sample range (0.00E+00,3.70E+01) resulting in fit range (9.54E-01,1.19E+00)
    min_max_norm_parameter = (5.758669341509425e-09, 1.0871784274707261)  # error of 4.26E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.8719669615585766, 1.3070605975374598)  # error of 2.41E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (7.58E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.8719669615585766, 1.0, 1.3070605975374598)  # error of 2.41E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (7.58E-01,1.00E+00)
    genlog_norm_parameter = (4.317780505773897, -0.4739394325838553, 0.05414752688497957, 0.0011076622058086279)  # error of 4.10E-03 with sample range (0.00E+00,3.70E+01) resulting in fit range (1.85E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SMR_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA9)
    # normalization
    linear_norm_parameter = (0.002463279237957446, 0.8723998320669256)  # error of 5.34E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (8.72E-01,1.54E+00)
    min_max_norm_parameter = (5e-324, 13.161474983617241)  # error of 4.95E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.1525795833229362, 0.15186000443615003)  # error of 1.83E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (4.56E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.1525795889639978, 1.0, 0.1518600046045379)  # error of 1.83E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (4.56E-01,1.00E+00)
    genlog_norm_parameter = (0.1391729597386993, -3.678587145902322, 0.0009810210159954982, 0.0006414479949079956)  # error of 1.79E-02 with sample range (0.00E+00,2.73E+02) resulting in fit range (4.00E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  HeavyAtomMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(HeavyAtomMolWt)
    # normalization
    linear_norm_parameter = (0.001100187734004554, 0.17368773471419774)  # error of 1.74E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (2.00E-01,1.54E+01)
    min_max_norm_parameter = (154.3124682840099, 516.5882134868882)  # error of 4.25E-02 with sample range (2.40E+01,1.38E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.0, 1.0)  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.0, 1.0, 1.0)  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    genlog_norm_parameter = (1.0, 1.0, 1.0, 1.0)  # error of 4.34E-01 with sample range (2.40E+01,1.38E+04) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcSpherocityIndex)
    # normalization
    linear_norm_parameter = (2.014040859270808, -0.004922239185380645)  # error of 8.41E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (-4.92E-03,1.89E+00)
    min_max_norm_parameter = (0.04763405868523997, 0.42911985390392243)  # error of 3.56E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.2351671867619151, 12.82810781848393)  # error of 2.66E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (4.67E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.2212419123873155, 16.425058420900328, 10.08081100416669)  # error of 1.03E-02 with sample range (0.00E+00,9.39E-01) resulting in fit range (2.57E-02,9.99E-01)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_allylic_oxid_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_allylic_oxid)
    # normalization
    linear_norm_parameter = (0.004305773194698159, 0.9499732166962105)  # error of 1.75E-02 with sample range (0.00E+00,5.60E+01) resulting in fit range (9.50E-01,1.19E+00)
    min_max_norm_parameter = (6.2907096383782455e-09, 1.107924268433823)  # error of 1.06E-02 with sample range (0.00E+00,5.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.624628166420796, 0.6192650652321926)  # error of 1.97E-03 with sample range (0.00E+00,5.60E+01) resulting in fit range (8.36E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.624628166420796, 1.0, 0.6192650652321926)  # error of 1.97E-03 with sample range (0.00E+00,5.60E+01) resulting in fit range (8.36E-01,1.00E+00)
    genlog_norm_parameter = (3.9521439344063825, -0.3039075954435677, 0.01726230829074985, 0.0009515842210063363)  # error of 1.04E-02 with sample range (0.00E+00,5.60E+01) resulting in fit range (4.32E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  VSA_EState7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState7)
    # normalization
    linear_norm_parameter = (0.006935598460676706, 0.5277845083933819)  # error of 1.85E-01 with sample range (-7.84E+02,1.08E+01) resulting in fit range (-4.91E+00,6.03E-01)
    min_max_norm_parameter = (-24.625854914081717, 6.979695541871978)  # error of 5.05E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-8.479594411787089, 0.14779915592252638)  # error of 3.33E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.74E-50,9.46E-01)
    dual_sigmoidal_norm_parameter = (-7.0464457823056605, 0.11243413657001933, 0.23343926406202142)  # error of 1.51E-02 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.19E-38,9.85E-01)
    genlog_norm_parameter = (0.5395522034797766, 2.8636999681039996, 0.38276869755674797, 6.8178631448799)  # error of 9.45E-03 with sample range (-7.84E+02,1.08E+01) resulting in fit range (1.07E-27,9.99E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SMR_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA8)
    # normalization
    linear_norm_parameter = (1.0, 1.0)  # error of 0.00E+00 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    dual_sigmoidal_norm_parameter = (0.0, 1.0, 0.0)  # error of 5.00E-01 with sample range (0.00E+00,0.00E+00) resulting in fit range (5.00E-01,5.00E-01)
    genlog_norm_parameter = (0.33705420273837067, 0.3370542027383706, -9.284058023162583e-10, 2.1473819766010673)  # error of 4.84E-10 with sample range (0.00E+00,0.00E+00) resulting in fit range (1.00E+00,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  MolLogP_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MolLogP)
    # normalization
    linear_norm_parameter = (0.08784472608787079, 0.1981370072747647)  # error of 1.64E-01 with sample range (-5.89E+01,9.69E+01) resulting in fit range (-4.98E+00,8.71E+00)
    min_max_norm_parameter = (0.46977870811162403, 5.966058231862306)  # error of 3.39E-02 with sample range (-5.89E+01,9.69E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (3.206291545692704, 0.8985363235365765)  # error of 1.25E-02 with sample range (-5.89E+01,9.69E+01) resulting in fit range (5.76E-25,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.1239114240932673, 0.9965869301073036, 0.811237464475052)  # error of 7.33E-03 with sample range (-5.89E+01,9.69E+01) resulting in fit range (1.42E-27,1.00E+00)
    genlog_norm_parameter = (0.7571489399076989, 1.390600008482986, 1.6113843793551472, 0.5076714309656658)  # error of 6.47E-03 with sample range (-5.89E+01,9.69E+01) resulting in fit range (3.42E-40,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_phenol_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_phenol)
    # normalization
    linear_norm_parameter = (0.010549050583289588, 0.9694394171190582)  # error of 3.47E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.69E-01,1.24E+00)
    min_max_norm_parameter = (6.645986623350427e-09, 1.0229791585226422)  # error of 1.21E-03 with sample range (0.00E+00,2.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3928295501175463, 1.5771919998858934)  # error of 2.42E-04 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3928295501175463, 1.0, 1.5771919998858934)  # error of 2.42E-04 with sample range (0.00E+00,2.60E+01) resulting in fit range (9.00E-01,1.00E+00)
    genlog_norm_parameter = (1.1092336835627035, 0.805783076032451, -0.9472745670283325, -63.49689033500014)  # error of 1.60E-08 with sample range (0.00E+00,2.60E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  SlogP_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA1)
    # normalization
    linear_norm_parameter = (0.007436684822960693, 0.6872466373090912)  # error of 1.45E-01 with sample range (0.00E+00,4.14E+02) resulting in fit range (6.87E-01,3.76E+00)
    min_max_norm_parameter = (1.9721522630525295e-31, 16.894446647493005)  # error of 4.50E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (7.760808021810809, 0.25639853999337753)  # error of 3.00E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (1.20E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (7.400786131696198, 0.3165029149114655, 0.2383662278955665)  # error of 2.94E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (8.77E-02,1.00E+00)
    genlog_norm_parameter = (0.2191296949501139, -0.07851957643167685, 1.2561993381152745, 0.30031793241554744)  # error of 2.92E-02 with sample range (0.00E+00,4.14E+02) resulting in fit range (6.87E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_azo_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_azo)
    # normalization
    linear_norm_parameter = (0.4990500854923059, 0.4990500854923058)  # error of 3.33E-16 with sample range (0.00E+00,4.00E+00) resulting in fit range (4.99E-01,2.50E+00)
    min_max_norm_parameter = (5.424965999671179e-12, 1.0019034438689793)  # error of 1.50E-04 with sample range (0.00E+00,4.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.502811180689176, 2.502811017959295)  # error of 5.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.77E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.502811180689176, 1.0, 2.502811017959295)  # error of 5.00E-08 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.77E-01,1.00E+00)
    genlog_norm_parameter = (1.1085421304163696, 0.27966912004024697, 0.007809921991104715, 1.844869681277523)  # error of 3.66E-15 with sample range (0.00E+00,4.00E+00) resulting in fit range (9.94E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BalabanJ)
    # normalization
    linear_norm_parameter = (0.17660976361946912, 0.011623897998747967)  # error of 1.60E-01 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.16E-02,8.95E+00)
    min_max_norm_parameter = (1.51069451962931, 3.7663482034024947)  # error of 4.43E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.6275405979312487, 2.067375370708902)  # error of 2.69E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (4.35E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.548536235030395, 2.66279982408227, 1.6191594985042048)  # error of 1.47E-02 with sample range (-7.44E-05,5.06E+01) resulting in fit range (1.13E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PMI2)
    # normalization
    linear_norm_parameter = (3.364102817740711e-05, 0.2861109449057394)  # error of 1.90E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.86E-01,2.45E+01)
    min_max_norm_parameter = (44.665447243295176, 10171.614030140174)  # error of 5.66E-02 with sample range (0.00E+00,7.21E+05) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.8655544226961647, 2.8655544226961647)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.86555442244817, 1.0, 2.8655544224481697)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (2.71E-04,1.00E+00)
    genlog_norm_parameter = (1.3122264001263977, 1.3122264001263975, 1.279660664863473, 0.2618729161321768)  # error of 5.77E-01 with sample range (0.00E+00,7.21E+05) resulting in fit range (3.30E-04,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  fr_isocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_isocyan)
    # normalization
    linear_norm_parameter = (0.49999999999890943, 0.1665333453304355)  # error of 2.36E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.67E-01,1.17E+00)
    min_max_norm_parameter = (5.002000657445256e-11, 1.000400131389011)  # error of 4.24E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6485353213537604, 22.257526471575645)  # error of 3.87E-07 with sample range (0.00E+00,2.00E+00) resulting in fit range (5.38E-07,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.49999969269758465, 8.490066047780406, 13.665791744120064)  # error of 7.34E-04 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.08E-03,1.00E+00)
    genlog_norm_parameter = (1.6834007612955404, 0.10832663972825836, 0.00014387991489756035, 0.00010533390176350755)  # error of 1.91E-01 with sample range (0.00E+00,2.00E+00) resulting in fit range (1.94E-01,9.45E-01)
    preferred_normalization = 'unity'
    # functions
    
    

class  TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcTPSA)
    # normalization
    linear_norm_parameter = (0.0021678074691265925, 0.4909310697599548)  # error of 1.90E-01 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.91E-01,6.20E+00)
    min_max_norm_parameter = (14.599527973028703, 118.60656324496115)  # error of 3.84E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (66.26559293182696, 0.047786843726724)  # error of 1.40E-02 with sample range (0.00E+00,2.63E+03) resulting in fit range (4.04E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (64.41269346904815, 0.05696911545435016, 0.043756733059762035)  # error of 9.63E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (2.49E-02,1.00E+00)
    genlog_norm_parameter = (0.039396470555296006, -1.565855746207205, 3.3127216535027557, 0.3094077325658369)  # error of 7.60E-03 with sample range (0.00E+00,2.63E+03) resulting in fit range (1.03E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(CalcInertialShapeFactor)
    # normalization
    linear_norm_parameter = (249.802812058766, 0.2626729794953536)  # error of 1.70E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (2.63E-01,2.41E+06)
    min_max_norm_parameter = (3.4403992924852414e-29, 0.0015516770560768962)  # error of 5.89E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.0007537390827077178, 2772.3791520268146)  # error of 5.07E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (1.10E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.0006221969804982748, 4615.762401524865, 1723.753663227779)  # error of 2.12E-02 with sample range (0.00E+00,9.63E+03) resulting in fit range (5.36E-02,1.00E+00)
    genlog_norm_parameter = (0.6335439368983742, 1.1952465398370342, 0.06337814286503886, 0.1824990146140799)  # error of 2.88E-01 with sample range (0.00E+00,9.63E+03) resulting in fit range (4.99E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa1)
    # normalization
    linear_norm_parameter = (0.07531844296316315, 0.10272005612925916)  # error of 1.58E-01 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.06E-01,3.83E+01)
    min_max_norm_parameter = (1.58553325314983, 8.60938851018364)  # error of 3.82E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.045698563226252, 0.6990022831903797)  # error of 2.71E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (2.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.767359124032586, 0.9160893907811511, 0.5589277261645443)  # error of 1.25E-02 with sample range (4.08E-02,5.07E+02) resulting in fit range (1.30E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MinEStateIndex)
    # normalization
    linear_norm_parameter = (0.22449008413602467, 1.529064938673871)  # error of 9.84E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (-3.78E+00,1.79E+00)
    min_max_norm_parameter = (-6.065158419096168, -2.9821960881671177)  # error of 4.20E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-4.49507192502996, 1.6028612470816288)  # error of 2.52E-02 with sample range (-2.36E+01,1.17E+00) resulting in fit range (4.77E-14,1.00E+00)
    dual_sigmoidal_norm_parameter = (-4.402227821606564, 1.2805551665118835, 2.027604245118143)  # error of 8.88E-03 with sample range (-2.36E+01,1.17E+00) resulting in fit range (2.02E-11,1.00E+00)
    genlog_norm_parameter = (2.5886519555655454, -2.465861347030154, 0.03346380192846074, 2.7129756634982467)  # error of 6.76E-03 with sample range (-2.36E+01,1.17E+00) resulting in fit range (5.93E-09,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  MaxEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(MaxEStateIndex)
    # normalization
    linear_norm_parameter = (0.1160824744026907, -0.9656546887853172)  # error of 9.45E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (-9.66E-01,2.80E+00)
    min_max_norm_parameter = (10.793338979130617, 15.53686323514367)  # error of 7.37E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (13.160091893100835, 0.9136778762386883)  # error of 6.33E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (6.00E-06,1.00E+00)
    dual_sigmoidal_norm_parameter = (13.444744292036967, 0.490702773968197, 1.4564642792613827)  # error of 3.13E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (1.36E-03,1.00E+00)
    genlog_norm_parameter = (2.973557817315883, 14.214531367680852, 9.116290253481058, 7.28923697651459)  # error of 3.39E-02 with sample range (0.00E+00,3.25E+01) resulting in fit range (2.24E-03,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_phenol_noOrthoHbond_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_phenol_noOrthoHbond)
    # normalization
    linear_norm_parameter = (0.01009659130462326, 0.9708109603468388)  # error of 3.45E-03 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.71E-01,1.20E+00)
    min_max_norm_parameter = (8.618314307387322e-09, 1.0220069386129595)  # error of 1.17E-03 with sample range (0.00E+00,2.30E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.3483707918108945, 1.6255249140159453)  # error of 3.04E-04 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.00E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.3483707918108945, 1.0, 1.6255249140159453)  # error of 3.04E-04 with sample range (0.00E+00,2.30E+01) resulting in fit range (9.00E-01,1.00E+00)
    genlog_norm_parameter = (0.9985301422047174, 0.8797776244208874, -0.9809659414331601, -93.72411900391694)  # error of 1.48E-09 with sample range (0.00E+00,2.30E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumLipinskiHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(CalcNumLipinskiHBA)
    # normalization
    linear_norm_parameter = (0.0021474686509596053, 0.8235464950362099)  # error of 1.92E-01 with sample range (0.00E+00,1.96E+02) resulting in fit range (8.24E-01,1.24E+00)
    min_max_norm_parameter = (1.2880062839960937, 9.187763624570907)  # error of 1.38E-02 with sample range (0.00E+00,1.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (5.184102720727361, 0.6674369952484219)  # error of 7.25E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (3.05E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (5.000220272676166, 0.7699798791319306, 0.5818113932872683)  # error of 5.06E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (2.08E-02,1.00E+00)
    genlog_norm_parameter = (0.5268272560910332, 2.9870414098551117, 0.8114362377392241, 0.34748243027398057)  # error of 3.62E-03 with sample range (0.00E+00,1.96E+02) resulting in fit range (1.02E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_sulfide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_sulfide)
    # normalization
    linear_norm_parameter = (0.041736243738063505, 0.9134677878990892)  # error of 0.00E+00 with sample range (0.00E+00,1.60E+01) resulting in fit range (9.13E-01,1.58E+00)
    min_max_norm_parameter = (9.293070357320419e-09, 1.0468967534295035)  # error of 8.64E-04 with sample range (0.00E+00,1.60E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.12222280200277533, 2.7265591239312834)  # error of 1.11E-16 with sample range (0.00E+00,1.60E+01) resulting in fit range (5.83E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.12222320614972192, 1.0, 2.726557767129511)  # error of 1.33E-08 with sample range (0.00E+00,1.60E+01) resulting in fit range (5.83E-01,1.00E+00)
    genlog_norm_parameter = (2.743426527038699, 0.4809352743948252, 0.3530749343716047, 1.780066358823601)  # error of 5.89E-09 with sample range (0.00E+00,1.60E+01) resulting in fit range (6.23E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Kappa3)
    # normalization
    linear_norm_parameter = (0.03561704845372893, 0.3431665835165363)  # error of 2.45E-01 with sample range (-4.32E+00,2.80E+04) resulting in fit range (1.89E-01,9.97E+02)
    min_max_norm_parameter = (1.6209280367339833, 6.869092211937303)  # error of 4.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.207644140181935, 0.935275527719581)  # error of 3.22E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (3.42E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (3.9792547209665945, 1.2655211736041962, 0.6994417885081644)  # error of 1.42E-02 with sample range (-4.32E+00,2.80E+04) resulting in fit range (2.73E-05,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_alkyl_carbamate_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_alkyl_carbamate)
    # normalization
    linear_norm_parameter = (0.4965278124968755, 0.4965278124968754)  # error of 4.44E-16 with sample range (0.00E+00,5.00E+00) resulting in fit range (4.97E-01,2.98E+00)
    min_max_norm_parameter = (6.897442569832465e-09, 1.0069929365320245)  # error of 2.31E-04 with sample range (0.00E+00,5.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.227745991027422, 2.22774593441806)  # error of 1.81E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.39E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.227745991027422, 1.0, 2.22774593441806)  # error of 1.81E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.39E-01,1.00E+00)
    genlog_norm_parameter = (1.1060801945544534, 0.28774995092742695, 0.028317304354023017, 1.8364810017645563)  # error of 2.59E-08 with sample range (0.00E+00,5.00E+00) resulting in fit range (9.79E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  SlogP_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA7)
    # normalization
    linear_norm_parameter = (0.0036630233160711567, 0.8928265874311769)  # error of 5.13E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (8.93E-01,3.81E+00)
    min_max_norm_parameter = (5e-324, 6.023897866072181)  # error of 4.56E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4655912538215367, 0.3384327438176033)  # error of 1.07E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (4.61E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.7059211488293957, 4.566242269850751, 0.35450788290713264)  # error of 1.03E-02 with sample range (0.00E+00,7.96E+02) resulting in fit range (3.83E-02,1.00E+00)
    genlog_norm_parameter = (0.3105599135425584, -7.8817149140594465, 0.0042533429031192385, 0.00040502865079720707)  # error of 9.68E-03 with sample range (0.00E+00,7.96E+02) resulting in fit range (4.03E-01,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_thiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_thiocyan)
    # normalization
    linear_norm_parameter = (0.11905202341120957, 0.45235119315332584)  # error of 3.36E-01 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.52E-01,1.17E+00)
    min_max_norm_parameter = (9.608678732145031e-12, 1.000064998295486)  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6659376309621101, 28.86030957889578)  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.50E-09,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4999998913713622, 4.196819719282997, 12.934269311991448)  # error of 1.07E-03 with sample range (0.00E+00,6.00E+00) resulting in fit range (1.55E-03,1.00E+00)
    genlog_norm_parameter = (11.51809750434588, 0.08697144539747265, -0.3591860288535241, -0.14970429382995373)  # error of 5.00E-06 with sample range (0.00E+00,6.00E+00) resulting in fit range (8.28E-12,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_term_acetylene_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_term_acetylene)
    # normalization
    linear_norm_parameter = (0.4985401313881752, 0.4985401313881751)  # error of 2.22E-16 with sample range (0.00E+00,6.00E+00) resulting in fit range (4.99E-01,3.49E+00)
    min_max_norm_parameter = (1.033924484011466e-08, 1.0029282870220821)  # error of 7.59E-05 with sample range (0.00E+00,6.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.4152302622952124, 2.415230200746865)  # error of 1.69E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.68E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.4152302622952124, 1.0, 2.415230200746865)  # error of 1.69E-09 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.68E-01,1.00E+00)
    genlog_norm_parameter = (1.108049480477119, 0.2812914897623567, 0.011983232018832704, 1.8431803563406333)  # error of 3.11E-15 with sample range (0.00E+00,6.00E+00) resulting in fit range (9.91E-01,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(FractionCSP3)
    # normalization
    linear_norm_parameter = (1.1210114366616568, 0.024238385885020208)  # error of 6.63E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.42E-02,1.15E+00)
    min_max_norm_parameter = (0.030077967569291073, 0.7816451738469055)  # error of 3.99E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.4036392556168974, 6.810271411900117)  # error of 2.19E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (6.01E-02,9.83E-01)
    dual_sigmoidal_norm_parameter = (0.37933375807687925, 8.369078319667627, 5.735309352343304)  # error of 1.04E-02 with sample range (0.00E+00,1.00E+00) resulting in fit range (4.01E-02,9.72E-01)
    genlog_norm_parameter = (5.121341713037929, 0.3541805503951657, 0.1654208707795316, 0.18747997008450548)  # error of 4.98E-03 with sample range (0.00E+00,1.00E+00) resulting in fit range (2.38E-02,9.68E-01)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_ester)
    # normalization
    linear_norm_parameter = (0.020566648999453596, 0.9270090691836141)  # error of 1.36E-02 with sample range (0.00E+00,2.10E+01) resulting in fit range (9.27E-01,1.36E+00)
    min_max_norm_parameter = (5.795842863633832e-09, 1.0721261560339788)  # error of 3.48E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.6092514604520473, 1.6347138056465544)  # error of 1.00E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.30E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.6092514604520473, 1.0, 1.6347138056465544)  # error of 1.00E-03 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.30E-01,1.00E+00)
    genlog_norm_parameter = (1.6078371541949399, 0.36532117342377624, 0.0018136162209850346, 0.009394534550233024)  # error of 9.71E-04 with sample range (0.00E+00,2.10E+01) resulting in fit range (7.07E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumHAcceptors_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumHAcceptors)
    # normalization
    linear_norm_parameter = (0.018653528399554697, 0.5574729713710027)  # error of 1.93E-01 with sample range (0.00E+00,1.71E+02) resulting in fit range (5.57E-01,3.75E+00)
    min_max_norm_parameter = (1.2630439561525542, 7.515443068519551)  # error of 1.49E-02 with sample range (0.00E+00,1.71E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (4.367102956502307, 0.8007786530186061)  # error of 9.05E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (2.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (4.18861863906966, 0.9639214876904849, 0.6754396233905521)  # error of 5.33E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (1.73E-02,1.00E+00)
    genlog_norm_parameter = (0.5934193512469899, 0.22514360787120383, 1.466124180627735, 0.18363318748043794)  # error of 3.35E-03 with sample range (0.00E+00,1.71E+02) resulting in fit range (4.70E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  fr_unbrch_alkane_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_unbrch_alkane)
    # normalization
    linear_norm_parameter = (0.03889834636895251, 0.7073099439562576)  # error of 2.80E-01 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.07E-01,1.17E+00)
    min_max_norm_parameter = (7.511636441371496e-09, 1.000225030377411)  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.6677048142926155, 25.276494721014547)  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (4.68E-08,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.4999997545511281, 6.600348494015017, 14.304121514205056)  # error of 3.23E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (7.83E-04,1.00E+00)
    genlog_norm_parameter = (11.130654497305729, 0.08127151505956189, 0.007557498776757308, 0.0012163465757203487)  # error of 4.28E-05 with sample range (0.00E+00,1.20E+01) resulting in fit range (2.48E-07,1.00E+00)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_sulfone_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_sulfone)
    # normalization
    linear_norm_parameter = (0.49420552150306496, 0.49420552150306485)  # error of 4.44E-16 with sample range (0.00E+00,3.00E+00) resulting in fit range (4.94E-01,1.98E+00)
    min_max_norm_parameter = (7.860937357634984e-09, 1.0117248355173414)  # error of 1.67E-04 with sample range (0.00E+00,3.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.1085649360924517, 2.108564882621699)  # error of 9.49E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.12E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.1085649360924517, 1.0, 2.108564882621699)  # error of 9.49E-11 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.12E-01,1.00E+00)
    genlog_norm_parameter = (1.1037625598041962, 0.2953092425545398, 0.04689730611865527, 1.8286860729451104)  # error of 1.33E-08 with sample range (0.00E+00,3.00E+00) resulting in fit range (9.66E-01,9.99E-01)
    preferred_normalization = 'min_max'
    # functions
    
    

class  HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(HallKierAlpha)
    # normalization
    linear_norm_parameter = (0.10941027797520916, 0.8206630827667591)  # error of 1.86E-01 with sample range (-7.86E+01,6.10E+01) resulting in fit range (-7.78E+00,7.50E+00)
    min_max_norm_parameter = (-4.025047303628864, -0.27476699527518833)  # error of 2.37E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.136641082271498, 1.3958219191517867)  # error of 1.43E-02 with sample range (-7.86E+01,6.10E+01) resulting in fit range (4.45E-47,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.0451007845368863, 1.1887380822440157, 1.5856961746935594)  # error of 8.43E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (3.00E-40,1.00E+00)
    genlog_norm_parameter = (1.7618968864206013, -2.0459113897478645, 2.4289416394683374, 1.8402620037112714)  # error of 7.26E-03 with sample range (-7.86E+01,6.10E+01) resulting in fit range (9.11E-33,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SMR_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SMR_VSA7)
    # normalization
    linear_norm_parameter = (0.004721924676151512, 0.3518106416338208)  # error of 1.44E-01 with sample range (0.00E+00,2.00E+03) resulting in fit range (3.52E-01,9.80E+00)
    min_max_norm_parameter = (0.027182952918228913, 90.6963985858831)  # error of 3.59E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (45.068602585969984, 0.05384149034564499)  # error of 1.60E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (8.12E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (43.693612281700155, 0.06291259209686757, 0.04995417494201176)  # error of 1.34E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (6.02E-02,1.00E+00)
    genlog_norm_parameter = (0.045747875021952156, -9.19552940034082, 3.664875844611263, 0.3961271397231768)  # error of 1.31E-02 with sample range (0.00E+00,2.00E+03) resulting in fit range (4.53E-02,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BCUT2D_LOGPLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_LOGPLOW)
    # normalization
    linear_norm_parameter = (1.7051602905317909, 4.918269713274668)  # error of 6.24E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (-9.14E-01,2.89E+00)
    min_max_norm_parameter = (-2.845292207717101, -2.3529603749416776)  # error of 3.19E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-2.603156460495485, 9.951733084541459)  # error of 2.24E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (2.94E-04,1.00E+00)
    dual_sigmoidal_norm_parameter = (-2.6168884537842767, 11.99429838644893, 8.316962700747439)  # error of 1.22E-02 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (6.54E-05,1.00E+00)
    genlog_norm_parameter = (7.177091347052052, -2.084683810964218, 0.00137344728212155, 0.08451640469156121)  # error of 7.48E-03 with sample range (-3.42E+00,-1.19E+00) resulting in fit range (2.29E-16,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  BCUT2D_LOGPHI_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(BCUT2D_LOGPHI)
    # normalization
    linear_norm_parameter = (0.9221221942827517, 0.3462995547380183)  # error of 1.18E-01 with sample range (-1.76E+00,2.74E+00) resulting in fit range (-1.27E+00,2.88E+00)
    min_max_norm_parameter = (-0.2074553870185299, 0.45765097151660844)  # error of 5.19E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.1195132064019319, 7.293530987809198)  # error of 3.57E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (1.14E-06,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.10216399172035487, 8.565792715629511, 6.032209966300347)  # error of 3.18E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (1.22E-07,1.00E+00)
    genlog_norm_parameter = (5.109888583609547, -0.016450255662772126, 0.057013183310217694, 0.0428114239970375)  # error of 2.84E-02 with sample range (-1.76E+00,2.74E+00) resulting in fit range (6.89E-62,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  VSA_EState10_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(VSA_EState10)
    # normalization
    linear_norm_parameter = (0.05651052449377392, 0.4477716191034624)  # error of 2.35E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (-1.74E+00,7.28E+00)
    min_max_norm_parameter = (-1.6960388214922417, 3.3674968629997517)  # error of 1.76E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (0.09768644304876672, 6.040221189979557)  # error of 1.42E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (1.42E-102,1.00E+00)
    dual_sigmoidal_norm_parameter = (0.005214427377489521, 150.37215166432333, 0.3998539942243761)  # error of 7.98E-02 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    genlog_norm_parameter = (4.164014290286187, -1.264438614178022, 0.027160888482521385, 0.00013893809910547775)  # error of 1.38E-01 with sample range (-3.87E+01,1.21E+02) resulting in fit range (0.00E+00,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_para_hydroxylation_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_para_hydroxylation)
    # normalization
    linear_norm_parameter = (0.029445349918507113, 0.8952769250767446)  # error of 1.78E-02 with sample range (0.00E+00,1.20E+01) resulting in fit range (8.95E-01,1.25E+00)
    min_max_norm_parameter = (8.207457396571228e-09, 1.1044006396378632)  # error of 6.70E-03 with sample range (0.00E+00,1.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.5046217800819417, 1.5020028251556437)  # error of 4.95E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.81E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.5046217800819417, 1.0, 1.5020028251556437)  # error of 4.95E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.81E-01,1.00E+00)
    genlog_norm_parameter = (1.4666659998544667, 0.20562880715116938, 0.003383564502110456, 0.010625020417625317)  # error of 4.48E-04 with sample range (0.00E+00,1.20E+01) resulting in fit range (6.51E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumSaturatedCarbocycles)
    # normalization
    linear_norm_parameter = (0.020972112507738894, 0.9098571128596467)  # error of 1.95E-02 with sample range (0.00E+00,1.80E+01) resulting in fit range (9.10E-01,1.29E+00)
    min_max_norm_parameter = (8.1643240157463e-09, 1.1069248512275087)  # error of 6.94E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.6176260310959208, 1.385111849563034)  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.6176260310959208, 1.0, 1.385111849563034)  # error of 2.60E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (7.02E-01,1.00E+00)
    genlog_norm_parameter = (4.069531491711097, -0.26470302233247034, 0.01816997314515009, 0.0010314014664690044)  # error of 6.57E-03 with sample range (0.00E+00,1.80E+01) resulting in fit range (2.53E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  SlogP_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(SlogP_VSA2)
    # normalization
    linear_norm_parameter = (0.007179097183038996, 0.31036360002709074)  # error of 1.50E-01 with sample range (0.00E+00,1.09E+03) resulting in fit range (3.10E-01,8.13E+00)
    min_max_norm_parameter = (2.8149378256505875, 62.887170739624814)  # error of 3.91E-02 with sample range (0.00E+00,1.09E+03) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (32.49634012337661, 0.07987637026985167)  # error of 1.84E-02 with sample range (0.00E+00,1.09E+03) resulting in fit range (6.94E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (31.06377443005473, 0.10301921701855828, 0.06990432005737299)  # error of 7.32E-03 with sample range (0.00E+00,1.09E+03) resulting in fit range (3.92E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_COO)
    # normalization
    linear_norm_parameter = (0.025487706106450125, 0.9292380352434951)  # error of 1.01E-02 with sample range (0.00E+00,3.20E+01) resulting in fit range (9.29E-01,1.74E+00)
    min_max_norm_parameter = (6.888269928388699e-09, 1.0553034775863013)  # error of 1.49E-03 with sample range (0.00E+00,3.20E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-0.27734172080976294, 2.2665201420240852)  # error of 4.93E-04 with sample range (0.00E+00,3.20E+01) resulting in fit range (6.52E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-0.27734172080976294, 1.0, 2.2665201420240852)  # error of 4.93E-04 with sample range (0.00E+00,3.20E+01) resulting in fit range (6.52E-01,1.00E+00)
    genlog_norm_parameter = (1.2369826026454291, 0.8812100223159962, -1.1093942149999834, -58.799677765337265)  # error of 2.00E-13 with sample range (0.00E+00,3.20E+01) resulting in fit range (NAN,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(NumHeteroatoms)
    # normalization
    linear_norm_parameter = (0.016077495185775725, 0.5033528543882042)  # error of 2.02E-01 with sample range (0.00E+00,2.15E+02) resulting in fit range (5.03E-01,3.96E+00)
    min_max_norm_parameter = (1.9439247544979925, 10.63914346171618)  # error of 1.53E-02 with sample range (0.00E+00,2.15E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (6.309633287283373, 0.5961308359128846)  # error of 8.73E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (2.27E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (6.045320930283959, 0.7179354032850181, 0.5008061943406604)  # error of 5.10E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (1.29E-02,1.00E+00)
    genlog_norm_parameter = (0.45068040681824034, 1.1172662980244155, 1.7144405539074021, 0.23628999990547292)  # error of 3.24E-03 with sample range (0.00E+00,2.15E+02) resulting in fit range (3.38E-03,1.00E+00)
    preferred_normalization = 'genlog'
    # functions
    
    

class  Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi3n)
    # normalization
    linear_norm_parameter = (0.16045617502782894, 0.06275167455712582)  # error of 1.47E-01 with sample range (0.00E+00,6.23E+01) resulting in fit range (6.28E-02,1.01E+01)
    min_max_norm_parameter = (0.8698290958911558, 4.249166143207709)  # error of 3.75E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (2.5307853903255304, 1.4301006733612187)  # error of 2.83E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (2.61E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (2.390012863113382, 1.9077779294117436, 1.1008446640044367)  # error of 1.03E-02 with sample range (0.00E+00,6.23E+01) resulting in fit range (1.04E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  PEOE_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(PEOE_VSA4)
    # normalization
    linear_norm_parameter = (0.005054555707632469, 0.8464248056257232)  # error of 4.88E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (8.46E-01,1.98E+00)
    min_max_norm_parameter = (4.919925289519856e-27, 6.338586899034091)  # error of 6.33E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.5531847169925812, 0.19004989659369026)  # error of 1.60E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.73E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.5531846677647467, 0.910463122499567, 0.1900498974926005)  # error of 1.60E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.73E-01,1.00E+00)
    genlog_norm_parameter = (0.2001724916564023, 1.5197898172299915, 1.0901780403502739, 1.7121894041115937)  # error of 1.59E-02 with sample range (0.00E+00,2.24E+02) resulting in fit range (5.89E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

class  Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.float32
    featurize=staticmethod(Chi4n)
    # normalization
    linear_norm_parameter = (0.2235474556767324, 0.12057837229443968)  # error of 1.52E-01 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.21E-01,8.63E+00)
    min_max_norm_parameter = (0.40708839430881494, 2.71836618230583)  # error of 3.94E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (1.5414132407381294, 2.0927265327645506)  # error of 3.03E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (3.82E-02,1.00E+00)
    dual_sigmoidal_norm_parameter = (1.441026134320466, 2.8235640784391123, 1.5793253223087178)  # error of 1.12E-02 with sample range (0.00E+00,3.81E+01) resulting in fit range (1.68E-02,1.00E+00)
    preferred_normalization = 'dual_sig'
    # functions
    
    

class  fr_dihydropyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_dihydropyridine)
    # normalization
    linear_norm_parameter = (0.4987476127148559, 0.4987476127148558)  # error of 3.33E-16 with sample range (0.00E+00,2.00E+00) resulting in fit range (4.99E-01,1.50E+00)
    min_max_norm_parameter = (5.0125553583537566e-11, 1.002511071570499)  # error of 4.22E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.4468440944675611, 2.4468441050172953)  # error of 6.43E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.72E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.4468440944675611, 1.0, 2.4468441050172953)  # error of 6.43E-09 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.72E-01,1.00E+00)
    genlog_norm_parameter = (1.1082502806056016, 0.2806306676780934, 0.010286911137338478, 1.8438681116157882)  # error of 2.78E-15 with sample range (0.00E+00,2.00E+00) resulting in fit range (9.92E-01,9.99E-01)
    preferred_normalization = 'unity'
    # functions
    
    

class  fr_priamide_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype=np.int32
    featurize=staticmethod(fr_priamide)
    # normalization
    linear_norm_parameter = (0.49340809327160573, 0.4934080932716056)  # error of 3.33E-16 with sample range (0.00E+00,9.00E+00) resulting in fit range (4.93E-01,4.93E+00)
    min_max_norm_parameter = (3.082917443384843e-09, 1.0133599484847597)  # error of 3.39E-04 with sample range (0.00E+00,9.00E+00) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (-1.0773767211105616, 2.0773766830617735)  # error of 1.77E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.04E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (-1.0773767211105616, 1.0, 2.0773766830617735)  # error of 1.77E-11 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.04E-01,1.00E+00)
    genlog_norm_parameter = (1.1029569625796818, 0.29792952181563015, 0.05320949757471211, 1.8259942909746336)  # error of 1.04E-08 with sample range (0.00E+00,9.00E+00) resulting in fit range (9.62E-01,1.00E+00)
    preferred_normalization = 'min_max'
    # functions
    
    

molecule_fr_nitro=fr_nitro_Featurizer()
molecule_PEOE_VSA8=PEOE_VSA8_Featurizer()
molecule_fr_Imine=fr_Imine_Featurizer()
molecule_PMI3=PMI3_Featurizer()
molecule_NumRotatableBonds=NumRotatableBonds_Featurizer()
molecule_Chi0v=Chi0v_Featurizer()
molecule_fr_quatN=fr_quatN_Featurizer()
molecule_fr_Ndealkylation2=fr_Ndealkylation2_Featurizer()
molecule_EState_VSA8=EState_VSA8_Featurizer()
molecule_fr_phos_ester=fr_phos_ester_Featurizer()
molecule_RingCount=RingCount_Featurizer()
molecule_PEOE_VSA5=PEOE_VSA5_Featurizer()
molecule_EState_VSA11=EState_VSA11_Featurizer()
molecule_VSA_EState1=VSA_EState1_Featurizer()
molecule_fr_diazo=fr_diazo_Featurizer()
molecule_Kappa1=Kappa1_Featurizer()
molecule_Chi4v=Chi4v_Featurizer()
molecule_Chi1=Chi1_Featurizer()
molecule_fr_ketone=fr_ketone_Featurizer()
molecule_fr_lactone=fr_lactone_Featurizer()
molecule_fr_thiazole=fr_thiazole_Featurizer()
molecule_MinAbsEStateIndex=MinAbsEStateIndex_Featurizer()
molecule_fr_COO2=fr_COO2_Featurizer()
molecule_SlogP_VSA9=SlogP_VSA9_Featurizer()
molecule_VSA_EState9=VSA_EState9_Featurizer()
molecule_fr_isothiocyan=fr_isothiocyan_Featurizer()
molecule_Chi2n=Chi2n_Featurizer()
molecule_fr_Al_OH_noTert=fr_Al_OH_noTert_Featurizer()
molecule_FractionCSP3=FractionCSP3_Featurizer()
molecule_NumAliphaticHeterocycles=NumAliphaticHeterocycles_Featurizer()
molecule_Chi0=Chi0_Featurizer()
molecule_Kappa1=Kappa1_Featurizer()
molecule_MaxAbsEStateIndex=MaxAbsEStateIndex_Featurizer()
molecule_PEOE_VSA14=PEOE_VSA14_Featurizer()
molecule_PEOE_VSA7=PEOE_VSA7_Featurizer()
molecule_NPR1=NPR1_Featurizer()
molecule_Chi3v=Chi3v_Featurizer()
molecule_Chi4n=Chi4n_Featurizer()
molecule_NumSaturatedHeterocycles=NumSaturatedHeterocycles_Featurizer()
molecule_Chi2v=Chi2v_Featurizer()
molecule_Chi3n=Chi3n_Featurizer()
molecule_fr_amidine=fr_amidine_Featurizer()
molecule_Asphericity=Asphericity_Featurizer()
molecule_Kappa2=Kappa2_Featurizer()
molecule_BertzCT=BertzCT_Featurizer()
molecule_MaxPartialCharge=MaxPartialCharge_Featurizer()
molecule_SlogP_VSA12=SlogP_VSA12_Featurizer()
molecule_VSA_EState2=VSA_EState2_Featurizer()
molecule_fr_HOCCN=fr_HOCCN_Featurizer()
molecule_MinPartialCharge=MinPartialCharge_Featurizer()
molecule_Chi1n=Chi1n_Featurizer()
molecule_PEOE_VSA6=PEOE_VSA6_Featurizer()
molecule_EState_VSA10=EState_VSA10_Featurizer()
molecule_EState_VSA7=EState_VSA7_Featurizer()
molecule_fr_pyridine=fr_pyridine_Featurizer()
molecule_BCUT2D_CHGHI=BCUT2D_CHGHI_Featurizer()
molecule_PEOE_VSA2=PEOE_VSA2_Featurizer()
molecule_qed=qed_Featurizer()
molecule_SMR_VSA1=SMR_VSA1_Featurizer()
molecule_fr_lactam=fr_lactam_Featurizer()
molecule_fr_imidazole=fr_imidazole_Featurizer()
molecule_fr_Al_COO=fr_Al_COO_Featurizer()
molecule_Chi0v=Chi0v_Featurizer()
molecule_NumHBD=NumHBD_Featurizer()
molecule_NumAliphaticRings=NumAliphaticRings_Featurizer()
molecule_NumRings=NumRings_Featurizer()
molecule_fr_nitroso=fr_nitroso_Featurizer()
molecule_fr_prisulfonamd=fr_prisulfonamd_Featurizer()
molecule_SMR_VSA2=SMR_VSA2_Featurizer()
molecule_Chi4n=Chi4n_Featurizer()
molecule_InertialShapeFactor=InertialShapeFactor_Featurizer()
molecule_NumAmideBonds=NumAmideBonds_Featurizer()
molecule_NumHDonors=NumHDonors_Featurizer()
molecule_EState_VSA5=EState_VSA5_Featurizer()
molecule_fr_methoxy=fr_methoxy_Featurizer()
molecule_NumHBA=NumHBA_Featurizer()
molecule_NPR1=NPR1_Featurizer()
molecule_MaxAbsPartialCharge=MaxAbsPartialCharge_Featurizer()
molecule_Chi2n=Chi2n_Featurizer()
molecule_Chi1v=Chi1v_Featurizer()
molecule_fr_sulfonamd=fr_sulfonamd_Featurizer()
molecule_PEOE_VSA9=PEOE_VSA9_Featurizer()
molecule_fr_Ndealkylation1=fr_Ndealkylation1_Featurizer()
molecule_LabuteASA=LabuteASA_Featurizer()
molecule_PMI1=PMI1_Featurizer()
molecule_LabuteASA=LabuteASA_Featurizer()
molecule_fr_imide=fr_imide_Featurizer()
molecule_NumAromaticRings=NumAromaticRings_Featurizer()
molecule_Ipc=Ipc_Featurizer()
molecule_ExactMolWt=ExactMolWt_Featurizer()
molecule_SMR_VSA3=SMR_VSA3_Featurizer()
molecule_fr_C_S=fr_C_S_Featurizer()
molecule_NumSaturatedRings=NumSaturatedRings_Featurizer()
molecule_NumAliphaticRings=NumAliphaticRings_Featurizer()
molecule_fr_NH0=fr_NH0_Featurizer()
molecule_fr_alkyl_halide=fr_alkyl_halide_Featurizer()
molecule_fr_hdrzone=fr_hdrzone_Featurizer()
molecule_Chi0n=Chi0n_Featurizer()
molecule_NumAromaticCarbocycles=NumAromaticCarbocycles_Featurizer()
molecule_BCUT2D_MWLOW=BCUT2D_MWLOW_Featurizer()
molecule_PEOE_VSA10=PEOE_VSA10_Featurizer()
molecule_PEOE_VSA1=PEOE_VSA1_Featurizer()
molecule_GetSSSR=GetSSSR_Featurizer()
molecule_Chi4v=Chi4v_Featurizer()
molecule_EState_VSA3=EState_VSA3_Featurizer()
molecule_Asphericity=Asphericity_Featurizer()
molecule_Phi=Phi_Featurizer()
molecule_VSA_EState4=VSA_EState4_Featurizer()
molecule_fr_ketone_Topliss=fr_ketone_Topliss_Featurizer()
molecule_NumSaturatedHeterocycles=NumSaturatedHeterocycles_Featurizer()
molecule_HallKierAlpha=HallKierAlpha_Featurizer()
molecule_GetFormalCharge=GetFormalCharge_Featurizer()
molecule_VSA_EState5=VSA_EState5_Featurizer()
molecule_fr_guanido=fr_guanido_Featurizer()
molecule_NumRadicalElectrons=NumRadicalElectrons_Featurizer()
molecule_SMR_VSA6=SMR_VSA6_Featurizer()
molecule_BalabanJ=BalabanJ_Featurizer()
molecule_TPSA=TPSA_Featurizer()
molecule_Chi2n=Chi2n_Featurizer()
molecule_fr_nitrile=fr_nitrile_Featurizer()
molecule_NumAliphaticCarbocycles=NumAliphaticCarbocycles_Featurizer()
molecule_EState_VSA6=EState_VSA6_Featurizer()
molecule_EState_VSA1=EState_VSA1_Featurizer()
molecule_NumAromaticCarbocycles=NumAromaticCarbocycles_Featurizer()
molecule_NumAromaticHeterocycles=NumAromaticHeterocycles_Featurizer()
molecule_SlogP_VSA6=SlogP_VSA6_Featurizer()
molecule_MolWt=MolWt_Featurizer()
molecule_ExactMolWt=ExactMolWt_Featurizer()
molecule_PMI1=PMI1_Featurizer()
molecule_RadiusOfGyration=RadiusOfGyration_Featurizer()
molecule_BCUT2D_MRLOW=BCUT2D_MRLOW_Featurizer()
molecule_VSA_EState3=VSA_EState3_Featurizer()
molecule_fr_ether=fr_ether_Featurizer()
molecule_NumValenceElectrons=NumValenceElectrons_Featurizer()
molecule_fr_N_O=fr_N_O_Featurizer()
molecule_NPR2=NPR2_Featurizer()
molecule_Chi0n=Chi0n_Featurizer()
molecule_MolMR=MolMR_Featurizer()
molecule_Eccentricity=Eccentricity_Featurizer()
molecule_FpDensityMorgan3=FpDensityMorgan3_Featurizer()
molecule_NHOHCount=NHOHCount_Featurizer()
molecule_PEOE_VSA12=PEOE_VSA12_Featurizer()
molecule_fr_azide=fr_azide_Featurizer()
molecule_Chi3n=Chi3n_Featurizer()
molecule_Eccentricity=Eccentricity_Featurizer()
molecule_NOCount=NOCount_Featurizer()
molecule_PEOE_VSA3=PEOE_VSA3_Featurizer()
molecule_fr_NH1=fr_NH1_Featurizer()
molecule_PEOE_VSA11=PEOE_VSA11_Featurizer()
molecule_NumHeteroatoms=NumHeteroatoms_Featurizer()
molecule_SlogP_VSA10=SlogP_VSA10_Featurizer()
molecule_Chi0=Chi0_Featurizer()
molecule_fr_morpholine=fr_morpholine_Featurizer()
molecule_EState_VSA9=EState_VSA9_Featurizer()
molecule_PBF=PBF_Featurizer()
molecule_MinAbsPartialCharge=MinAbsPartialCharge_Featurizer()
molecule_SlogP_VSA11=SlogP_VSA11_Featurizer()
molecule_fr_aniline=fr_aniline_Featurizer()
molecule_BCUT2D_MWHI=BCUT2D_MWHI_Featurizer()
molecule_Chi4v=Chi4v_Featurizer()
molecule_fr_piperdine=fr_piperdine_Featurizer()
molecule_Chi3v=Chi3v_Featurizer()
molecule_fr_Nhpyrrole=fr_Nhpyrrole_Featurizer()
molecule_Chi1=Chi1_Featurizer()
molecule_FpDensityMorgan1=FpDensityMorgan1_Featurizer()
molecule_fr_epoxide=fr_epoxide_Featurizer()
molecule_SlogP_VSA5=SlogP_VSA5_Featurizer()
molecule_fr_hdrzine=fr_hdrzine_Featurizer()
molecule_fr_Ar_N=fr_Ar_N_Featurizer()
molecule_SMR_VSA5=SMR_VSA5_Featurizer()
molecule_SlogP_VSA3=SlogP_VSA3_Featurizer()
molecule_BertzCT=BertzCT_Featurizer()
molecule_fr_C_O_noCOO=fr_C_O_noCOO_Featurizer()
molecule_NumSaturatedRings=NumSaturatedRings_Featurizer()
molecule_SMR_VSA10=SMR_VSA10_Featurizer()
molecule_NumSpiroAtoms=NumSpiroAtoms_Featurizer()
molecule_Chi3v=Chi3v_Featurizer()
molecule_NumSaturatedCarbocycles=NumSaturatedCarbocycles_Featurizer()
molecule_Chi0n=Chi0n_Featurizer()
molecule_fr_oxime=fr_oxime_Featurizer()
molecule_Chi2v=Chi2v_Featurizer()
molecule_BCUT2D_MRHI=BCUT2D_MRHI_Featurizer()
molecule_SMR_VSA4=SMR_VSA4_Featurizer()
molecule_PMI2=PMI2_Featurizer()
molecule_fr_Ar_COO=fr_Ar_COO_Featurizer()
molecule_HallKierAlpha=HallKierAlpha_Featurizer()
molecule_fr_nitro_arom=fr_nitro_arom_Featurizer()
molecule_fr_SH=fr_SH_Featurizer()
molecule_fr_piperzine=fr_piperzine_Featurizer()
molecule_Chi1n=Chi1n_Featurizer()
molecule_NumAromaticRings=NumAromaticRings_Featurizer()
molecule_EState_VSA2=EState_VSA2_Featurizer()
molecule_Chi2v=Chi2v_Featurizer()
molecule_fr_urea=fr_urea_Featurizer()
molecule_fr_benzene=fr_benzene_Featurizer()
molecule_fr_bicyclic=fr_bicyclic_Featurizer()
molecule_fr_ArN=fr_ArN_Featurizer()
molecule_Kappa2=Kappa2_Featurizer()
molecule_fr_halogen=fr_halogen_Featurizer()
molecule_VSA_EState6=VSA_EState6_Featurizer()
molecule_fr_Ar_OH=fr_Ar_OH_Featurizer()
molecule_PMI3=PMI3_Featurizer()
molecule_VSA_EState8=VSA_EState8_Featurizer()
molecule_BCUT2D_CHGLO=BCUT2D_CHGLO_Featurizer()
molecule_fr_phos_acid=fr_phos_acid_Featurizer()
molecule_SpherocityIndex=SpherocityIndex_Featurizer()
molecule_NumAliphaticHeterocycles=NumAliphaticHeterocycles_Featurizer()
molecule_fr_NH2=fr_NH2_Featurizer()
molecule_fr_furan=fr_furan_Featurizer()
molecule_fr_nitro_arom_nonortho=fr_nitro_arom_nonortho_Featurizer()
molecule_SlogP_VSA4=SlogP_VSA4_Featurizer()
molecule_NumRotatableBonds=NumRotatableBonds_Featurizer()
molecule_Chi1n=Chi1n_Featurizer()
molecule_fr_thiophene=fr_thiophene_Featurizer()
molecule_PEOE_VSA13=PEOE_VSA13_Featurizer()
molecule_Kappa2=Kappa2_Featurizer()
molecule_HeavyAtomCount=HeavyAtomCount_Featurizer()
molecule_fr_amide=fr_amide_Featurizer()
molecule_fr_barbitur=fr_barbitur_Featurizer()
molecule_SlogP_VSA8=SlogP_VSA8_Featurizer()
molecule_RadiusOfGyration=RadiusOfGyration_Featurizer()
molecule_Kappa3=Kappa3_Featurizer()
molecule_Chi0v=Chi0v_Featurizer()
molecule_NumBridgeheadAtoms=NumBridgeheadAtoms_Featurizer()
molecule_NumAromaticHeterocycles=NumAromaticHeterocycles_Featurizer()
molecule_Ipc=Ipc_Featurizer()
molecule_fr_C_O=fr_C_O_Featurizer()
molecule_fr_aldehyde=fr_aldehyde_Featurizer()
molecule_Kappa3=Kappa3_Featurizer()
molecule_EState_VSA4=EState_VSA4_Featurizer()
molecule_NumAliphaticCarbocycles=NumAliphaticCarbocycles_Featurizer()
molecule_fr_tetrazole=fr_tetrazole_Featurizer()
molecule_fr_oxazole=fr_oxazole_Featurizer()
molecule_fr_benzodiazepine=fr_benzodiazepine_Featurizer()
molecule_NPR2=NPR2_Featurizer()
molecule_FpDensityMorgan2=FpDensityMorgan2_Featurizer()
molecule_NumHeterocycles=NumHeterocycles_Featurizer()
molecule_NumLipinskiHBD=NumLipinskiHBD_Featurizer()
molecule_Chi1v=Chi1v_Featurizer()
molecule_fr_Ar_NH=fr_Ar_NH_Featurizer()
molecule_fr_aryl_methyl=fr_aryl_methyl_Featurizer()
molecule_Chi1v=Chi1v_Featurizer()
molecule_fr_Al_OH=fr_Al_OH_Featurizer()
molecule_SMR_VSA9=SMR_VSA9_Featurizer()
molecule_HeavyAtomMolWt=HeavyAtomMolWt_Featurizer()
molecule_SpherocityIndex=SpherocityIndex_Featurizer()
molecule_fr_allylic_oxid=fr_allylic_oxid_Featurizer()
molecule_VSA_EState7=VSA_EState7_Featurizer()
molecule_SMR_VSA8=SMR_VSA8_Featurizer()
molecule_MolLogP=MolLogP_Featurizer()
molecule_fr_phenol=fr_phenol_Featurizer()
molecule_SlogP_VSA1=SlogP_VSA1_Featurizer()
molecule_fr_azo=fr_azo_Featurizer()
molecule_BalabanJ=BalabanJ_Featurizer()
molecule_PMI2=PMI2_Featurizer()
molecule_fr_isocyan=fr_isocyan_Featurizer()
molecule_TPSA=TPSA_Featurizer()
molecule_InertialShapeFactor=InertialShapeFactor_Featurizer()
molecule_Kappa1=Kappa1_Featurizer()
molecule_MinEStateIndex=MinEStateIndex_Featurizer()
molecule_MaxEStateIndex=MaxEStateIndex_Featurizer()
molecule_fr_phenol_noOrthoHbond=fr_phenol_noOrthoHbond_Featurizer()
molecule_NumLipinskiHBA=NumLipinskiHBA_Featurizer()
molecule_fr_sulfide=fr_sulfide_Featurizer()
molecule_Kappa3=Kappa3_Featurizer()
molecule_fr_alkyl_carbamate=fr_alkyl_carbamate_Featurizer()
molecule_SlogP_VSA7=SlogP_VSA7_Featurizer()
molecule_fr_thiocyan=fr_thiocyan_Featurizer()
molecule_fr_term_acetylene=fr_term_acetylene_Featurizer()
molecule_FractionCSP3=FractionCSP3_Featurizer()
molecule_fr_ester=fr_ester_Featurizer()
molecule_NumHAcceptors=NumHAcceptors_Featurizer()
molecule_fr_unbrch_alkane=fr_unbrch_alkane_Featurizer()
molecule_fr_sulfone=fr_sulfone_Featurizer()
molecule_HallKierAlpha=HallKierAlpha_Featurizer()
molecule_SMR_VSA7=SMR_VSA7_Featurizer()
molecule_BCUT2D_LOGPLOW=BCUT2D_LOGPLOW_Featurizer()
molecule_BCUT2D_LOGPHI=BCUT2D_LOGPHI_Featurizer()
molecule_VSA_EState10=VSA_EState10_Featurizer()
molecule_fr_para_hydroxylation=fr_para_hydroxylation_Featurizer()
molecule_NumSaturatedCarbocycles=NumSaturatedCarbocycles_Featurizer()
molecule_SlogP_VSA2=SlogP_VSA2_Featurizer()
molecule_fr_COO=fr_COO_Featurizer()
molecule_NumHeteroatoms=NumHeteroatoms_Featurizer()
molecule_Chi3n=Chi3n_Featurizer()
molecule_PEOE_VSA4=PEOE_VSA4_Featurizer()
molecule_Chi4n=Chi4n_Featurizer()
molecule_fr_dihydropyridine=fr_dihydropyridine_Featurizer()
molecule_fr_priamide=fr_priamide_Featurizer()

_available_featurizer={
'molecule_fr_nitro':molecule_fr_nitro,
'molecule_PEOE_VSA8':molecule_PEOE_VSA8,
'molecule_fr_Imine':molecule_fr_Imine,
'molecule_PMI3':molecule_PMI3,
'molecule_NumRotatableBonds':molecule_NumRotatableBonds,
'molecule_Chi0v':molecule_Chi0v,
'molecule_fr_quatN':molecule_fr_quatN,
'molecule_fr_Ndealkylation2':molecule_fr_Ndealkylation2,
'molecule_EState_VSA8':molecule_EState_VSA8,
'molecule_fr_phos_ester':molecule_fr_phos_ester,
'molecule_RingCount':molecule_RingCount,
'molecule_PEOE_VSA5':molecule_PEOE_VSA5,
'molecule_EState_VSA11':molecule_EState_VSA11,
'molecule_VSA_EState1':molecule_VSA_EState1,
'molecule_fr_diazo':molecule_fr_diazo,
'molecule_Kappa1':molecule_Kappa1,
'molecule_Chi4v':molecule_Chi4v,
'molecule_Chi1':molecule_Chi1,
'molecule_fr_ketone':molecule_fr_ketone,
'molecule_fr_lactone':molecule_fr_lactone,
'molecule_fr_thiazole':molecule_fr_thiazole,
'molecule_MinAbsEStateIndex':molecule_MinAbsEStateIndex,
'molecule_fr_COO2':molecule_fr_COO2,
'molecule_SlogP_VSA9':molecule_SlogP_VSA9,
'molecule_VSA_EState9':molecule_VSA_EState9,
'molecule_fr_isothiocyan':molecule_fr_isothiocyan,
'molecule_Chi2n':molecule_Chi2n,
'molecule_fr_Al_OH_noTert':molecule_fr_Al_OH_noTert,
'molecule_FractionCSP3':molecule_FractionCSP3,
'molecule_NumAliphaticHeterocycles':molecule_NumAliphaticHeterocycles,
'molecule_Chi0':molecule_Chi0,
'molecule_Kappa1':molecule_Kappa1,
'molecule_MaxAbsEStateIndex':molecule_MaxAbsEStateIndex,
'molecule_PEOE_VSA14':molecule_PEOE_VSA14,
'molecule_PEOE_VSA7':molecule_PEOE_VSA7,
'molecule_NPR1':molecule_NPR1,
'molecule_Chi3v':molecule_Chi3v,
'molecule_Chi4n':molecule_Chi4n,
'molecule_NumSaturatedHeterocycles':molecule_NumSaturatedHeterocycles,
'molecule_Chi2v':molecule_Chi2v,
'molecule_Chi3n':molecule_Chi3n,
'molecule_fr_amidine':molecule_fr_amidine,
'molecule_Asphericity':molecule_Asphericity,
'molecule_Kappa2':molecule_Kappa2,
'molecule_BertzCT':molecule_BertzCT,
'molecule_MaxPartialCharge':molecule_MaxPartialCharge,
'molecule_SlogP_VSA12':molecule_SlogP_VSA12,
'molecule_VSA_EState2':molecule_VSA_EState2,
'molecule_fr_HOCCN':molecule_fr_HOCCN,
'molecule_MinPartialCharge':molecule_MinPartialCharge,
'molecule_Chi1n':molecule_Chi1n,
'molecule_PEOE_VSA6':molecule_PEOE_VSA6,
'molecule_EState_VSA10':molecule_EState_VSA10,
'molecule_EState_VSA7':molecule_EState_VSA7,
'molecule_fr_pyridine':molecule_fr_pyridine,
'molecule_BCUT2D_CHGHI':molecule_BCUT2D_CHGHI,
'molecule_PEOE_VSA2':molecule_PEOE_VSA2,
'molecule_qed':molecule_qed,
'molecule_SMR_VSA1':molecule_SMR_VSA1,
'molecule_fr_lactam':molecule_fr_lactam,
'molecule_fr_imidazole':molecule_fr_imidazole,
'molecule_fr_Al_COO':molecule_fr_Al_COO,
'molecule_Chi0v':molecule_Chi0v,
'molecule_NumHBD':molecule_NumHBD,
'molecule_NumAliphaticRings':molecule_NumAliphaticRings,
'molecule_NumRings':molecule_NumRings,
'molecule_fr_nitroso':molecule_fr_nitroso,
'molecule_fr_prisulfonamd':molecule_fr_prisulfonamd,
'molecule_SMR_VSA2':molecule_SMR_VSA2,
'molecule_Chi4n':molecule_Chi4n,
'molecule_InertialShapeFactor':molecule_InertialShapeFactor,
'molecule_NumAmideBonds':molecule_NumAmideBonds,
'molecule_NumHDonors':molecule_NumHDonors,
'molecule_EState_VSA5':molecule_EState_VSA5,
'molecule_fr_methoxy':molecule_fr_methoxy,
'molecule_NumHBA':molecule_NumHBA,
'molecule_NPR1':molecule_NPR1,
'molecule_MaxAbsPartialCharge':molecule_MaxAbsPartialCharge,
'molecule_Chi2n':molecule_Chi2n,
'molecule_Chi1v':molecule_Chi1v,
'molecule_fr_sulfonamd':molecule_fr_sulfonamd,
'molecule_PEOE_VSA9':molecule_PEOE_VSA9,
'molecule_fr_Ndealkylation1':molecule_fr_Ndealkylation1,
'molecule_LabuteASA':molecule_LabuteASA,
'molecule_PMI1':molecule_PMI1,
'molecule_LabuteASA':molecule_LabuteASA,
'molecule_fr_imide':molecule_fr_imide,
'molecule_NumAromaticRings':molecule_NumAromaticRings,
'molecule_Ipc':molecule_Ipc,
'molecule_ExactMolWt':molecule_ExactMolWt,
'molecule_SMR_VSA3':molecule_SMR_VSA3,
'molecule_fr_C_S':molecule_fr_C_S,
'molecule_NumSaturatedRings':molecule_NumSaturatedRings,
'molecule_NumAliphaticRings':molecule_NumAliphaticRings,
'molecule_fr_NH0':molecule_fr_NH0,
'molecule_fr_alkyl_halide':molecule_fr_alkyl_halide,
'molecule_fr_hdrzone':molecule_fr_hdrzone,
'molecule_Chi0n':molecule_Chi0n,
'molecule_NumAromaticCarbocycles':molecule_NumAromaticCarbocycles,
'molecule_BCUT2D_MWLOW':molecule_BCUT2D_MWLOW,
'molecule_PEOE_VSA10':molecule_PEOE_VSA10,
'molecule_PEOE_VSA1':molecule_PEOE_VSA1,
'molecule_GetSSSR':molecule_GetSSSR,
'molecule_Chi4v':molecule_Chi4v,
'molecule_EState_VSA3':molecule_EState_VSA3,
'molecule_Asphericity':molecule_Asphericity,
'molecule_Phi':molecule_Phi,
'molecule_VSA_EState4':molecule_VSA_EState4,
'molecule_fr_ketone_Topliss':molecule_fr_ketone_Topliss,
'molecule_NumSaturatedHeterocycles':molecule_NumSaturatedHeterocycles,
'molecule_HallKierAlpha':molecule_HallKierAlpha,
'molecule_GetFormalCharge':molecule_GetFormalCharge,
'molecule_VSA_EState5':molecule_VSA_EState5,
'molecule_fr_guanido':molecule_fr_guanido,
'molecule_NumRadicalElectrons':molecule_NumRadicalElectrons,
'molecule_SMR_VSA6':molecule_SMR_VSA6,
'molecule_BalabanJ':molecule_BalabanJ,
'molecule_TPSA':molecule_TPSA,
'molecule_Chi2n':molecule_Chi2n,
'molecule_fr_nitrile':molecule_fr_nitrile,
'molecule_NumAliphaticCarbocycles':molecule_NumAliphaticCarbocycles,
'molecule_EState_VSA6':molecule_EState_VSA6,
'molecule_EState_VSA1':molecule_EState_VSA1,
'molecule_NumAromaticCarbocycles':molecule_NumAromaticCarbocycles,
'molecule_NumAromaticHeterocycles':molecule_NumAromaticHeterocycles,
'molecule_SlogP_VSA6':molecule_SlogP_VSA6,
'molecule_MolWt':molecule_MolWt,
'molecule_ExactMolWt':molecule_ExactMolWt,
'molecule_PMI1':molecule_PMI1,
'molecule_RadiusOfGyration':molecule_RadiusOfGyration,
'molecule_BCUT2D_MRLOW':molecule_BCUT2D_MRLOW,
'molecule_VSA_EState3':molecule_VSA_EState3,
'molecule_fr_ether':molecule_fr_ether,
'molecule_NumValenceElectrons':molecule_NumValenceElectrons,
'molecule_fr_N_O':molecule_fr_N_O,
'molecule_NPR2':molecule_NPR2,
'molecule_Chi0n':molecule_Chi0n,
'molecule_MolMR':molecule_MolMR,
'molecule_Eccentricity':molecule_Eccentricity,
'molecule_FpDensityMorgan3':molecule_FpDensityMorgan3,
'molecule_NHOHCount':molecule_NHOHCount,
'molecule_PEOE_VSA12':molecule_PEOE_VSA12,
'molecule_fr_azide':molecule_fr_azide,
'molecule_Chi3n':molecule_Chi3n,
'molecule_Eccentricity':molecule_Eccentricity,
'molecule_NOCount':molecule_NOCount,
'molecule_PEOE_VSA3':molecule_PEOE_VSA3,
'molecule_fr_NH1':molecule_fr_NH1,
'molecule_PEOE_VSA11':molecule_PEOE_VSA11,
'molecule_NumHeteroatoms':molecule_NumHeteroatoms,
'molecule_SlogP_VSA10':molecule_SlogP_VSA10,
'molecule_Chi0':molecule_Chi0,
'molecule_fr_morpholine':molecule_fr_morpholine,
'molecule_EState_VSA9':molecule_EState_VSA9,
'molecule_PBF':molecule_PBF,
'molecule_MinAbsPartialCharge':molecule_MinAbsPartialCharge,
'molecule_SlogP_VSA11':molecule_SlogP_VSA11,
'molecule_fr_aniline':molecule_fr_aniline,
'molecule_BCUT2D_MWHI':molecule_BCUT2D_MWHI,
'molecule_Chi4v':molecule_Chi4v,
'molecule_fr_piperdine':molecule_fr_piperdine,
'molecule_Chi3v':molecule_Chi3v,
'molecule_fr_Nhpyrrole':molecule_fr_Nhpyrrole,
'molecule_Chi1':molecule_Chi1,
'molecule_FpDensityMorgan1':molecule_FpDensityMorgan1,
'molecule_fr_epoxide':molecule_fr_epoxide,
'molecule_SlogP_VSA5':molecule_SlogP_VSA5,
'molecule_fr_hdrzine':molecule_fr_hdrzine,
'molecule_fr_Ar_N':molecule_fr_Ar_N,
'molecule_SMR_VSA5':molecule_SMR_VSA5,
'molecule_SlogP_VSA3':molecule_SlogP_VSA3,
'molecule_BertzCT':molecule_BertzCT,
'molecule_fr_C_O_noCOO':molecule_fr_C_O_noCOO,
'molecule_NumSaturatedRings':molecule_NumSaturatedRings,
'molecule_SMR_VSA10':molecule_SMR_VSA10,
'molecule_NumSpiroAtoms':molecule_NumSpiroAtoms,
'molecule_Chi3v':molecule_Chi3v,
'molecule_NumSaturatedCarbocycles':molecule_NumSaturatedCarbocycles,
'molecule_Chi0n':molecule_Chi0n,
'molecule_fr_oxime':molecule_fr_oxime,
'molecule_Chi2v':molecule_Chi2v,
'molecule_BCUT2D_MRHI':molecule_BCUT2D_MRHI,
'molecule_SMR_VSA4':molecule_SMR_VSA4,
'molecule_PMI2':molecule_PMI2,
'molecule_fr_Ar_COO':molecule_fr_Ar_COO,
'molecule_HallKierAlpha':molecule_HallKierAlpha,
'molecule_fr_nitro_arom':molecule_fr_nitro_arom,
'molecule_fr_SH':molecule_fr_SH,
'molecule_fr_piperzine':molecule_fr_piperzine,
'molecule_Chi1n':molecule_Chi1n,
'molecule_NumAromaticRings':molecule_NumAromaticRings,
'molecule_EState_VSA2':molecule_EState_VSA2,
'molecule_Chi2v':molecule_Chi2v,
'molecule_fr_urea':molecule_fr_urea,
'molecule_fr_benzene':molecule_fr_benzene,
'molecule_fr_bicyclic':molecule_fr_bicyclic,
'molecule_fr_ArN':molecule_fr_ArN,
'molecule_Kappa2':molecule_Kappa2,
'molecule_fr_halogen':molecule_fr_halogen,
'molecule_VSA_EState6':molecule_VSA_EState6,
'molecule_fr_Ar_OH':molecule_fr_Ar_OH,
'molecule_PMI3':molecule_PMI3,
'molecule_VSA_EState8':molecule_VSA_EState8,
'molecule_BCUT2D_CHGLO':molecule_BCUT2D_CHGLO,
'molecule_fr_phos_acid':molecule_fr_phos_acid,
'molecule_SpherocityIndex':molecule_SpherocityIndex,
'molecule_NumAliphaticHeterocycles':molecule_NumAliphaticHeterocycles,
'molecule_fr_NH2':molecule_fr_NH2,
'molecule_fr_furan':molecule_fr_furan,
'molecule_fr_nitro_arom_nonortho':molecule_fr_nitro_arom_nonortho,
'molecule_SlogP_VSA4':molecule_SlogP_VSA4,
'molecule_NumRotatableBonds':molecule_NumRotatableBonds,
'molecule_Chi1n':molecule_Chi1n,
'molecule_fr_thiophene':molecule_fr_thiophene,
'molecule_PEOE_VSA13':molecule_PEOE_VSA13,
'molecule_Kappa2':molecule_Kappa2,
'molecule_HeavyAtomCount':molecule_HeavyAtomCount,
'molecule_fr_amide':molecule_fr_amide,
'molecule_fr_barbitur':molecule_fr_barbitur,
'molecule_SlogP_VSA8':molecule_SlogP_VSA8,
'molecule_RadiusOfGyration':molecule_RadiusOfGyration,
'molecule_Kappa3':molecule_Kappa3,
'molecule_Chi0v':molecule_Chi0v,
'molecule_NumBridgeheadAtoms':molecule_NumBridgeheadAtoms,
'molecule_NumAromaticHeterocycles':molecule_NumAromaticHeterocycles,
'molecule_Ipc':molecule_Ipc,
'molecule_fr_C_O':molecule_fr_C_O,
'molecule_fr_aldehyde':molecule_fr_aldehyde,
'molecule_Kappa3':molecule_Kappa3,
'molecule_EState_VSA4':molecule_EState_VSA4,
'molecule_NumAliphaticCarbocycles':molecule_NumAliphaticCarbocycles,
'molecule_fr_tetrazole':molecule_fr_tetrazole,
'molecule_fr_oxazole':molecule_fr_oxazole,
'molecule_fr_benzodiazepine':molecule_fr_benzodiazepine,
'molecule_NPR2':molecule_NPR2,
'molecule_FpDensityMorgan2':molecule_FpDensityMorgan2,
'molecule_NumHeterocycles':molecule_NumHeterocycles,
'molecule_NumLipinskiHBD':molecule_NumLipinskiHBD,
'molecule_Chi1v':molecule_Chi1v,
'molecule_fr_Ar_NH':molecule_fr_Ar_NH,
'molecule_fr_aryl_methyl':molecule_fr_aryl_methyl,
'molecule_Chi1v':molecule_Chi1v,
'molecule_fr_Al_OH':molecule_fr_Al_OH,
'molecule_SMR_VSA9':molecule_SMR_VSA9,
'molecule_HeavyAtomMolWt':molecule_HeavyAtomMolWt,
'molecule_SpherocityIndex':molecule_SpherocityIndex,
'molecule_fr_allylic_oxid':molecule_fr_allylic_oxid,
'molecule_VSA_EState7':molecule_VSA_EState7,
'molecule_SMR_VSA8':molecule_SMR_VSA8,
'molecule_MolLogP':molecule_MolLogP,
'molecule_fr_phenol':molecule_fr_phenol,
'molecule_SlogP_VSA1':molecule_SlogP_VSA1,
'molecule_fr_azo':molecule_fr_azo,
'molecule_BalabanJ':molecule_BalabanJ,
'molecule_PMI2':molecule_PMI2,
'molecule_fr_isocyan':molecule_fr_isocyan,
'molecule_TPSA':molecule_TPSA,
'molecule_InertialShapeFactor':molecule_InertialShapeFactor,
'molecule_Kappa1':molecule_Kappa1,
'molecule_MinEStateIndex':molecule_MinEStateIndex,
'molecule_MaxEStateIndex':molecule_MaxEStateIndex,
'molecule_fr_phenol_noOrthoHbond':molecule_fr_phenol_noOrthoHbond,
'molecule_NumLipinskiHBA':molecule_NumLipinskiHBA,
'molecule_fr_sulfide':molecule_fr_sulfide,
'molecule_Kappa3':molecule_Kappa3,
'molecule_fr_alkyl_carbamate':molecule_fr_alkyl_carbamate,
'molecule_SlogP_VSA7':molecule_SlogP_VSA7,
'molecule_fr_thiocyan':molecule_fr_thiocyan,
'molecule_fr_term_acetylene':molecule_fr_term_acetylene,
'molecule_FractionCSP3':molecule_FractionCSP3,
'molecule_fr_ester':molecule_fr_ester,
'molecule_NumHAcceptors':molecule_NumHAcceptors,
'molecule_fr_unbrch_alkane':molecule_fr_unbrch_alkane,
'molecule_fr_sulfone':molecule_fr_sulfone,
'molecule_HallKierAlpha':molecule_HallKierAlpha,
'molecule_SMR_VSA7':molecule_SMR_VSA7,
'molecule_BCUT2D_LOGPLOW':molecule_BCUT2D_LOGPLOW,
'molecule_BCUT2D_LOGPHI':molecule_BCUT2D_LOGPHI,
'molecule_VSA_EState10':molecule_VSA_EState10,
'molecule_fr_para_hydroxylation':molecule_fr_para_hydroxylation,
'molecule_NumSaturatedCarbocycles':molecule_NumSaturatedCarbocycles,
'molecule_SlogP_VSA2':molecule_SlogP_VSA2,
'molecule_fr_COO':molecule_fr_COO,
'molecule_NumHeteroatoms':molecule_NumHeteroatoms,
'molecule_Chi3n':molecule_Chi3n,
'molecule_PEOE_VSA4':molecule_PEOE_VSA4,
'molecule_Chi4n':molecule_Chi4n,
'molecule_fr_dihydropyridine':molecule_fr_dihydropyridine,
'molecule_fr_priamide':molecule_fr_priamide
}




def main():
    from rdkit import Chem
    testmol=Chem.MolFromSmiles("c1ccccc1")
    for k,f in _available_featurizer.items():
        print(k)
        f(testmol)

if __name__=='__main__':
    main()
    