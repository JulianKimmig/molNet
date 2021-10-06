from molNet.utils.smiles import mol_from_smiles
from ._molecule_featurizer import MoleculeFeaturizer, SingleValueMoleculeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import (
    MolWt,
    fr_sulfone,
    fr_Ar_N,
    fr_ester,
    VSA_EState7,
    fr_lactam,
    fr_aniline,
    fr_nitro_arom_nonortho,
    fr_Ndealkylation1,
    EState_VSA9,
    SlogP_VSA10,
    HeavyAtomMolWt,
    fr_NH2,
    Chi3n,
    Chi4v,
    SMR_VSA8,
    NumRadicalElectrons,
    Chi1n,
    Chi0,
    PEOE_VSA2,
    fr_Ndealkylation2,
    fr_oxazole,
    BalabanJ,
    SlogP_VSA2,
    EState_VSA1,
    MolLogP,
    fr_C_S,
    SlogP_VSA9,
    BCUT2D_MWHI,
    VSA_EState1,
    SMR_VSA1,
    Kappa1,
    BCUT2D_MRHI,
    SMR_VSA3,
    EState_VSA8,
    fr_thiophene,
    fr_para_hydroxylation,
    VSA_EState8,
    FpDensityMorgan2,
    MinPartialCharge,
    fr_morpholine,
    PEOE_VSA13,
    EState_VSA5,
    fr_lactone,
    PEOE_VSA5,
    MaxEStateIndex,
    fr_bicyclic,
    NumAromaticCarbocycles,
    HeavyAtomCount,
    fr_Ar_COO,
    PEOE_VSA6,
    BertzCT,
    EState_VSA10,
    PEOE_VSA9,
    SlogP_VSA12,
    fr_C_O_noCOO,
    fr_quatN,
    fr_COO,
    MaxAbsEStateIndex,
    VSA_EState6,
    SlogP_VSA7,
    SlogP_VSA1,
    SMR_VSA9,
    fr_term_acetylene,
    fr_phos_acid,
    PEOE_VSA4,
    NOCount,
    SlogP_VSA5,
    fr_alkyl_halide,
    NumHDonors,
    NumAliphaticHeterocycles,
    fr_tetrazole,
    fr_imide,
    EState_VSA4,
    Kappa2,
    EState_VSA7,
    BCUT2D_MRLOW,
    fr_phenol,
    fr_imidazole,
    fr_NH1,
    VSA_EState3,
    fr_nitrile,
    fr_SH,
    SlogP_VSA8,
    fr_phos_ester,
    Ipc,
    PEOE_VSA10,
    Chi1v,
    NumValenceElectrons,
    fr_furan,
    fr_Ar_NH,
    fr_ether,
    fr_piperzine,
    fr_Al_COO,
    SlogP_VSA6,
    fr_sulfide,
    PEOE_VSA1,
    fr_Ar_OH,
    fr_C_O,
    fr_barbitur,
    fr_isocyan,
    VSA_EState4,
    VSA_EState2,
    fr_nitro_arom,
    MinAbsEStateIndex,
    SMR_VSA10,
    qed,
    fr_azide,
    fr_epoxide,
    fr_urea,
    MaxAbsPartialCharge,
    MinEStateIndex,
    VSA_EState10,
    TPSA,
    PEOE_VSA8,
    VSA_EState5,
    SlogP_VSA3,
    fr_alkyl_carbamate,
    fr_phenol_noOrthoHbond,
    fr_isothiocyan,
    EState_VSA11,
    BCUT2D_CHGHI,
    fr_oxime,
    fr_ArN,
    fr_unbrch_alkane,
    SMR_VSA2,
    fr_Al_OH_noTert,
    fr_dihydropyridine,
    fr_guanido,
    fr_piperdine,
    fr_aldehyde,
    PEOE_VSA12,
    PEOE_VSA11,
    fr_Nhpyrrole,
    SMR_VSA4,
    PEOE_VSA14,
    fr_ketone,
    fr_methoxy,
    fr_aryl_methyl,
    fr_halogen,
    fr_hdrzine,
    BCUT2D_CHGLO,
    fr_N_O,
    RingCount,
    PEOE_VSA3,
    BCUT2D_MWLOW,
    fr_nitroso,
    fr_pyridine,
    fr_amidine,
    SMR_VSA7,
    fr_HOCCN,
    FpDensityMorgan3,
    fr_diazo,
    fr_Imine,
    EState_VSA3,
    fr_prisulfonamd,
    fr_sulfonamd,
    fr_nitro,
    NumHAcceptors,
    fr_COO2,
    fr_allylic_oxid,
    fr_benzodiazepine,
    VSA_EState9,
    BCUT2D_LOGPHI,
    EState_VSA6,
    MolMR,
    fr_thiocyan,
    PEOE_VSA7,
    fr_NH0,
    MaxPartialCharge,
    SMR_VSA6,
    fr_priamide,
    fr_thiazole,
    fr_Al_OH,
    EState_VSA2,
    MinAbsPartialCharge,
    SlogP_VSA11,
    fr_azo,
    SMR_VSA5,
    fr_amide,
    NHOHCount,
    FpDensityMorgan1,
    fr_benzene,
    fr_hdrzone,
    SlogP_VSA4,
    BCUT2D_LOGPLOW,
    fr_ketone_Topliss,
)
from rdkit.Chem.rdMolDescriptors import (
    CalcRadiusOfGyration,
    CalcPMI2,
    CalcChi2v,
    CalcChi4n,
    CalcNumBridgeheadAtoms,
    CalcNumAliphaticCarbocycles,
    CalcPMI1,
    CalcNumSaturatedRings,
    CalcNumHBA,
    CalcNumSaturatedCarbocycles,
    CalcNumRotatableBonds,
    CalcNumAromaticRings,
    CalcNumLipinskiHBD,
    CalcChi0n,
    CalcNumUnspecifiedAtomStereoCenters,
    CalcLabuteASA,
    CalcExactMolWt,
    CalcNumHeteroatoms,
    CalcInertialShapeFactor,
    CalcNumHBD,
    CalcNPR1,
    CalcNumHeterocycles,
    CalcNumAmideBonds,
    CalcNumSaturatedHeterocycles,
    CalcNumLipinskiHBA,
    CalcAsphericity,
    CalcNumRings,
    CalcFractionCSP3,
    CalcNumAromaticHeterocycles,
    CalcNumAliphaticRings,
    CalcNumAtomStereoCenters,
    CalcNumSpiroAtoms,
    CalcPBF,
    CalcWHIM,
    GetUSRCAT,
    CalcMORSE,
    CalcRDF,
    CalcGETAWAY,
    CalcAUTOCORR3D,
    GetUSR,
    CalcCrippenDescriptors,
    BCUT2D,
    CalcAUTOCORR2D,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetMACCSKeysFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
)
from rdkit.Chem.GraphDescriptors import (
    HallKierAlpha,
    Kappa3,
    Chi2n,
    Chi3v,
    Chi0v,
    Chi1,
)
from rdkit.Chem.Descriptors3D import (
    PMI3,
    Eccentricity,
    SpherocityIndex,
    NPR2,
)
from rdkit.Chem.rdmolops import (
    GetFormalCharge,
    GetSSSR,
    LayeredFingerprint,
    PatternFingerprint,
    RDKFingerprint,
)


class MolWtFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MolWt)


class fr_sulfoneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_sulfone)


class fr_Ar_NFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ar_N)


class fr_esterFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_ester)


class VSA_EState7Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState7)


class fr_lactamFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_lactam)


class fr_anilineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_aniline)


class RadiusOfGyrationFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcRadiusOfGyration)


class HallKierAlphaFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(HallKierAlpha)


class fr_nitro_arom_nonorthoFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom_nonortho)


class fr_Ndealkylation1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation1)


class EState_VSA9Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA9)


class SlogP_VSA10Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA10)


class HeavyAtomMolWtFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(HeavyAtomMolWt)


class PMI2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcPMI2)


class Chi2vFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcChi2v)


class fr_NH2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_NH2)


class Chi4nFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcChi4n)


class Chi3nFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi3n)


class Chi4vFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi4v)


class NumBridgeheadAtomsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumBridgeheadAtoms)


class SMR_VSA8Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA8)


class PMI3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PMI3)


class NumRadicalElectronsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumRadicalElectrons)


class Chi1nFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi1n)


class Chi0Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi0)


class PEOE_VSA2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA2)


class fr_Ndealkylation2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation2)


class fr_oxazoleFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_oxazole)


class BalabanJFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BalabanJ)


class NumAliphaticCarbocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticCarbocycles)


class SlogP_VSA2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA2)


class EState_VSA1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA1)


class Kappa3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Kappa3)


class MolLogPFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MolLogP)


class fr_C_SFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_C_S)


class SlogP_VSA9Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA9)


class BCUT2D_MWHIFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWHI)


class VSA_EState1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState1)


class PMI1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcPMI1)


class SMR_VSA1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA1)


class NumSaturatedRingsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedRings)


class Kappa1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Kappa1)


class BCUT2D_MRHIFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRHI)


class SMR_VSA3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA3)


class EState_VSA8Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA8)


class fr_thiopheneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_thiophene)


class fr_para_hydroxylationFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_para_hydroxylation)


class VSA_EState8Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState8)


class GetFormalChargeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(GetFormalCharge)


class FpDensityMorgan2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan2)


class MinPartialChargeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MinPartialCharge)


class fr_morpholineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_morpholine)


class PEOE_VSA13Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA13)


class EState_VSA5Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA5)


class NumHBAFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumHBA)


class NumSaturatedCarbocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedCarbocycles)


class fr_lactoneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_lactone)


class PEOE_VSA5Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA5)


class MaxEStateIndexFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MaxEStateIndex)


class fr_bicyclicFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_bicyclic)


class NumAromaticCarbocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumAromaticCarbocycles)


class HeavyAtomCountFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(HeavyAtomCount)


class fr_Ar_COOFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ar_COO)


class PEOE_VSA6Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA6)


class NumRotatableBondsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumRotatableBonds)


class NumAromaticRingsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticRings)


class BertzCTFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BertzCT)


class EState_VSA10Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA10)


class PEOE_VSA9Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA9)


class SlogP_VSA12Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA12)


class NumLipinskiHBDFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBD)


class Chi0nFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcChi0n)


class fr_C_O_noCOOFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_C_O_noCOO)


class EccentricityFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Eccentricity)


class NumUnspecifiedAtomStereoCentersFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumUnspecifiedAtomStereoCenters)


class fr_quatNFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_quatN)


class fr_COOFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_COO)


class LabuteASAFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcLabuteASA)


class Chi2nFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi2n)


class MaxAbsEStateIndexFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MaxAbsEStateIndex)


class VSA_EState6Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState6)


class ExactMolWtFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcExactMolWt)


class SlogP_VSA7Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA7)


class SlogP_VSA1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA1)


class NumHeteroatomsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumHeteroatoms)


class SMR_VSA9Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA9)


class fr_term_acetyleneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_term_acetylene)


class fr_phos_acidFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_phos_acid)


class PEOE_VSA4Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA4)


class NOCountFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NOCount)


class SlogP_VSA5Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA5)


class fr_alkyl_halideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_halide)


class InertialShapeFactorFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcInertialShapeFactor)


class NumHDonorsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumHDonors)


class NumAliphaticHeterocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumAliphaticHeterocycles)


class Chi3vFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi3v)


class fr_tetrazoleFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_tetrazole)


class NumHBDFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumHBD)


class fr_imideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_imide)


class SpherocityIndexFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SpherocityIndex)


class EState_VSA4Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA4)


class Kappa2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Kappa2)


class EState_VSA7Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA7)


class BCUT2D_MRLOWFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRLOW)


class fr_phenolFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_phenol)


class GetSSSRFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(GetSSSR)


class NPR1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcNPR1)


class fr_imidazoleFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_imidazole)


class fr_NH1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_NH1)


class VSA_EState3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState3)


class fr_nitrileFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_nitrile)


class fr_SHFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_SH)


class Chi0vFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi0v)


class SlogP_VSA8Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA8)


class NumHeterocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumHeterocycles)


class fr_phos_esterFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_phos_ester)


class NumAmideBondsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAmideBonds)


class IpcFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Ipc)


class PEOE_VSA10Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA10)


class Chi1vFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi1v)


class NumValenceElectronsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumValenceElectrons)


class fr_furanFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_furan)


class NumSaturatedHeterocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumSaturatedHeterocycles)


class NumLipinskiHBAFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBA)


class fr_Ar_NHFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ar_NH)


class fr_etherFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_ether)


class NPR2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(NPR2)


class fr_piperzineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_piperzine)


class fr_Al_COOFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Al_COO)


class SlogP_VSA6Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA6)


class fr_sulfideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_sulfide)


class PEOE_VSA1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA1)


class fr_Ar_OHFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Ar_OH)


class fr_C_OFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_C_O)


class fr_barbiturFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_barbitur)


class fr_isocyanFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_isocyan)


class VSA_EState4Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState4)


class VSA_EState2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState2)


class fr_nitro_aromFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom)


class Chi1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(Chi1)


class MinAbsEStateIndexFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MinAbsEStateIndex)


class SMR_VSA10Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA10)


class qedFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(qed)


class fr_azideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_azide)


class fr_epoxideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_epoxide)


class fr_ureaFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_urea)


class MaxAbsPartialChargeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MaxAbsPartialCharge)


class MinEStateIndexFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MinEStateIndex)


class VSA_EState10Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState10)


class TPSAFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(TPSA)


class PEOE_VSA8Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA8)


class AsphericityFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcAsphericity)


class VSA_EState5Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState5)


class SlogP_VSA3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA3)


class NumRingsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumRings)


class fr_alkyl_carbamateFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_carbamate)


class fr_phenol_noOrthoHbondFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_phenol_noOrthoHbond)


class fr_isothiocyanFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_isothiocyan)


class FractionCSP3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcFractionCSP3)


class EState_VSA11Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA11)


class BCUT2D_CHGHIFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGHI)


class fr_oximeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_oxime)


class fr_ArNFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_ArN)


class NumAromaticHeterocyclesFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAromaticHeterocycles)


class fr_unbrch_alkaneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_unbrch_alkane)


class SMR_VSA2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA2)


class fr_Al_OH_noTertFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH_noTert)


class NumAliphaticRingsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAliphaticRings)


class NumAtomStereoCentersFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumAtomStereoCenters)


class fr_dihydropyridineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_dihydropyridine)


class fr_guanidoFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_guanido)


class fr_piperdineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_piperdine)


class fr_aldehydeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_aldehyde)


class PEOE_VSA12Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA12)


class PEOE_VSA11Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA11)


class fr_NhpyrroleFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Nhpyrrole)


class SMR_VSA4Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA4)


class PEOE_VSA14Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA14)


class NumSpiroAtomsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(CalcNumSpiroAtoms)


class fr_ketoneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_ketone)


class fr_methoxyFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_methoxy)


class fr_aryl_methylFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_aryl_methyl)


class fr_halogenFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_halogen)


class fr_hdrzineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_hdrzine)


class BCUT2D_CHGLOFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGLO)


class fr_N_OFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_N_O)


class RingCountFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(RingCount)


class PEOE_VSA3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA3)


class BCUT2D_MWLOWFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWLOW)


class fr_nitrosoFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_nitroso)


class fr_pyridineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_pyridine)


class fr_amidineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_amidine)


class SMR_VSA7Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA7)


class fr_HOCCNFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_HOCCN)


class FpDensityMorgan3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan3)


class fr_diazoFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_diazo)


class fr_ImineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Imine)


class EState_VSA3Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA3)


class fr_prisulfonamdFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_prisulfonamd)


class fr_sulfonamdFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_sulfonamd)


class PBFFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(CalcPBF)


class fr_nitroFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_nitro)


class NumHAcceptorsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NumHAcceptors)


class fr_COO2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_COO2)


class fr_allylic_oxidFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_allylic_oxid)


class fr_benzodiazepineFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_benzodiazepine)


class VSA_EState9Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(VSA_EState9)


class BCUT2D_LOGPHIFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPHI)


class EState_VSA6Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA6)


class MolMRFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MolMR)


class fr_thiocyanFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_thiocyan)


class PEOE_VSA7Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA7)


class fr_NH0Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_NH0)


class MaxPartialChargeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MaxPartialCharge)


class SMR_VSA6Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA6)


class fr_priamideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_priamide)


class fr_thiazoleFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_thiazole)


class fr_Al_OHFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH)


class EState_VSA2Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(EState_VSA2)


class MinAbsPartialChargeFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(MinAbsPartialCharge)


class SlogP_VSA11Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA11)


class fr_azoFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_azo)


class SMR_VSA5Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SMR_VSA5)


class fr_amideFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_amide)


class NHOHCountFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(NHOHCount)


class FpDensityMorgan1Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan1)


class fr_benzeneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_benzene)


class fr_hdrzoneFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_hdrzone)


class SlogP_VSA4Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA4)


class BCUT2D_LOGPLOWFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPLOW)


class fr_ketone_ToplissFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32
    featurize = staticmethod(fr_ketone_Topliss)


class WHIMFeaturizer(MoleculeFeaturizer):
    _LENGTH = 114
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcWHIM(mol), dtype=self.dtype)


class GetUSRCATFeaturizer(MoleculeFeaturizer):
    _LENGTH = 60
    dtype = np.float32

    def featurize(self, mol):
        return np.array(GetUSRCAT(mol), dtype=self.dtype)


class MORSEFeaturizer(MoleculeFeaturizer):
    _LENGTH = 224
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcMORSE(mol), dtype=self.dtype)


class RDFFeaturizer(MoleculeFeaturizer):
    _LENGTH = 210
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcRDF(mol), dtype=self.dtype)


class GETAWAYFeaturizer(MoleculeFeaturizer):
    _LENGTH = 273
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcGETAWAY(mol), dtype=self.dtype)


class AUTOCORR3DFeaturizer(MoleculeFeaturizer):
    _LENGTH = 80
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcAUTOCORR3D(mol), dtype=self.dtype)


class GetUSRFeaturizer(MoleculeFeaturizer):
    _LENGTH = 12
    dtype = np.float32

    def featurize(self, mol):
        return np.array(GetUSR(mol), dtype=self.dtype)


class CrippenDescriptorsFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcCrippenDescriptors(mol), dtype=self.dtype)


class BCUT2DFeaturizer(MoleculeFeaturizer):
    _LENGTH = 8
    dtype = np.float32

    def featurize(self, mol):
        return np.array(BCUT2D(mol), dtype=self.dtype)


class AUTOCORR2DFeaturizer(MoleculeFeaturizer):
    _LENGTH = 192
    dtype = np.float32

    def featurize(self, mol):
        return np.array(CalcAUTOCORR2D(mol), dtype=self.dtype)


class GetHashedTopologicalTorsionFingerprintAsBitVectFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2048
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


class LayeredFingerprintFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2048
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


class GetMACCSKeysFingerprintFeaturizer(MoleculeFeaturizer):
    _LENGTH = 167
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
        return a


class GetHashedAtomPairFingerprintAsBitVectFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2048
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class PatternFingerprintFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2048
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class RDKFingerprintFeaturizer(MoleculeFeaturizer):
    _LENGTH = 2048
    dtype = np.bool_

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


molecule_mol_wt = MolWtFeaturizer()
molecule_fr_sulfone = fr_sulfoneFeaturizer()
molecule_fr_ar_n = fr_Ar_NFeaturizer()
molecule_fr_ester = fr_esterFeaturizer()
molecule_vsa_e_state7 = VSA_EState7Featurizer()
molecule_fr_lactam = fr_lactamFeaturizer()
molecule_fr_aniline = fr_anilineFeaturizer()
molecule_radius_of_gyration = RadiusOfGyrationFeaturizer()
molecule_hall_kier_alpha = HallKierAlphaFeaturizer()
molecule_fr_nitro_arom_nonortho = fr_nitro_arom_nonorthoFeaturizer()
molecule_fr_ndealkylation1 = fr_Ndealkylation1Featurizer()
molecule_e_state_vsa9 = EState_VSA9Featurizer()
molecule_slog_p_vsa10 = SlogP_VSA10Featurizer()
molecule_heavy_atom_mol_wt = HeavyAtomMolWtFeaturizer()
molecule_pmi2 = PMI2Featurizer()
molecule_chi2v = Chi2vFeaturizer()
molecule_fr_nh2 = fr_NH2Featurizer()
molecule_chi4n = Chi4nFeaturizer()
molecule_chi3n = Chi3nFeaturizer()
molecule_chi4v = Chi4vFeaturizer()
molecule_num_bridgehead_atoms = NumBridgeheadAtomsFeaturizer()
molecule_smr_vsa8 = SMR_VSA8Featurizer()
molecule_pmi3 = PMI3Featurizer()
molecule_num_radical_electrons = NumRadicalElectronsFeaturizer()
molecule_chi1n = Chi1nFeaturizer()
molecule_chi0 = Chi0Featurizer()
molecule_peoe_vsa2 = PEOE_VSA2Featurizer()
molecule_fr_ndealkylation2 = fr_Ndealkylation2Featurizer()
molecule_fr_oxazole = fr_oxazoleFeaturizer()
molecule_balaban_j = BalabanJFeaturizer()
molecule_num_aliphatic_carbocycles = NumAliphaticCarbocyclesFeaturizer()
molecule_slog_p_vsa2 = SlogP_VSA2Featurizer()
molecule_e_state_vsa1 = EState_VSA1Featurizer()
molecule_kappa3 = Kappa3Featurizer()
molecule_mol_log_p = MolLogPFeaturizer()
molecule_fr_c_s = fr_C_SFeaturizer()
molecule_slog_p_vsa9 = SlogP_VSA9Featurizer()
molecule_bcut2d_mwhi = BCUT2D_MWHIFeaturizer()
molecule_vsa_e_state1 = VSA_EState1Featurizer()
molecule_pmi1 = PMI1Featurizer()
molecule_smr_vsa1 = SMR_VSA1Featurizer()
molecule_num_saturated_rings = NumSaturatedRingsFeaturizer()
molecule_kappa1 = Kappa1Featurizer()
molecule_bcut2d_mrhi = BCUT2D_MRHIFeaturizer()
molecule_smr_vsa3 = SMR_VSA3Featurizer()
molecule_e_state_vsa8 = EState_VSA8Featurizer()
molecule_fr_thiophene = fr_thiopheneFeaturizer()
molecule_fr_para_hydroxylation = fr_para_hydroxylationFeaturizer()
molecule_vsa_e_state8 = VSA_EState8Featurizer()
molecule_get_formal_charge = GetFormalChargeFeaturizer()
molecule_fp_density_morgan2 = FpDensityMorgan2Featurizer()
molecule_min_partial_charge = MinPartialChargeFeaturizer()
molecule_fr_morpholine = fr_morpholineFeaturizer()
molecule_peoe_vsa13 = PEOE_VSA13Featurizer()
molecule_e_state_vsa5 = EState_VSA5Featurizer()
molecule_num_hba = NumHBAFeaturizer()
molecule_num_saturated_carbocycles = NumSaturatedCarbocyclesFeaturizer()
molecule_fr_lactone = fr_lactoneFeaturizer()
molecule_peoe_vsa5 = PEOE_VSA5Featurizer()
molecule_max_e_state_index = MaxEStateIndexFeaturizer()
molecule_fr_bicyclic = fr_bicyclicFeaturizer()
molecule_num_aromatic_carbocycles = NumAromaticCarbocyclesFeaturizer()
molecule_heavy_atom_count = HeavyAtomCountFeaturizer()
molecule_fr_ar_coo = fr_Ar_COOFeaturizer()
molecule_peoe_vsa6 = PEOE_VSA6Featurizer()
molecule_num_rotatable_bonds = NumRotatableBondsFeaturizer()
molecule_num_aromatic_rings = NumAromaticRingsFeaturizer()
molecule_bertz_ct = BertzCTFeaturizer()
molecule_e_state_vsa10 = EState_VSA10Featurizer()
molecule_peoe_vsa9 = PEOE_VSA9Featurizer()
molecule_slog_p_vsa12 = SlogP_VSA12Featurizer()
molecule_num_lipinski_hbd = NumLipinskiHBDFeaturizer()
molecule_chi0n = Chi0nFeaturizer()
molecule_fr_c_o_no_coo = fr_C_O_noCOOFeaturizer()
molecule_eccentricity = EccentricityFeaturizer()
molecule_num_unspecified_atom_stereo_centers = (
    NumUnspecifiedAtomStereoCentersFeaturizer()
)
molecule_fr_quat_n = fr_quatNFeaturizer()
molecule_fr_coo = fr_COOFeaturizer()
molecule_labute_asa = LabuteASAFeaturizer()
molecule_chi2n = Chi2nFeaturizer()
molecule_max_abs_e_state_index = MaxAbsEStateIndexFeaturizer()
molecule_vsa_e_state6 = VSA_EState6Featurizer()
molecule_exact_mol_wt = ExactMolWtFeaturizer()
molecule_slog_p_vsa7 = SlogP_VSA7Featurizer()
molecule_slog_p_vsa1 = SlogP_VSA1Featurizer()
molecule_num_heteroatoms = NumHeteroatomsFeaturizer()
molecule_smr_vsa9 = SMR_VSA9Featurizer()
molecule_fr_term_acetylene = fr_term_acetyleneFeaturizer()
molecule_fr_phos_acid = fr_phos_acidFeaturizer()
molecule_peoe_vsa4 = PEOE_VSA4Featurizer()
molecule_no_count = NOCountFeaturizer()
molecule_slog_p_vsa5 = SlogP_VSA5Featurizer()
molecule_fr_alkyl_halide = fr_alkyl_halideFeaturizer()
molecule_inertial_shape_factor = InertialShapeFactorFeaturizer()
molecule_num_h_donors = NumHDonorsFeaturizer()
molecule_num_aliphatic_heterocycles = NumAliphaticHeterocyclesFeaturizer()
molecule_chi3v = Chi3vFeaturizer()
molecule_fr_tetrazole = fr_tetrazoleFeaturizer()
molecule_num_hbd = NumHBDFeaturizer()
molecule_fr_imide = fr_imideFeaturizer()
molecule_spherocity_index = SpherocityIndexFeaturizer()
molecule_e_state_vsa4 = EState_VSA4Featurizer()
molecule_kappa2 = Kappa2Featurizer()
molecule_e_state_vsa7 = EState_VSA7Featurizer()
molecule_bcut2d_mrlow = BCUT2D_MRLOWFeaturizer()
molecule_fr_phenol = fr_phenolFeaturizer()
molecule_get_sssr = GetSSSRFeaturizer()
molecule_npr1 = NPR1Featurizer()
molecule_fr_imidazole = fr_imidazoleFeaturizer()
molecule_fr_nh1 = fr_NH1Featurizer()
molecule_vsa_e_state3 = VSA_EState3Featurizer()
molecule_fr_nitrile = fr_nitrileFeaturizer()
molecule_fr_sh = fr_SHFeaturizer()
molecule_chi0v = Chi0vFeaturizer()
molecule_slog_p_vsa8 = SlogP_VSA8Featurizer()
molecule_num_heterocycles = NumHeterocyclesFeaturizer()
molecule_fr_phos_ester = fr_phos_esterFeaturizer()
molecule_num_amide_bonds = NumAmideBondsFeaturizer()
molecule_ipc = IpcFeaturizer()
molecule_peoe_vsa10 = PEOE_VSA10Featurizer()
molecule_chi1v = Chi1vFeaturizer()
molecule_num_valence_electrons = NumValenceElectronsFeaturizer()
molecule_fr_furan = fr_furanFeaturizer()
molecule_num_saturated_heterocycles = NumSaturatedHeterocyclesFeaturizer()
molecule_num_lipinski_hba = NumLipinskiHBAFeaturizer()
molecule_fr_ar_nh = fr_Ar_NHFeaturizer()
molecule_fr_ether = fr_etherFeaturizer()
molecule_npr2 = NPR2Featurizer()
molecule_fr_piperzine = fr_piperzineFeaturizer()
molecule_fr_al_coo = fr_Al_COOFeaturizer()
molecule_slog_p_vsa6 = SlogP_VSA6Featurizer()
molecule_fr_sulfide = fr_sulfideFeaturizer()
molecule_peoe_vsa1 = PEOE_VSA1Featurizer()
molecule_fr_ar_oh = fr_Ar_OHFeaturizer()
molecule_fr_c_o = fr_C_OFeaturizer()
molecule_fr_barbitur = fr_barbiturFeaturizer()
molecule_fr_isocyan = fr_isocyanFeaturizer()
molecule_vsa_e_state4 = VSA_EState4Featurizer()
molecule_vsa_e_state2 = VSA_EState2Featurizer()
molecule_fr_nitro_arom = fr_nitro_aromFeaturizer()
molecule_chi1 = Chi1Featurizer()
molecule_min_abs_e_state_index = MinAbsEStateIndexFeaturizer()
molecule_smr_vsa10 = SMR_VSA10Featurizer()
molecule_qed = qedFeaturizer()
molecule_fr_azide = fr_azideFeaturizer()
molecule_fr_epoxide = fr_epoxideFeaturizer()
molecule_fr_urea = fr_ureaFeaturizer()
molecule_max_abs_partial_charge = MaxAbsPartialChargeFeaturizer()
molecule_min_e_state_index = MinEStateIndexFeaturizer()
molecule_vsa_e_state10 = VSA_EState10Featurizer()
molecule_tpsa = TPSAFeaturizer()
molecule_peoe_vsa8 = PEOE_VSA8Featurizer()
molecule_asphericity = AsphericityFeaturizer()
molecule_vsa_e_state5 = VSA_EState5Featurizer()
molecule_slog_p_vsa3 = SlogP_VSA3Featurizer()
molecule_num_rings = NumRingsFeaturizer()
molecule_fr_alkyl_carbamate = fr_alkyl_carbamateFeaturizer()
molecule_fr_phenol_no_ortho_hbond = fr_phenol_noOrthoHbondFeaturizer()
molecule_fr_isothiocyan = fr_isothiocyanFeaturizer()
molecule_fraction_csp3 = FractionCSP3Featurizer()
molecule_e_state_vsa11 = EState_VSA11Featurizer()
molecule_bcut2d_chghi = BCUT2D_CHGHIFeaturizer()
molecule_fr_oxime = fr_oximeFeaturizer()
molecule_fr_ar_n = fr_ArNFeaturizer()
molecule_num_aromatic_heterocycles = NumAromaticHeterocyclesFeaturizer()
molecule_fr_unbrch_alkane = fr_unbrch_alkaneFeaturizer()
molecule_smr_vsa2 = SMR_VSA2Featurizer()
molecule_fr_al_oh_no_tert = fr_Al_OH_noTertFeaturizer()
molecule_num_aliphatic_rings = NumAliphaticRingsFeaturizer()
molecule_num_atom_stereo_centers = NumAtomStereoCentersFeaturizer()
molecule_fr_dihydropyridine = fr_dihydropyridineFeaturizer()
molecule_fr_guanido = fr_guanidoFeaturizer()
molecule_fr_piperdine = fr_piperdineFeaturizer()
molecule_fr_aldehyde = fr_aldehydeFeaturizer()
molecule_peoe_vsa12 = PEOE_VSA12Featurizer()
molecule_peoe_vsa11 = PEOE_VSA11Featurizer()
molecule_fr_nhpyrrole = fr_NhpyrroleFeaturizer()
molecule_smr_vsa4 = SMR_VSA4Featurizer()
molecule_peoe_vsa14 = PEOE_VSA14Featurizer()
molecule_num_spiro_atoms = NumSpiroAtomsFeaturizer()
molecule_fr_ketone = fr_ketoneFeaturizer()
molecule_fr_methoxy = fr_methoxyFeaturizer()
molecule_fr_aryl_methyl = fr_aryl_methylFeaturizer()
molecule_fr_halogen = fr_halogenFeaturizer()
molecule_fr_hdrzine = fr_hdrzineFeaturizer()
molecule_bcut2d_chglo = BCUT2D_CHGLOFeaturizer()
molecule_fr_n_o = fr_N_OFeaturizer()
molecule_ring_count = RingCountFeaturizer()
molecule_peoe_vsa3 = PEOE_VSA3Featurizer()
molecule_bcut2d_mwlow = BCUT2D_MWLOWFeaturizer()
molecule_fr_nitroso = fr_nitrosoFeaturizer()
molecule_fr_pyridine = fr_pyridineFeaturizer()
molecule_fr_amidine = fr_amidineFeaturizer()
molecule_smr_vsa7 = SMR_VSA7Featurizer()
molecule_fr_hoccn = fr_HOCCNFeaturizer()
molecule_fp_density_morgan3 = FpDensityMorgan3Featurizer()
molecule_fr_diazo = fr_diazoFeaturizer()
molecule_fr_imine = fr_ImineFeaturizer()
molecule_e_state_vsa3 = EState_VSA3Featurizer()
molecule_fr_prisulfonamd = fr_prisulfonamdFeaturizer()
molecule_fr_sulfonamd = fr_sulfonamdFeaturizer()
molecule_pbf = PBFFeaturizer()
molecule_fr_nitro = fr_nitroFeaturizer()
molecule_num_h_acceptors = NumHAcceptorsFeaturizer()
molecule_fr_coo2 = fr_COO2Featurizer()
molecule_fr_allylic_oxid = fr_allylic_oxidFeaturizer()
molecule_fr_benzodiazepine = fr_benzodiazepineFeaturizer()
molecule_vsa_e_state9 = VSA_EState9Featurizer()
molecule_bcut2d_logphi = BCUT2D_LOGPHIFeaturizer()
molecule_e_state_vsa6 = EState_VSA6Featurizer()
molecule_mol_mr = MolMRFeaturizer()
molecule_fr_thiocyan = fr_thiocyanFeaturizer()
molecule_peoe_vsa7 = PEOE_VSA7Featurizer()
molecule_fr_nh0 = fr_NH0Featurizer()
molecule_max_partial_charge = MaxPartialChargeFeaturizer()
molecule_smr_vsa6 = SMR_VSA6Featurizer()
molecule_fr_priamide = fr_priamideFeaturizer()
molecule_fr_thiazole = fr_thiazoleFeaturizer()
molecule_fr_al_oh = fr_Al_OHFeaturizer()
molecule_e_state_vsa2 = EState_VSA2Featurizer()
molecule_min_abs_partial_charge = MinAbsPartialChargeFeaturizer()
molecule_slog_p_vsa11 = SlogP_VSA11Featurizer()
molecule_fr_azo = fr_azoFeaturizer()
molecule_smr_vsa5 = SMR_VSA5Featurizer()
molecule_fr_amide = fr_amideFeaturizer()
molecule_nhoh_count = NHOHCountFeaturizer()
molecule_fp_density_morgan1 = FpDensityMorgan1Featurizer()
molecule_fr_benzene = fr_benzeneFeaturizer()
molecule_fr_hdrzone = fr_hdrzoneFeaturizer()
molecule_slog_p_vsa4 = SlogP_VSA4Featurizer()
molecule_bcut2d_logplow = BCUT2D_LOGPLOWFeaturizer()
molecule_fr_ketone_topliss = fr_ketone_ToplissFeaturizer()
molecule_whim = WHIMFeaturizer()
molecule_get_usrcat = GetUSRCATFeaturizer()
molecule_morse = MORSEFeaturizer()
molecule_rdf = RDFFeaturizer()
molecule_getaway = GETAWAYFeaturizer()
molecule_autocorr3d = AUTOCORR3DFeaturizer()
molecule_get_usr = GetUSRFeaturizer()
molecule_crippen_descriptors = CrippenDescriptorsFeaturizer()
molecule_bcut2d = BCUT2DFeaturizer()
molecule_autocorr2d = AUTOCORR2DFeaturizer()
molecule_get_hashed_topological_torsion_fingerprint_as_bit_vect = (
    GetHashedTopologicalTorsionFingerprintAsBitVectFeaturizer()
)
molecule_layered_fingerprint = LayeredFingerprintFeaturizer()
molecule_get_maccs_keys_fingerprint = GetMACCSKeysFingerprintFeaturizer()
molecule_get_hashed_atom_pair_fingerprint_as_bit_vect = (
    GetHashedAtomPairFingerprintAsBitVectFeaturizer()
)
molecule_pattern_fingerprint = PatternFingerprintFeaturizer()
molecule_rdk_fingerprint = RDKFingerprintFeaturizer()

_available_featurizer = {
    "molecule_mol_wt": molecule_mol_wt,
    "molecule_fr_sulfone": molecule_fr_sulfone,
    "molecule_fr_ar_n": molecule_fr_ar_n,
    "molecule_fr_ester": molecule_fr_ester,
    "molecule_vsa_e_state7": molecule_vsa_e_state7,
    "molecule_fr_lactam": molecule_fr_lactam,
    "molecule_fr_aniline": molecule_fr_aniline,
    "molecule_radius_of_gyration": molecule_radius_of_gyration,
    "molecule_hall_kier_alpha": molecule_hall_kier_alpha,
    "molecule_fr_nitro_arom_nonortho": molecule_fr_nitro_arom_nonortho,
    "molecule_fr_ndealkylation1": molecule_fr_ndealkylation1,
    "molecule_e_state_vsa9": molecule_e_state_vsa9,
    "molecule_slog_p_vsa10": molecule_slog_p_vsa10,
    "molecule_heavy_atom_mol_wt": molecule_heavy_atom_mol_wt,
    "molecule_pmi2": molecule_pmi2,
    "molecule_chi2v": molecule_chi2v,
    "molecule_fr_nh2": molecule_fr_nh2,
    "molecule_chi4n": molecule_chi4n,
    "molecule_chi3n": molecule_chi3n,
    "molecule_chi4v": molecule_chi4v,
    "molecule_num_bridgehead_atoms": molecule_num_bridgehead_atoms,
    "molecule_smr_vsa8": molecule_smr_vsa8,
    "molecule_pmi3": molecule_pmi3,
    "molecule_num_radical_electrons": molecule_num_radical_electrons,
    "molecule_chi1n": molecule_chi1n,
    "molecule_chi0": molecule_chi0,
    "molecule_peoe_vsa2": molecule_peoe_vsa2,
    "molecule_fr_ndealkylation2": molecule_fr_ndealkylation2,
    "molecule_fr_oxazole": molecule_fr_oxazole,
    "molecule_balaban_j": molecule_balaban_j,
    "molecule_num_aliphatic_carbocycles": molecule_num_aliphatic_carbocycles,
    "molecule_slog_p_vsa2": molecule_slog_p_vsa2,
    "molecule_e_state_vsa1": molecule_e_state_vsa1,
    "molecule_kappa3": molecule_kappa3,
    "molecule_mol_log_p": molecule_mol_log_p,
    "molecule_fr_c_s": molecule_fr_c_s,
    "molecule_slog_p_vsa9": molecule_slog_p_vsa9,
    "molecule_bcut2d_mwhi": molecule_bcut2d_mwhi,
    "molecule_vsa_e_state1": molecule_vsa_e_state1,
    "molecule_pmi1": molecule_pmi1,
    "molecule_smr_vsa1": molecule_smr_vsa1,
    "molecule_num_saturated_rings": molecule_num_saturated_rings,
    "molecule_kappa1": molecule_kappa1,
    "molecule_bcut2d_mrhi": molecule_bcut2d_mrhi,
    "molecule_smr_vsa3": molecule_smr_vsa3,
    "molecule_e_state_vsa8": molecule_e_state_vsa8,
    "molecule_fr_thiophene": molecule_fr_thiophene,
    "molecule_fr_para_hydroxylation": molecule_fr_para_hydroxylation,
    "molecule_vsa_e_state8": molecule_vsa_e_state8,
    "molecule_get_formal_charge": molecule_get_formal_charge,
    "molecule_fp_density_morgan2": molecule_fp_density_morgan2,
    "molecule_min_partial_charge": molecule_min_partial_charge,
    "molecule_fr_morpholine": molecule_fr_morpholine,
    "molecule_peoe_vsa13": molecule_peoe_vsa13,
    "molecule_e_state_vsa5": molecule_e_state_vsa5,
    "molecule_num_hba": molecule_num_hba,
    "molecule_num_saturated_carbocycles": molecule_num_saturated_carbocycles,
    "molecule_fr_lactone": molecule_fr_lactone,
    "molecule_peoe_vsa5": molecule_peoe_vsa5,
    "molecule_max_e_state_index": molecule_max_e_state_index,
    "molecule_fr_bicyclic": molecule_fr_bicyclic,
    "molecule_num_aromatic_carbocycles": molecule_num_aromatic_carbocycles,
    "molecule_heavy_atom_count": molecule_heavy_atom_count,
    "molecule_fr_ar_coo": molecule_fr_ar_coo,
    "molecule_peoe_vsa6": molecule_peoe_vsa6,
    "molecule_num_rotatable_bonds": molecule_num_rotatable_bonds,
    "molecule_num_aromatic_rings": molecule_num_aromatic_rings,
    "molecule_bertz_ct": molecule_bertz_ct,
    "molecule_e_state_vsa10": molecule_e_state_vsa10,
    "molecule_peoe_vsa9": molecule_peoe_vsa9,
    "molecule_slog_p_vsa12": molecule_slog_p_vsa12,
    "molecule_num_lipinski_hbd": molecule_num_lipinski_hbd,
    "molecule_chi0n": molecule_chi0n,
    "molecule_fr_c_o_no_coo": molecule_fr_c_o_no_coo,
    "molecule_eccentricity": molecule_eccentricity,
    "molecule_num_unspecified_atom_stereo_centers": molecule_num_unspecified_atom_stereo_centers,
    "molecule_fr_quat_n": molecule_fr_quat_n,
    "molecule_fr_coo": molecule_fr_coo,
    "molecule_labute_asa": molecule_labute_asa,
    "molecule_chi2n": molecule_chi2n,
    "molecule_max_abs_e_state_index": molecule_max_abs_e_state_index,
    "molecule_vsa_e_state6": molecule_vsa_e_state6,
    "molecule_exact_mol_wt": molecule_exact_mol_wt,
    "molecule_slog_p_vsa7": molecule_slog_p_vsa7,
    "molecule_slog_p_vsa1": molecule_slog_p_vsa1,
    "molecule_num_heteroatoms": molecule_num_heteroatoms,
    "molecule_smr_vsa9": molecule_smr_vsa9,
    "molecule_fr_term_acetylene": molecule_fr_term_acetylene,
    "molecule_fr_phos_acid": molecule_fr_phos_acid,
    "molecule_peoe_vsa4": molecule_peoe_vsa4,
    "molecule_no_count": molecule_no_count,
    "molecule_slog_p_vsa5": molecule_slog_p_vsa5,
    "molecule_fr_alkyl_halide": molecule_fr_alkyl_halide,
    "molecule_inertial_shape_factor": molecule_inertial_shape_factor,
    "molecule_num_h_donors": molecule_num_h_donors,
    "molecule_num_aliphatic_heterocycles": molecule_num_aliphatic_heterocycles,
    "molecule_chi3v": molecule_chi3v,
    "molecule_fr_tetrazole": molecule_fr_tetrazole,
    "molecule_num_hbd": molecule_num_hbd,
    "molecule_fr_imide": molecule_fr_imide,
    "molecule_spherocity_index": molecule_spherocity_index,
    "molecule_e_state_vsa4": molecule_e_state_vsa4,
    "molecule_kappa2": molecule_kappa2,
    "molecule_e_state_vsa7": molecule_e_state_vsa7,
    "molecule_bcut2d_mrlow": molecule_bcut2d_mrlow,
    "molecule_fr_phenol": molecule_fr_phenol,
    "molecule_get_sssr": molecule_get_sssr,
    "molecule_npr1": molecule_npr1,
    "molecule_fr_imidazole": molecule_fr_imidazole,
    "molecule_fr_nh1": molecule_fr_nh1,
    "molecule_vsa_e_state3": molecule_vsa_e_state3,
    "molecule_fr_nitrile": molecule_fr_nitrile,
    "molecule_fr_sh": molecule_fr_sh,
    "molecule_chi0v": molecule_chi0v,
    "molecule_slog_p_vsa8": molecule_slog_p_vsa8,
    "molecule_num_heterocycles": molecule_num_heterocycles,
    "molecule_fr_phos_ester": molecule_fr_phos_ester,
    "molecule_num_amide_bonds": molecule_num_amide_bonds,
    "molecule_ipc": molecule_ipc,
    "molecule_peoe_vsa10": molecule_peoe_vsa10,
    "molecule_chi1v": molecule_chi1v,
    "molecule_num_valence_electrons": molecule_num_valence_electrons,
    "molecule_fr_furan": molecule_fr_furan,
    "molecule_num_saturated_heterocycles": molecule_num_saturated_heterocycles,
    "molecule_num_lipinski_hba": molecule_num_lipinski_hba,
    "molecule_fr_ar_nh": molecule_fr_ar_nh,
    "molecule_fr_ether": molecule_fr_ether,
    "molecule_npr2": molecule_npr2,
    "molecule_fr_piperzine": molecule_fr_piperzine,
    "molecule_fr_al_coo": molecule_fr_al_coo,
    "molecule_slog_p_vsa6": molecule_slog_p_vsa6,
    "molecule_fr_sulfide": molecule_fr_sulfide,
    "molecule_peoe_vsa1": molecule_peoe_vsa1,
    "molecule_fr_ar_oh": molecule_fr_ar_oh,
    "molecule_fr_c_o": molecule_fr_c_o,
    "molecule_fr_barbitur": molecule_fr_barbitur,
    "molecule_fr_isocyan": molecule_fr_isocyan,
    "molecule_vsa_e_state4": molecule_vsa_e_state4,
    "molecule_vsa_e_state2": molecule_vsa_e_state2,
    "molecule_fr_nitro_arom": molecule_fr_nitro_arom,
    "molecule_chi1": molecule_chi1,
    "molecule_min_abs_e_state_index": molecule_min_abs_e_state_index,
    "molecule_smr_vsa10": molecule_smr_vsa10,
    "molecule_qed": molecule_qed,
    "molecule_fr_azide": molecule_fr_azide,
    "molecule_fr_epoxide": molecule_fr_epoxide,
    "molecule_fr_urea": molecule_fr_urea,
    "molecule_max_abs_partial_charge": molecule_max_abs_partial_charge,
    "molecule_min_e_state_index": molecule_min_e_state_index,
    "molecule_vsa_e_state10": molecule_vsa_e_state10,
    "molecule_tpsa": molecule_tpsa,
    "molecule_peoe_vsa8": molecule_peoe_vsa8,
    "molecule_asphericity": molecule_asphericity,
    "molecule_vsa_e_state5": molecule_vsa_e_state5,
    "molecule_slog_p_vsa3": molecule_slog_p_vsa3,
    "molecule_num_rings": molecule_num_rings,
    "molecule_fr_alkyl_carbamate": molecule_fr_alkyl_carbamate,
    "molecule_fr_phenol_no_ortho_hbond": molecule_fr_phenol_no_ortho_hbond,
    "molecule_fr_isothiocyan": molecule_fr_isothiocyan,
    "molecule_fraction_csp3": molecule_fraction_csp3,
    "molecule_e_state_vsa11": molecule_e_state_vsa11,
    "molecule_bcut2d_chghi": molecule_bcut2d_chghi,
    "molecule_fr_oxime": molecule_fr_oxime,
    "molecule_fr_ar_n": molecule_fr_ar_n,
    "molecule_num_aromatic_heterocycles": molecule_num_aromatic_heterocycles,
    "molecule_fr_unbrch_alkane": molecule_fr_unbrch_alkane,
    "molecule_smr_vsa2": molecule_smr_vsa2,
    "molecule_fr_al_oh_no_tert": molecule_fr_al_oh_no_tert,
    "molecule_num_aliphatic_rings": molecule_num_aliphatic_rings,
    "molecule_num_atom_stereo_centers": molecule_num_atom_stereo_centers,
    "molecule_fr_dihydropyridine": molecule_fr_dihydropyridine,
    "molecule_fr_guanido": molecule_fr_guanido,
    "molecule_fr_piperdine": molecule_fr_piperdine,
    "molecule_fr_aldehyde": molecule_fr_aldehyde,
    "molecule_peoe_vsa12": molecule_peoe_vsa12,
    "molecule_peoe_vsa11": molecule_peoe_vsa11,
    "molecule_fr_nhpyrrole": molecule_fr_nhpyrrole,
    "molecule_smr_vsa4": molecule_smr_vsa4,
    "molecule_peoe_vsa14": molecule_peoe_vsa14,
    "molecule_num_spiro_atoms": molecule_num_spiro_atoms,
    "molecule_fr_ketone": molecule_fr_ketone,
    "molecule_fr_methoxy": molecule_fr_methoxy,
    "molecule_fr_aryl_methyl": molecule_fr_aryl_methyl,
    "molecule_fr_halogen": molecule_fr_halogen,
    "molecule_fr_hdrzine": molecule_fr_hdrzine,
    "molecule_bcut2d_chglo": molecule_bcut2d_chglo,
    "molecule_fr_n_o": molecule_fr_n_o,
    "molecule_ring_count": molecule_ring_count,
    "molecule_peoe_vsa3": molecule_peoe_vsa3,
    "molecule_bcut2d_mwlow": molecule_bcut2d_mwlow,
    "molecule_fr_nitroso": molecule_fr_nitroso,
    "molecule_fr_pyridine": molecule_fr_pyridine,
    "molecule_fr_amidine": molecule_fr_amidine,
    "molecule_smr_vsa7": molecule_smr_vsa7,
    "molecule_fr_hoccn": molecule_fr_hoccn,
    "molecule_fp_density_morgan3": molecule_fp_density_morgan3,
    "molecule_fr_diazo": molecule_fr_diazo,
    "molecule_fr_imine": molecule_fr_imine,
    "molecule_e_state_vsa3": molecule_e_state_vsa3,
    "molecule_fr_prisulfonamd": molecule_fr_prisulfonamd,
    "molecule_fr_sulfonamd": molecule_fr_sulfonamd,
    "molecule_pbf": molecule_pbf,
    "molecule_fr_nitro": molecule_fr_nitro,
    "molecule_num_h_acceptors": molecule_num_h_acceptors,
    "molecule_fr_coo2": molecule_fr_coo2,
    "molecule_fr_allylic_oxid": molecule_fr_allylic_oxid,
    "molecule_fr_benzodiazepine": molecule_fr_benzodiazepine,
    "molecule_vsa_e_state9": molecule_vsa_e_state9,
    "molecule_bcut2d_logphi": molecule_bcut2d_logphi,
    "molecule_e_state_vsa6": molecule_e_state_vsa6,
    "molecule_mol_mr": molecule_mol_mr,
    "molecule_fr_thiocyan": molecule_fr_thiocyan,
    "molecule_peoe_vsa7": molecule_peoe_vsa7,
    "molecule_fr_nh0": molecule_fr_nh0,
    "molecule_max_partial_charge": molecule_max_partial_charge,
    "molecule_smr_vsa6": molecule_smr_vsa6,
    "molecule_fr_priamide": molecule_fr_priamide,
    "molecule_fr_thiazole": molecule_fr_thiazole,
    "molecule_fr_al_oh": molecule_fr_al_oh,
    "molecule_e_state_vsa2": molecule_e_state_vsa2,
    "molecule_min_abs_partial_charge": molecule_min_abs_partial_charge,
    "molecule_slog_p_vsa11": molecule_slog_p_vsa11,
    "molecule_fr_azo": molecule_fr_azo,
    "molecule_smr_vsa5": molecule_smr_vsa5,
    "molecule_fr_amide": molecule_fr_amide,
    "molecule_nhoh_count": molecule_nhoh_count,
    "molecule_fp_density_morgan1": molecule_fp_density_morgan1,
    "molecule_fr_benzene": molecule_fr_benzene,
    "molecule_fr_hdrzone": molecule_fr_hdrzone,
    "molecule_slog_p_vsa4": molecule_slog_p_vsa4,
    "molecule_bcut2d_logplow": molecule_bcut2d_logplow,
    "molecule_fr_ketone_topliss": molecule_fr_ketone_topliss,
    "molecule_whim": molecule_whim,
    "molecule_get_usrcat": molecule_get_usrcat,
    "molecule_morse": molecule_morse,
    "molecule_rdf": molecule_rdf,
    "molecule_getaway": molecule_getaway,
    "molecule_autocorr3d": molecule_autocorr3d,
    "molecule_get_usr": molecule_get_usr,
    "molecule_crippen_descriptors": molecule_crippen_descriptors,
    "molecule_bcut2d": molecule_bcut2d,
    "molecule_autocorr2d": molecule_autocorr2d,
    "molecule_get_hashed_topological_torsion_fingerprint_as_bit_vect": molecule_get_hashed_topological_torsion_fingerprint_as_bit_vect,
    "molecule_layered_fingerprint": molecule_layered_fingerprint,
    "molecule_get_maccs_keys_fingerprint": molecule_get_maccs_keys_fingerprint,
    "molecule_get_hashed_atom_pair_fingerprint_as_bit_vect": molecule_get_hashed_atom_pair_fingerprint_as_bit_vect,
    "molecule_pattern_fingerprint": molecule_pattern_fingerprint,
    "molecule_rdk_fingerprint": molecule_rdk_fingerprint,
}


def main():
    from rdkit import Chem

    testmol = mol_from_smiles("c1ccccc1")
    return list(zip(_available_featurizer, [f(testmol) for f in _available_featurizer]))


if __name__ == "__main__":
    main()
