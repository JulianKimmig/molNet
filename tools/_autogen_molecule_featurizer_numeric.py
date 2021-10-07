from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,SingleValueMoleculeFeaturizer
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import (PEOE_VSA3,fr_bicyclic,fr_tetrazole,fr_SH,fr_oxime,PEOE_VSA2,fr_benzene,SlogP_VSA2,Chi3n,SlogP_VSA7,NumSaturatedHeterocycles,HeavyAtomMolWt,VSA_EState7,fr_thiophene,MinAbsEStateIndex,NumHAcceptors,SMR_VSA4,VSA_EState9,EState_VSA3,fr_isothiocyan,fr_piperzine,PEOE_VSA1,Chi2v,fr_nitro,fr_C_O_noCOO,VSA_EState2,fr_quatN,SlogP_VSA4,PEOE_VSA13,fr_hdrzine,fr_phenol,RingCount,PEOE_VSA10,fr_Imine,fr_Al_COO,fr_ester,fr_sulfide,fr_Ar_COO,fr_amide,BCUT2D_CHGHI,fr_sulfonamd,BCUT2D_LOGPLOW,SMR_VSA8,BCUT2D_LOGPHI,Chi0v,fr_alkyl_carbamate,SlogP_VSA11,Chi4v,fr_thiocyan,fr_Ndealkylation1,EState_VSA10,EState_VSA11,FpDensityMorgan1,NumAliphaticHeterocycles,fr_ketone_Topliss,NumAliphaticRings,fr_benzodiazepine,fr_C_S,fr_epoxide,Chi0n,SlogP_VSA1,fr_nitroso,PEOE_VSA4,SMR_VSA10,SlogP_VSA10,MinEStateIndex,VSA_EState4,fr_aniline,fr_furan,Kappa1,TPSA,MaxAbsEStateIndex,EState_VSA6,MinAbsPartialCharge,Chi2n,fr_morpholine,SlogP_VSA6,NumAromaticCarbocycles,fr_diazo,fr_hdrzone,NumAromaticRings,fr_nitro_arom,NumAromaticHeterocycles,NHOHCount,fr_guanido,EState_VSA9,NumSaturatedRings,PEOE_VSA11,SMR_VSA7,PEOE_VSA7,SMR_VSA1,fr_phenol_noOrthoHbond,Ipc,fr_prisulfonamd,Chi3v,fr_term_acetylene,fr_halogen,fr_dihydropyridine,PEOE_VSA5,fr_piperdine,SlogP_VSA5,LabuteASA,Chi1v,VSA_EState6,EState_VSA8,SMR_VSA9,fr_oxazole,fr_alkyl_halide,fr_Nhpyrrole,Chi0,NumSaturatedCarbocycles,HallKierAlpha,fr_Ndealkylation2,NumHeteroatoms,fr_para_hydroxylation,SMR_VSA5,VSA_EState5,fr_ArN,fr_ether,fr_pyridine,fr_HOCCN,fr_sulfone,MaxAbsPartialCharge,NumAliphaticCarbocycles,Chi4n,fr_isocyan,fr_Al_OH,MaxPartialCharge,fr_NH2,VSA_EState3,fr_lactone,Kappa2,SMR_VSA6,Chi1n,BCUT2D_MWLOW,fr_nitrile,BCUT2D_MWHI,EState_VSA5,ExactMolWt,PEOE_VSA8,MolMR,fr_amidine,VSA_EState1,SlogP_VSA8,fr_azo,FractionCSP3,NumRotatableBonds,fr_NH0,PEOE_VSA6,SlogP_VSA12,BalabanJ,SlogP_VSA9,fr_aryl_methyl,qed,BCUT2D_MRHI,Chi1,FpDensityMorgan3,fr_lactam,fr_urea,FpDensityMorgan2,fr_priamide,NumValenceElectrons,PEOE_VSA9,NumHDonors,fr_methoxy,EState_VSA2,HeavyAtomCount,fr_azide,VSA_EState8,BCUT2D_CHGLO,NumRadicalElectrons,fr_nitro_arom_nonortho,fr_Ar_NH,SlogP_VSA3,EState_VSA4,VSA_EState10,MinPartialCharge,fr_aldehyde,BertzCT,Kappa3,fr_thiazole,fr_imidazole,fr_imide,fr_N_O,fr_allylic_oxid,fr_phos_ester,BCUT2D_MRLOW,EState_VSA7,fr_barbitur,fr_COO2,PEOE_VSA14,NOCount,MaxEStateIndex,fr_NH1,SMR_VSA3,fr_C_O,EState_VSA1,fr_ketone,fr_unbrch_alkane,fr_Ar_N,SMR_VSA2,fr_Ar_OH,fr_Al_OH_noTert,fr_phos_acid,fr_COO,PEOE_VSA12,MolLogP,MolWt,)
from rdkit.Chem.rdMolDescriptors import (CalcNumAromaticHeterocycles,CalcExactMolWt,CalcNumLipinskiHBD,CalcNPR2,CalcChi4v,CalcNumSaturatedHeterocycles,CalcChi2v,CalcRadiusOfGyration,CalcKappa1,CalcNumSpiroAtoms,CalcChi3v,CalcPBF,CalcNumAliphaticRings,CalcNumAromaticCarbocycles,CalcChi0n,CalcFractionCSP3,CalcChi1n,CalcKappa2,CalcNumSaturatedCarbocycles,CalcNumLipinskiHBA,CalcChi0v,CalcPMI2,CalcHallKierAlpha,CalcNumSaturatedRings,CalcNumRotatableBonds,CalcPMI3,CalcNumHeteroatoms,CalcLabuteASA,CalcNumAromaticRings,CalcChi3n,CalcPMI1,CalcEccentricity,CalcPhi,CalcNumRings,CalcNumAmideBonds,CalcNumAliphaticCarbocycles,CalcChi4n,CalcKappa3,CalcNumHeterocycles,CalcInertialShapeFactor,CalcTPSA,CalcChi1v,CalcNumBridgeheadAtoms,CalcAsphericity,CalcSpherocityIndex,CalcNumHBD,CalcChi2n,CalcNumHBA,CalcNPR1,CalcNumAliphaticHeterocycles,)
from rdkit.Chem.GraphDescriptors import (Ipc,Chi3n,Chi3v,Chi0n,Chi1v,BalabanJ,Chi1,Chi2v,Chi0,HallKierAlpha,Kappa1,Chi2n,BertzCT,Kappa3,Chi4n,Kappa2,Chi0v,Chi1n,Chi4v,)
from rdkit.Chem.Descriptors3D import (NPR1,NPR2,PMI2,PMI1,Eccentricity,PMI3,InertialShapeFactor,RadiusOfGyration,Asphericity,SpherocityIndex,)
from rdkit.Chem.rdmolops import (GetFormalCharge,GetSSSR,)


class fr_nitro_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_nitro)

class PEOE_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA8)

class fr_Imine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Imine)

class PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcPMI3)

class NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumRotatableBonds)

class Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi0v)

class fr_quatN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_quatN)

class fr_Ndealkylation2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ndealkylation2)

class EState_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA8)

class fr_phos_ester_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_phos_ester)

class RingCount_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(RingCount)

class PEOE_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA5)

class EState_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA11)

class VSA_EState1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState1)

class fr_diazo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_diazo)

class Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa1)

class Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi4v)

class Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1)

class fr_ketone_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_ketone)

class fr_lactone_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_lactone)

class fr_thiazole_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_thiazole)

class MinAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MinAbsEStateIndex)

class fr_COO2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_COO2)

class SlogP_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA9)

class VSA_EState9_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState9)

class fr_isothiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_isothiocyan)

class Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi2n)

class fr_Al_OH_noTert_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Al_OH_noTert)

class FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcFractionCSP3)

class NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAliphaticHeterocycles)

class Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0)

class Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcKappa1)

class MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MaxAbsEStateIndex)

class PEOE_VSA14_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA14)

class PEOE_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA7)

class NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(NPR1)

class Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi3v)

class Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi4n)

class NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumSaturatedHeterocycles)

class Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi2v)

class Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi3n)

class fr_amidine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_amidine)

class Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcAsphericity)

class Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa2)

class BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BertzCT)

class MaxPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MaxPartialCharge)

class SlogP_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA12)

class VSA_EState2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState2)

class fr_HOCCN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_HOCCN)

class MinPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MinPartialCharge)

class Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1n)

class PEOE_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA6)

class EState_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA10)

class EState_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA7)

class fr_pyridine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_pyridine)

class BCUT2D_CHGHI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_CHGHI)

class PEOE_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA2)

class qed_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(qed)

class SMR_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA1)

class fr_lactam_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_lactam)

class fr_imidazole_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_imidazole)

class fr_Al_COO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Al_COO)

class Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0v)

class NumHBD_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumHBD)

class NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAliphaticRings)

class NumRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumRings)

class fr_nitroso_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_nitroso)

class fr_prisulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_prisulfonamd)

class SMR_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA2)

class Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi4n)

class InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(InertialShapeFactor)

class NumAmideBonds_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAmideBonds)

class NumHDonors_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumHDonors)

class EState_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA5)

class fr_methoxy_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_methoxy)

class NumHBA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumHBA)

class NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcNPR1)

class MaxAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MaxAbsPartialCharge)

class Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi2n)

class Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1v)

class fr_sulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_sulfonamd)

class PEOE_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA9)

class fr_Ndealkylation1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ndealkylation1)

class LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(LabuteASA)

class PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcPMI1)

class LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcLabuteASA)

class fr_imide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_imide)

class NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAromaticRings)

class Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Ipc)

class ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcExactMolWt)

class SMR_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA3)

class fr_C_S_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_C_S)

class NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumSaturatedRings)

class NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAliphaticRings)

class fr_NH0_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_NH0)

class fr_alkyl_halide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_alkyl_halide)

class fr_hdrzone_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_hdrzone)

class Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0n)

class NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAromaticCarbocycles)

class BCUT2D_MWLOW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_MWLOW)

class PEOE_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA10)

class PEOE_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA1)

class GetSSSR_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(GetSSSR)

class Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi4v)

class EState_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA3)

class Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Asphericity)

class Phi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcPhi)

class VSA_EState4_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState4)

class fr_ketone_Topliss_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_ketone_Topliss)

class NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumSaturatedHeterocycles)

class HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(HallKierAlpha)

class GetFormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(GetFormalCharge)

class VSA_EState5_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState5)

class fr_guanido_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_guanido)

class NumRadicalElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumRadicalElectrons)

class SMR_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA6)

class BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BalabanJ)

class TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(TPSA)

class Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi2n)

class fr_nitrile_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_nitrile)

class NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAliphaticCarbocycles)

class EState_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA6)

class EState_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA1)

class NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAromaticCarbocycles)

class NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAromaticHeterocycles)

class SlogP_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA6)

class MolWt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MolWt)

class ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(ExactMolWt)

class PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PMI1)

class RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcRadiusOfGyration)

class BCUT2D_MRLOW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_MRLOW)

class VSA_EState3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState3)

class fr_ether_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_ether)

class NumValenceElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumValenceElectrons)

class fr_N_O_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_N_O)

class NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcNPR2)

class Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0n)

class MolMR_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MolMR)

class Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Eccentricity)

class FpDensityMorgan3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(FpDensityMorgan3)

class NHOHCount_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NHOHCount)

class PEOE_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA12)

class fr_azide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_azide)

class Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi3n)

class Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcEccentricity)

class NOCount_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NOCount)

class PEOE_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA3)

class fr_NH1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_NH1)

class PEOE_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA11)

class NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumHeteroatoms)

class SlogP_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA10)

class Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0)

class fr_morpholine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_morpholine)

class EState_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA9)

class PBF_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcPBF)

class MinAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MinAbsPartialCharge)

class SlogP_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA11)

class fr_aniline_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_aniline)

class BCUT2D_MWHI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_MWHI)

class Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi4v)

class fr_piperdine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_piperdine)

class Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi3v)

class fr_Nhpyrrole_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Nhpyrrole)

class Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1)

class FpDensityMorgan1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(FpDensityMorgan1)

class fr_epoxide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_epoxide)

class SlogP_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA5)

class fr_hdrzine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_hdrzine)

class fr_Ar_N_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ar_N)

class SMR_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA5)

class SlogP_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA3)

class BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BertzCT)

class fr_C_O_noCOO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_C_O_noCOO)

class NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumSaturatedRings)

class SMR_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA10)

class NumSpiroAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumSpiroAtoms)

class Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi3v)

class NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumSaturatedCarbocycles)

class Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi0n)

class fr_oxime_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_oxime)

class Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi2v)

class BCUT2D_MRHI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_MRHI)

class SMR_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA4)

class PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcPMI2)

class fr_Ar_COO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ar_COO)

class HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcHallKierAlpha)

class fr_nitro_arom_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_nitro_arom)

class fr_SH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_SH)

class fr_piperzine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_piperzine)

class Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1n)

class NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAromaticRings)

class EState_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA2)

class Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi2v)

class fr_urea_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_urea)

class fr_benzene_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_benzene)

class fr_bicyclic_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_bicyclic)

class fr_ArN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_ArN)

class Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa2)

class fr_halogen_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_halogen)

class VSA_EState6_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState6)

class fr_Ar_OH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ar_OH)

class PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PMI3)

class VSA_EState8_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState8)

class BCUT2D_CHGLO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_CHGLO)

class fr_phos_acid_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_phos_acid)

class SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SpherocityIndex)

class NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumAliphaticHeterocycles)

class fr_NH2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_NH2)

class fr_furan_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_furan)

class fr_nitro_arom_nonortho_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_nitro_arom_nonortho)

class SlogP_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA4)

class NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumRotatableBonds)

class Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi1n)

class fr_thiophene_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_thiophene)

class PEOE_VSA13_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA13)

class Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcKappa2)

class HeavyAtomCount_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(HeavyAtomCount)

class fr_amide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_amide)

class fr_barbitur_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_barbitur)

class SlogP_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA8)

class RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(RadiusOfGyration)

class Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa3)

class Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi0v)

class NumBridgeheadAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumBridgeheadAtoms)

class NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAromaticHeterocycles)

class Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Ipc)

class fr_C_O_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_C_O)

class fr_aldehyde_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_aldehyde)

class Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcKappa3)

class EState_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(EState_VSA4)

class NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumAliphaticCarbocycles)

class fr_tetrazole_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_tetrazole)

class fr_oxazole_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_oxazole)

class fr_benzodiazepine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_benzodiazepine)

class NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(NPR2)

class FpDensityMorgan2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(FpDensityMorgan2)

class NumHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumHeterocycles)

class NumLipinskiHBD_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumLipinskiHBD)

class Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi1v)

class fr_Ar_NH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Ar_NH)

class fr_aryl_methyl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_aryl_methyl)

class Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcChi1v)

class fr_Al_OH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_Al_OH)

class SMR_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA9)

class HeavyAtomMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(HeavyAtomMolWt)

class SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcSpherocityIndex)

class fr_allylic_oxid_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_allylic_oxid)

class VSA_EState7_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState7)

class SMR_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA8)

class MolLogP_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MolLogP)

class fr_phenol_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_phenol)

class SlogP_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA1)

class fr_azo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_azo)

class BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BalabanJ)

class PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PMI2)

class fr_isocyan_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_isocyan)

class TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcTPSA)

class InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(CalcInertialShapeFactor)

class Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa1)

class MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MinEStateIndex)

class MaxEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(MaxEStateIndex)

class fr_phenol_noOrthoHbond_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_phenol_noOrthoHbond)

class NumLipinskiHBA_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(CalcNumLipinskiHBA)

class fr_sulfide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_sulfide)

class Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Kappa3)

class fr_alkyl_carbamate_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_alkyl_carbamate)

class SlogP_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA7)

class fr_thiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_thiocyan)

class fr_term_acetylene_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_term_acetylene)

class FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(FractionCSP3)

class fr_ester_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_ester)

class NumHAcceptors_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumHAcceptors)

class fr_unbrch_alkane_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_unbrch_alkane)

class fr_sulfone_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_sulfone)

class HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(HallKierAlpha)

class SMR_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SMR_VSA7)

class BCUT2D_LOGPLOW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_LOGPLOW)

class BCUT2D_LOGPHI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(BCUT2D_LOGPHI)

class VSA_EState10_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(VSA_EState10)

class fr_para_hydroxylation_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_para_hydroxylation)

class NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumSaturatedCarbocycles)

class SlogP_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(SlogP_VSA2)

class fr_COO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_COO)

class NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(NumHeteroatoms)

class Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi3n)

class PEOE_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(PEOE_VSA4)

class Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.float64
    featurize=staticmethod(Chi4n)

class fr_dihydropyridine_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_dihydropyridine)

class fr_priamide_Featurizer(SingleValueMoleculeFeaturizer):
    dtype=np.int64
    featurize=staticmethod(fr_priamide)

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
    