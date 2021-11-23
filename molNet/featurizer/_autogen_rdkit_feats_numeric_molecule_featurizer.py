import numpy as np
from rdkit.Chem import (GetSSSR, GetFormalCharge)
from rdkit.Chem.AllChem import (CalcNumLipinskiHBA, CalcNumHeterocycles, CalcRadiusOfGyration, CalcPBF, CalcExactMolWt,
                                CalcNumAmideBonds, CalcChi0v, UFFOptimizeMolecule, CalcChi3v, CalcChi4v,
                                ComputeMolVolume, MMFFHasAllMoleculeParams, CalcEccentricity, CalcNumLipinskiHBD,
                                CalcPMI3, UFFHasAllMoleculeParams, CalcNumHBA, CalcAsphericity, CalcPhi,
                                CalcNumBridgeheadAtoms, CalcKappa2, EmbedMolecule, CalcChi4n, CalcNumRings, CalcNPR1,
                                Compute2DCoords, CalcInertialShapeFactor, CalcSpherocityIndex, CalcNumHBD, CalcNPR2,
                                CalcPMI2, CalcNumSpiroAtoms, CalcChi1v, CalcPMI1, MMFFOptimizeMolecule)
from rdkit.Chem.Crippen import (MolLogP, MolMR)
from rdkit.Chem.Descriptors import (EState_VSA11, HeavyAtomMolWt, BertzCT, Chi0n, EState_VSA4, EState_VSA5,
                                    BCUT2D_CHGHI, FpDensityMorgan2, Chi1, FpDensityMorgan1, NumRadicalElectrons,
                                    VSA_EState9, EState_VSA10, MaxAbsPartialCharge, Chi1n, EState_VSA7, EState_VSA6,
                                    HallKierAlpha, EState_VSA9, MolWt, BCUT2D_MWLOW, Chi2n, Ipc, EState_VSA1, Chi2v,
                                    BCUT2D_MRHI, VSA_EState3, Chi0, EState_VSA8, BCUT2D_LOGPHI, MinPartialCharge,
                                    MaxPartialCharge, BCUT2D_MWHI, BCUT2D_MRLOW, VSA_EState2, Kappa3, VSA_EState8,
                                    VSA_EState1, MinAbsPartialCharge, BCUT2D_CHGLO, NumValenceElectrons, VSA_EState6,
                                    VSA_EState7, FpDensityMorgan3, EState_VSA3, Chi3n, VSA_EState4, BalabanJ,
                                    VSA_EState5, VSA_EState10, Kappa1, EState_VSA2, BCUT2D_LOGPLOW)
from rdkit.Chem.EState import (MinAbsEStateIndex, MaxEStateIndex, MaxAbsEStateIndex, MinEStateIndex)
from rdkit.Chem.EnumerateStereoisomers import (GetStereoisomerCount)
from rdkit.Chem.Fragments import (fr_ketone, fr_aryl_methyl, fr_azo, fr_C_O_noCOO, fr_barbitur, fr_urea,
                                  fr_Ndealkylation1, fr_nitro, fr_alkyl_halide, fr_isothiocyan, fr_benzene,
                                  fr_unbrch_alkane, fr_nitro_arom_nonortho, fr_sulfone, fr_morpholine, fr_C_O, fr_ArN,
                                  fr_Imine, fr_imide, fr_prisulfonamd, fr_bicyclic, fr_phos_acid, fr_SH, fr_sulfonamd,
                                  fr_nitrile, fr_tetrazole, fr_phenol, fr_ester, fr_HOCCN, fr_allylic_oxid,
                                  fr_dihydropyridine, fr_COO, fr_nitroso, fr_Ar_N, fr_piperdine, fr_epoxide,
                                  fr_thiocyan, fr_oxime, fr_hdrzone, fr_benzodiazepine, fr_priamide,
                                  fr_phenol_noOrthoHbond, fr_N_O, fr_ketone_Topliss, fr_NH1, fr_Nhpyrrole, fr_aniline,
                                  fr_pyridine, fr_furan, fr_COO2, fr_NH2, fr_piperzine, fr_hdrzine, fr_Ndealkylation2,
                                  fr_imidazole, fr_thiazole, fr_isocyan, fr_diazo, fr_azide, fr_quatN, fr_ether,
                                  fr_Al_OH, fr_C_S, fr_Ar_NH, fr_term_acetylene, fr_Ar_OH, fr_Al_OH_noTert,
                                  fr_thiophene, fr_alkyl_carbamate, fr_amidine, fr_lactone, fr_methoxy, fr_Al_COO,
                                  fr_phos_ester, fr_Ar_COO, fr_NH0, fr_halogen, fr_sulfide, fr_aldehyde, fr_oxazole,
                                  fr_para_hydroxylation, fr_lactam, fr_guanido, fr_amide, fr_nitro_arom)
from rdkit.Chem.Lipinski import (NumAromaticHeterocycles, FractionCSP3, NumHDonors, RingCount, NumAliphaticHeterocycles,
                                 NumAromaticCarbocycles, NumHeteroatoms, NumRotatableBonds, NHOHCount,
                                 NumAliphaticCarbocycles, NumHAcceptors, HeavyAtomCount, NumAliphaticRings,
                                 NumSaturatedCarbocycles, NumAromaticRings, NumSaturatedHeterocycles, NOCount,
                                 NumSaturatedRings)
from rdkit.Chem.MolSurf import (PEOE_VSA1, pyLabuteASA, SlogP_VSA11, PEOE_VSA6, SMR_VSA7, SlogP_VSA12, PEOE_VSA2,
                                SlogP_VSA7, PEOE_VSA8, PEOE_VSA7, SlogP_VSA1, SMR_VSA3, SlogP_VSA3, LabuteASA, SMR_VSA9,
                                PEOE_VSA10, SMR_VSA6, SlogP_VSA5, PEOE_VSA9, SMR_VSA1, PEOE_VSA11, PEOE_VSA5,
                                SlogP_VSA10, SlogP_VSA9, PEOE_VSA4, TPSA, SlogP_VSA2, SMR_VSA8, SlogP_VSA8, SlogP_VSA6,
                                PEOE_VSA12, PEOE_VSA3, PEOE_VSA13, SMR_VSA10, SMR_VSA2, SMR_VSA4, SMR_VSA5, PEOE_VSA14,
                                SlogP_VSA4)
from rdkit.Chem.QED import (weights_mean, qed, default, weights_max, weights_none)

from molNet.featurizer._molecule_featurizer import (SingleValueMoleculeFeaturizer)
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization


class Molecule_NumAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumAtoms
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumAtoms()


class Molecule_NumBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumBonds
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumBonds()


class Molecule_NumHeavyAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=GetNumHeavyAtoms
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumHeavyAtoms()


class Molecule_fr_thiophene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiophene
    dtype = np.int32
    featurize = staticmethod(fr_thiophene)


class Molecule_VSA_EState8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState8
    dtype = np.float32
    featurize = staticmethod(VSA_EState8)


class Molecule_fr_amide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_amide
    dtype = np.int32
    featurize = staticmethod(fr_amide)


class Molecule_NumAromaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumAromaticCarbocycles)


class Molecule_fr_urea_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_urea
    dtype = np.int32
    featurize = staticmethod(fr_urea)


class Molecule_BalabanJ_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BalabanJ
    dtype = np.float32
    featurize = staticmethod(BalabanJ)


class Molecule_default_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.default
    dtype = np.float32
    featurize = staticmethod(default)


class Molecule_TPSA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.TPSA
    dtype = np.float32
    featurize = staticmethod(TPSA)


class Molecule_Chi4n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi4n
    dtype = np.float32
    featurize = staticmethod(CalcChi4n)


class Molecule_fr_Ar_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_OH
    dtype = np.int32
    featurize = staticmethod(fr_Ar_OH)


class Molecule_VSA_EState9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState9
    dtype = np.float32
    featurize = staticmethod(VSA_EState9)


class Molecule_BCUT2D_LOGPLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_LOGPLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPLOW)


class Molecule_InertialShapeFactor_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcInertialShapeFactor
    dtype = np.float32
    featurize = staticmethod(CalcInertialShapeFactor)


class Molecule_fr_nitrile_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitrile
    dtype = np.int32
    featurize = staticmethod(fr_nitrile)


class Molecule_fr_NH1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH1
    dtype = np.int32
    featurize = staticmethod(fr_NH1)


class Molecule_fr_lactone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_lactone
    dtype = np.int32
    featurize = staticmethod(fr_lactone)


class Molecule_SlogP_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA1
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA1)


class Molecule_fr_barbitur_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_barbitur
    dtype = np.int32
    featurize = staticmethod(fr_barbitur)


class Molecule_NPR2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNPR2
    dtype = np.float32
    featurize = staticmethod(CalcNPR2)


class Molecule_VSA_EState10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState10
    dtype = np.float32
    featurize = staticmethod(VSA_EState10)


class Molecule_fr_phos_acid_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phos_acid
    dtype = np.int32
    featurize = staticmethod(fr_phos_acid)


class Molecule_fr_Ndealkylation2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ndealkylation2
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation2)


class Molecule_MaxEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MaxEStateIndex
    dtype = np.float32
    featurize = staticmethod(MaxEStateIndex)


class Molecule_fr_prisulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_prisulfonamd
    dtype = np.int32
    featurize = staticmethod(fr_prisulfonamd)


class Molecule_fr_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_COO
    dtype = np.int32
    featurize = staticmethod(fr_COO)


class Molecule_fr_benzodiazepine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_benzodiazepine
    dtype = np.int32
    featurize = staticmethod(fr_benzodiazepine)


class Molecule_PEOE_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA12
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA12)


class Molecule_fr_nitro_arom_nonortho_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro_arom_nonortho
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom_nonortho)


class Molecule_fr_halogen_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_halogen
    dtype = np.int32
    featurize = staticmethod(fr_halogen)


class Molecule_PEOE_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA6
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA6)


class Molecule_EState_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA2
    dtype = np.float32
    featurize = staticmethod(EState_VSA2)


class Molecule_PMI3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI3
    dtype = np.float32
    featurize = staticmethod(CalcPMI3)


class Molecule_EState_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA7
    dtype = np.float32
    featurize = staticmethod(EState_VSA7)


class Molecule_NHOHCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NHOHCount
    dtype = np.int32
    featurize = staticmethod(NHOHCount)


class Molecule_SlogP_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA11
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA11)


class Molecule_MMFFOptimizeMolecule_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MMFFOptimizeMolecule
    dtype = np.int32
    featurize = staticmethod(MMFFOptimizeMolecule)


class Molecule_NumLipinskiHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumLipinskiHBA
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBA)


class Molecule_SlogP_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA4
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA4)


class Molecule_fr_Ndealkylation1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ndealkylation1
    dtype = np.int32
    featurize = staticmethod(fr_Ndealkylation1)


class Molecule_Eccentricity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcEccentricity
    dtype = np.float32
    featurize = staticmethod(CalcEccentricity)


class Molecule_SMR_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA3
    dtype = np.float32
    featurize = staticmethod(SMR_VSA3)


class Molecule_SlogP_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA2
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA2)


class Molecule_HeavyAtomCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.HeavyAtomCount
    dtype = np.int32
    featurize = staticmethod(HeavyAtomCount)


class Molecule_fr_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ester
    dtype = np.int32
    featurize = staticmethod(fr_ester)


class Molecule_PEOE_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA10
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA10)


class Molecule_BCUT2D_CHGLO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_CHGLO
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGLO)


class Molecule_fr_hdrzone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_hdrzone
    dtype = np.int32
    featurize = staticmethod(fr_hdrzone)


class Molecule_SSSR_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetSSSR
    dtype = np.int32
    featurize = staticmethod(GetSSSR)


class Molecule_Ipc_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Ipc
    dtype = np.float32
    featurize = staticmethod(Ipc)


class Molecule_NumHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHBD
    dtype = np.int32
    featurize = staticmethod(CalcNumHBD)


class Molecule_fr_C_O_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_O
    dtype = np.int32
    featurize = staticmethod(fr_C_O)


class Molecule_Asphericity_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAsphericity
    dtype = np.float32
    featurize = staticmethod(CalcAsphericity)


class Molecule_fr_C_S_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_S
    dtype = np.int32
    featurize = staticmethod(fr_C_S)


class Molecule_SMR_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA8
    dtype = np.float32
    featurize = staticmethod(SMR_VSA8)


class Molecule_MMFFHasAllMoleculeParams_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MMFFHasAllMoleculeParams
    dtype = bool
    featurize = staticmethod(MMFFHasAllMoleculeParams)


class Molecule_fr_thiazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiazole
    dtype = np.int32
    featurize = staticmethod(fr_thiazole)


class Molecule_NumAromaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumAromaticHeterocycles)


class Molecule_fr_Imine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Imine
    dtype = np.int32
    featurize = staticmethod(fr_Imine)


class Molecule_SlogP_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA8
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA8)


class Molecule_Chi2n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi2n
    dtype = np.float32
    featurize = staticmethod(Chi2n)


class Molecule_fr_nitro_arom_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro_arom
    dtype = np.int32
    featurize = staticmethod(fr_nitro_arom)


class Molecule_SlogP_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA10
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA10)


class Molecule_SMR_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA5
    dtype = np.float32
    featurize = staticmethod(SMR_VSA5)


class Molecule_fr_nitroso_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitroso
    dtype = np.int32
    featurize = staticmethod(fr_nitroso)


class Molecule_MaxAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MaxAbsEStateIndex
    dtype = np.float32
    featurize = staticmethod(MaxAbsEStateIndex)


class Molecule_NumRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumRings
    dtype = np.int32
    featurize = staticmethod(CalcNumRings)


class Molecule_PEOE_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA5
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA5)


class Molecule_NumHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHeterocycles
    dtype = np.int32
    featurize = staticmethod(CalcNumHeterocycles)


class Molecule_StereoisomerCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EnumerateStereoisomers.GetStereoisomerCount
    dtype = np.int32
    featurize = staticmethod(GetStereoisomerCount)


class Molecule_NOCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NOCount
    dtype = np.int32
    featurize = staticmethod(NOCount)


class Molecule_NumHDonors_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHDonors
    dtype = np.int32
    featurize = staticmethod(NumHDonors)


class Molecule_BCUT2D_CHGHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_CHGHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_CHGHI)


class Molecule_SMR_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA6
    dtype = np.float32
    featurize = staticmethod(SMR_VSA6)


class Molecule_EState_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA10
    dtype = np.float32
    featurize = staticmethod(EState_VSA10)


class Molecule_MaxAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MaxAbsPartialCharge
    dtype = np.float32
    featurize = staticmethod(MaxAbsPartialCharge)


class Molecule_VSA_EState6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState6
    dtype = np.float32
    featurize = staticmethod(VSA_EState6)


class Molecule_BCUT2D_MWHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MWHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWHI)


class Molecule_NumSpiroAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumSpiroAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumSpiroAtoms)


class Molecule_EState_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA9
    dtype = np.float32
    featurize = staticmethod(EState_VSA9)


class Molecule_fr_ether_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ether
    dtype = np.int32
    featurize = staticmethod(fr_ether)


class Molecule_fr_term_acetylene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_term_acetylene
    dtype = np.int32
    featurize = staticmethod(fr_term_acetylene)


class Molecule_PEOE_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA8
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA8)


class Molecule_Compute2DCoords_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.Compute2DCoords
    dtype = np.int32
    featurize = staticmethod(Compute2DCoords)


class Molecule_fr_diazo_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_diazo
    dtype = np.int32
    featurize = staticmethod(fr_diazo)


class Molecule_fr_Ar_NH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_NH
    dtype = np.int32
    featurize = staticmethod(fr_Ar_NH)


class Molecule_fr_isocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_isocyan
    dtype = np.int32
    featurize = staticmethod(fr_isocyan)


class Molecule_fr_priamide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_priamide
    dtype = np.int32
    featurize = staticmethod(fr_priamide)


class Molecule_BCUT2D_MRLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MRLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRLOW)


class Molecule_fr_COO2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_COO2
    dtype = np.int32
    featurize = staticmethod(fr_COO2)


class Molecule_MinAbsPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MinAbsPartialCharge
    dtype = np.float32
    featurize = staticmethod(MinAbsPartialCharge)


class Molecule_weights_mean_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_mean
    dtype = np.float32
    featurize = staticmethod(weights_mean)


class Molecule_NumAmideBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumAmideBonds
    dtype = np.int32
    featurize = staticmethod(CalcNumAmideBonds)


class Molecule_fr_Al_OH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_OH
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH)


class Molecule_fr_furan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_furan
    dtype = np.int32
    featurize = staticmethod(fr_furan)


class Molecule_fr_NH0_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH0
    dtype = np.int32
    featurize = staticmethod(fr_NH0)


class Molecule_fr_quatN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_quatN
    dtype = np.int32
    featurize = staticmethod(fr_quatN)


class Molecule_NumSaturatedCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumSaturatedCarbocycles)


class Molecule_SMR_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA9
    dtype = np.float32
    featurize = staticmethod(SMR_VSA9)


class Molecule_MinAbsEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MinAbsEStateIndex
    dtype = np.float32
    featurize = staticmethod(MinAbsEStateIndex)


class Molecule_LabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.LabuteASA
    dtype = np.float32
    featurize = staticmethod(LabuteASA)


class Molecule_SlogP_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA3
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA3)


class Molecule_SlogP_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA5
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA5)


class Molecule_SlogP_VSA12_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA12
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA12)


class Molecule_BCUT2D_MRHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MRHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MRHI)


class Molecule_HeavyAtomMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.HeavyAtomMolWt
    dtype = np.float32
    featurize = staticmethod(HeavyAtomMolWt)


class Molecule_SpherocityIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcSpherocityIndex
    dtype = np.float32
    featurize = staticmethod(CalcSpherocityIndex)


class Molecule_fr_piperzine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_piperzine
    dtype = np.int32
    featurize = staticmethod(fr_piperzine)


class Molecule_Chi4v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi4v
    dtype = np.float32
    featurize = staticmethod(CalcChi4v)


class Molecule_SMR_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA7
    dtype = np.float32
    featurize = staticmethod(SMR_VSA7)


class Molecule_PEOE_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA7
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA7)


class Molecule_EState_VSA8_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA8
    dtype = np.float32
    featurize = staticmethod(EState_VSA8)


class Molecule_MinEStateIndex_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.MinEStateIndex
    dtype = np.float32
    featurize = staticmethod(MinEStateIndex)


class Molecule_Chi3n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi3n
    dtype = np.float32
    featurize = staticmethod(Chi3n)


class Molecule_fr_methoxy_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_methoxy
    dtype = np.int32
    featurize = staticmethod(fr_methoxy)


class Molecule_MinPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MinPartialCharge
    dtype = np.float32
    featurize = staticmethod(MinPartialCharge)


class Molecule_Chi0v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi0v
    dtype = np.float32
    featurize = staticmethod(CalcChi0v)


class Molecule_EState_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA4
    dtype = np.float32
    featurize = staticmethod(EState_VSA4)


class Molecule_fr_oxazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_oxazole
    dtype = np.int32
    featurize = staticmethod(fr_oxazole)


class Molecule_pyLabuteASA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.pyLabuteASA
    dtype = np.float32
    featurize = staticmethod(pyLabuteASA)


class Molecule_fr_N_O_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_N_O
    dtype = np.int32
    featurize = staticmethod(fr_N_O)


class Molecule_Chi0_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi0
    dtype = np.float32
    featurize = staticmethod(Chi0)


class Molecule_fr_ketone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ketone
    dtype = np.int32
    featurize = staticmethod(fr_ketone)


class Molecule_fr_aniline_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aniline
    dtype = np.int32
    featurize = staticmethod(fr_aniline)


class Molecule_EState_VSA5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA5
    dtype = np.float32
    featurize = staticmethod(EState_VSA5)


class Molecule_fr_Al_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_COO
    dtype = np.int32
    featurize = staticmethod(fr_Al_COO)


class Molecule_PEOE_VSA13_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA13
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA13)


class Molecule_fr_HOCCN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_HOCCN
    dtype = np.int32
    featurize = staticmethod(fr_HOCCN)


class Molecule_fr_piperdine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_piperdine
    dtype = np.int32
    featurize = staticmethod(fr_piperdine)


class Molecule_fr_thiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_thiocyan
    dtype = np.int32
    featurize = staticmethod(fr_thiocyan)


class Molecule_VSA_EState4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState4
    dtype = np.float32
    featurize = staticmethod(VSA_EState4)


class Molecule_fr_Ar_N_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_N
    dtype = np.int32
    featurize = staticmethod(fr_Ar_N)


class Molecule_NumHBA_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumHBA
    dtype = np.int32
    featurize = staticmethod(CalcNumHBA)


class Molecule_weights_none_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_none
    dtype = np.float32
    featurize = staticmethod(weights_none)


class Molecule_VSA_EState5_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState5
    dtype = np.float32
    featurize = staticmethod(VSA_EState5)


class Molecule_NumSaturatedHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumSaturatedHeterocycles)


class Molecule_fr_guanido_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_guanido
    dtype = np.int32
    featurize = staticmethod(fr_guanido)


class Molecule_fr_tetrazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_tetrazole
    dtype = np.int32
    featurize = staticmethod(fr_tetrazole)


class Molecule_fr_morpholine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_morpholine
    dtype = np.int32
    featurize = staticmethod(fr_morpholine)


class Molecule_NumLipinskiHBD_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumLipinskiHBD
    dtype = np.int32
    featurize = staticmethod(CalcNumLipinskiHBD)


class Molecule_ExactMolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcExactMolWt
    dtype = np.float32
    featurize = staticmethod(CalcExactMolWt)


class Molecule_SMR_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA1
    dtype = np.float32
    featurize = staticmethod(SMR_VSA1)


class Molecule_PEOE_VSA14_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA14
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA14)


class Molecule_NumHeteroatoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHeteroatoms
    dtype = np.int32
    featurize = staticmethod(NumHeteroatoms)


class Molecule_VSA_EState7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState7
    dtype = np.float32
    featurize = staticmethod(VSA_EState7)


class Molecule_fr_epoxide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_epoxide
    dtype = np.int32
    featurize = staticmethod(fr_epoxide)


class Molecule_PBF_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPBF
    dtype = np.float32
    featurize = staticmethod(CalcPBF)


class Molecule_SMR_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA2
    dtype = np.float32
    featurize = staticmethod(SMR_VSA2)


class Molecule_BertzCT_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BertzCT
    dtype = np.float32
    featurize = staticmethod(BertzCT)


class Molecule_NumSaturatedRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumSaturatedRings
    dtype = np.int32
    featurize = staticmethod(NumSaturatedRings)


class Molecule_fr_pyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_pyridine
    dtype = np.int32
    featurize = staticmethod(fr_pyridine)


class Molecule_NPR1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNPR1
    dtype = np.float32
    featurize = staticmethod(CalcNPR1)


class Molecule_fr_allylic_oxid_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_allylic_oxid
    dtype = np.int32
    featurize = staticmethod(fr_allylic_oxid)


class Molecule_Chi0n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi0n
    dtype = np.float32
    featurize = staticmethod(Chi0n)


class Molecule_NumAliphaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticRings
    dtype = np.int32
    featurize = staticmethod(NumAliphaticRings)


class Molecule_Chi1n_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1n
    dtype = np.float32
    featurize = staticmethod(Chi1n)


class Molecule_PEOE_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA3
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA3)


class Molecule_fr_sulfonamd_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfonamd
    dtype = np.int32
    featurize = staticmethod(fr_sulfonamd)


class Molecule_RingCount_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.RingCount
    dtype = np.int32
    featurize = staticmethod(RingCount)


class Molecule_fr_amidine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_amidine
    dtype = np.int32
    featurize = staticmethod(fr_amidine)


class Molecule_Kappa1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Kappa1
    dtype = np.float32
    featurize = staticmethod(Kappa1)


class Molecule_VSA_EState2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState2
    dtype = np.float32
    featurize = staticmethod(VSA_EState2)


class Molecule_fr_imide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_imide
    dtype = np.int32
    featurize = staticmethod(fr_imide)


class Molecule_fr_Al_OH_noTert_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Al_OH_noTert
    dtype = np.int32
    featurize = staticmethod(fr_Al_OH_noTert)


class Molecule_fr_Nhpyrrole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Nhpyrrole
    dtype = np.int32
    featurize = staticmethod(fr_Nhpyrrole)


class Molecule_NumAliphaticHeterocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticHeterocycles
    dtype = np.int32
    featurize = staticmethod(NumAliphaticHeterocycles)


class Molecule_fr_para_hydroxylation_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_para_hydroxylation
    dtype = np.int32
    featurize = staticmethod(fr_para_hydroxylation)


class Molecule_PEOE_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA1
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA1)


class Molecule_PEOE_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA4
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA4)


class Molecule_PEOE_VSA2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA2
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA2)


class Molecule_Chi3v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi3v
    dtype = np.float32
    featurize = staticmethod(CalcChi3v)


class Molecule_SlogP_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA6
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA6)


class Molecule_fr_ketone_Topliss_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ketone_Topliss
    dtype = np.int32
    featurize = staticmethod(fr_ketone_Topliss)


class Molecule_EState_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA11
    dtype = np.float32
    featurize = staticmethod(EState_VSA11)


class Molecule_fr_sulfone_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfone
    dtype = np.int32
    featurize = staticmethod(fr_sulfone)


class Molecule_Kappa2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcKappa2
    dtype = np.float32
    featurize = staticmethod(CalcKappa2)


class Molecule_fr_unbrch_alkane_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_unbrch_alkane
    dtype = np.int32
    featurize = staticmethod(fr_unbrch_alkane)


class Molecule_SMR_VSA4_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA4
    dtype = np.float32
    featurize = staticmethod(SMR_VSA4)


class Molecule_SlogP_VSA7_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA7
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA7)


class Molecule_fr_azo_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_azo
    dtype = np.int32
    featurize = staticmethod(fr_azo)


class Molecule_VSA_EState3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState3
    dtype = np.float32
    featurize = staticmethod(VSA_EState3)


class Molecule_fr_ArN_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_ArN
    dtype = np.int32
    featurize = staticmethod(fr_ArN)


class Molecule_FormalCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetFormalCharge
    dtype = np.int32
    featurize = staticmethod(GetFormalCharge)


class Molecule_Phi_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPhi
    dtype = np.float32
    featurize = staticmethod(CalcPhi)


class Molecule_NumValenceElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.NumValenceElectrons
    dtype = np.int32
    featurize = staticmethod(NumValenceElectrons)


class Molecule_BCUT2D_MWLOW_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_MWLOW
    dtype = np.float32
    featurize = staticmethod(BCUT2D_MWLOW)


class Molecule_EmbedMolecule_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.EmbedMolecule
    dtype = np.int32
    featurize = staticmethod(EmbedMolecule)


class Molecule_NumAromaticRings_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAromaticRings
    dtype = np.int32
    featurize = staticmethod(NumAromaticRings)


class Molecule_SMR_VSA10_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SMR_VSA10
    dtype = np.float32
    featurize = staticmethod(SMR_VSA10)


class Molecule_NumRotatableBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumRotatableBonds
    dtype = np.int32
    featurize = staticmethod(NumRotatableBonds)


class Molecule_MaxPartialCharge_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MaxPartialCharge
    dtype = np.float32
    featurize = staticmethod(MaxPartialCharge)


class Molecule_FpDensityMorgan2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan2
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan2)


class Molecule_UFFHasAllMoleculeParams_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.UFFHasAllMoleculeParams
    dtype = bool
    featurize = staticmethod(UFFHasAllMoleculeParams)


class Molecule_PMI1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI1
    dtype = np.float32
    featurize = staticmethod(CalcPMI1)


class Molecule_fr_lactam_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_lactam
    dtype = np.int32
    featurize = staticmethod(fr_lactam)


class Molecule_fr_phenol_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phenol
    dtype = np.int32
    featurize = staticmethod(fr_phenol)


class Molecule_VSA_EState1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.VSA_EState1
    dtype = np.float32
    featurize = staticmethod(VSA_EState1)


class Molecule_NumAliphaticCarbocycles_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumAliphaticCarbocycles
    dtype = np.int32
    featurize = staticmethod(NumAliphaticCarbocycles)


class Molecule_PEOE_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA9
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA9)


class Molecule_NumHAcceptors_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.NumHAcceptors
    dtype = np.int32
    featurize = staticmethod(NumHAcceptors)


class Molecule_NumBridgeheadAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcNumBridgeheadAtoms
    dtype = np.int32
    featurize = staticmethod(CalcNumBridgeheadAtoms)


class Molecule_SlogP_VSA9_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.SlogP_VSA9
    dtype = np.float32
    featurize = staticmethod(SlogP_VSA9)


class Molecule_UFFOptimizeMolecule_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.UFFOptimizeMolecule
    dtype = np.int32
    featurize = staticmethod(UFFOptimizeMolecule)


class Molecule_fr_bicyclic_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_bicyclic
    dtype = np.int32
    featurize = staticmethod(fr_bicyclic)


class Molecule_fr_isothiocyan_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_isothiocyan
    dtype = np.int32
    featurize = staticmethod(fr_isothiocyan)


class Molecule_Kappa3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Kappa3
    dtype = np.float32
    featurize = staticmethod(Kappa3)


class Molecule_FpDensityMorgan3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan3
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan3)


class Molecule_FractionCSP3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Lipinski.FractionCSP3
    dtype = np.float32
    featurize = staticmethod(FractionCSP3)


class Molecule_fr_aryl_methyl_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aryl_methyl
    dtype = np.int32
    featurize = staticmethod(fr_aryl_methyl)


class Molecule_fr_aldehyde_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_aldehyde
    dtype = np.int32
    featurize = staticmethod(fr_aldehyde)


class Molecule_PMI2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcPMI2
    dtype = np.float32
    featurize = staticmethod(CalcPMI2)


class Molecule_EState_VSA6_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA6
    dtype = np.float32
    featurize = staticmethod(EState_VSA6)


class Molecule_fr_hdrzine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_hdrzine
    dtype = np.int32
    featurize = staticmethod(fr_hdrzine)


class Molecule_Chi1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1
    dtype = np.float32
    featurize = staticmethod(Chi1)


class Molecule_HallKierAlpha_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.HallKierAlpha
    dtype = np.float32
    featurize = staticmethod(HallKierAlpha)


class Molecule_fr_sulfide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_sulfide
    dtype = np.int32
    featurize = staticmethod(fr_sulfide)


class Molecule_fr_imidazole_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_imidazole
    dtype = np.int32
    featurize = staticmethod(fr_imidazole)


class Molecule_fr_NH2_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_NH2
    dtype = np.int32
    featurize = staticmethod(fr_NH2)


class Molecule_MolWt_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.MolWt
    dtype = np.float32
    featurize = staticmethod(MolWt)


class Molecule_fr_oxime_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_oxime
    dtype = np.int32
    featurize = staticmethod(fr_oxime)


class Molecule_fr_SH_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_SH
    dtype = np.int32
    featurize = staticmethod(fr_SH)


class Molecule_fr_phos_ester_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phos_ester
    dtype = np.int32
    featurize = staticmethod(fr_phos_ester)


class Molecule_fr_benzene_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_benzene
    dtype = np.int32
    featurize = staticmethod(fr_benzene)


class Molecule_fr_nitro_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_nitro
    dtype = np.int32
    featurize = staticmethod(fr_nitro)


class Molecule_fr_phenol_noOrthoHbond_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_phenol_noOrthoHbond
    dtype = np.int32
    featurize = staticmethod(fr_phenol_noOrthoHbond)


class Molecule_EState_VSA1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA1
    dtype = np.float32
    featurize = staticmethod(EState_VSA1)


class Molecule_fr_azide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_azide
    dtype = np.int32
    featurize = staticmethod(fr_azide)


class Molecule_ComputeMolVolume_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.ComputeMolVolume
    dtype = np.float32
    featurize = staticmethod(ComputeMolVolume)


class Molecule_EState_VSA3_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.EState_VSA3
    dtype = np.float32
    featurize = staticmethod(EState_VSA3)


class Molecule_BCUT2D_LOGPHI_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.BCUT2D_LOGPHI
    dtype = np.float32
    featurize = staticmethod(BCUT2D_LOGPHI)


class Molecule_weights_max_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.weights_max
    dtype = np.float32
    featurize = staticmethod(weights_max)


class Molecule_fr_C_O_noCOO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_C_O_noCOO
    dtype = np.int32
    featurize = staticmethod(fr_C_O_noCOO)


class Molecule_PEOE_VSA11_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolSurf.PEOE_VSA11
    dtype = np.float32
    featurize = staticmethod(PEOE_VSA11)


class Molecule_RadiusOfGyration_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcRadiusOfGyration
    dtype = np.float32
    featurize = staticmethod(CalcRadiusOfGyration)


class Molecule_fr_dihydropyridine_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_dihydropyridine
    dtype = np.int32
    featurize = staticmethod(fr_dihydropyridine)


class Molecule_fr_alkyl_halide_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_alkyl_halide
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_halide)


class Molecule_FpDensityMorgan1_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.FpDensityMorgan1
    dtype = np.float32
    featurize = staticmethod(FpDensityMorgan1)


class Molecule_MolMR_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Crippen.MolMR
    dtype = np.float32
    featurize = staticmethod(MolMR)


class Molecule_Chi2v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi2v
    dtype = np.float32
    featurize = staticmethod(Chi2v)


class Molecule_fr_Ar_COO_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_Ar_COO
    dtype = np.int32
    featurize = staticmethod(fr_Ar_COO)


class Molecule_MolLogP_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Crippen.MolLogP
    dtype = np.float32
    featurize = staticmethod(MolLogP)


class Molecule_fr_alkyl_carbamate_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Fragments.fr_alkyl_carbamate
    dtype = np.int32
    featurize = staticmethod(fr_alkyl_carbamate)


class Molecule_qed_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.qed
    dtype = np.float32
    featurize = staticmethod(qed)


class Molecule_Chi1v_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcChi1v
    dtype = np.float32
    featurize = staticmethod(CalcChi1v)


class Molecule_NumRadicalElectrons_Featurizer(SingleValueMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.NumRadicalElectrons
    dtype = np.int32
    featurize = staticmethod(NumRadicalElectrons)


molecule_NumAtoms_featurizer = Molecule_NumAtoms_Featurizer()
molecule_NumBonds_featurizer = Molecule_NumBonds_Featurizer()
molecule_NumHeavyAtoms_featurizer = Molecule_NumHeavyAtoms_Featurizer()
molecule_fr_thiophene_featurizer = Molecule_fr_thiophene_Featurizer()
molecule_VSA_EState8_featurizer = Molecule_VSA_EState8_Featurizer()
molecule_fr_amide_featurizer = Molecule_fr_amide_Featurizer()
molecule_NumAromaticCarbocycles_featurizer = Molecule_NumAromaticCarbocycles_Featurizer()
molecule_fr_urea_featurizer = Molecule_fr_urea_Featurizer()
molecule_BalabanJ_featurizer = Molecule_BalabanJ_Featurizer()
molecule_default_featurizer = Molecule_default_Featurizer()
molecule_TPSA_featurizer = Molecule_TPSA_Featurizer()
molecule_Chi4n_featurizer = Molecule_Chi4n_Featurizer()
molecule_fr_Ar_OH_featurizer = Molecule_fr_Ar_OH_Featurizer()
molecule_VSA_EState9_featurizer = Molecule_VSA_EState9_Featurizer()
molecule_BCUT2D_LOGPLOW_featurizer = Molecule_BCUT2D_LOGPLOW_Featurizer()
molecule_InertialShapeFactor_featurizer = Molecule_InertialShapeFactor_Featurizer()
molecule_fr_nitrile_featurizer = Molecule_fr_nitrile_Featurizer()
molecule_fr_NH1_featurizer = Molecule_fr_NH1_Featurizer()
molecule_fr_lactone_featurizer = Molecule_fr_lactone_Featurizer()
molecule_SlogP_VSA1_featurizer = Molecule_SlogP_VSA1_Featurizer()
molecule_fr_barbitur_featurizer = Molecule_fr_barbitur_Featurizer()
molecule_NPR2_featurizer = Molecule_NPR2_Featurizer()
molecule_VSA_EState10_featurizer = Molecule_VSA_EState10_Featurizer()
molecule_fr_phos_acid_featurizer = Molecule_fr_phos_acid_Featurizer()
molecule_fr_Ndealkylation2_featurizer = Molecule_fr_Ndealkylation2_Featurizer()
molecule_MaxEStateIndex_featurizer = Molecule_MaxEStateIndex_Featurizer()
molecule_fr_prisulfonamd_featurizer = Molecule_fr_prisulfonamd_Featurizer()
molecule_fr_COO_featurizer = Molecule_fr_COO_Featurizer()
molecule_fr_benzodiazepine_featurizer = Molecule_fr_benzodiazepine_Featurizer()
molecule_PEOE_VSA12_featurizer = Molecule_PEOE_VSA12_Featurizer()
molecule_fr_nitro_arom_nonortho_featurizer = Molecule_fr_nitro_arom_nonortho_Featurizer()
molecule_fr_halogen_featurizer = Molecule_fr_halogen_Featurizer()
molecule_PEOE_VSA6_featurizer = Molecule_PEOE_VSA6_Featurizer()
molecule_EState_VSA2_featurizer = Molecule_EState_VSA2_Featurizer()
molecule_PMI3_featurizer = Molecule_PMI3_Featurizer()
molecule_EState_VSA7_featurizer = Molecule_EState_VSA7_Featurizer()
molecule_NHOHCount_featurizer = Molecule_NHOHCount_Featurizer()
molecule_SlogP_VSA11_featurizer = Molecule_SlogP_VSA11_Featurizer()
molecule_MMFFOptimizeMolecule_featurizer = Molecule_MMFFOptimizeMolecule_Featurizer()
molecule_NumLipinskiHBA_featurizer = Molecule_NumLipinskiHBA_Featurizer()
molecule_SlogP_VSA4_featurizer = Molecule_SlogP_VSA4_Featurizer()
molecule_fr_Ndealkylation1_featurizer = Molecule_fr_Ndealkylation1_Featurizer()
molecule_Eccentricity_featurizer = Molecule_Eccentricity_Featurizer()
molecule_SMR_VSA3_featurizer = Molecule_SMR_VSA3_Featurizer()
molecule_SlogP_VSA2_featurizer = Molecule_SlogP_VSA2_Featurizer()
molecule_HeavyAtomCount_featurizer = Molecule_HeavyAtomCount_Featurizer()
molecule_fr_ester_featurizer = Molecule_fr_ester_Featurizer()
molecule_PEOE_VSA10_featurizer = Molecule_PEOE_VSA10_Featurizer()
molecule_BCUT2D_CHGLO_featurizer = Molecule_BCUT2D_CHGLO_Featurizer()
molecule_fr_hdrzone_featurizer = Molecule_fr_hdrzone_Featurizer()
molecule_SSSR_featurizer = Molecule_SSSR_Featurizer()
molecule_Ipc_featurizer = Molecule_Ipc_Featurizer()
molecule_NumHBD_featurizer = Molecule_NumHBD_Featurizer()
molecule_fr_C_O_featurizer = Molecule_fr_C_O_Featurizer()
molecule_Asphericity_featurizer = Molecule_Asphericity_Featurizer()
molecule_fr_C_S_featurizer = Molecule_fr_C_S_Featurizer()
molecule_SMR_VSA8_featurizer = Molecule_SMR_VSA8_Featurizer()
molecule_MMFFHasAllMoleculeParams_featurizer = Molecule_MMFFHasAllMoleculeParams_Featurizer()
molecule_fr_thiazole_featurizer = Molecule_fr_thiazole_Featurizer()
molecule_NumAromaticHeterocycles_featurizer = Molecule_NumAromaticHeterocycles_Featurizer()
molecule_fr_Imine_featurizer = Molecule_fr_Imine_Featurizer()
molecule_SlogP_VSA8_featurizer = Molecule_SlogP_VSA8_Featurizer()
molecule_Chi2n_featurizer = Molecule_Chi2n_Featurizer()
molecule_fr_nitro_arom_featurizer = Molecule_fr_nitro_arom_Featurizer()
molecule_SlogP_VSA10_featurizer = Molecule_SlogP_VSA10_Featurizer()
molecule_SMR_VSA5_featurizer = Molecule_SMR_VSA5_Featurizer()
molecule_fr_nitroso_featurizer = Molecule_fr_nitroso_Featurizer()
molecule_MaxAbsEStateIndex_featurizer = Molecule_MaxAbsEStateIndex_Featurizer()
molecule_NumRings_featurizer = Molecule_NumRings_Featurizer()
molecule_PEOE_VSA5_featurizer = Molecule_PEOE_VSA5_Featurizer()
molecule_NumHeterocycles_featurizer = Molecule_NumHeterocycles_Featurizer()
molecule_StereoisomerCount_featurizer = Molecule_StereoisomerCount_Featurizer()
molecule_NOCount_featurizer = Molecule_NOCount_Featurizer()
molecule_NumHDonors_featurizer = Molecule_NumHDonors_Featurizer()
molecule_BCUT2D_CHGHI_featurizer = Molecule_BCUT2D_CHGHI_Featurizer()
molecule_SMR_VSA6_featurizer = Molecule_SMR_VSA6_Featurizer()
molecule_EState_VSA10_featurizer = Molecule_EState_VSA10_Featurizer()
molecule_MaxAbsPartialCharge_featurizer = Molecule_MaxAbsPartialCharge_Featurizer()
molecule_VSA_EState6_featurizer = Molecule_VSA_EState6_Featurizer()
molecule_BCUT2D_MWHI_featurizer = Molecule_BCUT2D_MWHI_Featurizer()
molecule_NumSpiroAtoms_featurizer = Molecule_NumSpiroAtoms_Featurizer()
molecule_EState_VSA9_featurizer = Molecule_EState_VSA9_Featurizer()
molecule_fr_ether_featurizer = Molecule_fr_ether_Featurizer()
molecule_fr_term_acetylene_featurizer = Molecule_fr_term_acetylene_Featurizer()
molecule_PEOE_VSA8_featurizer = Molecule_PEOE_VSA8_Featurizer()
molecule_Compute2DCoords_featurizer = Molecule_Compute2DCoords_Featurizer()
molecule_fr_diazo_featurizer = Molecule_fr_diazo_Featurizer()
molecule_fr_Ar_NH_featurizer = Molecule_fr_Ar_NH_Featurizer()
molecule_fr_isocyan_featurizer = Molecule_fr_isocyan_Featurizer()
molecule_fr_priamide_featurizer = Molecule_fr_priamide_Featurizer()
molecule_BCUT2D_MRLOW_featurizer = Molecule_BCUT2D_MRLOW_Featurizer()
molecule_fr_COO2_featurizer = Molecule_fr_COO2_Featurizer()
molecule_MinAbsPartialCharge_featurizer = Molecule_MinAbsPartialCharge_Featurizer()
molecule_weights_mean_featurizer = Molecule_weights_mean_Featurizer()
molecule_NumAmideBonds_featurizer = Molecule_NumAmideBonds_Featurizer()
molecule_fr_Al_OH_featurizer = Molecule_fr_Al_OH_Featurizer()
molecule_fr_furan_featurizer = Molecule_fr_furan_Featurizer()
molecule_fr_NH0_featurizer = Molecule_fr_NH0_Featurizer()
molecule_fr_quatN_featurizer = Molecule_fr_quatN_Featurizer()
molecule_NumSaturatedCarbocycles_featurizer = Molecule_NumSaturatedCarbocycles_Featurizer()
molecule_SMR_VSA9_featurizer = Molecule_SMR_VSA9_Featurizer()
molecule_MinAbsEStateIndex_featurizer = Molecule_MinAbsEStateIndex_Featurizer()
molecule_LabuteASA_featurizer = Molecule_LabuteASA_Featurizer()
molecule_SlogP_VSA3_featurizer = Molecule_SlogP_VSA3_Featurizer()
molecule_SlogP_VSA5_featurizer = Molecule_SlogP_VSA5_Featurizer()
molecule_SlogP_VSA12_featurizer = Molecule_SlogP_VSA12_Featurizer()
molecule_BCUT2D_MRHI_featurizer = Molecule_BCUT2D_MRHI_Featurizer()
molecule_HeavyAtomMolWt_featurizer = Molecule_HeavyAtomMolWt_Featurizer()
molecule_SpherocityIndex_featurizer = Molecule_SpherocityIndex_Featurizer()
molecule_fr_piperzine_featurizer = Molecule_fr_piperzine_Featurizer()
molecule_Chi4v_featurizer = Molecule_Chi4v_Featurizer()
molecule_SMR_VSA7_featurizer = Molecule_SMR_VSA7_Featurizer()
molecule_PEOE_VSA7_featurizer = Molecule_PEOE_VSA7_Featurizer()
molecule_EState_VSA8_featurizer = Molecule_EState_VSA8_Featurizer()
molecule_MinEStateIndex_featurizer = Molecule_MinEStateIndex_Featurizer()
molecule_Chi3n_featurizer = Molecule_Chi3n_Featurizer()
molecule_fr_methoxy_featurizer = Molecule_fr_methoxy_Featurizer()
molecule_MinPartialCharge_featurizer = Molecule_MinPartialCharge_Featurizer()
molecule_Chi0v_featurizer = Molecule_Chi0v_Featurizer()
molecule_EState_VSA4_featurizer = Molecule_EState_VSA4_Featurizer()
molecule_fr_oxazole_featurizer = Molecule_fr_oxazole_Featurizer()
molecule_pyLabuteASA_featurizer = Molecule_pyLabuteASA_Featurizer()
molecule_fr_N_O_featurizer = Molecule_fr_N_O_Featurizer()
molecule_Chi0_featurizer = Molecule_Chi0_Featurizer()
molecule_fr_ketone_featurizer = Molecule_fr_ketone_Featurizer()
molecule_fr_aniline_featurizer = Molecule_fr_aniline_Featurizer()
molecule_EState_VSA5_featurizer = Molecule_EState_VSA5_Featurizer()
molecule_fr_Al_COO_featurizer = Molecule_fr_Al_COO_Featurizer()
molecule_PEOE_VSA13_featurizer = Molecule_PEOE_VSA13_Featurizer()
molecule_fr_HOCCN_featurizer = Molecule_fr_HOCCN_Featurizer()
molecule_fr_piperdine_featurizer = Molecule_fr_piperdine_Featurizer()
molecule_fr_thiocyan_featurizer = Molecule_fr_thiocyan_Featurizer()
molecule_VSA_EState4_featurizer = Molecule_VSA_EState4_Featurizer()
molecule_fr_Ar_N_featurizer = Molecule_fr_Ar_N_Featurizer()
molecule_NumHBA_featurizer = Molecule_NumHBA_Featurizer()
molecule_weights_none_featurizer = Molecule_weights_none_Featurizer()
molecule_VSA_EState5_featurizer = Molecule_VSA_EState5_Featurizer()
molecule_NumSaturatedHeterocycles_featurizer = Molecule_NumSaturatedHeterocycles_Featurizer()
molecule_fr_guanido_featurizer = Molecule_fr_guanido_Featurizer()
molecule_fr_tetrazole_featurizer = Molecule_fr_tetrazole_Featurizer()
molecule_fr_morpholine_featurizer = Molecule_fr_morpholine_Featurizer()
molecule_NumLipinskiHBD_featurizer = Molecule_NumLipinskiHBD_Featurizer()
molecule_ExactMolWt_featurizer = Molecule_ExactMolWt_Featurizer()
molecule_SMR_VSA1_featurizer = Molecule_SMR_VSA1_Featurizer()
molecule_PEOE_VSA14_featurizer = Molecule_PEOE_VSA14_Featurizer()
molecule_NumHeteroatoms_featurizer = Molecule_NumHeteroatoms_Featurizer()
molecule_VSA_EState7_featurizer = Molecule_VSA_EState7_Featurizer()
molecule_fr_epoxide_featurizer = Molecule_fr_epoxide_Featurizer()
molecule_PBF_featurizer = Molecule_PBF_Featurizer()
molecule_SMR_VSA2_featurizer = Molecule_SMR_VSA2_Featurizer()
molecule_BertzCT_featurizer = Molecule_BertzCT_Featurizer()
molecule_NumSaturatedRings_featurizer = Molecule_NumSaturatedRings_Featurizer()
molecule_fr_pyridine_featurizer = Molecule_fr_pyridine_Featurizer()
molecule_NPR1_featurizer = Molecule_NPR1_Featurizer()
molecule_fr_allylic_oxid_featurizer = Molecule_fr_allylic_oxid_Featurizer()
molecule_Chi0n_featurizer = Molecule_Chi0n_Featurizer()
molecule_NumAliphaticRings_featurizer = Molecule_NumAliphaticRings_Featurizer()
molecule_Chi1n_featurizer = Molecule_Chi1n_Featurizer()
molecule_PEOE_VSA3_featurizer = Molecule_PEOE_VSA3_Featurizer()
molecule_fr_sulfonamd_featurizer = Molecule_fr_sulfonamd_Featurizer()
molecule_RingCount_featurizer = Molecule_RingCount_Featurizer()
molecule_fr_amidine_featurizer = Molecule_fr_amidine_Featurizer()
molecule_Kappa1_featurizer = Molecule_Kappa1_Featurizer()
molecule_VSA_EState2_featurizer = Molecule_VSA_EState2_Featurizer()
molecule_fr_imide_featurizer = Molecule_fr_imide_Featurizer()
molecule_fr_Al_OH_noTert_featurizer = Molecule_fr_Al_OH_noTert_Featurizer()
molecule_fr_Nhpyrrole_featurizer = Molecule_fr_Nhpyrrole_Featurizer()
molecule_NumAliphaticHeterocycles_featurizer = Molecule_NumAliphaticHeterocycles_Featurizer()
molecule_fr_para_hydroxylation_featurizer = Molecule_fr_para_hydroxylation_Featurizer()
molecule_PEOE_VSA1_featurizer = Molecule_PEOE_VSA1_Featurizer()
molecule_PEOE_VSA4_featurizer = Molecule_PEOE_VSA4_Featurizer()
molecule_PEOE_VSA2_featurizer = Molecule_PEOE_VSA2_Featurizer()
molecule_Chi3v_featurizer = Molecule_Chi3v_Featurizer()
molecule_SlogP_VSA6_featurizer = Molecule_SlogP_VSA6_Featurizer()
molecule_fr_ketone_Topliss_featurizer = Molecule_fr_ketone_Topliss_Featurizer()
molecule_EState_VSA11_featurizer = Molecule_EState_VSA11_Featurizer()
molecule_fr_sulfone_featurizer = Molecule_fr_sulfone_Featurizer()
molecule_Kappa2_featurizer = Molecule_Kappa2_Featurizer()
molecule_fr_unbrch_alkane_featurizer = Molecule_fr_unbrch_alkane_Featurizer()
molecule_SMR_VSA4_featurizer = Molecule_SMR_VSA4_Featurizer()
molecule_SlogP_VSA7_featurizer = Molecule_SlogP_VSA7_Featurizer()
molecule_fr_azo_featurizer = Molecule_fr_azo_Featurizer()
molecule_VSA_EState3_featurizer = Molecule_VSA_EState3_Featurizer()
molecule_fr_ArN_featurizer = Molecule_fr_ArN_Featurizer()
molecule_FormalCharge_featurizer = Molecule_FormalCharge_Featurizer()
molecule_Phi_featurizer = Molecule_Phi_Featurizer()
molecule_NumValenceElectrons_featurizer = Molecule_NumValenceElectrons_Featurizer()
molecule_BCUT2D_MWLOW_featurizer = Molecule_BCUT2D_MWLOW_Featurizer()
molecule_EmbedMolecule_featurizer = Molecule_EmbedMolecule_Featurizer()
molecule_NumAromaticRings_featurizer = Molecule_NumAromaticRings_Featurizer()
molecule_SMR_VSA10_featurizer = Molecule_SMR_VSA10_Featurizer()
molecule_NumRotatableBonds_featurizer = Molecule_NumRotatableBonds_Featurizer()
molecule_MaxPartialCharge_featurizer = Molecule_MaxPartialCharge_Featurizer()
molecule_FpDensityMorgan2_featurizer = Molecule_FpDensityMorgan2_Featurizer()
molecule_UFFHasAllMoleculeParams_featurizer = Molecule_UFFHasAllMoleculeParams_Featurizer()
molecule_PMI1_featurizer = Molecule_PMI1_Featurizer()
molecule_fr_lactam_featurizer = Molecule_fr_lactam_Featurizer()
molecule_fr_phenol_featurizer = Molecule_fr_phenol_Featurizer()
molecule_VSA_EState1_featurizer = Molecule_VSA_EState1_Featurizer()
molecule_NumAliphaticCarbocycles_featurizer = Molecule_NumAliphaticCarbocycles_Featurizer()
molecule_PEOE_VSA9_featurizer = Molecule_PEOE_VSA9_Featurizer()
molecule_NumHAcceptors_featurizer = Molecule_NumHAcceptors_Featurizer()
molecule_NumBridgeheadAtoms_featurizer = Molecule_NumBridgeheadAtoms_Featurizer()
molecule_SlogP_VSA9_featurizer = Molecule_SlogP_VSA9_Featurizer()
molecule_UFFOptimizeMolecule_featurizer = Molecule_UFFOptimizeMolecule_Featurizer()
molecule_fr_bicyclic_featurizer = Molecule_fr_bicyclic_Featurizer()
molecule_fr_isothiocyan_featurizer = Molecule_fr_isothiocyan_Featurizer()
molecule_Kappa3_featurizer = Molecule_Kappa3_Featurizer()
molecule_FpDensityMorgan3_featurizer = Molecule_FpDensityMorgan3_Featurizer()
molecule_FractionCSP3_featurizer = Molecule_FractionCSP3_Featurizer()
molecule_fr_aryl_methyl_featurizer = Molecule_fr_aryl_methyl_Featurizer()
molecule_fr_aldehyde_featurizer = Molecule_fr_aldehyde_Featurizer()
molecule_PMI2_featurizer = Molecule_PMI2_Featurizer()
molecule_EState_VSA6_featurizer = Molecule_EState_VSA6_Featurizer()
molecule_fr_hdrzine_featurizer = Molecule_fr_hdrzine_Featurizer()
molecule_Chi1_featurizer = Molecule_Chi1_Featurizer()
molecule_HallKierAlpha_featurizer = Molecule_HallKierAlpha_Featurizer()
molecule_fr_sulfide_featurizer = Molecule_fr_sulfide_Featurizer()
molecule_fr_imidazole_featurizer = Molecule_fr_imidazole_Featurizer()
molecule_fr_NH2_featurizer = Molecule_fr_NH2_Featurizer()
molecule_MolWt_featurizer = Molecule_MolWt_Featurizer()
molecule_fr_oxime_featurizer = Molecule_fr_oxime_Featurizer()
molecule_fr_SH_featurizer = Molecule_fr_SH_Featurizer()
molecule_fr_phos_ester_featurizer = Molecule_fr_phos_ester_Featurizer()
molecule_fr_benzene_featurizer = Molecule_fr_benzene_Featurizer()
molecule_fr_nitro_featurizer = Molecule_fr_nitro_Featurizer()
molecule_fr_phenol_noOrthoHbond_featurizer = Molecule_fr_phenol_noOrthoHbond_Featurizer()
molecule_EState_VSA1_featurizer = Molecule_EState_VSA1_Featurizer()
molecule_fr_azide_featurizer = Molecule_fr_azide_Featurizer()
molecule_ComputeMolVolume_featurizer = Molecule_ComputeMolVolume_Featurizer()
molecule_EState_VSA3_featurizer = Molecule_EState_VSA3_Featurizer()
molecule_BCUT2D_LOGPHI_featurizer = Molecule_BCUT2D_LOGPHI_Featurizer()
molecule_weights_max_featurizer = Molecule_weights_max_Featurizer()
molecule_fr_C_O_noCOO_featurizer = Molecule_fr_C_O_noCOO_Featurizer()
molecule_PEOE_VSA11_featurizer = Molecule_PEOE_VSA11_Featurizer()
molecule_RadiusOfGyration_featurizer = Molecule_RadiusOfGyration_Featurizer()
molecule_fr_dihydropyridine_featurizer = Molecule_fr_dihydropyridine_Featurizer()
molecule_fr_alkyl_halide_featurizer = Molecule_fr_alkyl_halide_Featurizer()
molecule_FpDensityMorgan1_featurizer = Molecule_FpDensityMorgan1_Featurizer()
molecule_MolMR_featurizer = Molecule_MolMR_Featurizer()
molecule_Chi2v_featurizer = Molecule_Chi2v_Featurizer()
molecule_fr_Ar_COO_featurizer = Molecule_fr_Ar_COO_Featurizer()
molecule_MolLogP_featurizer = Molecule_MolLogP_Featurizer()
molecule_fr_alkyl_carbamate_featurizer = Molecule_fr_alkyl_carbamate_Featurizer()
molecule_qed_featurizer = Molecule_qed_Featurizer()
molecule_Chi1v_featurizer = Molecule_Chi1v_Featurizer()
molecule_NumRadicalElectrons_featurizer = Molecule_NumRadicalElectrons_Featurizer()
_available_featurizer = {
    'molecule_NumAtoms_featurizer': molecule_NumAtoms_featurizer,
    'molecule_NumBonds_featurizer': molecule_NumBonds_featurizer,
    'molecule_NumHeavyAtoms_featurizer': molecule_NumHeavyAtoms_featurizer,
    'molecule_fr_thiophene_featurizer': molecule_fr_thiophene_featurizer,
    'molecule_VSA_EState8_featurizer': molecule_VSA_EState8_featurizer,
    'molecule_fr_amide_featurizer': molecule_fr_amide_featurizer,
    'molecule_NumAromaticCarbocycles_featurizer': molecule_NumAromaticCarbocycles_featurizer,
    'molecule_fr_urea_featurizer': molecule_fr_urea_featurizer,
    'molecule_BalabanJ_featurizer': molecule_BalabanJ_featurizer,
    'molecule_default_featurizer': molecule_default_featurizer,
    'molecule_TPSA_featurizer': molecule_TPSA_featurizer,
    'molecule_Chi4n_featurizer': molecule_Chi4n_featurizer,
    'molecule_fr_Ar_OH_featurizer': molecule_fr_Ar_OH_featurizer,
    'molecule_VSA_EState9_featurizer': molecule_VSA_EState9_featurizer,
    'molecule_BCUT2D_LOGPLOW_featurizer': molecule_BCUT2D_LOGPLOW_featurizer,
    'molecule_InertialShapeFactor_featurizer': molecule_InertialShapeFactor_featurizer,
    'molecule_fr_nitrile_featurizer': molecule_fr_nitrile_featurizer,
    'molecule_fr_NH1_featurizer': molecule_fr_NH1_featurizer,
    'molecule_fr_lactone_featurizer': molecule_fr_lactone_featurizer,
    'molecule_SlogP_VSA1_featurizer': molecule_SlogP_VSA1_featurizer,
    'molecule_fr_barbitur_featurizer': molecule_fr_barbitur_featurizer,
    'molecule_NPR2_featurizer': molecule_NPR2_featurizer,
    'molecule_VSA_EState10_featurizer': molecule_VSA_EState10_featurizer,
    'molecule_fr_phos_acid_featurizer': molecule_fr_phos_acid_featurizer,
    'molecule_fr_Ndealkylation2_featurizer': molecule_fr_Ndealkylation2_featurizer,
    'molecule_MaxEStateIndex_featurizer': molecule_MaxEStateIndex_featurizer,
    'molecule_fr_prisulfonamd_featurizer': molecule_fr_prisulfonamd_featurizer,
    'molecule_fr_COO_featurizer': molecule_fr_COO_featurizer,
    'molecule_fr_benzodiazepine_featurizer': molecule_fr_benzodiazepine_featurizer,
    'molecule_PEOE_VSA12_featurizer': molecule_PEOE_VSA12_featurizer,
    'molecule_fr_nitro_arom_nonortho_featurizer': molecule_fr_nitro_arom_nonortho_featurizer,
    'molecule_fr_halogen_featurizer': molecule_fr_halogen_featurizer,
    'molecule_PEOE_VSA6_featurizer': molecule_PEOE_VSA6_featurizer,
    'molecule_EState_VSA2_featurizer': molecule_EState_VSA2_featurizer,
    'molecule_PMI3_featurizer': molecule_PMI3_featurizer,
    'molecule_EState_VSA7_featurizer': molecule_EState_VSA7_featurizer,
    'molecule_NHOHCount_featurizer': molecule_NHOHCount_featurizer,
    'molecule_SlogP_VSA11_featurizer': molecule_SlogP_VSA11_featurizer,
    'molecule_MMFFOptimizeMolecule_featurizer': molecule_MMFFOptimizeMolecule_featurizer,
    'molecule_NumLipinskiHBA_featurizer': molecule_NumLipinskiHBA_featurizer,
    'molecule_SlogP_VSA4_featurizer': molecule_SlogP_VSA4_featurizer,
    'molecule_fr_Ndealkylation1_featurizer': molecule_fr_Ndealkylation1_featurizer,
    'molecule_Eccentricity_featurizer': molecule_Eccentricity_featurizer,
    'molecule_SMR_VSA3_featurizer': molecule_SMR_VSA3_featurizer,
    'molecule_SlogP_VSA2_featurizer': molecule_SlogP_VSA2_featurizer,
    'molecule_HeavyAtomCount_featurizer': molecule_HeavyAtomCount_featurizer,
    'molecule_fr_ester_featurizer': molecule_fr_ester_featurizer,
    'molecule_PEOE_VSA10_featurizer': molecule_PEOE_VSA10_featurizer,
    'molecule_BCUT2D_CHGLO_featurizer': molecule_BCUT2D_CHGLO_featurizer,
    'molecule_fr_hdrzone_featurizer': molecule_fr_hdrzone_featurizer,
    'molecule_SSSR_featurizer': molecule_SSSR_featurizer,
    'molecule_Ipc_featurizer': molecule_Ipc_featurizer,
    'molecule_NumHBD_featurizer': molecule_NumHBD_featurizer,
    'molecule_fr_C_O_featurizer': molecule_fr_C_O_featurizer,
    'molecule_Asphericity_featurizer': molecule_Asphericity_featurizer,
    'molecule_fr_C_S_featurizer': molecule_fr_C_S_featurizer,
    'molecule_SMR_VSA8_featurizer': molecule_SMR_VSA8_featurizer,
    'molecule_MMFFHasAllMoleculeParams_featurizer': molecule_MMFFHasAllMoleculeParams_featurizer,
    'molecule_fr_thiazole_featurizer': molecule_fr_thiazole_featurizer,
    'molecule_NumAromaticHeterocycles_featurizer': molecule_NumAromaticHeterocycles_featurizer,
    'molecule_fr_Imine_featurizer': molecule_fr_Imine_featurizer,
    'molecule_SlogP_VSA8_featurizer': molecule_SlogP_VSA8_featurizer,
    'molecule_Chi2n_featurizer': molecule_Chi2n_featurizer,
    'molecule_fr_nitro_arom_featurizer': molecule_fr_nitro_arom_featurizer,
    'molecule_SlogP_VSA10_featurizer': molecule_SlogP_VSA10_featurizer,
    'molecule_SMR_VSA5_featurizer': molecule_SMR_VSA5_featurizer,
    'molecule_fr_nitroso_featurizer': molecule_fr_nitroso_featurizer,
    'molecule_MaxAbsEStateIndex_featurizer': molecule_MaxAbsEStateIndex_featurizer,
    'molecule_NumRings_featurizer': molecule_NumRings_featurizer,
    'molecule_PEOE_VSA5_featurizer': molecule_PEOE_VSA5_featurizer,
    'molecule_NumHeterocycles_featurizer': molecule_NumHeterocycles_featurizer,
    'molecule_StereoisomerCount_featurizer': molecule_StereoisomerCount_featurizer,
    'molecule_NOCount_featurizer': molecule_NOCount_featurizer,
    'molecule_NumHDonors_featurizer': molecule_NumHDonors_featurizer,
    'molecule_BCUT2D_CHGHI_featurizer': molecule_BCUT2D_CHGHI_featurizer,
    'molecule_SMR_VSA6_featurizer': molecule_SMR_VSA6_featurizer,
    'molecule_EState_VSA10_featurizer': molecule_EState_VSA10_featurizer,
    'molecule_MaxAbsPartialCharge_featurizer': molecule_MaxAbsPartialCharge_featurizer,
    'molecule_VSA_EState6_featurizer': molecule_VSA_EState6_featurizer,
    'molecule_BCUT2D_MWHI_featurizer': molecule_BCUT2D_MWHI_featurizer,
    'molecule_NumSpiroAtoms_featurizer': molecule_NumSpiroAtoms_featurizer,
    'molecule_EState_VSA9_featurizer': molecule_EState_VSA9_featurizer,
    'molecule_fr_ether_featurizer': molecule_fr_ether_featurizer,
    'molecule_fr_term_acetylene_featurizer': molecule_fr_term_acetylene_featurizer,
    'molecule_PEOE_VSA8_featurizer': molecule_PEOE_VSA8_featurizer,
    'molecule_Compute2DCoords_featurizer': molecule_Compute2DCoords_featurizer,
    'molecule_fr_diazo_featurizer': molecule_fr_diazo_featurizer,
    'molecule_fr_Ar_NH_featurizer': molecule_fr_Ar_NH_featurizer,
    'molecule_fr_isocyan_featurizer': molecule_fr_isocyan_featurizer,
    'molecule_fr_priamide_featurizer': molecule_fr_priamide_featurizer,
    'molecule_BCUT2D_MRLOW_featurizer': molecule_BCUT2D_MRLOW_featurizer,
    'molecule_fr_COO2_featurizer': molecule_fr_COO2_featurizer,
    'molecule_MinAbsPartialCharge_featurizer': molecule_MinAbsPartialCharge_featurizer,
    'molecule_weights_mean_featurizer': molecule_weights_mean_featurizer,
    'molecule_NumAmideBonds_featurizer': molecule_NumAmideBonds_featurizer,
    'molecule_fr_Al_OH_featurizer': molecule_fr_Al_OH_featurizer,
    'molecule_fr_furan_featurizer': molecule_fr_furan_featurizer,
    'molecule_fr_NH0_featurizer': molecule_fr_NH0_featurizer,
    'molecule_fr_quatN_featurizer': molecule_fr_quatN_featurizer,
    'molecule_NumSaturatedCarbocycles_featurizer': molecule_NumSaturatedCarbocycles_featurizer,
    'molecule_SMR_VSA9_featurizer': molecule_SMR_VSA9_featurizer,
    'molecule_MinAbsEStateIndex_featurizer': molecule_MinAbsEStateIndex_featurizer,
    'molecule_LabuteASA_featurizer': molecule_LabuteASA_featurizer,
    'molecule_SlogP_VSA3_featurizer': molecule_SlogP_VSA3_featurizer,
    'molecule_SlogP_VSA5_featurizer': molecule_SlogP_VSA5_featurizer,
    'molecule_SlogP_VSA12_featurizer': molecule_SlogP_VSA12_featurizer,
    'molecule_BCUT2D_MRHI_featurizer': molecule_BCUT2D_MRHI_featurizer,
    'molecule_HeavyAtomMolWt_featurizer': molecule_HeavyAtomMolWt_featurizer,
    'molecule_SpherocityIndex_featurizer': molecule_SpherocityIndex_featurizer,
    'molecule_fr_piperzine_featurizer': molecule_fr_piperzine_featurizer,
    'molecule_Chi4v_featurizer': molecule_Chi4v_featurizer,
    'molecule_SMR_VSA7_featurizer': molecule_SMR_VSA7_featurizer,
    'molecule_PEOE_VSA7_featurizer': molecule_PEOE_VSA7_featurizer,
    'molecule_EState_VSA8_featurizer': molecule_EState_VSA8_featurizer,
    'molecule_MinEStateIndex_featurizer': molecule_MinEStateIndex_featurizer,
    'molecule_Chi3n_featurizer': molecule_Chi3n_featurizer,
    'molecule_fr_methoxy_featurizer': molecule_fr_methoxy_featurizer,
    'molecule_MinPartialCharge_featurizer': molecule_MinPartialCharge_featurizer,
    'molecule_Chi0v_featurizer': molecule_Chi0v_featurizer,
    'molecule_EState_VSA4_featurizer': molecule_EState_VSA4_featurizer,
    'molecule_fr_oxazole_featurizer': molecule_fr_oxazole_featurizer,
    'molecule_pyLabuteASA_featurizer': molecule_pyLabuteASA_featurizer,
    'molecule_fr_N_O_featurizer': molecule_fr_N_O_featurizer,
    'molecule_Chi0_featurizer': molecule_Chi0_featurizer,
    'molecule_fr_ketone_featurizer': molecule_fr_ketone_featurizer,
    'molecule_fr_aniline_featurizer': molecule_fr_aniline_featurizer,
    'molecule_EState_VSA5_featurizer': molecule_EState_VSA5_featurizer,
    'molecule_fr_Al_COO_featurizer': molecule_fr_Al_COO_featurizer,
    'molecule_PEOE_VSA13_featurizer': molecule_PEOE_VSA13_featurizer,
    'molecule_fr_HOCCN_featurizer': molecule_fr_HOCCN_featurizer,
    'molecule_fr_piperdine_featurizer': molecule_fr_piperdine_featurizer,
    'molecule_fr_thiocyan_featurizer': molecule_fr_thiocyan_featurizer,
    'molecule_VSA_EState4_featurizer': molecule_VSA_EState4_featurizer,
    'molecule_fr_Ar_N_featurizer': molecule_fr_Ar_N_featurizer,
    'molecule_NumHBA_featurizer': molecule_NumHBA_featurizer,
    'molecule_weights_none_featurizer': molecule_weights_none_featurizer,
    'molecule_VSA_EState5_featurizer': molecule_VSA_EState5_featurizer,
    'molecule_NumSaturatedHeterocycles_featurizer': molecule_NumSaturatedHeterocycles_featurizer,
    'molecule_fr_guanido_featurizer': molecule_fr_guanido_featurizer,
    'molecule_fr_tetrazole_featurizer': molecule_fr_tetrazole_featurizer,
    'molecule_fr_morpholine_featurizer': molecule_fr_morpholine_featurizer,
    'molecule_NumLipinskiHBD_featurizer': molecule_NumLipinskiHBD_featurizer,
    'molecule_ExactMolWt_featurizer': molecule_ExactMolWt_featurizer,
    'molecule_SMR_VSA1_featurizer': molecule_SMR_VSA1_featurizer,
    'molecule_PEOE_VSA14_featurizer': molecule_PEOE_VSA14_featurizer,
    'molecule_NumHeteroatoms_featurizer': molecule_NumHeteroatoms_featurizer,
    'molecule_VSA_EState7_featurizer': molecule_VSA_EState7_featurizer,
    'molecule_fr_epoxide_featurizer': molecule_fr_epoxide_featurizer,
    'molecule_PBF_featurizer': molecule_PBF_featurizer,
    'molecule_SMR_VSA2_featurizer': molecule_SMR_VSA2_featurizer,
    'molecule_BertzCT_featurizer': molecule_BertzCT_featurizer,
    'molecule_NumSaturatedRings_featurizer': molecule_NumSaturatedRings_featurizer,
    'molecule_fr_pyridine_featurizer': molecule_fr_pyridine_featurizer,
    'molecule_NPR1_featurizer': molecule_NPR1_featurizer,
    'molecule_fr_allylic_oxid_featurizer': molecule_fr_allylic_oxid_featurizer,
    'molecule_Chi0n_featurizer': molecule_Chi0n_featurizer,
    'molecule_NumAliphaticRings_featurizer': molecule_NumAliphaticRings_featurizer,
    'molecule_Chi1n_featurizer': molecule_Chi1n_featurizer,
    'molecule_PEOE_VSA3_featurizer': molecule_PEOE_VSA3_featurizer,
    'molecule_fr_sulfonamd_featurizer': molecule_fr_sulfonamd_featurizer,
    'molecule_RingCount_featurizer': molecule_RingCount_featurizer,
    'molecule_fr_amidine_featurizer': molecule_fr_amidine_featurizer,
    'molecule_Kappa1_featurizer': molecule_Kappa1_featurizer,
    'molecule_VSA_EState2_featurizer': molecule_VSA_EState2_featurizer,
    'molecule_fr_imide_featurizer': molecule_fr_imide_featurizer,
    'molecule_fr_Al_OH_noTert_featurizer': molecule_fr_Al_OH_noTert_featurizer,
    'molecule_fr_Nhpyrrole_featurizer': molecule_fr_Nhpyrrole_featurizer,
    'molecule_NumAliphaticHeterocycles_featurizer': molecule_NumAliphaticHeterocycles_featurizer,
    'molecule_fr_para_hydroxylation_featurizer': molecule_fr_para_hydroxylation_featurizer,
    'molecule_PEOE_VSA1_featurizer': molecule_PEOE_VSA1_featurizer,
    'molecule_PEOE_VSA4_featurizer': molecule_PEOE_VSA4_featurizer,
    'molecule_PEOE_VSA2_featurizer': molecule_PEOE_VSA2_featurizer,
    'molecule_Chi3v_featurizer': molecule_Chi3v_featurizer,
    'molecule_SlogP_VSA6_featurizer': molecule_SlogP_VSA6_featurizer,
    'molecule_fr_ketone_Topliss_featurizer': molecule_fr_ketone_Topliss_featurizer,
    'molecule_EState_VSA11_featurizer': molecule_EState_VSA11_featurizer,
    'molecule_fr_sulfone_featurizer': molecule_fr_sulfone_featurizer,
    'molecule_Kappa2_featurizer': molecule_Kappa2_featurizer,
    'molecule_fr_unbrch_alkane_featurizer': molecule_fr_unbrch_alkane_featurizer,
    'molecule_SMR_VSA4_featurizer': molecule_SMR_VSA4_featurizer,
    'molecule_SlogP_VSA7_featurizer': molecule_SlogP_VSA7_featurizer,
    'molecule_fr_azo_featurizer': molecule_fr_azo_featurizer,
    'molecule_VSA_EState3_featurizer': molecule_VSA_EState3_featurizer,
    'molecule_fr_ArN_featurizer': molecule_fr_ArN_featurizer,
    'molecule_FormalCharge_featurizer': molecule_FormalCharge_featurizer,
    'molecule_Phi_featurizer': molecule_Phi_featurizer,
    'molecule_NumValenceElectrons_featurizer': molecule_NumValenceElectrons_featurizer,
    'molecule_BCUT2D_MWLOW_featurizer': molecule_BCUT2D_MWLOW_featurizer,
    'molecule_EmbedMolecule_featurizer': molecule_EmbedMolecule_featurizer,
    'molecule_NumAromaticRings_featurizer': molecule_NumAromaticRings_featurizer,
    'molecule_SMR_VSA10_featurizer': molecule_SMR_VSA10_featurizer,
    'molecule_NumRotatableBonds_featurizer': molecule_NumRotatableBonds_featurizer,
    'molecule_MaxPartialCharge_featurizer': molecule_MaxPartialCharge_featurizer,
    'molecule_FpDensityMorgan2_featurizer': molecule_FpDensityMorgan2_featurizer,
    'molecule_UFFHasAllMoleculeParams_featurizer': molecule_UFFHasAllMoleculeParams_featurizer,
    'molecule_PMI1_featurizer': molecule_PMI1_featurizer,
    'molecule_fr_lactam_featurizer': molecule_fr_lactam_featurizer,
    'molecule_fr_phenol_featurizer': molecule_fr_phenol_featurizer,
    'molecule_VSA_EState1_featurizer': molecule_VSA_EState1_featurizer,
    'molecule_NumAliphaticCarbocycles_featurizer': molecule_NumAliphaticCarbocycles_featurizer,
    'molecule_PEOE_VSA9_featurizer': molecule_PEOE_VSA9_featurizer,
    'molecule_NumHAcceptors_featurizer': molecule_NumHAcceptors_featurizer,
    'molecule_NumBridgeheadAtoms_featurizer': molecule_NumBridgeheadAtoms_featurizer,
    'molecule_SlogP_VSA9_featurizer': molecule_SlogP_VSA9_featurizer,
    'molecule_UFFOptimizeMolecule_featurizer': molecule_UFFOptimizeMolecule_featurizer,
    'molecule_fr_bicyclic_featurizer': molecule_fr_bicyclic_featurizer,
    'molecule_fr_isothiocyan_featurizer': molecule_fr_isothiocyan_featurizer,
    'molecule_Kappa3_featurizer': molecule_Kappa3_featurizer,
    'molecule_FpDensityMorgan3_featurizer': molecule_FpDensityMorgan3_featurizer,
    'molecule_FractionCSP3_featurizer': molecule_FractionCSP3_featurizer,
    'molecule_fr_aryl_methyl_featurizer': molecule_fr_aryl_methyl_featurizer,
    'molecule_fr_aldehyde_featurizer': molecule_fr_aldehyde_featurizer,
    'molecule_PMI2_featurizer': molecule_PMI2_featurizer,
    'molecule_EState_VSA6_featurizer': molecule_EState_VSA6_featurizer,
    'molecule_fr_hdrzine_featurizer': molecule_fr_hdrzine_featurizer,
    'molecule_Chi1_featurizer': molecule_Chi1_featurizer,
    'molecule_HallKierAlpha_featurizer': molecule_HallKierAlpha_featurizer,
    'molecule_fr_sulfide_featurizer': molecule_fr_sulfide_featurizer,
    'molecule_fr_imidazole_featurizer': molecule_fr_imidazole_featurizer,
    'molecule_fr_NH2_featurizer': molecule_fr_NH2_featurizer,
    'molecule_MolWt_featurizer': molecule_MolWt_featurizer,
    'molecule_fr_oxime_featurizer': molecule_fr_oxime_featurizer,
    'molecule_fr_SH_featurizer': molecule_fr_SH_featurizer,
    'molecule_fr_phos_ester_featurizer': molecule_fr_phos_ester_featurizer,
    'molecule_fr_benzene_featurizer': molecule_fr_benzene_featurizer,
    'molecule_fr_nitro_featurizer': molecule_fr_nitro_featurizer,
    'molecule_fr_phenol_noOrthoHbond_featurizer': molecule_fr_phenol_noOrthoHbond_featurizer,
    'molecule_EState_VSA1_featurizer': molecule_EState_VSA1_featurizer,
    'molecule_fr_azide_featurizer': molecule_fr_azide_featurizer,
    'molecule_ComputeMolVolume_featurizer': molecule_ComputeMolVolume_featurizer,
    'molecule_EState_VSA3_featurizer': molecule_EState_VSA3_featurizer,
    'molecule_BCUT2D_LOGPHI_featurizer': molecule_BCUT2D_LOGPHI_featurizer,
    'molecule_weights_max_featurizer': molecule_weights_max_featurizer,
    'molecule_fr_C_O_noCOO_featurizer': molecule_fr_C_O_noCOO_featurizer,
    'molecule_PEOE_VSA11_featurizer': molecule_PEOE_VSA11_featurizer,
    'molecule_RadiusOfGyration_featurizer': molecule_RadiusOfGyration_featurizer,
    'molecule_fr_dihydropyridine_featurizer': molecule_fr_dihydropyridine_featurizer,
    'molecule_fr_alkyl_halide_featurizer': molecule_fr_alkyl_halide_featurizer,
    'molecule_FpDensityMorgan1_featurizer': molecule_FpDensityMorgan1_featurizer,
    'molecule_MolMR_featurizer': molecule_MolMR_featurizer,
    'molecule_Chi2v_featurizer': molecule_Chi2v_featurizer,
    'molecule_fr_Ar_COO_featurizer': molecule_fr_Ar_COO_featurizer,
    'molecule_MolLogP_featurizer': molecule_MolLogP_featurizer,
    'molecule_fr_alkyl_carbamate_featurizer': molecule_fr_alkyl_carbamate_featurizer,
    'molecule_qed_featurizer': molecule_qed_featurizer,
    'molecule_Chi1v_featurizer': molecule_Chi1v_featurizer,
    'molecule_NumRadicalElectrons_featurizer': molecule_NumRadicalElectrons_featurizer,
}
__all__ = [
    'Molecule_NumAtoms_Featurizer',
    'molecule_NumAtoms_featurizer',
    'Molecule_NumBonds_Featurizer',
    'molecule_NumBonds_featurizer',
    'Molecule_NumHeavyAtoms_Featurizer',
    'molecule_NumHeavyAtoms_featurizer',
    'Molecule_fr_thiophene_Featurizer',
    'molecule_fr_thiophene_featurizer',
    'Molecule_VSA_EState8_Featurizer',
    'molecule_VSA_EState8_featurizer',
    'Molecule_fr_amide_Featurizer',
    'molecule_fr_amide_featurizer',
    'Molecule_NumAromaticCarbocycles_Featurizer',
    'molecule_NumAromaticCarbocycles_featurizer',
    'Molecule_fr_urea_Featurizer',
    'molecule_fr_urea_featurizer',
    'Molecule_BalabanJ_Featurizer',
    'molecule_BalabanJ_featurizer',
    'Molecule_default_Featurizer',
    'molecule_default_featurizer',
    'Molecule_TPSA_Featurizer',
    'molecule_TPSA_featurizer',
    'Molecule_Chi4n_Featurizer',
    'molecule_Chi4n_featurizer',
    'Molecule_fr_Ar_OH_Featurizer',
    'molecule_fr_Ar_OH_featurizer',
    'Molecule_VSA_EState9_Featurizer',
    'molecule_VSA_EState9_featurizer',
    'Molecule_BCUT2D_LOGPLOW_Featurizer',
    'molecule_BCUT2D_LOGPLOW_featurizer',
    'Molecule_InertialShapeFactor_Featurizer',
    'molecule_InertialShapeFactor_featurizer',
    'Molecule_fr_nitrile_Featurizer',
    'molecule_fr_nitrile_featurizer',
    'Molecule_fr_NH1_Featurizer',
    'molecule_fr_NH1_featurizer',
    'Molecule_fr_lactone_Featurizer',
    'molecule_fr_lactone_featurizer',
    'Molecule_SlogP_VSA1_Featurizer',
    'molecule_SlogP_VSA1_featurizer',
    'Molecule_fr_barbitur_Featurizer',
    'molecule_fr_barbitur_featurizer',
    'Molecule_NPR2_Featurizer',
    'molecule_NPR2_featurizer',
    'Molecule_VSA_EState10_Featurizer',
    'molecule_VSA_EState10_featurizer',
    'Molecule_fr_phos_acid_Featurizer',
    'molecule_fr_phos_acid_featurizer',
    'Molecule_fr_Ndealkylation2_Featurizer',
    'molecule_fr_Ndealkylation2_featurizer',
    'Molecule_MaxEStateIndex_Featurizer',
    'molecule_MaxEStateIndex_featurizer',
    'Molecule_fr_prisulfonamd_Featurizer',
    'molecule_fr_prisulfonamd_featurizer',
    'Molecule_fr_COO_Featurizer',
    'molecule_fr_COO_featurizer',
    'Molecule_fr_benzodiazepine_Featurizer',
    'molecule_fr_benzodiazepine_featurizer',
    'Molecule_PEOE_VSA12_Featurizer',
    'molecule_PEOE_VSA12_featurizer',
    'Molecule_fr_nitro_arom_nonortho_Featurizer',
    'molecule_fr_nitro_arom_nonortho_featurizer',
    'Molecule_fr_halogen_Featurizer',
    'molecule_fr_halogen_featurizer',
    'Molecule_PEOE_VSA6_Featurizer',
    'molecule_PEOE_VSA6_featurizer',
    'Molecule_EState_VSA2_Featurizer',
    'molecule_EState_VSA2_featurizer',
    'Molecule_PMI3_Featurizer',
    'molecule_PMI3_featurizer',
    'Molecule_EState_VSA7_Featurizer',
    'molecule_EState_VSA7_featurizer',
    'Molecule_NHOHCount_Featurizer',
    'molecule_NHOHCount_featurizer',
    'Molecule_SlogP_VSA11_Featurizer',
    'molecule_SlogP_VSA11_featurizer',
    'Molecule_MMFFOptimizeMolecule_Featurizer',
    'molecule_MMFFOptimizeMolecule_featurizer',
    'Molecule_NumLipinskiHBA_Featurizer',
    'molecule_NumLipinskiHBA_featurizer',
    'Molecule_SlogP_VSA4_Featurizer',
    'molecule_SlogP_VSA4_featurizer',
    'Molecule_fr_Ndealkylation1_Featurizer',
    'molecule_fr_Ndealkylation1_featurizer',
    'Molecule_Eccentricity_Featurizer',
    'molecule_Eccentricity_featurizer',
    'Molecule_SMR_VSA3_Featurizer',
    'molecule_SMR_VSA3_featurizer',
    'Molecule_SlogP_VSA2_Featurizer',
    'molecule_SlogP_VSA2_featurizer',
    'Molecule_HeavyAtomCount_Featurizer',
    'molecule_HeavyAtomCount_featurizer',
    'Molecule_fr_ester_Featurizer',
    'molecule_fr_ester_featurizer',
    'Molecule_PEOE_VSA10_Featurizer',
    'molecule_PEOE_VSA10_featurizer',
    'Molecule_BCUT2D_CHGLO_Featurizer',
    'molecule_BCUT2D_CHGLO_featurizer',
    'Molecule_fr_hdrzone_Featurizer',
    'molecule_fr_hdrzone_featurizer',
    'Molecule_SSSR_Featurizer',
    'molecule_SSSR_featurizer',
    'Molecule_Ipc_Featurizer',
    'molecule_Ipc_featurizer',
    'Molecule_NumHBD_Featurizer',
    'molecule_NumHBD_featurizer',
    'Molecule_fr_C_O_Featurizer',
    'molecule_fr_C_O_featurizer',
    'Molecule_Asphericity_Featurizer',
    'molecule_Asphericity_featurizer',
    'Molecule_fr_C_S_Featurizer',
    'molecule_fr_C_S_featurizer',
    'Molecule_SMR_VSA8_Featurizer',
    'molecule_SMR_VSA8_featurizer',
    'Molecule_MMFFHasAllMoleculeParams_Featurizer',
    'molecule_MMFFHasAllMoleculeParams_featurizer',
    'Molecule_fr_thiazole_Featurizer',
    'molecule_fr_thiazole_featurizer',
    'Molecule_NumAromaticHeterocycles_Featurizer',
    'molecule_NumAromaticHeterocycles_featurizer',
    'Molecule_fr_Imine_Featurizer',
    'molecule_fr_Imine_featurizer',
    'Molecule_SlogP_VSA8_Featurizer',
    'molecule_SlogP_VSA8_featurizer',
    'Molecule_Chi2n_Featurizer',
    'molecule_Chi2n_featurizer',
    'Molecule_fr_nitro_arom_Featurizer',
    'molecule_fr_nitro_arom_featurizer',
    'Molecule_SlogP_VSA10_Featurizer',
    'molecule_SlogP_VSA10_featurizer',
    'Molecule_SMR_VSA5_Featurizer',
    'molecule_SMR_VSA5_featurizer',
    'Molecule_fr_nitroso_Featurizer',
    'molecule_fr_nitroso_featurizer',
    'Molecule_MaxAbsEStateIndex_Featurizer',
    'molecule_MaxAbsEStateIndex_featurizer',
    'Molecule_NumRings_Featurizer',
    'molecule_NumRings_featurizer',
    'Molecule_PEOE_VSA5_Featurizer',
    'molecule_PEOE_VSA5_featurizer',
    'Molecule_NumHeterocycles_Featurizer',
    'molecule_NumHeterocycles_featurizer',
    'Molecule_StereoisomerCount_Featurizer',
    'molecule_StereoisomerCount_featurizer',
    'Molecule_NOCount_Featurizer',
    'molecule_NOCount_featurizer',
    'Molecule_NumHDonors_Featurizer',
    'molecule_NumHDonors_featurizer',
    'Molecule_BCUT2D_CHGHI_Featurizer',
    'molecule_BCUT2D_CHGHI_featurizer',
    'Molecule_SMR_VSA6_Featurizer',
    'molecule_SMR_VSA6_featurizer',
    'Molecule_EState_VSA10_Featurizer',
    'molecule_EState_VSA10_featurizer',
    'Molecule_MaxAbsPartialCharge_Featurizer',
    'molecule_MaxAbsPartialCharge_featurizer',
    'Molecule_VSA_EState6_Featurizer',
    'molecule_VSA_EState6_featurizer',
    'Molecule_BCUT2D_MWHI_Featurizer',
    'molecule_BCUT2D_MWHI_featurizer',
    'Molecule_NumSpiroAtoms_Featurizer',
    'molecule_NumSpiroAtoms_featurizer',
    'Molecule_EState_VSA9_Featurizer',
    'molecule_EState_VSA9_featurizer',
    'Molecule_fr_ether_Featurizer',
    'molecule_fr_ether_featurizer',
    'Molecule_fr_term_acetylene_Featurizer',
    'molecule_fr_term_acetylene_featurizer',
    'Molecule_PEOE_VSA8_Featurizer',
    'molecule_PEOE_VSA8_featurizer',
    'Molecule_Compute2DCoords_Featurizer',
    'molecule_Compute2DCoords_featurizer',
    'Molecule_fr_diazo_Featurizer',
    'molecule_fr_diazo_featurizer',
    'Molecule_fr_Ar_NH_Featurizer',
    'molecule_fr_Ar_NH_featurizer',
    'Molecule_fr_isocyan_Featurizer',
    'molecule_fr_isocyan_featurizer',
    'Molecule_fr_priamide_Featurizer',
    'molecule_fr_priamide_featurizer',
    'Molecule_BCUT2D_MRLOW_Featurizer',
    'molecule_BCUT2D_MRLOW_featurizer',
    'Molecule_fr_COO2_Featurizer',
    'molecule_fr_COO2_featurizer',
    'Molecule_MinAbsPartialCharge_Featurizer',
    'molecule_MinAbsPartialCharge_featurizer',
    'Molecule_weights_mean_Featurizer',
    'molecule_weights_mean_featurizer',
    'Molecule_NumAmideBonds_Featurizer',
    'molecule_NumAmideBonds_featurizer',
    'Molecule_fr_Al_OH_Featurizer',
    'molecule_fr_Al_OH_featurizer',
    'Molecule_fr_furan_Featurizer',
    'molecule_fr_furan_featurizer',
    'Molecule_fr_NH0_Featurizer',
    'molecule_fr_NH0_featurizer',
    'Molecule_fr_quatN_Featurizer',
    'molecule_fr_quatN_featurizer',
    'Molecule_NumSaturatedCarbocycles_Featurizer',
    'molecule_NumSaturatedCarbocycles_featurizer',
    'Molecule_SMR_VSA9_Featurizer',
    'molecule_SMR_VSA9_featurizer',
    'Molecule_MinAbsEStateIndex_Featurizer',
    'molecule_MinAbsEStateIndex_featurizer',
    'Molecule_LabuteASA_Featurizer',
    'molecule_LabuteASA_featurizer',
    'Molecule_SlogP_VSA3_Featurizer',
    'molecule_SlogP_VSA3_featurizer',
    'Molecule_SlogP_VSA5_Featurizer',
    'molecule_SlogP_VSA5_featurizer',
    'Molecule_SlogP_VSA12_Featurizer',
    'molecule_SlogP_VSA12_featurizer',
    'Molecule_BCUT2D_MRHI_Featurizer',
    'molecule_BCUT2D_MRHI_featurizer',
    'Molecule_HeavyAtomMolWt_Featurizer',
    'molecule_HeavyAtomMolWt_featurizer',
    'Molecule_SpherocityIndex_Featurizer',
    'molecule_SpherocityIndex_featurizer',
    'Molecule_fr_piperzine_Featurizer',
    'molecule_fr_piperzine_featurizer',
    'Molecule_Chi4v_Featurizer',
    'molecule_Chi4v_featurizer',
    'Molecule_SMR_VSA7_Featurizer',
    'molecule_SMR_VSA7_featurizer',
    'Molecule_PEOE_VSA7_Featurizer',
    'molecule_PEOE_VSA7_featurizer',
    'Molecule_EState_VSA8_Featurizer',
    'molecule_EState_VSA8_featurizer',
    'Molecule_MinEStateIndex_Featurizer',
    'molecule_MinEStateIndex_featurizer',
    'Molecule_Chi3n_Featurizer',
    'molecule_Chi3n_featurizer',
    'Molecule_fr_methoxy_Featurizer',
    'molecule_fr_methoxy_featurizer',
    'Molecule_MinPartialCharge_Featurizer',
    'molecule_MinPartialCharge_featurizer',
    'Molecule_Chi0v_Featurizer',
    'molecule_Chi0v_featurizer',
    'Molecule_EState_VSA4_Featurizer',
    'molecule_EState_VSA4_featurizer',
    'Molecule_fr_oxazole_Featurizer',
    'molecule_fr_oxazole_featurizer',
    'Molecule_pyLabuteASA_Featurizer',
    'molecule_pyLabuteASA_featurizer',
    'Molecule_fr_N_O_Featurizer',
    'molecule_fr_N_O_featurizer',
    'Molecule_Chi0_Featurizer',
    'molecule_Chi0_featurizer',
    'Molecule_fr_ketone_Featurizer',
    'molecule_fr_ketone_featurizer',
    'Molecule_fr_aniline_Featurizer',
    'molecule_fr_aniline_featurizer',
    'Molecule_EState_VSA5_Featurizer',
    'molecule_EState_VSA5_featurizer',
    'Molecule_fr_Al_COO_Featurizer',
    'molecule_fr_Al_COO_featurizer',
    'Molecule_PEOE_VSA13_Featurizer',
    'molecule_PEOE_VSA13_featurizer',
    'Molecule_fr_HOCCN_Featurizer',
    'molecule_fr_HOCCN_featurizer',
    'Molecule_fr_piperdine_Featurizer',
    'molecule_fr_piperdine_featurizer',
    'Molecule_fr_thiocyan_Featurizer',
    'molecule_fr_thiocyan_featurizer',
    'Molecule_VSA_EState4_Featurizer',
    'molecule_VSA_EState4_featurizer',
    'Molecule_fr_Ar_N_Featurizer',
    'molecule_fr_Ar_N_featurizer',
    'Molecule_NumHBA_Featurizer',
    'molecule_NumHBA_featurizer',
    'Molecule_weights_none_Featurizer',
    'molecule_weights_none_featurizer',
    'Molecule_VSA_EState5_Featurizer',
    'molecule_VSA_EState5_featurizer',
    'Molecule_NumSaturatedHeterocycles_Featurizer',
    'molecule_NumSaturatedHeterocycles_featurizer',
    'Molecule_fr_guanido_Featurizer',
    'molecule_fr_guanido_featurizer',
    'Molecule_fr_tetrazole_Featurizer',
    'molecule_fr_tetrazole_featurizer',
    'Molecule_fr_morpholine_Featurizer',
    'molecule_fr_morpholine_featurizer',
    'Molecule_NumLipinskiHBD_Featurizer',
    'molecule_NumLipinskiHBD_featurizer',
    'Molecule_ExactMolWt_Featurizer',
    'molecule_ExactMolWt_featurizer',
    'Molecule_SMR_VSA1_Featurizer',
    'molecule_SMR_VSA1_featurizer',
    'Molecule_PEOE_VSA14_Featurizer',
    'molecule_PEOE_VSA14_featurizer',
    'Molecule_NumHeteroatoms_Featurizer',
    'molecule_NumHeteroatoms_featurizer',
    'Molecule_VSA_EState7_Featurizer',
    'molecule_VSA_EState7_featurizer',
    'Molecule_fr_epoxide_Featurizer',
    'molecule_fr_epoxide_featurizer',
    'Molecule_PBF_Featurizer',
    'molecule_PBF_featurizer',
    'Molecule_SMR_VSA2_Featurizer',
    'molecule_SMR_VSA2_featurizer',
    'Molecule_BertzCT_Featurizer',
    'molecule_BertzCT_featurizer',
    'Molecule_NumSaturatedRings_Featurizer',
    'molecule_NumSaturatedRings_featurizer',
    'Molecule_fr_pyridine_Featurizer',
    'molecule_fr_pyridine_featurizer',
    'Molecule_NPR1_Featurizer',
    'molecule_NPR1_featurizer',
    'Molecule_fr_allylic_oxid_Featurizer',
    'molecule_fr_allylic_oxid_featurizer',
    'Molecule_Chi0n_Featurizer',
    'molecule_Chi0n_featurizer',
    'Molecule_NumAliphaticRings_Featurizer',
    'molecule_NumAliphaticRings_featurizer',
    'Molecule_Chi1n_Featurizer',
    'molecule_Chi1n_featurizer',
    'Molecule_PEOE_VSA3_Featurizer',
    'molecule_PEOE_VSA3_featurizer',
    'Molecule_fr_sulfonamd_Featurizer',
    'molecule_fr_sulfonamd_featurizer',
    'Molecule_RingCount_Featurizer',
    'molecule_RingCount_featurizer',
    'Molecule_fr_amidine_Featurizer',
    'molecule_fr_amidine_featurizer',
    'Molecule_Kappa1_Featurizer',
    'molecule_Kappa1_featurizer',
    'Molecule_VSA_EState2_Featurizer',
    'molecule_VSA_EState2_featurizer',
    'Molecule_fr_imide_Featurizer',
    'molecule_fr_imide_featurizer',
    'Molecule_fr_Al_OH_noTert_Featurizer',
    'molecule_fr_Al_OH_noTert_featurizer',
    'Molecule_fr_Nhpyrrole_Featurizer',
    'molecule_fr_Nhpyrrole_featurizer',
    'Molecule_NumAliphaticHeterocycles_Featurizer',
    'molecule_NumAliphaticHeterocycles_featurizer',
    'Molecule_fr_para_hydroxylation_Featurizer',
    'molecule_fr_para_hydroxylation_featurizer',
    'Molecule_PEOE_VSA1_Featurizer',
    'molecule_PEOE_VSA1_featurizer',
    'Molecule_PEOE_VSA4_Featurizer',
    'molecule_PEOE_VSA4_featurizer',
    'Molecule_PEOE_VSA2_Featurizer',
    'molecule_PEOE_VSA2_featurizer',
    'Molecule_Chi3v_Featurizer',
    'molecule_Chi3v_featurizer',
    'Molecule_SlogP_VSA6_Featurizer',
    'molecule_SlogP_VSA6_featurizer',
    'Molecule_fr_ketone_Topliss_Featurizer',
    'molecule_fr_ketone_Topliss_featurizer',
    'Molecule_EState_VSA11_Featurizer',
    'molecule_EState_VSA11_featurizer',
    'Molecule_fr_sulfone_Featurizer',
    'molecule_fr_sulfone_featurizer',
    'Molecule_Kappa2_Featurizer',
    'molecule_Kappa2_featurizer',
    'Molecule_fr_unbrch_alkane_Featurizer',
    'molecule_fr_unbrch_alkane_featurizer',
    'Molecule_SMR_VSA4_Featurizer',
    'molecule_SMR_VSA4_featurizer',
    'Molecule_SlogP_VSA7_Featurizer',
    'molecule_SlogP_VSA7_featurizer',
    'Molecule_fr_azo_Featurizer',
    'molecule_fr_azo_featurizer',
    'Molecule_VSA_EState3_Featurizer',
    'molecule_VSA_EState3_featurizer',
    'Molecule_fr_ArN_Featurizer',
    'molecule_fr_ArN_featurizer',
    'Molecule_FormalCharge_Featurizer',
    'molecule_FormalCharge_featurizer',
    'Molecule_Phi_Featurizer',
    'molecule_Phi_featurizer',
    'Molecule_NumValenceElectrons_Featurizer',
    'molecule_NumValenceElectrons_featurizer',
    'Molecule_BCUT2D_MWLOW_Featurizer',
    'molecule_BCUT2D_MWLOW_featurizer',
    'Molecule_EmbedMolecule_Featurizer',
    'molecule_EmbedMolecule_featurizer',
    'Molecule_NumAromaticRings_Featurizer',
    'molecule_NumAromaticRings_featurizer',
    'Molecule_SMR_VSA10_Featurizer',
    'molecule_SMR_VSA10_featurizer',
    'Molecule_NumRotatableBonds_Featurizer',
    'molecule_NumRotatableBonds_featurizer',
    'Molecule_MaxPartialCharge_Featurizer',
    'molecule_MaxPartialCharge_featurizer',
    'Molecule_FpDensityMorgan2_Featurizer',
    'molecule_FpDensityMorgan2_featurizer',
    'Molecule_UFFHasAllMoleculeParams_Featurizer',
    'molecule_UFFHasAllMoleculeParams_featurizer',
    'Molecule_PMI1_Featurizer',
    'molecule_PMI1_featurizer',
    'Molecule_fr_lactam_Featurizer',
    'molecule_fr_lactam_featurizer',
    'Molecule_fr_phenol_Featurizer',
    'molecule_fr_phenol_featurizer',
    'Molecule_VSA_EState1_Featurizer',
    'molecule_VSA_EState1_featurizer',
    'Molecule_NumAliphaticCarbocycles_Featurizer',
    'molecule_NumAliphaticCarbocycles_featurizer',
    'Molecule_PEOE_VSA9_Featurizer',
    'molecule_PEOE_VSA9_featurizer',
    'Molecule_NumHAcceptors_Featurizer',
    'molecule_NumHAcceptors_featurizer',
    'Molecule_NumBridgeheadAtoms_Featurizer',
    'molecule_NumBridgeheadAtoms_featurizer',
    'Molecule_SlogP_VSA9_Featurizer',
    'molecule_SlogP_VSA9_featurizer',
    'Molecule_UFFOptimizeMolecule_Featurizer',
    'molecule_UFFOptimizeMolecule_featurizer',
    'Molecule_fr_bicyclic_Featurizer',
    'molecule_fr_bicyclic_featurizer',
    'Molecule_fr_isothiocyan_Featurizer',
    'molecule_fr_isothiocyan_featurizer',
    'Molecule_Kappa3_Featurizer',
    'molecule_Kappa3_featurizer',
    'Molecule_FpDensityMorgan3_Featurizer',
    'molecule_FpDensityMorgan3_featurizer',
    'Molecule_FractionCSP3_Featurizer',
    'molecule_FractionCSP3_featurizer',
    'Molecule_fr_aryl_methyl_Featurizer',
    'molecule_fr_aryl_methyl_featurizer',
    'Molecule_fr_aldehyde_Featurizer',
    'molecule_fr_aldehyde_featurizer',
    'Molecule_PMI2_Featurizer',
    'molecule_PMI2_featurizer',
    'Molecule_EState_VSA6_Featurizer',
    'molecule_EState_VSA6_featurizer',
    'Molecule_fr_hdrzine_Featurizer',
    'molecule_fr_hdrzine_featurizer',
    'Molecule_Chi1_Featurizer',
    'molecule_Chi1_featurizer',
    'Molecule_HallKierAlpha_Featurizer',
    'molecule_HallKierAlpha_featurizer',
    'Molecule_fr_sulfide_Featurizer',
    'molecule_fr_sulfide_featurizer',
    'Molecule_fr_imidazole_Featurizer',
    'molecule_fr_imidazole_featurizer',
    'Molecule_fr_NH2_Featurizer',
    'molecule_fr_NH2_featurizer',
    'Molecule_MolWt_Featurizer',
    'molecule_MolWt_featurizer',
    'Molecule_fr_oxime_Featurizer',
    'molecule_fr_oxime_featurizer',
    'Molecule_fr_SH_Featurizer',
    'molecule_fr_SH_featurizer',
    'Molecule_fr_phos_ester_Featurizer',
    'molecule_fr_phos_ester_featurizer',
    'Molecule_fr_benzene_Featurizer',
    'molecule_fr_benzene_featurizer',
    'Molecule_fr_nitro_Featurizer',
    'molecule_fr_nitro_featurizer',
    'Molecule_fr_phenol_noOrthoHbond_Featurizer',
    'molecule_fr_phenol_noOrthoHbond_featurizer',
    'Molecule_EState_VSA1_Featurizer',
    'molecule_EState_VSA1_featurizer',
    'Molecule_fr_azide_Featurizer',
    'molecule_fr_azide_featurizer',
    'Molecule_ComputeMolVolume_Featurizer',
    'molecule_ComputeMolVolume_featurizer',
    'Molecule_EState_VSA3_Featurizer',
    'molecule_EState_VSA3_featurizer',
    'Molecule_BCUT2D_LOGPHI_Featurizer',
    'molecule_BCUT2D_LOGPHI_featurizer',
    'Molecule_weights_max_Featurizer',
    'molecule_weights_max_featurizer',
    'Molecule_fr_C_O_noCOO_Featurizer',
    'molecule_fr_C_O_noCOO_featurizer',
    'Molecule_PEOE_VSA11_Featurizer',
    'molecule_PEOE_VSA11_featurizer',
    'Molecule_RadiusOfGyration_Featurizer',
    'molecule_RadiusOfGyration_featurizer',
    'Molecule_fr_dihydropyridine_Featurizer',
    'molecule_fr_dihydropyridine_featurizer',
    'Molecule_fr_alkyl_halide_Featurizer',
    'molecule_fr_alkyl_halide_featurizer',
    'Molecule_FpDensityMorgan1_Featurizer',
    'molecule_FpDensityMorgan1_featurizer',
    'Molecule_MolMR_Featurizer',
    'molecule_MolMR_featurizer',
    'Molecule_Chi2v_Featurizer',
    'molecule_Chi2v_featurizer',
    'Molecule_fr_Ar_COO_Featurizer',
    'molecule_fr_Ar_COO_featurizer',
    'Molecule_MolLogP_Featurizer',
    'molecule_MolLogP_featurizer',
    'Molecule_fr_alkyl_carbamate_Featurizer',
    'molecule_fr_alkyl_carbamate_featurizer',
    'Molecule_qed_Featurizer',
    'molecule_qed_featurizer',
    'Molecule_Chi1v_Featurizer',
    'molecule_Chi1v_featurizer',
    'Molecule_NumRadicalElectrons_Featurizer',
    'molecule_NumRadicalElectrons_featurizer',
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))
    for n, f in get_available_featurizer().items():
        print(n, end=' ')
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == '__main__':
    main()
