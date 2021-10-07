import numpy as np
from molNet.mol.molgraph import mol_graph_from_smiles
from molNet import ConformerError
import os
import json


def func(d):
    f = d[0][3]()
    r = np.zeros((len(d), len(f))) * np.nan
    for i, data in enumerate(d):
        mg = mol_graph_from_smiles(data[0], *data[1], **data[2])
        try:
            mg.featurize_mol(f, name="para_feats")
            r[i] = mg.as_arrays()["graph_features"]["para_feats"]
        except ConformerError:
            pass
    return r


def generate_code_inject(classname, data_folder="ecdf_data"):
    df_cont = os.listdir(data_folder)
    avail_norms = [f.replace(".data", "") for f in df_cont if ".data" in f]
    avail_norms = [f for f in avail_norms if f + ".data" in df_cont]
    avail_norms = [f for f in avail_norms if f + ".ecdf" in df_cont]

    if classname in avail_norms:
        # print(s["classname"])
        with open(os.path.join(data_folder, classname + ".data"), "r") as f:
            ecdf_data = json.load(f)["0"]
        precode = ""
        best = None
        for datakey, parakey, best_key in [
            ("linear_norm", "linear_norm_parameter", "linear"),
            ("min_max_norm", "min_max_norm_parameter", "min_max"),
            ("sig_norm", "sigmoidal_norm_parameter", "sig"),
            ("dual_sig_norm", "dual_sigmoidal_norm_parameter", "dual_sig"),
            ("genlog_norm", "genlog_norm_parameter", "genlog"),
        ]:
            if datakey in ecdf_data:
                norm_data = ecdf_data[datakey]
                precode += (
                    f"    {parakey} = ({', '.join([str(i) for i in norm_data['parameter']])})"
                    + f"  # error of {norm_data['error']:.2E} with sample range ({norm_data['sample_bounds'][0][0]:.2E},{norm_data['sample_bounds'][0][1]:.2E}) resulting in fit range ({norm_data['sample_bounds'][1][0]:.2E},{norm_data['sample_bounds'][1][1]:.2E})\n"
                )

            if (
                "sample_bounds99" not in norm_data
                or norm_data["sample_bounds"][0][0] == norm_data["sample_bounds"][0][1]
            ):
                best = ("unity", 0, norm_data["sample_bounds"])
            else:
                if (
                    norm_data["sample_bounds"][1][0] <= 0.3
                    and norm_data["sample_bounds"][1][1] > 0.5
                ):
                    if best is None:
                        best = (
                            best_key,
                            norm_data["error"],
                            norm_data["sample_bounds"],
                        )
                    else:
                        if norm_data["error"] < best[1]:
                            best = (
                                best_key,
                                norm_data["error"],
                                norm_data["sample_bounds"],
                            )

        if best is not None:
            precode += f"    preferred_normalization = '{best[0]}'"

        return precode
    return None


from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    Asphericity_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_CHGHI_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_CHGLO_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_LOGPHI_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_LOGPLOW_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_MRHI_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_MRLOW_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_MWHI_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    BCUT2D_MWLOW_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import BalabanJ_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import BertzCT_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi0_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi0n_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi0v_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi1n_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi1v_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi2n_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi2v_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi3n_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi3v_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi4n_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Chi4v_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA10_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA11_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA1_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA2_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA3_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA4_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA5_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA6_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA7_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA8_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    EState_VSA9_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    Eccentricity_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import ExactMolWt_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    FpDensityMorgan1_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    FpDensityMorgan2_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    FpDensityMorgan3_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    FractionCSP3_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    GetFormalCharge_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import GetSSSR_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    HallKierAlpha_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    HeavyAtomCount_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    HeavyAtomMolWt_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    InertialShapeFactor_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import Ipc_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Kappa1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Kappa2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Kappa3_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import LabuteASA_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MaxAbsEStateIndex_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MaxAbsPartialCharge_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MaxEStateIndex_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MaxPartialCharge_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MinAbsEStateIndex_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MinAbsPartialCharge_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MinEStateIndex_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    MinPartialCharge_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import MolLogP_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import MolMR_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import MolWt_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NHOHCount_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NOCount_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NPR1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NPR2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAliphaticCarbocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAliphaticHeterocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAliphaticRings_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAmideBonds_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAromaticCarbocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAromaticHeterocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumAromaticRings_Featurizer,
)
from molNet.featurizer.molecule_featurizer import NumAtomsFeaturizer
from molNet.featurizer.molecule_featurizer import NumBondsFeaturizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumBridgeheadAtoms_Featurizer,
)
from molNet.featurizer.molecule_featurizer import NumFragmentsFeaturizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumHAcceptors_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import NumHBA_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NumHBD_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import NumHDonors_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumHeteroatoms_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumHeterocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumLipinskiHBA_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumLipinskiHBD_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumRadicalElectrons_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import NumRings_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumRotatableBonds_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumSaturatedCarbocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumSaturatedHeterocycles_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumSaturatedRings_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumSpiroAtoms_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    NumValenceElectrons_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import PBF_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA10_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA11_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA12_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA13_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA14_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA3_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA4_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA5_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA6_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA7_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA8_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PEOE_VSA9_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PMI1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PMI2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import PMI3_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import Phi_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    RadiusOfGyration_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import RingCount_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA10_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA3_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA4_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA5_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA6_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA7_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA8_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SMR_VSA9_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    SlogP_VSA10_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    SlogP_VSA11_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    SlogP_VSA12_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA3_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA4_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA5_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA6_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA7_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA8_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import SlogP_VSA9_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    SpherocityIndex_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import TPSA_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState10_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState1_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState2_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState3_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState4_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState5_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState6_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState7_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState8_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    VSA_EState9_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Al_COO_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Al_OH_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_Al_OH_noTert_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_ArN_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Ar_COO_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Ar_NH_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Ar_N_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Ar_OH_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_COO2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_COO_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_C_O_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_C_O_noCOO_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_C_S_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_HOCCN_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_Imine_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_NH0_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_NH1_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_NH2_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_N_O_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_Ndealkylation1_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_Ndealkylation2_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_Nhpyrrole_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_SH_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_aldehyde_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_alkyl_carbamate_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_alkyl_halide_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_allylic_oxid_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_amide_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_amidine_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_aniline_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_aryl_methyl_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_azide_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_azo_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_barbitur_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_benzene_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_benzodiazepine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_bicyclic_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_diazo_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_dihydropyridine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_epoxide_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_ester_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_ether_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_furan_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_guanido_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_halogen_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_hdrzine_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_hdrzone_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_imidazole_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_imide_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_isocyan_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_isothiocyan_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_ketone_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_ketone_Topliss_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_lactam_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_lactone_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_methoxy_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_morpholine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_nitrile_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_nitro_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_nitro_arom_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_nitro_arom_nonortho_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_nitroso_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_oxazole_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_oxime_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_para_hydroxylation_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_phenol_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_phenol_noOrthoHbond_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_phos_acid_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_phos_ester_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_piperdine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_piperzine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_priamide_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_prisulfonamd_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_pyridine_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_quatN_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_sulfide_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_sulfonamd_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_sulfone_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_term_acetylene_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_tetrazole_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_thiazole_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_thiocyan_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_thiophene_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import (
    fr_unbrch_alkane_Featurizer,
)
from molNet.featurizer._autogen_molecule_featurizer_numeric import fr_urea_Featurizer
from molNet.featurizer._autogen_molecule_featurizer_numeric import qed_Featurizer

classlist = [
    Asphericity_Featurizer,
    BCUT2D_CHGHI_Featurizer,
    BCUT2D_CHGLO_Featurizer,
    BCUT2D_LOGPHI_Featurizer,
    BCUT2D_LOGPLOW_Featurizer,
    BCUT2D_MRHI_Featurizer,
    BCUT2D_MRLOW_Featurizer,
    BCUT2D_MWHI_Featurizer,
    BCUT2D_MWLOW_Featurizer,
    BalabanJ_Featurizer,
    BertzCT_Featurizer,
    Chi0_Featurizer,
    Chi0n_Featurizer,
    Chi0v_Featurizer,
    Chi1_Featurizer,
    Chi1n_Featurizer,
    Chi1v_Featurizer,
    Chi2n_Featurizer,
    Chi2v_Featurizer,
    Chi3n_Featurizer,
    Chi3v_Featurizer,
    Chi4n_Featurizer,
    Chi4v_Featurizer,
    EState_VSA10_Featurizer,
    EState_VSA11_Featurizer,
    EState_VSA1_Featurizer,
    EState_VSA2_Featurizer,
    EState_VSA3_Featurizer,
    EState_VSA4_Featurizer,
    EState_VSA5_Featurizer,
    EState_VSA6_Featurizer,
    EState_VSA7_Featurizer,
    EState_VSA8_Featurizer,
    EState_VSA9_Featurizer,
    Eccentricity_Featurizer,
    ExactMolWt_Featurizer,
    FpDensityMorgan1_Featurizer,
    FpDensityMorgan2_Featurizer,
    FpDensityMorgan3_Featurizer,
    FractionCSP3_Featurizer,
    GetFormalCharge_Featurizer,
    GetSSSR_Featurizer,
    HallKierAlpha_Featurizer,
    HeavyAtomCount_Featurizer,
    HeavyAtomMolWt_Featurizer,
    InertialShapeFactor_Featurizer,
    Ipc_Featurizer,
    Kappa1_Featurizer,
    Kappa2_Featurizer,
    Kappa3_Featurizer,
    LabuteASA_Featurizer,
    MaxAbsEStateIndex_Featurizer,
    MaxAbsPartialCharge_Featurizer,
    MaxEStateIndex_Featurizer,
    MaxPartialCharge_Featurizer,
    MinAbsEStateIndex_Featurizer,
    MinAbsPartialCharge_Featurizer,
    MinEStateIndex_Featurizer,
    MinPartialCharge_Featurizer,
    MolLogP_Featurizer,
    MolMR_Featurizer,
    MolWt_Featurizer,
    NHOHCount_Featurizer,
    NOCount_Featurizer,
    NPR1_Featurizer,
    NPR2_Featurizer,
    NumAliphaticCarbocycles_Featurizer,
    NumAliphaticHeterocycles_Featurizer,
    NumAliphaticRings_Featurizer,
    NumAmideBonds_Featurizer,
    NumAromaticCarbocycles_Featurizer,
    NumAromaticHeterocycles_Featurizer,
    NumAromaticRings_Featurizer,
    NumAtomsFeaturizer,
    NumBondsFeaturizer,
    NumBridgeheadAtoms_Featurizer,
    NumFragmentsFeaturizer,
    NumHAcceptors_Featurizer,
    NumHBA_Featurizer,
    NumHBD_Featurizer,
    NumHDonors_Featurizer,
    NumHeteroatoms_Featurizer,
    NumHeterocycles_Featurizer,
    NumLipinskiHBA_Featurizer,
    NumLipinskiHBD_Featurizer,
    NumRadicalElectrons_Featurizer,
    NumRings_Featurizer,
    NumRotatableBonds_Featurizer,
    NumSaturatedCarbocycles_Featurizer,
    NumSaturatedHeterocycles_Featurizer,
    NumSaturatedRings_Featurizer,
    NumSpiroAtoms_Featurizer,
    NumValenceElectrons_Featurizer,
    PBF_Featurizer,
    PEOE_VSA10_Featurizer,
    PEOE_VSA11_Featurizer,
    PEOE_VSA12_Featurizer,
    PEOE_VSA13_Featurizer,
    PEOE_VSA14_Featurizer,
    PEOE_VSA1_Featurizer,
    PEOE_VSA2_Featurizer,
    PEOE_VSA3_Featurizer,
    PEOE_VSA4_Featurizer,
    PEOE_VSA5_Featurizer,
    PEOE_VSA6_Featurizer,
    PEOE_VSA7_Featurizer,
    PEOE_VSA8_Featurizer,
    PEOE_VSA9_Featurizer,
    PMI1_Featurizer,
    PMI2_Featurizer,
    PMI3_Featurizer,
    Phi_Featurizer,
    RadiusOfGyration_Featurizer,
    RingCount_Featurizer,
    SMR_VSA10_Featurizer,
    SMR_VSA1_Featurizer,
    SMR_VSA2_Featurizer,
    SMR_VSA3_Featurizer,
    SMR_VSA4_Featurizer,
    SMR_VSA5_Featurizer,
    SMR_VSA6_Featurizer,
    SMR_VSA7_Featurizer,
    SMR_VSA8_Featurizer,
    SMR_VSA9_Featurizer,
    SlogP_VSA10_Featurizer,
    SlogP_VSA11_Featurizer,
    SlogP_VSA12_Featurizer,
    SlogP_VSA1_Featurizer,
    SlogP_VSA2_Featurizer,
    SlogP_VSA3_Featurizer,
    SlogP_VSA4_Featurizer,
    SlogP_VSA5_Featurizer,
    SlogP_VSA6_Featurizer,
    SlogP_VSA7_Featurizer,
    SlogP_VSA8_Featurizer,
    SlogP_VSA9_Featurizer,
    SpherocityIndex_Featurizer,
    TPSA_Featurizer,
    VSA_EState10_Featurizer,
    VSA_EState1_Featurizer,
    VSA_EState2_Featurizer,
    VSA_EState3_Featurizer,
    VSA_EState4_Featurizer,
    VSA_EState5_Featurizer,
    VSA_EState6_Featurizer,
    VSA_EState7_Featurizer,
    VSA_EState8_Featurizer,
    VSA_EState9_Featurizer,
    fr_Al_COO_Featurizer,
    fr_Al_OH_Featurizer,
    fr_Al_OH_noTert_Featurizer,
    fr_ArN_Featurizer,
    fr_Ar_COO_Featurizer,
    fr_Ar_NH_Featurizer,
    fr_Ar_N_Featurizer,
    fr_Ar_OH_Featurizer,
    fr_COO2_Featurizer,
    fr_COO_Featurizer,
    fr_C_O_Featurizer,
    fr_C_O_noCOO_Featurizer,
    fr_C_S_Featurizer,
    fr_HOCCN_Featurizer,
    fr_Imine_Featurizer,
    fr_NH0_Featurizer,
    fr_NH1_Featurizer,
    fr_NH2_Featurizer,
    fr_N_O_Featurizer,
    fr_Ndealkylation1_Featurizer,
    fr_Ndealkylation2_Featurizer,
    fr_Nhpyrrole_Featurizer,
    fr_SH_Featurizer,
    fr_aldehyde_Featurizer,
    fr_alkyl_carbamate_Featurizer,
    fr_alkyl_halide_Featurizer,
    fr_allylic_oxid_Featurizer,
    fr_amide_Featurizer,
    fr_amidine_Featurizer,
    fr_aniline_Featurizer,
    fr_aryl_methyl_Featurizer,
    fr_azide_Featurizer,
    fr_azo_Featurizer,
    fr_barbitur_Featurizer,
    fr_benzene_Featurizer,
    fr_benzodiazepine_Featurizer,
    fr_bicyclic_Featurizer,
    fr_diazo_Featurizer,
    fr_dihydropyridine_Featurizer,
    fr_epoxide_Featurizer,
    fr_ester_Featurizer,
    fr_ether_Featurizer,
    fr_furan_Featurizer,
    fr_guanido_Featurizer,
    fr_halogen_Featurizer,
    fr_hdrzine_Featurizer,
    fr_hdrzone_Featurizer,
    fr_imidazole_Featurizer,
    fr_imide_Featurizer,
    fr_isocyan_Featurizer,
    fr_isothiocyan_Featurizer,
    fr_ketone_Featurizer,
    fr_ketone_Topliss_Featurizer,
    fr_lactam_Featurizer,
    fr_lactone_Featurizer,
    fr_methoxy_Featurizer,
    fr_morpholine_Featurizer,
    fr_nitrile_Featurizer,
    fr_nitro_Featurizer,
    fr_nitro_arom_Featurizer,
    fr_nitro_arom_nonortho_Featurizer,
    fr_nitroso_Featurizer,
    fr_oxazole_Featurizer,
    fr_oxime_Featurizer,
    fr_para_hydroxylation_Featurizer,
    fr_phenol_Featurizer,
    fr_phenol_noOrthoHbond_Featurizer,
    fr_phos_acid_Featurizer,
    fr_phos_ester_Featurizer,
    fr_piperdine_Featurizer,
    fr_piperzine_Featurizer,
    fr_priamide_Featurizer,
    fr_prisulfonamd_Featurizer,
    fr_pyridine_Featurizer,
    fr_quatN_Featurizer,
    fr_sulfide_Featurizer,
    fr_sulfonamd_Featurizer,
    fr_sulfone_Featurizer,
    fr_term_acetylene_Featurizer,
    fr_tetrazole_Featurizer,
    fr_thiazole_Featurizer,
    fr_thiocyan_Featurizer,
    fr_thiophene_Featurizer,
    fr_unbrch_alkane_Featurizer,
    fr_urea_Featurizer,
    qed_Featurizer,
]

from molNet.featurizer.featurizer import FeaturizerList


class FixSVFeaturizerList(FeaturizerList):
    def __init__(self):
        super().__init__(feature_list=[k() for k in classlist])
