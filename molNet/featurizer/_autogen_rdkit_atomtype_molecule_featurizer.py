import numpy as np
from rdkit.Chem import rdqueries

from molNet.featurizer._molecule_featurizer import SingleValueMoleculeFeaturizer


class Molecule_NumberAtomsRgroup_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(0)))


class Molecule_RelativeContentRgroup_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(0)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRgroup_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(0))) > 0


class Molecule_NumberAtomsH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1)))


class Molecule_RelativeContentH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1)))
                / mol.GetNumAtoms()
        )


class Molecule_HasH_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(1))) > 0


class Molecule_NumberAtomsHe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(2)))


class Molecule_RelativeContentHe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(2)))
                / mol.GetNumAtoms()
        )


class Molecule_HasHe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(2))) > 0


class Molecule_NumberAtomsLi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(3)))


class Molecule_RelativeContentLi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(3)))
                / mol.GetNumAtoms()
        )


class Molecule_HasLi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(3))) > 0


class Molecule_NumberAtomsBe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(4)))


class Molecule_RelativeContentBe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(4)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(4))) > 0


class Molecule_NumberAtomsB_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(5)))


class Molecule_RelativeContentB_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(5)))
                / mol.GetNumAtoms()
        )


class Molecule_HasB_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(5))) > 0


class Molecule_NumberAtomsC_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6)))


class Molecule_RelativeContentC_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6)))
                / mol.GetNumAtoms()
        )


class Molecule_HasC_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(6))) > 0


class Molecule_NumberAtomsN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(7)))


class Molecule_RelativeContentN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(7)))
                / mol.GetNumAtoms()
        )


class Molecule_HasN_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(7))) > 0


class Molecule_NumberAtomsO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(8)))


class Molecule_RelativeContentO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(8)))
                / mol.GetNumAtoms()
        )


class Molecule_HasO_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(8))) > 0


class Molecule_NumberAtomsF_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(9)))


class Molecule_RelativeContentF_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(9)))
                / mol.GetNumAtoms()
        )


class Molecule_HasF_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(9))) > 0


class Molecule_NumberAtomsNe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(10)))


class Molecule_RelativeContentNe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(10)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(10))) > 0


class Molecule_NumberAtomsNa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(11)))


class Molecule_RelativeContentNa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(11)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(11))) > 0


class Molecule_NumberAtomsMg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(12)))


class Molecule_RelativeContentMg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(12)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(12))) > 0


class Molecule_NumberAtomsAl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(13)))


class Molecule_RelativeContentAl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(13)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(13))) > 0


class Molecule_NumberAtomsSi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(14)))


class Molecule_RelativeContentSi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(14)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(14))) > 0


class Molecule_NumberAtomsP_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(15)))


class Molecule_RelativeContentP_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(15)))
                / mol.GetNumAtoms()
        )


class Molecule_HasP_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(15))) > 0


class Molecule_NumberAtomsS_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(16)))


class Molecule_RelativeContentS_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(16)))
                / mol.GetNumAtoms()
        )


class Molecule_HasS_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(16))) > 0


class Molecule_NumberAtomsCl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(17)))


class Molecule_RelativeContentCl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(17)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(17))) > 0


class Molecule_NumberAtomsAr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(18)))


class Molecule_RelativeContentAr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(18)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(18))) > 0


class Molecule_NumberAtomsK_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(19)))


class Molecule_RelativeContentK_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(19)))
                / mol.GetNumAtoms()
        )


class Molecule_HasK_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(19))) > 0


class Molecule_NumberAtomsCa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(20)))


class Molecule_RelativeContentCa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(20)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(20))) > 0


class Molecule_NumberAtomsSc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(21)))


class Molecule_RelativeContentSc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(21)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(21))) > 0


class Molecule_NumberAtomsTi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(22)))


class Molecule_RelativeContentTi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(22)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(22))) > 0


class Molecule_NumberAtomsV_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(23)))


class Molecule_RelativeContentV_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(23)))
                / mol.GetNumAtoms()
        )


class Molecule_HasV_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(23))) > 0


class Molecule_NumberAtomsCr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(24)))


class Molecule_RelativeContentCr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(24)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(24))) > 0


class Molecule_NumberAtomsMn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(25)))


class Molecule_RelativeContentMn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(25)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(25))) > 0


class Molecule_NumberAtomsFe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(26)))


class Molecule_RelativeContentFe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(26)))
                / mol.GetNumAtoms()
        )


class Molecule_HasFe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(26))) > 0


class Molecule_NumberAtomsCo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(27)))


class Molecule_RelativeContentCo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(27)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(27))) > 0


class Molecule_NumberAtomsNi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(28)))


class Molecule_RelativeContentNi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(28)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(28))) > 0


class Molecule_NumberAtomsCu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(29)))


class Molecule_RelativeContentCu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(29)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(29))) > 0


class Molecule_NumberAtomsZn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(30)))


class Molecule_RelativeContentZn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(30)))
                / mol.GetNumAtoms()
        )


class Molecule_HasZn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(30))) > 0


class Molecule_NumberAtomsGa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(31)))


class Molecule_RelativeContentGa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(31)))
                / mol.GetNumAtoms()
        )


class Molecule_HasGa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(31))) > 0


class Molecule_NumberAtomsGe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(32)))


class Molecule_RelativeContentGe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(32)))
                / mol.GetNumAtoms()
        )


class Molecule_HasGe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(32))) > 0


class Molecule_NumberAtomsAs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(33)))


class Molecule_RelativeContentAs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(33)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(33))) > 0


class Molecule_NumberAtomsSe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(34)))


class Molecule_RelativeContentSe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(34)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(34))) > 0


class Molecule_NumberAtomsBr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(35)))


class Molecule_RelativeContentBr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(35)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(35))) > 0


class Molecule_NumberAtomsKr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(36)))


class Molecule_RelativeContentKr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(36)))
                / mol.GetNumAtoms()
        )


class Molecule_HasKr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(36))) > 0


class Molecule_NumberAtomsRb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(37)))


class Molecule_RelativeContentRb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(37)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(37))) > 0


class Molecule_NumberAtomsSr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(38)))


class Molecule_RelativeContentSr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(38)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(38))) > 0


class Molecule_NumberAtomsY_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(39)))


class Molecule_RelativeContentY_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(39)))
                / mol.GetNumAtoms()
        )


class Molecule_HasY_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(39))) > 0


class Molecule_NumberAtomsZr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(40)))


class Molecule_RelativeContentZr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(40)))
                / mol.GetNumAtoms()
        )


class Molecule_HasZr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(40))) > 0


class Molecule_NumberAtomsNb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(41)))


class Molecule_RelativeContentNb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(41)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(41))) > 0


class Molecule_NumberAtomsMo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(42)))


class Molecule_RelativeContentMo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(42)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(42))) > 0


class Molecule_NumberAtomsTc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(43)))


class Molecule_RelativeContentTc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(43)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(43))) > 0


class Molecule_NumberAtomsRu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(44)))


class Molecule_RelativeContentRu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(44)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(44))) > 0


class Molecule_NumberAtomsRh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(45)))


class Molecule_RelativeContentRh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(45)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(45))) > 0


class Molecule_NumberAtomsPd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(46)))


class Molecule_RelativeContentPd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(46)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(46))) > 0


class Molecule_NumberAtomsAg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(47)))


class Molecule_RelativeContentAg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(47)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(47))) > 0


class Molecule_NumberAtomsCd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(48)))


class Molecule_RelativeContentCd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(48)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(48))) > 0


class Molecule_NumberAtomsIn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(49)))


class Molecule_RelativeContentIn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(49)))
                / mol.GetNumAtoms()
        )


class Molecule_HasIn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(49))) > 0


class Molecule_NumberAtomsSn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(50)))


class Molecule_RelativeContentSn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(50)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(50))) > 0


class Molecule_NumberAtomsSb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(51)))


class Molecule_RelativeContentSb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(51)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(51))) > 0


class Molecule_NumberAtomsTe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(52)))


class Molecule_RelativeContentTe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(52)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(52))) > 0


class Molecule_NumberAtomsI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(53)))


class Molecule_RelativeContentI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(53)))
                / mol.GetNumAtoms()
        )


class Molecule_HasI_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(53))) > 0


class Molecule_NumberAtomsXe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(54)))


class Molecule_RelativeContentXe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(54)))
                / mol.GetNumAtoms()
        )


class Molecule_HasXe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(54))) > 0


class Molecule_NumberAtomsCs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(55)))


class Molecule_RelativeContentCs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(55)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(55))) > 0


class Molecule_NumberAtomsBa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(56)))


class Molecule_RelativeContentBa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(56)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(56))) > 0


class Molecule_NumberAtomsLa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(57)))


class Molecule_RelativeContentLa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(57)))
                / mol.GetNumAtoms()
        )


class Molecule_HasLa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(57))) > 0


class Molecule_NumberAtomsCe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(58)))


class Molecule_RelativeContentCe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(58)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(58))) > 0


class Molecule_NumberAtomsPr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(59)))


class Molecule_RelativeContentPr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(59)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(59))) > 0


class Molecule_NumberAtomsNd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(60)))


class Molecule_RelativeContentNd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(60)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(60))) > 0


class Molecule_NumberAtomsPm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(61)))


class Molecule_RelativeContentPm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(61)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(61))) > 0


class Molecule_NumberAtomsSm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(62)))


class Molecule_RelativeContentSm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(62)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(62))) > 0


class Molecule_NumberAtomsEu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(63)))


class Molecule_RelativeContentEu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(63)))
                / mol.GetNumAtoms()
        )


class Molecule_HasEu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(63))) > 0


class Molecule_NumberAtomsGd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(64)))


class Molecule_RelativeContentGd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(64)))
                / mol.GetNumAtoms()
        )


class Molecule_HasGd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(64))) > 0


class Molecule_NumberAtomsTb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(65)))


class Molecule_RelativeContentTb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(65)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(65))) > 0


class Molecule_NumberAtomsDy_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(66)))


class Molecule_RelativeContentDy_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(66)))
                / mol.GetNumAtoms()
        )


class Molecule_HasDy_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(66))) > 0


class Molecule_NumberAtomsHo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(67)))


class Molecule_RelativeContentHo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(67)))
                / mol.GetNumAtoms()
        )


class Molecule_HasHo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(67))) > 0


class Molecule_NumberAtomsEr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(68)))


class Molecule_RelativeContentEr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(68)))
                / mol.GetNumAtoms()
        )


class Molecule_HasEr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(68))) > 0


class Molecule_NumberAtomsTm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(69)))


class Molecule_RelativeContentTm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(69)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(69))) > 0


class Molecule_NumberAtomsYb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(70)))


class Molecule_RelativeContentYb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(70)))
                / mol.GetNumAtoms()
        )


class Molecule_HasYb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(70))) > 0


class Molecule_NumberAtomsLu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(71)))


class Molecule_RelativeContentLu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(71)))
                / mol.GetNumAtoms()
        )


class Molecule_HasLu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(71))) > 0


class Molecule_NumberAtomsHf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(72)))


class Molecule_RelativeContentHf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(72)))
                / mol.GetNumAtoms()
        )


class Molecule_HasHf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(72))) > 0


class Molecule_NumberAtomsTa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(73)))


class Molecule_RelativeContentTa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(73)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(73))) > 0


class Molecule_NumberAtomsW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(74)))


class Molecule_RelativeContentW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(74)))
                / mol.GetNumAtoms()
        )


class Molecule_HasW_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(74))) > 0


class Molecule_NumberAtomsRe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(75)))


class Molecule_RelativeContentRe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(75)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRe_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(75))) > 0


class Molecule_NumberAtomsOs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(76)))


class Molecule_RelativeContentOs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(76)))
                / mol.GetNumAtoms()
        )


class Molecule_HasOs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(76))) > 0


class Molecule_NumberAtomsIr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(77)))


class Molecule_RelativeContentIr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(77)))
                / mol.GetNumAtoms()
        )


class Molecule_HasIr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(77))) > 0


class Molecule_NumberAtomsPt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(78)))


class Molecule_RelativeContentPt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(78)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(78))) > 0


class Molecule_NumberAtomsAu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(79)))


class Molecule_RelativeContentAu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(79)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(79))) > 0


class Molecule_NumberAtomsHg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(80)))


class Molecule_RelativeContentHg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(80)))
                / mol.GetNumAtoms()
        )


class Molecule_HasHg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(80))) > 0


class Molecule_NumberAtomsTl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(81)))


class Molecule_RelativeContentTl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(81)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(81))) > 0


class Molecule_NumberAtomsPb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(82)))


class Molecule_RelativeContentPb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(82)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(82))) > 0


class Molecule_NumberAtomsBi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(83)))


class Molecule_RelativeContentBi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(83)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBi_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(83))) > 0


class Molecule_NumberAtomsPo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(84)))


class Molecule_RelativeContentPo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(84)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(84))) > 0


class Molecule_NumberAtomsAt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(85)))


class Molecule_RelativeContentAt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(85)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(85))) > 0


class Molecule_NumberAtomsRn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(86)))


class Molecule_RelativeContentRn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(86)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(86))) > 0


class Molecule_NumberAtomsFr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(87)))


class Molecule_RelativeContentFr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(87)))
                / mol.GetNumAtoms()
        )


class Molecule_HasFr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(87))) > 0


class Molecule_NumberAtomsRa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(88)))


class Molecule_RelativeContentRa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(88)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(88))) > 0


class Molecule_NumberAtomsAc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(89)))


class Molecule_RelativeContentAc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(89)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(89))) > 0


class Molecule_NumberAtomsTh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(90)))


class Molecule_RelativeContentTh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(90)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(90))) > 0


class Molecule_NumberAtomsPa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(91)))


class Molecule_RelativeContentPa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(91)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPa_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(91))) > 0


class Molecule_NumberAtomsU_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(92)))


class Molecule_RelativeContentU_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(92)))
                / mol.GetNumAtoms()
        )


class Molecule_HasU_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(92))) > 0


class Molecule_NumberAtomsNp_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(93)))


class Molecule_RelativeContentNp_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(93)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNp_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(93))) > 0


class Molecule_NumberAtomsPu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(94)))


class Molecule_RelativeContentPu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(94)))
                / mol.GetNumAtoms()
        )


class Molecule_HasPu_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(94))) > 0


class Molecule_NumberAtomsAm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(95)))


class Molecule_RelativeContentAm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(95)))
                / mol.GetNumAtoms()
        )


class Molecule_HasAm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(95))) > 0


class Molecule_NumberAtomsCm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(96)))


class Molecule_RelativeContentCm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(96)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(96))) > 0


class Molecule_NumberAtomsBk_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(97)))


class Molecule_RelativeContentBk_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(97)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBk_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(97))) > 0


class Molecule_NumberAtomsCf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(98)))


class Molecule_RelativeContentCf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(98)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(98))) > 0


class Molecule_NumberAtomsEs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(99)))


class Molecule_RelativeContentEs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(99)))
                / mol.GetNumAtoms()
        )


class Molecule_HasEs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(99))) > 0


class Molecule_NumberAtomsFm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(100)))


class Molecule_RelativeContentFm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(100)))
                / mol.GetNumAtoms()
        )


class Molecule_HasFm_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(100))) > 0


class Molecule_NumberAtomsMd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(101)))


class Molecule_RelativeContentMd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(101)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMd_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(101))) > 0


class Molecule_NumberAtomsNo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(102)))


class Molecule_RelativeContentNo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(102)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNo_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(102))) > 0


class Molecule_NumberAtomsLr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(103)))


class Molecule_RelativeContentLr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(103)))
                / mol.GetNumAtoms()
        )


class Molecule_HasLr_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(103))) > 0


class Molecule_NumberAtomsRf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(104)))


class Molecule_RelativeContentRf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(104)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRf_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(104))) > 0


class Molecule_NumberAtomsDb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(105)))


class Molecule_RelativeContentDb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(105)))
                / mol.GetNumAtoms()
        )


class Molecule_HasDb_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(105))) > 0


class Molecule_NumberAtomsSg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(106)))


class Molecule_RelativeContentSg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(106)))
                / mol.GetNumAtoms()
        )


class Molecule_HasSg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(106))) > 0


class Molecule_NumberAtomsBh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(107)))


class Molecule_RelativeContentBh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(107)))
                / mol.GetNumAtoms()
        )


class Molecule_HasBh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(107))) > 0


class Molecule_NumberAtomsHs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(108)))


class Molecule_RelativeContentHs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(108)))
                / mol.GetNumAtoms()
        )


class Molecule_HasHs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(108))) > 0


class Molecule_NumberAtomsMt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(109)))


class Molecule_RelativeContentMt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(109)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMt_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(109))) > 0


class Molecule_NumberAtomsDs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(110)))


class Molecule_RelativeContentDs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(110)))
                / mol.GetNumAtoms()
        )


class Molecule_HasDs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(110))) > 0


class Molecule_NumberAtomsRg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(111)))


class Molecule_RelativeContentRg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(111)))
                / mol.GetNumAtoms()
        )


class Molecule_HasRg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(111))) > 0


class Molecule_NumberAtomsCn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(112)))


class Molecule_RelativeContentCn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(112)))
                / mol.GetNumAtoms()
        )


class Molecule_HasCn_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(112))) > 0


class Molecule_NumberAtomsNh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(113)))


class Molecule_RelativeContentNh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(113)))
                / mol.GetNumAtoms()
        )


class Molecule_HasNh_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(113))) > 0


class Molecule_NumberAtomsFl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(114)))


class Molecule_RelativeContentFl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(114)))
                / mol.GetNumAtoms()
        )


class Molecule_HasFl_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(114))) > 0


class Molecule_NumberAtomsMc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(115)))


class Molecule_RelativeContentMc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(115)))
                / mol.GetNumAtoms()
        )


class Molecule_HasMc_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(115))) > 0


class Molecule_NumberAtomsLv_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(116)))


class Molecule_RelativeContentLv_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(116)))
                / mol.GetNumAtoms()
        )


class Molecule_HasLv_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(116))) > 0


class Molecule_NumberAtomsTs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(117)))


class Molecule_RelativeContentTs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(117)))
                / mol.GetNumAtoms()
        )


class Molecule_HasTs_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(117))) > 0


class Molecule_NumberAtomsOg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(118)))


class Molecule_RelativeContentOg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32

    def featurize(self, mol):
        return (
                len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(118)))
                / mol.GetNumAtoms()
        )


class Molecule_HasOg_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool

    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(118))) > 0


molecule_NumberAtomsRgroup_featurizer = Molecule_NumberAtomsRgroup_Featurizer()
molecule_RelativeContentRgroup_featurizer = Molecule_RelativeContentRgroup_Featurizer()
molecule_HasRgroup_featurizer = Molecule_HasRgroup_Featurizer()
molecule_NumberAtomsH_featurizer = Molecule_NumberAtomsH_Featurizer()
molecule_RelativeContentH_featurizer = Molecule_RelativeContentH_Featurizer()
molecule_HasH_featurizer = Molecule_HasH_Featurizer()
molecule_NumberAtomsHe_featurizer = Molecule_NumberAtomsHe_Featurizer()
molecule_RelativeContentHe_featurizer = Molecule_RelativeContentHe_Featurizer()
molecule_HasHe_featurizer = Molecule_HasHe_Featurizer()
molecule_NumberAtomsLi_featurizer = Molecule_NumberAtomsLi_Featurizer()
molecule_RelativeContentLi_featurizer = Molecule_RelativeContentLi_Featurizer()
molecule_HasLi_featurizer = Molecule_HasLi_Featurizer()
molecule_NumberAtomsBe_featurizer = Molecule_NumberAtomsBe_Featurizer()
molecule_RelativeContentBe_featurizer = Molecule_RelativeContentBe_Featurizer()
molecule_HasBe_featurizer = Molecule_HasBe_Featurizer()
molecule_NumberAtomsB_featurizer = Molecule_NumberAtomsB_Featurizer()
molecule_RelativeContentB_featurizer = Molecule_RelativeContentB_Featurizer()
molecule_HasB_featurizer = Molecule_HasB_Featurizer()
molecule_NumberAtomsC_featurizer = Molecule_NumberAtomsC_Featurizer()
molecule_RelativeContentC_featurizer = Molecule_RelativeContentC_Featurizer()
molecule_HasC_featurizer = Molecule_HasC_Featurizer()
molecule_NumberAtomsN_featurizer = Molecule_NumberAtomsN_Featurizer()
molecule_RelativeContentN_featurizer = Molecule_RelativeContentN_Featurizer()
molecule_HasN_featurizer = Molecule_HasN_Featurizer()
molecule_NumberAtomsO_featurizer = Molecule_NumberAtomsO_Featurizer()
molecule_RelativeContentO_featurizer = Molecule_RelativeContentO_Featurizer()
molecule_HasO_featurizer = Molecule_HasO_Featurizer()
molecule_NumberAtomsF_featurizer = Molecule_NumberAtomsF_Featurizer()
molecule_RelativeContentF_featurizer = Molecule_RelativeContentF_Featurizer()
molecule_HasF_featurizer = Molecule_HasF_Featurizer()
molecule_NumberAtomsNe_featurizer = Molecule_NumberAtomsNe_Featurizer()
molecule_RelativeContentNe_featurizer = Molecule_RelativeContentNe_Featurizer()
molecule_HasNe_featurizer = Molecule_HasNe_Featurizer()
molecule_NumberAtomsNa_featurizer = Molecule_NumberAtomsNa_Featurizer()
molecule_RelativeContentNa_featurizer = Molecule_RelativeContentNa_Featurizer()
molecule_HasNa_featurizer = Molecule_HasNa_Featurizer()
molecule_NumberAtomsMg_featurizer = Molecule_NumberAtomsMg_Featurizer()
molecule_RelativeContentMg_featurizer = Molecule_RelativeContentMg_Featurizer()
molecule_HasMg_featurizer = Molecule_HasMg_Featurizer()
molecule_NumberAtomsAl_featurizer = Molecule_NumberAtomsAl_Featurizer()
molecule_RelativeContentAl_featurizer = Molecule_RelativeContentAl_Featurizer()
molecule_HasAl_featurizer = Molecule_HasAl_Featurizer()
molecule_NumberAtomsSi_featurizer = Molecule_NumberAtomsSi_Featurizer()
molecule_RelativeContentSi_featurizer = Molecule_RelativeContentSi_Featurizer()
molecule_HasSi_featurizer = Molecule_HasSi_Featurizer()
molecule_NumberAtomsP_featurizer = Molecule_NumberAtomsP_Featurizer()
molecule_RelativeContentP_featurizer = Molecule_RelativeContentP_Featurizer()
molecule_HasP_featurizer = Molecule_HasP_Featurizer()
molecule_NumberAtomsS_featurizer = Molecule_NumberAtomsS_Featurizer()
molecule_RelativeContentS_featurizer = Molecule_RelativeContentS_Featurizer()
molecule_HasS_featurizer = Molecule_HasS_Featurizer()
molecule_NumberAtomsCl_featurizer = Molecule_NumberAtomsCl_Featurizer()
molecule_RelativeContentCl_featurizer = Molecule_RelativeContentCl_Featurizer()
molecule_HasCl_featurizer = Molecule_HasCl_Featurizer()
molecule_NumberAtomsAr_featurizer = Molecule_NumberAtomsAr_Featurizer()
molecule_RelativeContentAr_featurizer = Molecule_RelativeContentAr_Featurizer()
molecule_HasAr_featurizer = Molecule_HasAr_Featurizer()
molecule_NumberAtomsK_featurizer = Molecule_NumberAtomsK_Featurizer()
molecule_RelativeContentK_featurizer = Molecule_RelativeContentK_Featurizer()
molecule_HasK_featurizer = Molecule_HasK_Featurizer()
molecule_NumberAtomsCa_featurizer = Molecule_NumberAtomsCa_Featurizer()
molecule_RelativeContentCa_featurizer = Molecule_RelativeContentCa_Featurizer()
molecule_HasCa_featurizer = Molecule_HasCa_Featurizer()
molecule_NumberAtomsSc_featurizer = Molecule_NumberAtomsSc_Featurizer()
molecule_RelativeContentSc_featurizer = Molecule_RelativeContentSc_Featurizer()
molecule_HasSc_featurizer = Molecule_HasSc_Featurizer()
molecule_NumberAtomsTi_featurizer = Molecule_NumberAtomsTi_Featurizer()
molecule_RelativeContentTi_featurizer = Molecule_RelativeContentTi_Featurizer()
molecule_HasTi_featurizer = Molecule_HasTi_Featurizer()
molecule_NumberAtomsV_featurizer = Molecule_NumberAtomsV_Featurizer()
molecule_RelativeContentV_featurizer = Molecule_RelativeContentV_Featurizer()
molecule_HasV_featurizer = Molecule_HasV_Featurizer()
molecule_NumberAtomsCr_featurizer = Molecule_NumberAtomsCr_Featurizer()
molecule_RelativeContentCr_featurizer = Molecule_RelativeContentCr_Featurizer()
molecule_HasCr_featurizer = Molecule_HasCr_Featurizer()
molecule_NumberAtomsMn_featurizer = Molecule_NumberAtomsMn_Featurizer()
molecule_RelativeContentMn_featurizer = Molecule_RelativeContentMn_Featurizer()
molecule_HasMn_featurizer = Molecule_HasMn_Featurizer()
molecule_NumberAtomsFe_featurizer = Molecule_NumberAtomsFe_Featurizer()
molecule_RelativeContentFe_featurizer = Molecule_RelativeContentFe_Featurizer()
molecule_HasFe_featurizer = Molecule_HasFe_Featurizer()
molecule_NumberAtomsCo_featurizer = Molecule_NumberAtomsCo_Featurizer()
molecule_RelativeContentCo_featurizer = Molecule_RelativeContentCo_Featurizer()
molecule_HasCo_featurizer = Molecule_HasCo_Featurizer()
molecule_NumberAtomsNi_featurizer = Molecule_NumberAtomsNi_Featurizer()
molecule_RelativeContentNi_featurizer = Molecule_RelativeContentNi_Featurizer()
molecule_HasNi_featurizer = Molecule_HasNi_Featurizer()
molecule_NumberAtomsCu_featurizer = Molecule_NumberAtomsCu_Featurizer()
molecule_RelativeContentCu_featurizer = Molecule_RelativeContentCu_Featurizer()
molecule_HasCu_featurizer = Molecule_HasCu_Featurizer()
molecule_NumberAtomsZn_featurizer = Molecule_NumberAtomsZn_Featurizer()
molecule_RelativeContentZn_featurizer = Molecule_RelativeContentZn_Featurizer()
molecule_HasZn_featurizer = Molecule_HasZn_Featurizer()
molecule_NumberAtomsGa_featurizer = Molecule_NumberAtomsGa_Featurizer()
molecule_RelativeContentGa_featurizer = Molecule_RelativeContentGa_Featurizer()
molecule_HasGa_featurizer = Molecule_HasGa_Featurizer()
molecule_NumberAtomsGe_featurizer = Molecule_NumberAtomsGe_Featurizer()
molecule_RelativeContentGe_featurizer = Molecule_RelativeContentGe_Featurizer()
molecule_HasGe_featurizer = Molecule_HasGe_Featurizer()
molecule_NumberAtomsAs_featurizer = Molecule_NumberAtomsAs_Featurizer()
molecule_RelativeContentAs_featurizer = Molecule_RelativeContentAs_Featurizer()
molecule_HasAs_featurizer = Molecule_HasAs_Featurizer()
molecule_NumberAtomsSe_featurizer = Molecule_NumberAtomsSe_Featurizer()
molecule_RelativeContentSe_featurizer = Molecule_RelativeContentSe_Featurizer()
molecule_HasSe_featurizer = Molecule_HasSe_Featurizer()
molecule_NumberAtomsBr_featurizer = Molecule_NumberAtomsBr_Featurizer()
molecule_RelativeContentBr_featurizer = Molecule_RelativeContentBr_Featurizer()
molecule_HasBr_featurizer = Molecule_HasBr_Featurizer()
molecule_NumberAtomsKr_featurizer = Molecule_NumberAtomsKr_Featurizer()
molecule_RelativeContentKr_featurizer = Molecule_RelativeContentKr_Featurizer()
molecule_HasKr_featurizer = Molecule_HasKr_Featurizer()
molecule_NumberAtomsRb_featurizer = Molecule_NumberAtomsRb_Featurizer()
molecule_RelativeContentRb_featurizer = Molecule_RelativeContentRb_Featurizer()
molecule_HasRb_featurizer = Molecule_HasRb_Featurizer()
molecule_NumberAtomsSr_featurizer = Molecule_NumberAtomsSr_Featurizer()
molecule_RelativeContentSr_featurizer = Molecule_RelativeContentSr_Featurizer()
molecule_HasSr_featurizer = Molecule_HasSr_Featurizer()
molecule_NumberAtomsY_featurizer = Molecule_NumberAtomsY_Featurizer()
molecule_RelativeContentY_featurizer = Molecule_RelativeContentY_Featurizer()
molecule_HasY_featurizer = Molecule_HasY_Featurizer()
molecule_NumberAtomsZr_featurizer = Molecule_NumberAtomsZr_Featurizer()
molecule_RelativeContentZr_featurizer = Molecule_RelativeContentZr_Featurizer()
molecule_HasZr_featurizer = Molecule_HasZr_Featurizer()
molecule_NumberAtomsNb_featurizer = Molecule_NumberAtomsNb_Featurizer()
molecule_RelativeContentNb_featurizer = Molecule_RelativeContentNb_Featurizer()
molecule_HasNb_featurizer = Molecule_HasNb_Featurizer()
molecule_NumberAtomsMo_featurizer = Molecule_NumberAtomsMo_Featurizer()
molecule_RelativeContentMo_featurizer = Molecule_RelativeContentMo_Featurizer()
molecule_HasMo_featurizer = Molecule_HasMo_Featurizer()
molecule_NumberAtomsTc_featurizer = Molecule_NumberAtomsTc_Featurizer()
molecule_RelativeContentTc_featurizer = Molecule_RelativeContentTc_Featurizer()
molecule_HasTc_featurizer = Molecule_HasTc_Featurizer()
molecule_NumberAtomsRu_featurizer = Molecule_NumberAtomsRu_Featurizer()
molecule_RelativeContentRu_featurizer = Molecule_RelativeContentRu_Featurizer()
molecule_HasRu_featurizer = Molecule_HasRu_Featurizer()
molecule_NumberAtomsRh_featurizer = Molecule_NumberAtomsRh_Featurizer()
molecule_RelativeContentRh_featurizer = Molecule_RelativeContentRh_Featurizer()
molecule_HasRh_featurizer = Molecule_HasRh_Featurizer()
molecule_NumberAtomsPd_featurizer = Molecule_NumberAtomsPd_Featurizer()
molecule_RelativeContentPd_featurizer = Molecule_RelativeContentPd_Featurizer()
molecule_HasPd_featurizer = Molecule_HasPd_Featurizer()
molecule_NumberAtomsAg_featurizer = Molecule_NumberAtomsAg_Featurizer()
molecule_RelativeContentAg_featurizer = Molecule_RelativeContentAg_Featurizer()
molecule_HasAg_featurizer = Molecule_HasAg_Featurizer()
molecule_NumberAtomsCd_featurizer = Molecule_NumberAtomsCd_Featurizer()
molecule_RelativeContentCd_featurizer = Molecule_RelativeContentCd_Featurizer()
molecule_HasCd_featurizer = Molecule_HasCd_Featurizer()
molecule_NumberAtomsIn_featurizer = Molecule_NumberAtomsIn_Featurizer()
molecule_RelativeContentIn_featurizer = Molecule_RelativeContentIn_Featurizer()
molecule_HasIn_featurizer = Molecule_HasIn_Featurizer()
molecule_NumberAtomsSn_featurizer = Molecule_NumberAtomsSn_Featurizer()
molecule_RelativeContentSn_featurizer = Molecule_RelativeContentSn_Featurizer()
molecule_HasSn_featurizer = Molecule_HasSn_Featurizer()
molecule_NumberAtomsSb_featurizer = Molecule_NumberAtomsSb_Featurizer()
molecule_RelativeContentSb_featurizer = Molecule_RelativeContentSb_Featurizer()
molecule_HasSb_featurizer = Molecule_HasSb_Featurizer()
molecule_NumberAtomsTe_featurizer = Molecule_NumberAtomsTe_Featurizer()
molecule_RelativeContentTe_featurizer = Molecule_RelativeContentTe_Featurizer()
molecule_HasTe_featurizer = Molecule_HasTe_Featurizer()
molecule_NumberAtomsI_featurizer = Molecule_NumberAtomsI_Featurizer()
molecule_RelativeContentI_featurizer = Molecule_RelativeContentI_Featurizer()
molecule_HasI_featurizer = Molecule_HasI_Featurizer()
molecule_NumberAtomsXe_featurizer = Molecule_NumberAtomsXe_Featurizer()
molecule_RelativeContentXe_featurizer = Molecule_RelativeContentXe_Featurizer()
molecule_HasXe_featurizer = Molecule_HasXe_Featurizer()
molecule_NumberAtomsCs_featurizer = Molecule_NumberAtomsCs_Featurizer()
molecule_RelativeContentCs_featurizer = Molecule_RelativeContentCs_Featurizer()
molecule_HasCs_featurizer = Molecule_HasCs_Featurizer()
molecule_NumberAtomsBa_featurizer = Molecule_NumberAtomsBa_Featurizer()
molecule_RelativeContentBa_featurizer = Molecule_RelativeContentBa_Featurizer()
molecule_HasBa_featurizer = Molecule_HasBa_Featurizer()
molecule_NumberAtomsLa_featurizer = Molecule_NumberAtomsLa_Featurizer()
molecule_RelativeContentLa_featurizer = Molecule_RelativeContentLa_Featurizer()
molecule_HasLa_featurizer = Molecule_HasLa_Featurizer()
molecule_NumberAtomsCe_featurizer = Molecule_NumberAtomsCe_Featurizer()
molecule_RelativeContentCe_featurizer = Molecule_RelativeContentCe_Featurizer()
molecule_HasCe_featurizer = Molecule_HasCe_Featurizer()
molecule_NumberAtomsPr_featurizer = Molecule_NumberAtomsPr_Featurizer()
molecule_RelativeContentPr_featurizer = Molecule_RelativeContentPr_Featurizer()
molecule_HasPr_featurizer = Molecule_HasPr_Featurizer()
molecule_NumberAtomsNd_featurizer = Molecule_NumberAtomsNd_Featurizer()
molecule_RelativeContentNd_featurizer = Molecule_RelativeContentNd_Featurizer()
molecule_HasNd_featurizer = Molecule_HasNd_Featurizer()
molecule_NumberAtomsPm_featurizer = Molecule_NumberAtomsPm_Featurizer()
molecule_RelativeContentPm_featurizer = Molecule_RelativeContentPm_Featurizer()
molecule_HasPm_featurizer = Molecule_HasPm_Featurizer()
molecule_NumberAtomsSm_featurizer = Molecule_NumberAtomsSm_Featurizer()
molecule_RelativeContentSm_featurizer = Molecule_RelativeContentSm_Featurizer()
molecule_HasSm_featurizer = Molecule_HasSm_Featurizer()
molecule_NumberAtomsEu_featurizer = Molecule_NumberAtomsEu_Featurizer()
molecule_RelativeContentEu_featurizer = Molecule_RelativeContentEu_Featurizer()
molecule_HasEu_featurizer = Molecule_HasEu_Featurizer()
molecule_NumberAtomsGd_featurizer = Molecule_NumberAtomsGd_Featurizer()
molecule_RelativeContentGd_featurizer = Molecule_RelativeContentGd_Featurizer()
molecule_HasGd_featurizer = Molecule_HasGd_Featurizer()
molecule_NumberAtomsTb_featurizer = Molecule_NumberAtomsTb_Featurizer()
molecule_RelativeContentTb_featurizer = Molecule_RelativeContentTb_Featurizer()
molecule_HasTb_featurizer = Molecule_HasTb_Featurizer()
molecule_NumberAtomsDy_featurizer = Molecule_NumberAtomsDy_Featurizer()
molecule_RelativeContentDy_featurizer = Molecule_RelativeContentDy_Featurizer()
molecule_HasDy_featurizer = Molecule_HasDy_Featurizer()
molecule_NumberAtomsHo_featurizer = Molecule_NumberAtomsHo_Featurizer()
molecule_RelativeContentHo_featurizer = Molecule_RelativeContentHo_Featurizer()
molecule_HasHo_featurizer = Molecule_HasHo_Featurizer()
molecule_NumberAtomsEr_featurizer = Molecule_NumberAtomsEr_Featurizer()
molecule_RelativeContentEr_featurizer = Molecule_RelativeContentEr_Featurizer()
molecule_HasEr_featurizer = Molecule_HasEr_Featurizer()
molecule_NumberAtomsTm_featurizer = Molecule_NumberAtomsTm_Featurizer()
molecule_RelativeContentTm_featurizer = Molecule_RelativeContentTm_Featurizer()
molecule_HasTm_featurizer = Molecule_HasTm_Featurizer()
molecule_NumberAtomsYb_featurizer = Molecule_NumberAtomsYb_Featurizer()
molecule_RelativeContentYb_featurizer = Molecule_RelativeContentYb_Featurizer()
molecule_HasYb_featurizer = Molecule_HasYb_Featurizer()
molecule_NumberAtomsLu_featurizer = Molecule_NumberAtomsLu_Featurizer()
molecule_RelativeContentLu_featurizer = Molecule_RelativeContentLu_Featurizer()
molecule_HasLu_featurizer = Molecule_HasLu_Featurizer()
molecule_NumberAtomsHf_featurizer = Molecule_NumberAtomsHf_Featurizer()
molecule_RelativeContentHf_featurizer = Molecule_RelativeContentHf_Featurizer()
molecule_HasHf_featurizer = Molecule_HasHf_Featurizer()
molecule_NumberAtomsTa_featurizer = Molecule_NumberAtomsTa_Featurizer()
molecule_RelativeContentTa_featurizer = Molecule_RelativeContentTa_Featurizer()
molecule_HasTa_featurizer = Molecule_HasTa_Featurizer()
molecule_NumberAtomsW_featurizer = Molecule_NumberAtomsW_Featurizer()
molecule_RelativeContentW_featurizer = Molecule_RelativeContentW_Featurizer()
molecule_HasW_featurizer = Molecule_HasW_Featurizer()
molecule_NumberAtomsRe_featurizer = Molecule_NumberAtomsRe_Featurizer()
molecule_RelativeContentRe_featurizer = Molecule_RelativeContentRe_Featurizer()
molecule_HasRe_featurizer = Molecule_HasRe_Featurizer()
molecule_NumberAtomsOs_featurizer = Molecule_NumberAtomsOs_Featurizer()
molecule_RelativeContentOs_featurizer = Molecule_RelativeContentOs_Featurizer()
molecule_HasOs_featurizer = Molecule_HasOs_Featurizer()
molecule_NumberAtomsIr_featurizer = Molecule_NumberAtomsIr_Featurizer()
molecule_RelativeContentIr_featurizer = Molecule_RelativeContentIr_Featurizer()
molecule_HasIr_featurizer = Molecule_HasIr_Featurizer()
molecule_NumberAtomsPt_featurizer = Molecule_NumberAtomsPt_Featurizer()
molecule_RelativeContentPt_featurizer = Molecule_RelativeContentPt_Featurizer()
molecule_HasPt_featurizer = Molecule_HasPt_Featurizer()
molecule_NumberAtomsAu_featurizer = Molecule_NumberAtomsAu_Featurizer()
molecule_RelativeContentAu_featurizer = Molecule_RelativeContentAu_Featurizer()
molecule_HasAu_featurizer = Molecule_HasAu_Featurizer()
molecule_NumberAtomsHg_featurizer = Molecule_NumberAtomsHg_Featurizer()
molecule_RelativeContentHg_featurizer = Molecule_RelativeContentHg_Featurizer()
molecule_HasHg_featurizer = Molecule_HasHg_Featurizer()
molecule_NumberAtomsTl_featurizer = Molecule_NumberAtomsTl_Featurizer()
molecule_RelativeContentTl_featurizer = Molecule_RelativeContentTl_Featurizer()
molecule_HasTl_featurizer = Molecule_HasTl_Featurizer()
molecule_NumberAtomsPb_featurizer = Molecule_NumberAtomsPb_Featurizer()
molecule_RelativeContentPb_featurizer = Molecule_RelativeContentPb_Featurizer()
molecule_HasPb_featurizer = Molecule_HasPb_Featurizer()
molecule_NumberAtomsBi_featurizer = Molecule_NumberAtomsBi_Featurizer()
molecule_RelativeContentBi_featurizer = Molecule_RelativeContentBi_Featurizer()
molecule_HasBi_featurizer = Molecule_HasBi_Featurizer()
molecule_NumberAtomsPo_featurizer = Molecule_NumberAtomsPo_Featurizer()
molecule_RelativeContentPo_featurizer = Molecule_RelativeContentPo_Featurizer()
molecule_HasPo_featurizer = Molecule_HasPo_Featurizer()
molecule_NumberAtomsAt_featurizer = Molecule_NumberAtomsAt_Featurizer()
molecule_RelativeContentAt_featurizer = Molecule_RelativeContentAt_Featurizer()
molecule_HasAt_featurizer = Molecule_HasAt_Featurizer()
molecule_NumberAtomsRn_featurizer = Molecule_NumberAtomsRn_Featurizer()
molecule_RelativeContentRn_featurizer = Molecule_RelativeContentRn_Featurizer()
molecule_HasRn_featurizer = Molecule_HasRn_Featurizer()
molecule_NumberAtomsFr_featurizer = Molecule_NumberAtomsFr_Featurizer()
molecule_RelativeContentFr_featurizer = Molecule_RelativeContentFr_Featurizer()
molecule_HasFr_featurizer = Molecule_HasFr_Featurizer()
molecule_NumberAtomsRa_featurizer = Molecule_NumberAtomsRa_Featurizer()
molecule_RelativeContentRa_featurizer = Molecule_RelativeContentRa_Featurizer()
molecule_HasRa_featurizer = Molecule_HasRa_Featurizer()
molecule_NumberAtomsAc_featurizer = Molecule_NumberAtomsAc_Featurizer()
molecule_RelativeContentAc_featurizer = Molecule_RelativeContentAc_Featurizer()
molecule_HasAc_featurizer = Molecule_HasAc_Featurizer()
molecule_NumberAtomsTh_featurizer = Molecule_NumberAtomsTh_Featurizer()
molecule_RelativeContentTh_featurizer = Molecule_RelativeContentTh_Featurizer()
molecule_HasTh_featurizer = Molecule_HasTh_Featurizer()
molecule_NumberAtomsPa_featurizer = Molecule_NumberAtomsPa_Featurizer()
molecule_RelativeContentPa_featurizer = Molecule_RelativeContentPa_Featurizer()
molecule_HasPa_featurizer = Molecule_HasPa_Featurizer()
molecule_NumberAtomsU_featurizer = Molecule_NumberAtomsU_Featurizer()
molecule_RelativeContentU_featurizer = Molecule_RelativeContentU_Featurizer()
molecule_HasU_featurizer = Molecule_HasU_Featurizer()
molecule_NumberAtomsNp_featurizer = Molecule_NumberAtomsNp_Featurizer()
molecule_RelativeContentNp_featurizer = Molecule_RelativeContentNp_Featurizer()
molecule_HasNp_featurizer = Molecule_HasNp_Featurizer()
molecule_NumberAtomsPu_featurizer = Molecule_NumberAtomsPu_Featurizer()
molecule_RelativeContentPu_featurizer = Molecule_RelativeContentPu_Featurizer()
molecule_HasPu_featurizer = Molecule_HasPu_Featurizer()
molecule_NumberAtomsAm_featurizer = Molecule_NumberAtomsAm_Featurizer()
molecule_RelativeContentAm_featurizer = Molecule_RelativeContentAm_Featurizer()
molecule_HasAm_featurizer = Molecule_HasAm_Featurizer()
molecule_NumberAtomsCm_featurizer = Molecule_NumberAtomsCm_Featurizer()
molecule_RelativeContentCm_featurizer = Molecule_RelativeContentCm_Featurizer()
molecule_HasCm_featurizer = Molecule_HasCm_Featurizer()
molecule_NumberAtomsBk_featurizer = Molecule_NumberAtomsBk_Featurizer()
molecule_RelativeContentBk_featurizer = Molecule_RelativeContentBk_Featurizer()
molecule_HasBk_featurizer = Molecule_HasBk_Featurizer()
molecule_NumberAtomsCf_featurizer = Molecule_NumberAtomsCf_Featurizer()
molecule_RelativeContentCf_featurizer = Molecule_RelativeContentCf_Featurizer()
molecule_HasCf_featurizer = Molecule_HasCf_Featurizer()
molecule_NumberAtomsEs_featurizer = Molecule_NumberAtomsEs_Featurizer()
molecule_RelativeContentEs_featurizer = Molecule_RelativeContentEs_Featurizer()
molecule_HasEs_featurizer = Molecule_HasEs_Featurizer()
molecule_NumberAtomsFm_featurizer = Molecule_NumberAtomsFm_Featurizer()
molecule_RelativeContentFm_featurizer = Molecule_RelativeContentFm_Featurizer()
molecule_HasFm_featurizer = Molecule_HasFm_Featurizer()
molecule_NumberAtomsMd_featurizer = Molecule_NumberAtomsMd_Featurizer()
molecule_RelativeContentMd_featurizer = Molecule_RelativeContentMd_Featurizer()
molecule_HasMd_featurizer = Molecule_HasMd_Featurizer()
molecule_NumberAtomsNo_featurizer = Molecule_NumberAtomsNo_Featurizer()
molecule_RelativeContentNo_featurizer = Molecule_RelativeContentNo_Featurizer()
molecule_HasNo_featurizer = Molecule_HasNo_Featurizer()
molecule_NumberAtomsLr_featurizer = Molecule_NumberAtomsLr_Featurizer()
molecule_RelativeContentLr_featurizer = Molecule_RelativeContentLr_Featurizer()
molecule_HasLr_featurizer = Molecule_HasLr_Featurizer()
molecule_NumberAtomsRf_featurizer = Molecule_NumberAtomsRf_Featurizer()
molecule_RelativeContentRf_featurizer = Molecule_RelativeContentRf_Featurizer()
molecule_HasRf_featurizer = Molecule_HasRf_Featurizer()
molecule_NumberAtomsDb_featurizer = Molecule_NumberAtomsDb_Featurizer()
molecule_RelativeContentDb_featurizer = Molecule_RelativeContentDb_Featurizer()
molecule_HasDb_featurizer = Molecule_HasDb_Featurizer()
molecule_NumberAtomsSg_featurizer = Molecule_NumberAtomsSg_Featurizer()
molecule_RelativeContentSg_featurizer = Molecule_RelativeContentSg_Featurizer()
molecule_HasSg_featurizer = Molecule_HasSg_Featurizer()
molecule_NumberAtomsBh_featurizer = Molecule_NumberAtomsBh_Featurizer()
molecule_RelativeContentBh_featurizer = Molecule_RelativeContentBh_Featurizer()
molecule_HasBh_featurizer = Molecule_HasBh_Featurizer()
molecule_NumberAtomsHs_featurizer = Molecule_NumberAtomsHs_Featurizer()
molecule_RelativeContentHs_featurizer = Molecule_RelativeContentHs_Featurizer()
molecule_HasHs_featurizer = Molecule_HasHs_Featurizer()
molecule_NumberAtomsMt_featurizer = Molecule_NumberAtomsMt_Featurizer()
molecule_RelativeContentMt_featurizer = Molecule_RelativeContentMt_Featurizer()
molecule_HasMt_featurizer = Molecule_HasMt_Featurizer()
molecule_NumberAtomsDs_featurizer = Molecule_NumberAtomsDs_Featurizer()
molecule_RelativeContentDs_featurizer = Molecule_RelativeContentDs_Featurizer()
molecule_HasDs_featurizer = Molecule_HasDs_Featurizer()
molecule_NumberAtomsRg_featurizer = Molecule_NumberAtomsRg_Featurizer()
molecule_RelativeContentRg_featurizer = Molecule_RelativeContentRg_Featurizer()
molecule_HasRg_featurizer = Molecule_HasRg_Featurizer()
molecule_NumberAtomsCn_featurizer = Molecule_NumberAtomsCn_Featurizer()
molecule_RelativeContentCn_featurizer = Molecule_RelativeContentCn_Featurizer()
molecule_HasCn_featurizer = Molecule_HasCn_Featurizer()
molecule_NumberAtomsNh_featurizer = Molecule_NumberAtomsNh_Featurizer()
molecule_RelativeContentNh_featurizer = Molecule_RelativeContentNh_Featurizer()
molecule_HasNh_featurizer = Molecule_HasNh_Featurizer()
molecule_NumberAtomsFl_featurizer = Molecule_NumberAtomsFl_Featurizer()
molecule_RelativeContentFl_featurizer = Molecule_RelativeContentFl_Featurizer()
molecule_HasFl_featurizer = Molecule_HasFl_Featurizer()
molecule_NumberAtomsMc_featurizer = Molecule_NumberAtomsMc_Featurizer()
molecule_RelativeContentMc_featurizer = Molecule_RelativeContentMc_Featurizer()
molecule_HasMc_featurizer = Molecule_HasMc_Featurizer()
molecule_NumberAtomsLv_featurizer = Molecule_NumberAtomsLv_Featurizer()
molecule_RelativeContentLv_featurizer = Molecule_RelativeContentLv_Featurizer()
molecule_HasLv_featurizer = Molecule_HasLv_Featurizer()
molecule_NumberAtomsTs_featurizer = Molecule_NumberAtomsTs_Featurizer()
molecule_RelativeContentTs_featurizer = Molecule_RelativeContentTs_Featurizer()
molecule_HasTs_featurizer = Molecule_HasTs_Featurizer()
molecule_NumberAtomsOg_featurizer = Molecule_NumberAtomsOg_Featurizer()
molecule_RelativeContentOg_featurizer = Molecule_RelativeContentOg_Featurizer()
molecule_HasOg_featurizer = Molecule_HasOg_Featurizer()
_available_featurizer = {
    "molecule_NumberAtomsRgroup_featurizer": molecule_NumberAtomsRgroup_featurizer,
    "molecule_RelativeContentRgroup_featurizer": molecule_RelativeContentRgroup_featurizer,
    "molecule_HasRgroup_featurizer": molecule_HasRgroup_featurizer,
    "molecule_NumberAtomsH_featurizer": molecule_NumberAtomsH_featurizer,
    "molecule_RelativeContentH_featurizer": molecule_RelativeContentH_featurizer,
    "molecule_HasH_featurizer": molecule_HasH_featurizer,
    "molecule_NumberAtomsHe_featurizer": molecule_NumberAtomsHe_featurizer,
    "molecule_RelativeContentHe_featurizer": molecule_RelativeContentHe_featurizer,
    "molecule_HasHe_featurizer": molecule_HasHe_featurizer,
    "molecule_NumberAtomsLi_featurizer": molecule_NumberAtomsLi_featurizer,
    "molecule_RelativeContentLi_featurizer": molecule_RelativeContentLi_featurizer,
    "molecule_HasLi_featurizer": molecule_HasLi_featurizer,
    "molecule_NumberAtomsBe_featurizer": molecule_NumberAtomsBe_featurizer,
    "molecule_RelativeContentBe_featurizer": molecule_RelativeContentBe_featurizer,
    "molecule_HasBe_featurizer": molecule_HasBe_featurizer,
    "molecule_NumberAtomsB_featurizer": molecule_NumberAtomsB_featurizer,
    "molecule_RelativeContentB_featurizer": molecule_RelativeContentB_featurizer,
    "molecule_HasB_featurizer": molecule_HasB_featurizer,
    "molecule_NumberAtomsC_featurizer": molecule_NumberAtomsC_featurizer,
    "molecule_RelativeContentC_featurizer": molecule_RelativeContentC_featurizer,
    "molecule_HasC_featurizer": molecule_HasC_featurizer,
    "molecule_NumberAtomsN_featurizer": molecule_NumberAtomsN_featurizer,
    "molecule_RelativeContentN_featurizer": molecule_RelativeContentN_featurizer,
    "molecule_HasN_featurizer": molecule_HasN_featurizer,
    "molecule_NumberAtomsO_featurizer": molecule_NumberAtomsO_featurizer,
    "molecule_RelativeContentO_featurizer": molecule_RelativeContentO_featurizer,
    "molecule_HasO_featurizer": molecule_HasO_featurizer,
    "molecule_NumberAtomsF_featurizer": molecule_NumberAtomsF_featurizer,
    "molecule_RelativeContentF_featurizer": molecule_RelativeContentF_featurizer,
    "molecule_HasF_featurizer": molecule_HasF_featurizer,
    "molecule_NumberAtomsNe_featurizer": molecule_NumberAtomsNe_featurizer,
    "molecule_RelativeContentNe_featurizer": molecule_RelativeContentNe_featurizer,
    "molecule_HasNe_featurizer": molecule_HasNe_featurizer,
    "molecule_NumberAtomsNa_featurizer": molecule_NumberAtomsNa_featurizer,
    "molecule_RelativeContentNa_featurizer": molecule_RelativeContentNa_featurizer,
    "molecule_HasNa_featurizer": molecule_HasNa_featurizer,
    "molecule_NumberAtomsMg_featurizer": molecule_NumberAtomsMg_featurizer,
    "molecule_RelativeContentMg_featurizer": molecule_RelativeContentMg_featurizer,
    "molecule_HasMg_featurizer": molecule_HasMg_featurizer,
    "molecule_NumberAtomsAl_featurizer": molecule_NumberAtomsAl_featurizer,
    "molecule_RelativeContentAl_featurizer": molecule_RelativeContentAl_featurizer,
    "molecule_HasAl_featurizer": molecule_HasAl_featurizer,
    "molecule_NumberAtomsSi_featurizer": molecule_NumberAtomsSi_featurizer,
    "molecule_RelativeContentSi_featurizer": molecule_RelativeContentSi_featurizer,
    "molecule_HasSi_featurizer": molecule_HasSi_featurizer,
    "molecule_NumberAtomsP_featurizer": molecule_NumberAtomsP_featurizer,
    "molecule_RelativeContentP_featurizer": molecule_RelativeContentP_featurizer,
    "molecule_HasP_featurizer": molecule_HasP_featurizer,
    "molecule_NumberAtomsS_featurizer": molecule_NumberAtomsS_featurizer,
    "molecule_RelativeContentS_featurizer": molecule_RelativeContentS_featurizer,
    "molecule_HasS_featurizer": molecule_HasS_featurizer,
    "molecule_NumberAtomsCl_featurizer": molecule_NumberAtomsCl_featurizer,
    "molecule_RelativeContentCl_featurizer": molecule_RelativeContentCl_featurizer,
    "molecule_HasCl_featurizer": molecule_HasCl_featurizer,
    "molecule_NumberAtomsAr_featurizer": molecule_NumberAtomsAr_featurizer,
    "molecule_RelativeContentAr_featurizer": molecule_RelativeContentAr_featurizer,
    "molecule_HasAr_featurizer": molecule_HasAr_featurizer,
    "molecule_NumberAtomsK_featurizer": molecule_NumberAtomsK_featurizer,
    "molecule_RelativeContentK_featurizer": molecule_RelativeContentK_featurizer,
    "molecule_HasK_featurizer": molecule_HasK_featurizer,
    "molecule_NumberAtomsCa_featurizer": molecule_NumberAtomsCa_featurizer,
    "molecule_RelativeContentCa_featurizer": molecule_RelativeContentCa_featurizer,
    "molecule_HasCa_featurizer": molecule_HasCa_featurizer,
    "molecule_NumberAtomsSc_featurizer": molecule_NumberAtomsSc_featurizer,
    "molecule_RelativeContentSc_featurizer": molecule_RelativeContentSc_featurizer,
    "molecule_HasSc_featurizer": molecule_HasSc_featurizer,
    "molecule_NumberAtomsTi_featurizer": molecule_NumberAtomsTi_featurizer,
    "molecule_RelativeContentTi_featurizer": molecule_RelativeContentTi_featurizer,
    "molecule_HasTi_featurizer": molecule_HasTi_featurizer,
    "molecule_NumberAtomsV_featurizer": molecule_NumberAtomsV_featurizer,
    "molecule_RelativeContentV_featurizer": molecule_RelativeContentV_featurizer,
    "molecule_HasV_featurizer": molecule_HasV_featurizer,
    "molecule_NumberAtomsCr_featurizer": molecule_NumberAtomsCr_featurizer,
    "molecule_RelativeContentCr_featurizer": molecule_RelativeContentCr_featurizer,
    "molecule_HasCr_featurizer": molecule_HasCr_featurizer,
    "molecule_NumberAtomsMn_featurizer": molecule_NumberAtomsMn_featurizer,
    "molecule_RelativeContentMn_featurizer": molecule_RelativeContentMn_featurizer,
    "molecule_HasMn_featurizer": molecule_HasMn_featurizer,
    "molecule_NumberAtomsFe_featurizer": molecule_NumberAtomsFe_featurizer,
    "molecule_RelativeContentFe_featurizer": molecule_RelativeContentFe_featurizer,
    "molecule_HasFe_featurizer": molecule_HasFe_featurizer,
    "molecule_NumberAtomsCo_featurizer": molecule_NumberAtomsCo_featurizer,
    "molecule_RelativeContentCo_featurizer": molecule_RelativeContentCo_featurizer,
    "molecule_HasCo_featurizer": molecule_HasCo_featurizer,
    "molecule_NumberAtomsNi_featurizer": molecule_NumberAtomsNi_featurizer,
    "molecule_RelativeContentNi_featurizer": molecule_RelativeContentNi_featurizer,
    "molecule_HasNi_featurizer": molecule_HasNi_featurizer,
    "molecule_NumberAtomsCu_featurizer": molecule_NumberAtomsCu_featurizer,
    "molecule_RelativeContentCu_featurizer": molecule_RelativeContentCu_featurizer,
    "molecule_HasCu_featurizer": molecule_HasCu_featurizer,
    "molecule_NumberAtomsZn_featurizer": molecule_NumberAtomsZn_featurizer,
    "molecule_RelativeContentZn_featurizer": molecule_RelativeContentZn_featurizer,
    "molecule_HasZn_featurizer": molecule_HasZn_featurizer,
    "molecule_NumberAtomsGa_featurizer": molecule_NumberAtomsGa_featurizer,
    "molecule_RelativeContentGa_featurizer": molecule_RelativeContentGa_featurizer,
    "molecule_HasGa_featurizer": molecule_HasGa_featurizer,
    "molecule_NumberAtomsGe_featurizer": molecule_NumberAtomsGe_featurizer,
    "molecule_RelativeContentGe_featurizer": molecule_RelativeContentGe_featurizer,
    "molecule_HasGe_featurizer": molecule_HasGe_featurizer,
    "molecule_NumberAtomsAs_featurizer": molecule_NumberAtomsAs_featurizer,
    "molecule_RelativeContentAs_featurizer": molecule_RelativeContentAs_featurizer,
    "molecule_HasAs_featurizer": molecule_HasAs_featurizer,
    "molecule_NumberAtomsSe_featurizer": molecule_NumberAtomsSe_featurizer,
    "molecule_RelativeContentSe_featurizer": molecule_RelativeContentSe_featurizer,
    "molecule_HasSe_featurizer": molecule_HasSe_featurizer,
    "molecule_NumberAtomsBr_featurizer": molecule_NumberAtomsBr_featurizer,
    "molecule_RelativeContentBr_featurizer": molecule_RelativeContentBr_featurizer,
    "molecule_HasBr_featurizer": molecule_HasBr_featurizer,
    "molecule_NumberAtomsKr_featurizer": molecule_NumberAtomsKr_featurizer,
    "molecule_RelativeContentKr_featurizer": molecule_RelativeContentKr_featurizer,
    "molecule_HasKr_featurizer": molecule_HasKr_featurizer,
    "molecule_NumberAtomsRb_featurizer": molecule_NumberAtomsRb_featurizer,
    "molecule_RelativeContentRb_featurizer": molecule_RelativeContentRb_featurizer,
    "molecule_HasRb_featurizer": molecule_HasRb_featurizer,
    "molecule_NumberAtomsSr_featurizer": molecule_NumberAtomsSr_featurizer,
    "molecule_RelativeContentSr_featurizer": molecule_RelativeContentSr_featurizer,
    "molecule_HasSr_featurizer": molecule_HasSr_featurizer,
    "molecule_NumberAtomsY_featurizer": molecule_NumberAtomsY_featurizer,
    "molecule_RelativeContentY_featurizer": molecule_RelativeContentY_featurizer,
    "molecule_HasY_featurizer": molecule_HasY_featurizer,
    "molecule_NumberAtomsZr_featurizer": molecule_NumberAtomsZr_featurizer,
    "molecule_RelativeContentZr_featurizer": molecule_RelativeContentZr_featurizer,
    "molecule_HasZr_featurizer": molecule_HasZr_featurizer,
    "molecule_NumberAtomsNb_featurizer": molecule_NumberAtomsNb_featurizer,
    "molecule_RelativeContentNb_featurizer": molecule_RelativeContentNb_featurizer,
    "molecule_HasNb_featurizer": molecule_HasNb_featurizer,
    "molecule_NumberAtomsMo_featurizer": molecule_NumberAtomsMo_featurizer,
    "molecule_RelativeContentMo_featurizer": molecule_RelativeContentMo_featurizer,
    "molecule_HasMo_featurizer": molecule_HasMo_featurizer,
    "molecule_NumberAtomsTc_featurizer": molecule_NumberAtomsTc_featurizer,
    "molecule_RelativeContentTc_featurizer": molecule_RelativeContentTc_featurizer,
    "molecule_HasTc_featurizer": molecule_HasTc_featurizer,
    "molecule_NumberAtomsRu_featurizer": molecule_NumberAtomsRu_featurizer,
    "molecule_RelativeContentRu_featurizer": molecule_RelativeContentRu_featurizer,
    "molecule_HasRu_featurizer": molecule_HasRu_featurizer,
    "molecule_NumberAtomsRh_featurizer": molecule_NumberAtomsRh_featurizer,
    "molecule_RelativeContentRh_featurizer": molecule_RelativeContentRh_featurizer,
    "molecule_HasRh_featurizer": molecule_HasRh_featurizer,
    "molecule_NumberAtomsPd_featurizer": molecule_NumberAtomsPd_featurizer,
    "molecule_RelativeContentPd_featurizer": molecule_RelativeContentPd_featurizer,
    "molecule_HasPd_featurizer": molecule_HasPd_featurizer,
    "molecule_NumberAtomsAg_featurizer": molecule_NumberAtomsAg_featurizer,
    "molecule_RelativeContentAg_featurizer": molecule_RelativeContentAg_featurizer,
    "molecule_HasAg_featurizer": molecule_HasAg_featurizer,
    "molecule_NumberAtomsCd_featurizer": molecule_NumberAtomsCd_featurizer,
    "molecule_RelativeContentCd_featurizer": molecule_RelativeContentCd_featurizer,
    "molecule_HasCd_featurizer": molecule_HasCd_featurizer,
    "molecule_NumberAtomsIn_featurizer": molecule_NumberAtomsIn_featurizer,
    "molecule_RelativeContentIn_featurizer": molecule_RelativeContentIn_featurizer,
    "molecule_HasIn_featurizer": molecule_HasIn_featurizer,
    "molecule_NumberAtomsSn_featurizer": molecule_NumberAtomsSn_featurizer,
    "molecule_RelativeContentSn_featurizer": molecule_RelativeContentSn_featurizer,
    "molecule_HasSn_featurizer": molecule_HasSn_featurizer,
    "molecule_NumberAtomsSb_featurizer": molecule_NumberAtomsSb_featurizer,
    "molecule_RelativeContentSb_featurizer": molecule_RelativeContentSb_featurizer,
    "molecule_HasSb_featurizer": molecule_HasSb_featurizer,
    "molecule_NumberAtomsTe_featurizer": molecule_NumberAtomsTe_featurizer,
    "molecule_RelativeContentTe_featurizer": molecule_RelativeContentTe_featurizer,
    "molecule_HasTe_featurizer": molecule_HasTe_featurizer,
    "molecule_NumberAtomsI_featurizer": molecule_NumberAtomsI_featurizer,
    "molecule_RelativeContentI_featurizer": molecule_RelativeContentI_featurizer,
    "molecule_HasI_featurizer": molecule_HasI_featurizer,
    "molecule_NumberAtomsXe_featurizer": molecule_NumberAtomsXe_featurizer,
    "molecule_RelativeContentXe_featurizer": molecule_RelativeContentXe_featurizer,
    "molecule_HasXe_featurizer": molecule_HasXe_featurizer,
    "molecule_NumberAtomsCs_featurizer": molecule_NumberAtomsCs_featurizer,
    "molecule_RelativeContentCs_featurizer": molecule_RelativeContentCs_featurizer,
    "molecule_HasCs_featurizer": molecule_HasCs_featurizer,
    "molecule_NumberAtomsBa_featurizer": molecule_NumberAtomsBa_featurizer,
    "molecule_RelativeContentBa_featurizer": molecule_RelativeContentBa_featurizer,
    "molecule_HasBa_featurizer": molecule_HasBa_featurizer,
    "molecule_NumberAtomsLa_featurizer": molecule_NumberAtomsLa_featurizer,
    "molecule_RelativeContentLa_featurizer": molecule_RelativeContentLa_featurizer,
    "molecule_HasLa_featurizer": molecule_HasLa_featurizer,
    "molecule_NumberAtomsCe_featurizer": molecule_NumberAtomsCe_featurizer,
    "molecule_RelativeContentCe_featurizer": molecule_RelativeContentCe_featurizer,
    "molecule_HasCe_featurizer": molecule_HasCe_featurizer,
    "molecule_NumberAtomsPr_featurizer": molecule_NumberAtomsPr_featurizer,
    "molecule_RelativeContentPr_featurizer": molecule_RelativeContentPr_featurizer,
    "molecule_HasPr_featurizer": molecule_HasPr_featurizer,
    "molecule_NumberAtomsNd_featurizer": molecule_NumberAtomsNd_featurizer,
    "molecule_RelativeContentNd_featurizer": molecule_RelativeContentNd_featurizer,
    "molecule_HasNd_featurizer": molecule_HasNd_featurizer,
    "molecule_NumberAtomsPm_featurizer": molecule_NumberAtomsPm_featurizer,
    "molecule_RelativeContentPm_featurizer": molecule_RelativeContentPm_featurizer,
    "molecule_HasPm_featurizer": molecule_HasPm_featurizer,
    "molecule_NumberAtomsSm_featurizer": molecule_NumberAtomsSm_featurizer,
    "molecule_RelativeContentSm_featurizer": molecule_RelativeContentSm_featurizer,
    "molecule_HasSm_featurizer": molecule_HasSm_featurizer,
    "molecule_NumberAtomsEu_featurizer": molecule_NumberAtomsEu_featurizer,
    "molecule_RelativeContentEu_featurizer": molecule_RelativeContentEu_featurizer,
    "molecule_HasEu_featurizer": molecule_HasEu_featurizer,
    "molecule_NumberAtomsGd_featurizer": molecule_NumberAtomsGd_featurizer,
    "molecule_RelativeContentGd_featurizer": molecule_RelativeContentGd_featurizer,
    "molecule_HasGd_featurizer": molecule_HasGd_featurizer,
    "molecule_NumberAtomsTb_featurizer": molecule_NumberAtomsTb_featurizer,
    "molecule_RelativeContentTb_featurizer": molecule_RelativeContentTb_featurizer,
    "molecule_HasTb_featurizer": molecule_HasTb_featurizer,
    "molecule_NumberAtomsDy_featurizer": molecule_NumberAtomsDy_featurizer,
    "molecule_RelativeContentDy_featurizer": molecule_RelativeContentDy_featurizer,
    "molecule_HasDy_featurizer": molecule_HasDy_featurizer,
    "molecule_NumberAtomsHo_featurizer": molecule_NumberAtomsHo_featurizer,
    "molecule_RelativeContentHo_featurizer": molecule_RelativeContentHo_featurizer,
    "molecule_HasHo_featurizer": molecule_HasHo_featurizer,
    "molecule_NumberAtomsEr_featurizer": molecule_NumberAtomsEr_featurizer,
    "molecule_RelativeContentEr_featurizer": molecule_RelativeContentEr_featurizer,
    "molecule_HasEr_featurizer": molecule_HasEr_featurizer,
    "molecule_NumberAtomsTm_featurizer": molecule_NumberAtomsTm_featurizer,
    "molecule_RelativeContentTm_featurizer": molecule_RelativeContentTm_featurizer,
    "molecule_HasTm_featurizer": molecule_HasTm_featurizer,
    "molecule_NumberAtomsYb_featurizer": molecule_NumberAtomsYb_featurizer,
    "molecule_RelativeContentYb_featurizer": molecule_RelativeContentYb_featurizer,
    "molecule_HasYb_featurizer": molecule_HasYb_featurizer,
    "molecule_NumberAtomsLu_featurizer": molecule_NumberAtomsLu_featurizer,
    "molecule_RelativeContentLu_featurizer": molecule_RelativeContentLu_featurizer,
    "molecule_HasLu_featurizer": molecule_HasLu_featurizer,
    "molecule_NumberAtomsHf_featurizer": molecule_NumberAtomsHf_featurizer,
    "molecule_RelativeContentHf_featurizer": molecule_RelativeContentHf_featurizer,
    "molecule_HasHf_featurizer": molecule_HasHf_featurizer,
    "molecule_NumberAtomsTa_featurizer": molecule_NumberAtomsTa_featurizer,
    "molecule_RelativeContentTa_featurizer": molecule_RelativeContentTa_featurizer,
    "molecule_HasTa_featurizer": molecule_HasTa_featurizer,
    "molecule_NumberAtomsW_featurizer": molecule_NumberAtomsW_featurizer,
    "molecule_RelativeContentW_featurizer": molecule_RelativeContentW_featurizer,
    "molecule_HasW_featurizer": molecule_HasW_featurizer,
    "molecule_NumberAtomsRe_featurizer": molecule_NumberAtomsRe_featurizer,
    "molecule_RelativeContentRe_featurizer": molecule_RelativeContentRe_featurizer,
    "molecule_HasRe_featurizer": molecule_HasRe_featurizer,
    "molecule_NumberAtomsOs_featurizer": molecule_NumberAtomsOs_featurizer,
    "molecule_RelativeContentOs_featurizer": molecule_RelativeContentOs_featurizer,
    "molecule_HasOs_featurizer": molecule_HasOs_featurizer,
    "molecule_NumberAtomsIr_featurizer": molecule_NumberAtomsIr_featurizer,
    "molecule_RelativeContentIr_featurizer": molecule_RelativeContentIr_featurizer,
    "molecule_HasIr_featurizer": molecule_HasIr_featurizer,
    "molecule_NumberAtomsPt_featurizer": molecule_NumberAtomsPt_featurizer,
    "molecule_RelativeContentPt_featurizer": molecule_RelativeContentPt_featurizer,
    "molecule_HasPt_featurizer": molecule_HasPt_featurizer,
    "molecule_NumberAtomsAu_featurizer": molecule_NumberAtomsAu_featurizer,
    "molecule_RelativeContentAu_featurizer": molecule_RelativeContentAu_featurizer,
    "molecule_HasAu_featurizer": molecule_HasAu_featurizer,
    "molecule_NumberAtomsHg_featurizer": molecule_NumberAtomsHg_featurizer,
    "molecule_RelativeContentHg_featurizer": molecule_RelativeContentHg_featurizer,
    "molecule_HasHg_featurizer": molecule_HasHg_featurizer,
    "molecule_NumberAtomsTl_featurizer": molecule_NumberAtomsTl_featurizer,
    "molecule_RelativeContentTl_featurizer": molecule_RelativeContentTl_featurizer,
    "molecule_HasTl_featurizer": molecule_HasTl_featurizer,
    "molecule_NumberAtomsPb_featurizer": molecule_NumberAtomsPb_featurizer,
    "molecule_RelativeContentPb_featurizer": molecule_RelativeContentPb_featurizer,
    "molecule_HasPb_featurizer": molecule_HasPb_featurizer,
    "molecule_NumberAtomsBi_featurizer": molecule_NumberAtomsBi_featurizer,
    "molecule_RelativeContentBi_featurizer": molecule_RelativeContentBi_featurizer,
    "molecule_HasBi_featurizer": molecule_HasBi_featurizer,
    "molecule_NumberAtomsPo_featurizer": molecule_NumberAtomsPo_featurizer,
    "molecule_RelativeContentPo_featurizer": molecule_RelativeContentPo_featurizer,
    "molecule_HasPo_featurizer": molecule_HasPo_featurizer,
    "molecule_NumberAtomsAt_featurizer": molecule_NumberAtomsAt_featurizer,
    "molecule_RelativeContentAt_featurizer": molecule_RelativeContentAt_featurizer,
    "molecule_HasAt_featurizer": molecule_HasAt_featurizer,
    "molecule_NumberAtomsRn_featurizer": molecule_NumberAtomsRn_featurizer,
    "molecule_RelativeContentRn_featurizer": molecule_RelativeContentRn_featurizer,
    "molecule_HasRn_featurizer": molecule_HasRn_featurizer,
    "molecule_NumberAtomsFr_featurizer": molecule_NumberAtomsFr_featurizer,
    "molecule_RelativeContentFr_featurizer": molecule_RelativeContentFr_featurizer,
    "molecule_HasFr_featurizer": molecule_HasFr_featurizer,
    "molecule_NumberAtomsRa_featurizer": molecule_NumberAtomsRa_featurizer,
    "molecule_RelativeContentRa_featurizer": molecule_RelativeContentRa_featurizer,
    "molecule_HasRa_featurizer": molecule_HasRa_featurizer,
    "molecule_NumberAtomsAc_featurizer": molecule_NumberAtomsAc_featurizer,
    "molecule_RelativeContentAc_featurizer": molecule_RelativeContentAc_featurizer,
    "molecule_HasAc_featurizer": molecule_HasAc_featurizer,
    "molecule_NumberAtomsTh_featurizer": molecule_NumberAtomsTh_featurizer,
    "molecule_RelativeContentTh_featurizer": molecule_RelativeContentTh_featurizer,
    "molecule_HasTh_featurizer": molecule_HasTh_featurizer,
    "molecule_NumberAtomsPa_featurizer": molecule_NumberAtomsPa_featurizer,
    "molecule_RelativeContentPa_featurizer": molecule_RelativeContentPa_featurizer,
    "molecule_HasPa_featurizer": molecule_HasPa_featurizer,
    "molecule_NumberAtomsU_featurizer": molecule_NumberAtomsU_featurizer,
    "molecule_RelativeContentU_featurizer": molecule_RelativeContentU_featurizer,
    "molecule_HasU_featurizer": molecule_HasU_featurizer,
    "molecule_NumberAtomsNp_featurizer": molecule_NumberAtomsNp_featurizer,
    "molecule_RelativeContentNp_featurizer": molecule_RelativeContentNp_featurizer,
    "molecule_HasNp_featurizer": molecule_HasNp_featurizer,
    "molecule_NumberAtomsPu_featurizer": molecule_NumberAtomsPu_featurizer,
    "molecule_RelativeContentPu_featurizer": molecule_RelativeContentPu_featurizer,
    "molecule_HasPu_featurizer": molecule_HasPu_featurizer,
    "molecule_NumberAtomsAm_featurizer": molecule_NumberAtomsAm_featurizer,
    "molecule_RelativeContentAm_featurizer": molecule_RelativeContentAm_featurizer,
    "molecule_HasAm_featurizer": molecule_HasAm_featurizer,
    "molecule_NumberAtomsCm_featurizer": molecule_NumberAtomsCm_featurizer,
    "molecule_RelativeContentCm_featurizer": molecule_RelativeContentCm_featurizer,
    "molecule_HasCm_featurizer": molecule_HasCm_featurizer,
    "molecule_NumberAtomsBk_featurizer": molecule_NumberAtomsBk_featurizer,
    "molecule_RelativeContentBk_featurizer": molecule_RelativeContentBk_featurizer,
    "molecule_HasBk_featurizer": molecule_HasBk_featurizer,
    "molecule_NumberAtomsCf_featurizer": molecule_NumberAtomsCf_featurizer,
    "molecule_RelativeContentCf_featurizer": molecule_RelativeContentCf_featurizer,
    "molecule_HasCf_featurizer": molecule_HasCf_featurizer,
    "molecule_NumberAtomsEs_featurizer": molecule_NumberAtomsEs_featurizer,
    "molecule_RelativeContentEs_featurizer": molecule_RelativeContentEs_featurizer,
    "molecule_HasEs_featurizer": molecule_HasEs_featurizer,
    "molecule_NumberAtomsFm_featurizer": molecule_NumberAtomsFm_featurizer,
    "molecule_RelativeContentFm_featurizer": molecule_RelativeContentFm_featurizer,
    "molecule_HasFm_featurizer": molecule_HasFm_featurizer,
    "molecule_NumberAtomsMd_featurizer": molecule_NumberAtomsMd_featurizer,
    "molecule_RelativeContentMd_featurizer": molecule_RelativeContentMd_featurizer,
    "molecule_HasMd_featurizer": molecule_HasMd_featurizer,
    "molecule_NumberAtomsNo_featurizer": molecule_NumberAtomsNo_featurizer,
    "molecule_RelativeContentNo_featurizer": molecule_RelativeContentNo_featurizer,
    "molecule_HasNo_featurizer": molecule_HasNo_featurizer,
    "molecule_NumberAtomsLr_featurizer": molecule_NumberAtomsLr_featurizer,
    "molecule_RelativeContentLr_featurizer": molecule_RelativeContentLr_featurizer,
    "molecule_HasLr_featurizer": molecule_HasLr_featurizer,
    "molecule_NumberAtomsRf_featurizer": molecule_NumberAtomsRf_featurizer,
    "molecule_RelativeContentRf_featurizer": molecule_RelativeContentRf_featurizer,
    "molecule_HasRf_featurizer": molecule_HasRf_featurizer,
    "molecule_NumberAtomsDb_featurizer": molecule_NumberAtomsDb_featurizer,
    "molecule_RelativeContentDb_featurizer": molecule_RelativeContentDb_featurizer,
    "molecule_HasDb_featurizer": molecule_HasDb_featurizer,
    "molecule_NumberAtomsSg_featurizer": molecule_NumberAtomsSg_featurizer,
    "molecule_RelativeContentSg_featurizer": molecule_RelativeContentSg_featurizer,
    "molecule_HasSg_featurizer": molecule_HasSg_featurizer,
    "molecule_NumberAtomsBh_featurizer": molecule_NumberAtomsBh_featurizer,
    "molecule_RelativeContentBh_featurizer": molecule_RelativeContentBh_featurizer,
    "molecule_HasBh_featurizer": molecule_HasBh_featurizer,
    "molecule_NumberAtomsHs_featurizer": molecule_NumberAtomsHs_featurizer,
    "molecule_RelativeContentHs_featurizer": molecule_RelativeContentHs_featurizer,
    "molecule_HasHs_featurizer": molecule_HasHs_featurizer,
    "molecule_NumberAtomsMt_featurizer": molecule_NumberAtomsMt_featurizer,
    "molecule_RelativeContentMt_featurizer": molecule_RelativeContentMt_featurizer,
    "molecule_HasMt_featurizer": molecule_HasMt_featurizer,
    "molecule_NumberAtomsDs_featurizer": molecule_NumberAtomsDs_featurizer,
    "molecule_RelativeContentDs_featurizer": molecule_RelativeContentDs_featurizer,
    "molecule_HasDs_featurizer": molecule_HasDs_featurizer,
    "molecule_NumberAtomsRg_featurizer": molecule_NumberAtomsRg_featurizer,
    "molecule_RelativeContentRg_featurizer": molecule_RelativeContentRg_featurizer,
    "molecule_HasRg_featurizer": molecule_HasRg_featurizer,
    "molecule_NumberAtomsCn_featurizer": molecule_NumberAtomsCn_featurizer,
    "molecule_RelativeContentCn_featurizer": molecule_RelativeContentCn_featurizer,
    "molecule_HasCn_featurizer": molecule_HasCn_featurizer,
    "molecule_NumberAtomsNh_featurizer": molecule_NumberAtomsNh_featurizer,
    "molecule_RelativeContentNh_featurizer": molecule_RelativeContentNh_featurizer,
    "molecule_HasNh_featurizer": molecule_HasNh_featurizer,
    "molecule_NumberAtomsFl_featurizer": molecule_NumberAtomsFl_featurizer,
    "molecule_RelativeContentFl_featurizer": molecule_RelativeContentFl_featurizer,
    "molecule_HasFl_featurizer": molecule_HasFl_featurizer,
    "molecule_NumberAtomsMc_featurizer": molecule_NumberAtomsMc_featurizer,
    "molecule_RelativeContentMc_featurizer": molecule_RelativeContentMc_featurizer,
    "molecule_HasMc_featurizer": molecule_HasMc_featurizer,
    "molecule_NumberAtomsLv_featurizer": molecule_NumberAtomsLv_featurizer,
    "molecule_RelativeContentLv_featurizer": molecule_RelativeContentLv_featurizer,
    "molecule_HasLv_featurizer": molecule_HasLv_featurizer,
    "molecule_NumberAtomsTs_featurizer": molecule_NumberAtomsTs_featurizer,
    "molecule_RelativeContentTs_featurizer": molecule_RelativeContentTs_featurizer,
    "molecule_HasTs_featurizer": molecule_HasTs_featurizer,
    "molecule_NumberAtomsOg_featurizer": molecule_NumberAtomsOg_featurizer,
    "molecule_RelativeContentOg_featurizer": molecule_RelativeContentOg_featurizer,
    "molecule_HasOg_featurizer": molecule_HasOg_featurizer,
}

__all__ = [
    "Molecule_NumberAtomsRgroup_Featurizer",
    "molecule_NumberAtomsRgroup_featurizer",
    "Molecule_RelativeContentRgroup_Featurizer",
    "molecule_RelativeContentRgroup_featurizer",
    "Molecule_HasRgroup_Featurizer",
    "molecule_HasRgroup_featurizer",
    "Molecule_NumberAtomsH_Featurizer",
    "molecule_NumberAtomsH_featurizer",
    "Molecule_RelativeContentH_Featurizer",
    "molecule_RelativeContentH_featurizer",
    "Molecule_HasH_Featurizer",
    "molecule_HasH_featurizer",
    "Molecule_NumberAtomsHe_Featurizer",
    "molecule_NumberAtomsHe_featurizer",
    "Molecule_RelativeContentHe_Featurizer",
    "molecule_RelativeContentHe_featurizer",
    "Molecule_HasHe_Featurizer",
    "molecule_HasHe_featurizer",
    "Molecule_NumberAtomsLi_Featurizer",
    "molecule_NumberAtomsLi_featurizer",
    "Molecule_RelativeContentLi_Featurizer",
    "molecule_RelativeContentLi_featurizer",
    "Molecule_HasLi_Featurizer",
    "molecule_HasLi_featurizer",
    "Molecule_NumberAtomsBe_Featurizer",
    "molecule_NumberAtomsBe_featurizer",
    "Molecule_RelativeContentBe_Featurizer",
    "molecule_RelativeContentBe_featurizer",
    "Molecule_HasBe_Featurizer",
    "molecule_HasBe_featurizer",
    "Molecule_NumberAtomsB_Featurizer",
    "molecule_NumberAtomsB_featurizer",
    "Molecule_RelativeContentB_Featurizer",
    "molecule_RelativeContentB_featurizer",
    "Molecule_HasB_Featurizer",
    "molecule_HasB_featurizer",
    "Molecule_NumberAtomsC_Featurizer",
    "molecule_NumberAtomsC_featurizer",
    "Molecule_RelativeContentC_Featurizer",
    "molecule_RelativeContentC_featurizer",
    "Molecule_HasC_Featurizer",
    "molecule_HasC_featurizer",
    "Molecule_NumberAtomsN_Featurizer",
    "molecule_NumberAtomsN_featurizer",
    "Molecule_RelativeContentN_Featurizer",
    "molecule_RelativeContentN_featurizer",
    "Molecule_HasN_Featurizer",
    "molecule_HasN_featurizer",
    "Molecule_NumberAtomsO_Featurizer",
    "molecule_NumberAtomsO_featurizer",
    "Molecule_RelativeContentO_Featurizer",
    "molecule_RelativeContentO_featurizer",
    "Molecule_HasO_Featurizer",
    "molecule_HasO_featurizer",
    "Molecule_NumberAtomsF_Featurizer",
    "molecule_NumberAtomsF_featurizer",
    "Molecule_RelativeContentF_Featurizer",
    "molecule_RelativeContentF_featurizer",
    "Molecule_HasF_Featurizer",
    "molecule_HasF_featurizer",
    "Molecule_NumberAtomsNe_Featurizer",
    "molecule_NumberAtomsNe_featurizer",
    "Molecule_RelativeContentNe_Featurizer",
    "molecule_RelativeContentNe_featurizer",
    "Molecule_HasNe_Featurizer",
    "molecule_HasNe_featurizer",
    "Molecule_NumberAtomsNa_Featurizer",
    "molecule_NumberAtomsNa_featurizer",
    "Molecule_RelativeContentNa_Featurizer",
    "molecule_RelativeContentNa_featurizer",
    "Molecule_HasNa_Featurizer",
    "molecule_HasNa_featurizer",
    "Molecule_NumberAtomsMg_Featurizer",
    "molecule_NumberAtomsMg_featurizer",
    "Molecule_RelativeContentMg_Featurizer",
    "molecule_RelativeContentMg_featurizer",
    "Molecule_HasMg_Featurizer",
    "molecule_HasMg_featurizer",
    "Molecule_NumberAtomsAl_Featurizer",
    "molecule_NumberAtomsAl_featurizer",
    "Molecule_RelativeContentAl_Featurizer",
    "molecule_RelativeContentAl_featurizer",
    "Molecule_HasAl_Featurizer",
    "molecule_HasAl_featurizer",
    "Molecule_NumberAtomsSi_Featurizer",
    "molecule_NumberAtomsSi_featurizer",
    "Molecule_RelativeContentSi_Featurizer",
    "molecule_RelativeContentSi_featurizer",
    "Molecule_HasSi_Featurizer",
    "molecule_HasSi_featurizer",
    "Molecule_NumberAtomsP_Featurizer",
    "molecule_NumberAtomsP_featurizer",
    "Molecule_RelativeContentP_Featurizer",
    "molecule_RelativeContentP_featurizer",
    "Molecule_HasP_Featurizer",
    "molecule_HasP_featurizer",
    "Molecule_NumberAtomsS_Featurizer",
    "molecule_NumberAtomsS_featurizer",
    "Molecule_RelativeContentS_Featurizer",
    "molecule_RelativeContentS_featurizer",
    "Molecule_HasS_Featurizer",
    "molecule_HasS_featurizer",
    "Molecule_NumberAtomsCl_Featurizer",
    "molecule_NumberAtomsCl_featurizer",
    "Molecule_RelativeContentCl_Featurizer",
    "molecule_RelativeContentCl_featurizer",
    "Molecule_HasCl_Featurizer",
    "molecule_HasCl_featurizer",
    "Molecule_NumberAtomsAr_Featurizer",
    "molecule_NumberAtomsAr_featurizer",
    "Molecule_RelativeContentAr_Featurizer",
    "molecule_RelativeContentAr_featurizer",
    "Molecule_HasAr_Featurizer",
    "molecule_HasAr_featurizer",
    "Molecule_NumberAtomsK_Featurizer",
    "molecule_NumberAtomsK_featurizer",
    "Molecule_RelativeContentK_Featurizer",
    "molecule_RelativeContentK_featurizer",
    "Molecule_HasK_Featurizer",
    "molecule_HasK_featurizer",
    "Molecule_NumberAtomsCa_Featurizer",
    "molecule_NumberAtomsCa_featurizer",
    "Molecule_RelativeContentCa_Featurizer",
    "molecule_RelativeContentCa_featurizer",
    "Molecule_HasCa_Featurizer",
    "molecule_HasCa_featurizer",
    "Molecule_NumberAtomsSc_Featurizer",
    "molecule_NumberAtomsSc_featurizer",
    "Molecule_RelativeContentSc_Featurizer",
    "molecule_RelativeContentSc_featurizer",
    "Molecule_HasSc_Featurizer",
    "molecule_HasSc_featurizer",
    "Molecule_NumberAtomsTi_Featurizer",
    "molecule_NumberAtomsTi_featurizer",
    "Molecule_RelativeContentTi_Featurizer",
    "molecule_RelativeContentTi_featurizer",
    "Molecule_HasTi_Featurizer",
    "molecule_HasTi_featurizer",
    "Molecule_NumberAtomsV_Featurizer",
    "molecule_NumberAtomsV_featurizer",
    "Molecule_RelativeContentV_Featurizer",
    "molecule_RelativeContentV_featurizer",
    "Molecule_HasV_Featurizer",
    "molecule_HasV_featurizer",
    "Molecule_NumberAtomsCr_Featurizer",
    "molecule_NumberAtomsCr_featurizer",
    "Molecule_RelativeContentCr_Featurizer",
    "molecule_RelativeContentCr_featurizer",
    "Molecule_HasCr_Featurizer",
    "molecule_HasCr_featurizer",
    "Molecule_NumberAtomsMn_Featurizer",
    "molecule_NumberAtomsMn_featurizer",
    "Molecule_RelativeContentMn_Featurizer",
    "molecule_RelativeContentMn_featurizer",
    "Molecule_HasMn_Featurizer",
    "molecule_HasMn_featurizer",
    "Molecule_NumberAtomsFe_Featurizer",
    "molecule_NumberAtomsFe_featurizer",
    "Molecule_RelativeContentFe_Featurizer",
    "molecule_RelativeContentFe_featurizer",
    "Molecule_HasFe_Featurizer",
    "molecule_HasFe_featurizer",
    "Molecule_NumberAtomsCo_Featurizer",
    "molecule_NumberAtomsCo_featurizer",
    "Molecule_RelativeContentCo_Featurizer",
    "molecule_RelativeContentCo_featurizer",
    "Molecule_HasCo_Featurizer",
    "molecule_HasCo_featurizer",
    "Molecule_NumberAtomsNi_Featurizer",
    "molecule_NumberAtomsNi_featurizer",
    "Molecule_RelativeContentNi_Featurizer",
    "molecule_RelativeContentNi_featurizer",
    "Molecule_HasNi_Featurizer",
    "molecule_HasNi_featurizer",
    "Molecule_NumberAtomsCu_Featurizer",
    "molecule_NumberAtomsCu_featurizer",
    "Molecule_RelativeContentCu_Featurizer",
    "molecule_RelativeContentCu_featurizer",
    "Molecule_HasCu_Featurizer",
    "molecule_HasCu_featurizer",
    "Molecule_NumberAtomsZn_Featurizer",
    "molecule_NumberAtomsZn_featurizer",
    "Molecule_RelativeContentZn_Featurizer",
    "molecule_RelativeContentZn_featurizer",
    "Molecule_HasZn_Featurizer",
    "molecule_HasZn_featurizer",
    "Molecule_NumberAtomsGa_Featurizer",
    "molecule_NumberAtomsGa_featurizer",
    "Molecule_RelativeContentGa_Featurizer",
    "molecule_RelativeContentGa_featurizer",
    "Molecule_HasGa_Featurizer",
    "molecule_HasGa_featurizer",
    "Molecule_NumberAtomsGe_Featurizer",
    "molecule_NumberAtomsGe_featurizer",
    "Molecule_RelativeContentGe_Featurizer",
    "molecule_RelativeContentGe_featurizer",
    "Molecule_HasGe_Featurizer",
    "molecule_HasGe_featurizer",
    "Molecule_NumberAtomsAs_Featurizer",
    "molecule_NumberAtomsAs_featurizer",
    "Molecule_RelativeContentAs_Featurizer",
    "molecule_RelativeContentAs_featurizer",
    "Molecule_HasAs_Featurizer",
    "molecule_HasAs_featurizer",
    "Molecule_NumberAtomsSe_Featurizer",
    "molecule_NumberAtomsSe_featurizer",
    "Molecule_RelativeContentSe_Featurizer",
    "molecule_RelativeContentSe_featurizer",
    "Molecule_HasSe_Featurizer",
    "molecule_HasSe_featurizer",
    "Molecule_NumberAtomsBr_Featurizer",
    "molecule_NumberAtomsBr_featurizer",
    "Molecule_RelativeContentBr_Featurizer",
    "molecule_RelativeContentBr_featurizer",
    "Molecule_HasBr_Featurizer",
    "molecule_HasBr_featurizer",
    "Molecule_NumberAtomsKr_Featurizer",
    "molecule_NumberAtomsKr_featurizer",
    "Molecule_RelativeContentKr_Featurizer",
    "molecule_RelativeContentKr_featurizer",
    "Molecule_HasKr_Featurizer",
    "molecule_HasKr_featurizer",
    "Molecule_NumberAtomsRb_Featurizer",
    "molecule_NumberAtomsRb_featurizer",
    "Molecule_RelativeContentRb_Featurizer",
    "molecule_RelativeContentRb_featurizer",
    "Molecule_HasRb_Featurizer",
    "molecule_HasRb_featurizer",
    "Molecule_NumberAtomsSr_Featurizer",
    "molecule_NumberAtomsSr_featurizer",
    "Molecule_RelativeContentSr_Featurizer",
    "molecule_RelativeContentSr_featurizer",
    "Molecule_HasSr_Featurizer",
    "molecule_HasSr_featurizer",
    "Molecule_NumberAtomsY_Featurizer",
    "molecule_NumberAtomsY_featurizer",
    "Molecule_RelativeContentY_Featurizer",
    "molecule_RelativeContentY_featurizer",
    "Molecule_HasY_Featurizer",
    "molecule_HasY_featurizer",
    "Molecule_NumberAtomsZr_Featurizer",
    "molecule_NumberAtomsZr_featurizer",
    "Molecule_RelativeContentZr_Featurizer",
    "molecule_RelativeContentZr_featurizer",
    "Molecule_HasZr_Featurizer",
    "molecule_HasZr_featurizer",
    "Molecule_NumberAtomsNb_Featurizer",
    "molecule_NumberAtomsNb_featurizer",
    "Molecule_RelativeContentNb_Featurizer",
    "molecule_RelativeContentNb_featurizer",
    "Molecule_HasNb_Featurizer",
    "molecule_HasNb_featurizer",
    "Molecule_NumberAtomsMo_Featurizer",
    "molecule_NumberAtomsMo_featurizer",
    "Molecule_RelativeContentMo_Featurizer",
    "molecule_RelativeContentMo_featurizer",
    "Molecule_HasMo_Featurizer",
    "molecule_HasMo_featurizer",
    "Molecule_NumberAtomsTc_Featurizer",
    "molecule_NumberAtomsTc_featurizer",
    "Molecule_RelativeContentTc_Featurizer",
    "molecule_RelativeContentTc_featurizer",
    "Molecule_HasTc_Featurizer",
    "molecule_HasTc_featurizer",
    "Molecule_NumberAtomsRu_Featurizer",
    "molecule_NumberAtomsRu_featurizer",
    "Molecule_RelativeContentRu_Featurizer",
    "molecule_RelativeContentRu_featurizer",
    "Molecule_HasRu_Featurizer",
    "molecule_HasRu_featurizer",
    "Molecule_NumberAtomsRh_Featurizer",
    "molecule_NumberAtomsRh_featurizer",
    "Molecule_RelativeContentRh_Featurizer",
    "molecule_RelativeContentRh_featurizer",
    "Molecule_HasRh_Featurizer",
    "molecule_HasRh_featurizer",
    "Molecule_NumberAtomsPd_Featurizer",
    "molecule_NumberAtomsPd_featurizer",
    "Molecule_RelativeContentPd_Featurizer",
    "molecule_RelativeContentPd_featurizer",
    "Molecule_HasPd_Featurizer",
    "molecule_HasPd_featurizer",
    "Molecule_NumberAtomsAg_Featurizer",
    "molecule_NumberAtomsAg_featurizer",
    "Molecule_RelativeContentAg_Featurizer",
    "molecule_RelativeContentAg_featurizer",
    "Molecule_HasAg_Featurizer",
    "molecule_HasAg_featurizer",
    "Molecule_NumberAtomsCd_Featurizer",
    "molecule_NumberAtomsCd_featurizer",
    "Molecule_RelativeContentCd_Featurizer",
    "molecule_RelativeContentCd_featurizer",
    "Molecule_HasCd_Featurizer",
    "molecule_HasCd_featurizer",
    "Molecule_NumberAtomsIn_Featurizer",
    "molecule_NumberAtomsIn_featurizer",
    "Molecule_RelativeContentIn_Featurizer",
    "molecule_RelativeContentIn_featurizer",
    "Molecule_HasIn_Featurizer",
    "molecule_HasIn_featurizer",
    "Molecule_NumberAtomsSn_Featurizer",
    "molecule_NumberAtomsSn_featurizer",
    "Molecule_RelativeContentSn_Featurizer",
    "molecule_RelativeContentSn_featurizer",
    "Molecule_HasSn_Featurizer",
    "molecule_HasSn_featurizer",
    "Molecule_NumberAtomsSb_Featurizer",
    "molecule_NumberAtomsSb_featurizer",
    "Molecule_RelativeContentSb_Featurizer",
    "molecule_RelativeContentSb_featurizer",
    "Molecule_HasSb_Featurizer",
    "molecule_HasSb_featurizer",
    "Molecule_NumberAtomsTe_Featurizer",
    "molecule_NumberAtomsTe_featurizer",
    "Molecule_RelativeContentTe_Featurizer",
    "molecule_RelativeContentTe_featurizer",
    "Molecule_HasTe_Featurizer",
    "molecule_HasTe_featurizer",
    "Molecule_NumberAtomsI_Featurizer",
    "molecule_NumberAtomsI_featurizer",
    "Molecule_RelativeContentI_Featurizer",
    "molecule_RelativeContentI_featurizer",
    "Molecule_HasI_Featurizer",
    "molecule_HasI_featurizer",
    "Molecule_NumberAtomsXe_Featurizer",
    "molecule_NumberAtomsXe_featurizer",
    "Molecule_RelativeContentXe_Featurizer",
    "molecule_RelativeContentXe_featurizer",
    "Molecule_HasXe_Featurizer",
    "molecule_HasXe_featurizer",
    "Molecule_NumberAtomsCs_Featurizer",
    "molecule_NumberAtomsCs_featurizer",
    "Molecule_RelativeContentCs_Featurizer",
    "molecule_RelativeContentCs_featurizer",
    "Molecule_HasCs_Featurizer",
    "molecule_HasCs_featurizer",
    "Molecule_NumberAtomsBa_Featurizer",
    "molecule_NumberAtomsBa_featurizer",
    "Molecule_RelativeContentBa_Featurizer",
    "molecule_RelativeContentBa_featurizer",
    "Molecule_HasBa_Featurizer",
    "molecule_HasBa_featurizer",
    "Molecule_NumberAtomsLa_Featurizer",
    "molecule_NumberAtomsLa_featurizer",
    "Molecule_RelativeContentLa_Featurizer",
    "molecule_RelativeContentLa_featurizer",
    "Molecule_HasLa_Featurizer",
    "molecule_HasLa_featurizer",
    "Molecule_NumberAtomsCe_Featurizer",
    "molecule_NumberAtomsCe_featurizer",
    "Molecule_RelativeContentCe_Featurizer",
    "molecule_RelativeContentCe_featurizer",
    "Molecule_HasCe_Featurizer",
    "molecule_HasCe_featurizer",
    "Molecule_NumberAtomsPr_Featurizer",
    "molecule_NumberAtomsPr_featurizer",
    "Molecule_RelativeContentPr_Featurizer",
    "molecule_RelativeContentPr_featurizer",
    "Molecule_HasPr_Featurizer",
    "molecule_HasPr_featurizer",
    "Molecule_NumberAtomsNd_Featurizer",
    "molecule_NumberAtomsNd_featurizer",
    "Molecule_RelativeContentNd_Featurizer",
    "molecule_RelativeContentNd_featurizer",
    "Molecule_HasNd_Featurizer",
    "molecule_HasNd_featurizer",
    "Molecule_NumberAtomsPm_Featurizer",
    "molecule_NumberAtomsPm_featurizer",
    "Molecule_RelativeContentPm_Featurizer",
    "molecule_RelativeContentPm_featurizer",
    "Molecule_HasPm_Featurizer",
    "molecule_HasPm_featurizer",
    "Molecule_NumberAtomsSm_Featurizer",
    "molecule_NumberAtomsSm_featurizer",
    "Molecule_RelativeContentSm_Featurizer",
    "molecule_RelativeContentSm_featurizer",
    "Molecule_HasSm_Featurizer",
    "molecule_HasSm_featurizer",
    "Molecule_NumberAtomsEu_Featurizer",
    "molecule_NumberAtomsEu_featurizer",
    "Molecule_RelativeContentEu_Featurizer",
    "molecule_RelativeContentEu_featurizer",
    "Molecule_HasEu_Featurizer",
    "molecule_HasEu_featurizer",
    "Molecule_NumberAtomsGd_Featurizer",
    "molecule_NumberAtomsGd_featurizer",
    "Molecule_RelativeContentGd_Featurizer",
    "molecule_RelativeContentGd_featurizer",
    "Molecule_HasGd_Featurizer",
    "molecule_HasGd_featurizer",
    "Molecule_NumberAtomsTb_Featurizer",
    "molecule_NumberAtomsTb_featurizer",
    "Molecule_RelativeContentTb_Featurizer",
    "molecule_RelativeContentTb_featurizer",
    "Molecule_HasTb_Featurizer",
    "molecule_HasTb_featurizer",
    "Molecule_NumberAtomsDy_Featurizer",
    "molecule_NumberAtomsDy_featurizer",
    "Molecule_RelativeContentDy_Featurizer",
    "molecule_RelativeContentDy_featurizer",
    "Molecule_HasDy_Featurizer",
    "molecule_HasDy_featurizer",
    "Molecule_NumberAtomsHo_Featurizer",
    "molecule_NumberAtomsHo_featurizer",
    "Molecule_RelativeContentHo_Featurizer",
    "molecule_RelativeContentHo_featurizer",
    "Molecule_HasHo_Featurizer",
    "molecule_HasHo_featurizer",
    "Molecule_NumberAtomsEr_Featurizer",
    "molecule_NumberAtomsEr_featurizer",
    "Molecule_RelativeContentEr_Featurizer",
    "molecule_RelativeContentEr_featurizer",
    "Molecule_HasEr_Featurizer",
    "molecule_HasEr_featurizer",
    "Molecule_NumberAtomsTm_Featurizer",
    "molecule_NumberAtomsTm_featurizer",
    "Molecule_RelativeContentTm_Featurizer",
    "molecule_RelativeContentTm_featurizer",
    "Molecule_HasTm_Featurizer",
    "molecule_HasTm_featurizer",
    "Molecule_NumberAtomsYb_Featurizer",
    "molecule_NumberAtomsYb_featurizer",
    "Molecule_RelativeContentYb_Featurizer",
    "molecule_RelativeContentYb_featurizer",
    "Molecule_HasYb_Featurizer",
    "molecule_HasYb_featurizer",
    "Molecule_NumberAtomsLu_Featurizer",
    "molecule_NumberAtomsLu_featurizer",
    "Molecule_RelativeContentLu_Featurizer",
    "molecule_RelativeContentLu_featurizer",
    "Molecule_HasLu_Featurizer",
    "molecule_HasLu_featurizer",
    "Molecule_NumberAtomsHf_Featurizer",
    "molecule_NumberAtomsHf_featurizer",
    "Molecule_RelativeContentHf_Featurizer",
    "molecule_RelativeContentHf_featurizer",
    "Molecule_HasHf_Featurizer",
    "molecule_HasHf_featurizer",
    "Molecule_NumberAtomsTa_Featurizer",
    "molecule_NumberAtomsTa_featurizer",
    "Molecule_RelativeContentTa_Featurizer",
    "molecule_RelativeContentTa_featurizer",
    "Molecule_HasTa_Featurizer",
    "molecule_HasTa_featurizer",
    "Molecule_NumberAtomsW_Featurizer",
    "molecule_NumberAtomsW_featurizer",
    "Molecule_RelativeContentW_Featurizer",
    "molecule_RelativeContentW_featurizer",
    "Molecule_HasW_Featurizer",
    "molecule_HasW_featurizer",
    "Molecule_NumberAtomsRe_Featurizer",
    "molecule_NumberAtomsRe_featurizer",
    "Molecule_RelativeContentRe_Featurizer",
    "molecule_RelativeContentRe_featurizer",
    "Molecule_HasRe_Featurizer",
    "molecule_HasRe_featurizer",
    "Molecule_NumberAtomsOs_Featurizer",
    "molecule_NumberAtomsOs_featurizer",
    "Molecule_RelativeContentOs_Featurizer",
    "molecule_RelativeContentOs_featurizer",
    "Molecule_HasOs_Featurizer",
    "molecule_HasOs_featurizer",
    "Molecule_NumberAtomsIr_Featurizer",
    "molecule_NumberAtomsIr_featurizer",
    "Molecule_RelativeContentIr_Featurizer",
    "molecule_RelativeContentIr_featurizer",
    "Molecule_HasIr_Featurizer",
    "molecule_HasIr_featurizer",
    "Molecule_NumberAtomsPt_Featurizer",
    "molecule_NumberAtomsPt_featurizer",
    "Molecule_RelativeContentPt_Featurizer",
    "molecule_RelativeContentPt_featurizer",
    "Molecule_HasPt_Featurizer",
    "molecule_HasPt_featurizer",
    "Molecule_NumberAtomsAu_Featurizer",
    "molecule_NumberAtomsAu_featurizer",
    "Molecule_RelativeContentAu_Featurizer",
    "molecule_RelativeContentAu_featurizer",
    "Molecule_HasAu_Featurizer",
    "molecule_HasAu_featurizer",
    "Molecule_NumberAtomsHg_Featurizer",
    "molecule_NumberAtomsHg_featurizer",
    "Molecule_RelativeContentHg_Featurizer",
    "molecule_RelativeContentHg_featurizer",
    "Molecule_HasHg_Featurizer",
    "molecule_HasHg_featurizer",
    "Molecule_NumberAtomsTl_Featurizer",
    "molecule_NumberAtomsTl_featurizer",
    "Molecule_RelativeContentTl_Featurizer",
    "molecule_RelativeContentTl_featurizer",
    "Molecule_HasTl_Featurizer",
    "molecule_HasTl_featurizer",
    "Molecule_NumberAtomsPb_Featurizer",
    "molecule_NumberAtomsPb_featurizer",
    "Molecule_RelativeContentPb_Featurizer",
    "molecule_RelativeContentPb_featurizer",
    "Molecule_HasPb_Featurizer",
    "molecule_HasPb_featurizer",
    "Molecule_NumberAtomsBi_Featurizer",
    "molecule_NumberAtomsBi_featurizer",
    "Molecule_RelativeContentBi_Featurizer",
    "molecule_RelativeContentBi_featurizer",
    "Molecule_HasBi_Featurizer",
    "molecule_HasBi_featurizer",
    "Molecule_NumberAtomsPo_Featurizer",
    "molecule_NumberAtomsPo_featurizer",
    "Molecule_RelativeContentPo_Featurizer",
    "molecule_RelativeContentPo_featurizer",
    "Molecule_HasPo_Featurizer",
    "molecule_HasPo_featurizer",
    "Molecule_NumberAtomsAt_Featurizer",
    "molecule_NumberAtomsAt_featurizer",
    "Molecule_RelativeContentAt_Featurizer",
    "molecule_RelativeContentAt_featurizer",
    "Molecule_HasAt_Featurizer",
    "molecule_HasAt_featurizer",
    "Molecule_NumberAtomsRn_Featurizer",
    "molecule_NumberAtomsRn_featurizer",
    "Molecule_RelativeContentRn_Featurizer",
    "molecule_RelativeContentRn_featurizer",
    "Molecule_HasRn_Featurizer",
    "molecule_HasRn_featurizer",
    "Molecule_NumberAtomsFr_Featurizer",
    "molecule_NumberAtomsFr_featurizer",
    "Molecule_RelativeContentFr_Featurizer",
    "molecule_RelativeContentFr_featurizer",
    "Molecule_HasFr_Featurizer",
    "molecule_HasFr_featurizer",
    "Molecule_NumberAtomsRa_Featurizer",
    "molecule_NumberAtomsRa_featurizer",
    "Molecule_RelativeContentRa_Featurizer",
    "molecule_RelativeContentRa_featurizer",
    "Molecule_HasRa_Featurizer",
    "molecule_HasRa_featurizer",
    "Molecule_NumberAtomsAc_Featurizer",
    "molecule_NumberAtomsAc_featurizer",
    "Molecule_RelativeContentAc_Featurizer",
    "molecule_RelativeContentAc_featurizer",
    "Molecule_HasAc_Featurizer",
    "molecule_HasAc_featurizer",
    "Molecule_NumberAtomsTh_Featurizer",
    "molecule_NumberAtomsTh_featurizer",
    "Molecule_RelativeContentTh_Featurizer",
    "molecule_RelativeContentTh_featurizer",
    "Molecule_HasTh_Featurizer",
    "molecule_HasTh_featurizer",
    "Molecule_NumberAtomsPa_Featurizer",
    "molecule_NumberAtomsPa_featurizer",
    "Molecule_RelativeContentPa_Featurizer",
    "molecule_RelativeContentPa_featurizer",
    "Molecule_HasPa_Featurizer",
    "molecule_HasPa_featurizer",
    "Molecule_NumberAtomsU_Featurizer",
    "molecule_NumberAtomsU_featurizer",
    "Molecule_RelativeContentU_Featurizer",
    "molecule_RelativeContentU_featurizer",
    "Molecule_HasU_Featurizer",
    "molecule_HasU_featurizer",
    "Molecule_NumberAtomsNp_Featurizer",
    "molecule_NumberAtomsNp_featurizer",
    "Molecule_RelativeContentNp_Featurizer",
    "molecule_RelativeContentNp_featurizer",
    "Molecule_HasNp_Featurizer",
    "molecule_HasNp_featurizer",
    "Molecule_NumberAtomsPu_Featurizer",
    "molecule_NumberAtomsPu_featurizer",
    "Molecule_RelativeContentPu_Featurizer",
    "molecule_RelativeContentPu_featurizer",
    "Molecule_HasPu_Featurizer",
    "molecule_HasPu_featurizer",
    "Molecule_NumberAtomsAm_Featurizer",
    "molecule_NumberAtomsAm_featurizer",
    "Molecule_RelativeContentAm_Featurizer",
    "molecule_RelativeContentAm_featurizer",
    "Molecule_HasAm_Featurizer",
    "molecule_HasAm_featurizer",
    "Molecule_NumberAtomsCm_Featurizer",
    "molecule_NumberAtomsCm_featurizer",
    "Molecule_RelativeContentCm_Featurizer",
    "molecule_RelativeContentCm_featurizer",
    "Molecule_HasCm_Featurizer",
    "molecule_HasCm_featurizer",
    "Molecule_NumberAtomsBk_Featurizer",
    "molecule_NumberAtomsBk_featurizer",
    "Molecule_RelativeContentBk_Featurizer",
    "molecule_RelativeContentBk_featurizer",
    "Molecule_HasBk_Featurizer",
    "molecule_HasBk_featurizer",
    "Molecule_NumberAtomsCf_Featurizer",
    "molecule_NumberAtomsCf_featurizer",
    "Molecule_RelativeContentCf_Featurizer",
    "molecule_RelativeContentCf_featurizer",
    "Molecule_HasCf_Featurizer",
    "molecule_HasCf_featurizer",
    "Molecule_NumberAtomsEs_Featurizer",
    "molecule_NumberAtomsEs_featurizer",
    "Molecule_RelativeContentEs_Featurizer",
    "molecule_RelativeContentEs_featurizer",
    "Molecule_HasEs_Featurizer",
    "molecule_HasEs_featurizer",
    "Molecule_NumberAtomsFm_Featurizer",
    "molecule_NumberAtomsFm_featurizer",
    "Molecule_RelativeContentFm_Featurizer",
    "molecule_RelativeContentFm_featurizer",
    "Molecule_HasFm_Featurizer",
    "molecule_HasFm_featurizer",
    "Molecule_NumberAtomsMd_Featurizer",
    "molecule_NumberAtomsMd_featurizer",
    "Molecule_RelativeContentMd_Featurizer",
    "molecule_RelativeContentMd_featurizer",
    "Molecule_HasMd_Featurizer",
    "molecule_HasMd_featurizer",
    "Molecule_NumberAtomsNo_Featurizer",
    "molecule_NumberAtomsNo_featurizer",
    "Molecule_RelativeContentNo_Featurizer",
    "molecule_RelativeContentNo_featurizer",
    "Molecule_HasNo_Featurizer",
    "molecule_HasNo_featurizer",
    "Molecule_NumberAtomsLr_Featurizer",
    "molecule_NumberAtomsLr_featurizer",
    "Molecule_RelativeContentLr_Featurizer",
    "molecule_RelativeContentLr_featurizer",
    "Molecule_HasLr_Featurizer",
    "molecule_HasLr_featurizer",
    "Molecule_NumberAtomsRf_Featurizer",
    "molecule_NumberAtomsRf_featurizer",
    "Molecule_RelativeContentRf_Featurizer",
    "molecule_RelativeContentRf_featurizer",
    "Molecule_HasRf_Featurizer",
    "molecule_HasRf_featurizer",
    "Molecule_NumberAtomsDb_Featurizer",
    "molecule_NumberAtomsDb_featurizer",
    "Molecule_RelativeContentDb_Featurizer",
    "molecule_RelativeContentDb_featurizer",
    "Molecule_HasDb_Featurizer",
    "molecule_HasDb_featurizer",
    "Molecule_NumberAtomsSg_Featurizer",
    "molecule_NumberAtomsSg_featurizer",
    "Molecule_RelativeContentSg_Featurizer",
    "molecule_RelativeContentSg_featurizer",
    "Molecule_HasSg_Featurizer",
    "molecule_HasSg_featurizer",
    "Molecule_NumberAtomsBh_Featurizer",
    "molecule_NumberAtomsBh_featurizer",
    "Molecule_RelativeContentBh_Featurizer",
    "molecule_RelativeContentBh_featurizer",
    "Molecule_HasBh_Featurizer",
    "molecule_HasBh_featurizer",
    "Molecule_NumberAtomsHs_Featurizer",
    "molecule_NumberAtomsHs_featurizer",
    "Molecule_RelativeContentHs_Featurizer",
    "molecule_RelativeContentHs_featurizer",
    "Molecule_HasHs_Featurizer",
    "molecule_HasHs_featurizer",
    "Molecule_NumberAtomsMt_Featurizer",
    "molecule_NumberAtomsMt_featurizer",
    "Molecule_RelativeContentMt_Featurizer",
    "molecule_RelativeContentMt_featurizer",
    "Molecule_HasMt_Featurizer",
    "molecule_HasMt_featurizer",
    "Molecule_NumberAtomsDs_Featurizer",
    "molecule_NumberAtomsDs_featurizer",
    "Molecule_RelativeContentDs_Featurizer",
    "molecule_RelativeContentDs_featurizer",
    "Molecule_HasDs_Featurizer",
    "molecule_HasDs_featurizer",
    "Molecule_NumberAtomsRg_Featurizer",
    "molecule_NumberAtomsRg_featurizer",
    "Molecule_RelativeContentRg_Featurizer",
    "molecule_RelativeContentRg_featurizer",
    "Molecule_HasRg_Featurizer",
    "molecule_HasRg_featurizer",
    "Molecule_NumberAtomsCn_Featurizer",
    "molecule_NumberAtomsCn_featurizer",
    "Molecule_RelativeContentCn_Featurizer",
    "molecule_RelativeContentCn_featurizer",
    "Molecule_HasCn_Featurizer",
    "molecule_HasCn_featurizer",
    "Molecule_NumberAtomsNh_Featurizer",
    "molecule_NumberAtomsNh_featurizer",
    "Molecule_RelativeContentNh_Featurizer",
    "molecule_RelativeContentNh_featurizer",
    "Molecule_HasNh_Featurizer",
    "molecule_HasNh_featurizer",
    "Molecule_NumberAtomsFl_Featurizer",
    "molecule_NumberAtomsFl_featurizer",
    "Molecule_RelativeContentFl_Featurizer",
    "molecule_RelativeContentFl_featurizer",
    "Molecule_HasFl_Featurizer",
    "molecule_HasFl_featurizer",
    "Molecule_NumberAtomsMc_Featurizer",
    "molecule_NumberAtomsMc_featurizer",
    "Molecule_RelativeContentMc_Featurizer",
    "molecule_RelativeContentMc_featurizer",
    "Molecule_HasMc_Featurizer",
    "molecule_HasMc_featurizer",
    "Molecule_NumberAtomsLv_Featurizer",
    "molecule_NumberAtomsLv_featurizer",
    "Molecule_RelativeContentLv_Featurizer",
    "molecule_RelativeContentLv_featurizer",
    "Molecule_HasLv_Featurizer",
    "molecule_HasLv_featurizer",
    "Molecule_NumberAtomsTs_Featurizer",
    "molecule_NumberAtomsTs_featurizer",
    "Molecule_RelativeContentTs_Featurizer",
    "molecule_RelativeContentTs_featurizer",
    "Molecule_HasTs_Featurizer",
    "molecule_HasTs_featurizer",
    "Molecule_NumberAtomsOg_Featurizer",
    "molecule_NumberAtomsOg_featurizer",
    "Molecule_RelativeContentOg_Featurizer",
    "molecule_RelativeContentOg_featurizer",
    "Molecule_HasOg_Featurizer",
    "molecule_HasOg_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
