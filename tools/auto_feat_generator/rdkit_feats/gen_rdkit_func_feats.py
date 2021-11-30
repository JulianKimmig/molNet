import enum
import importlib
import re
from dataclasses import dataclass, field
from inspect import isfunction, ismodule, isgenerator
from typing import Callable, Any

import black
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import (
    rdMolDescriptors,
    Descriptors3D,
    GraphDescriptors,
    Descriptors,
    rdmolops,
    SanitizeMol, MolToSmiles,
)
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdchem import Mol, MolBundle
from rdkit.Chem.rdchem import SubstanceGroup_VECT
from rdkit.DataStructs import ExplicitBitVect, LongSparseIntVect, IntSparseIntVect, ULongSparseIntVect, \
    ConvertToNumpyArray
from rdkit.ForceField import ForceField, MMFFMolProperties
from rdkit.Geometry import UniformGrid3D_
from rdkit.rdBase import _vectdouble, _vectint

from molNet import ConformerError
from molNet.featurizer._molecule_featurizer import prepare_mol_for_featurization
from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS

BAD_LIST = [
    "^SplitMolByPDBResidues$",  # creashes syste,
    "^SplitMolByPDBChainId$",  # creashes system
    "SanitizeMol",  #
    "AUTOCORR2D_[0-9]",
    "^_",
    "_$",  # internal use,
]

BAD_LIST = [re.compile(s) for s in BAD_LIST]

_MOL_FUNCTIONS_PRERUN_FAILED = ['rdkit.Chem.MolToTPLFile', 'rdkit.Chem.AllChem.GetBestRMS',
                                'rdkit.Chem.AllChem.GetUSRDistributionsFromPoints',
                                'rdkit.Geometry.rdGeometry.TanimotoDistance',
                                'rdkit.Chem.AllChem.ExplicitValenceLessQueryAtom',
                                'rdkit.Chem.rdChemReactions.HasProductTemplateSubstructMatch',
                                'rdkit.Chem.rdqueries.QHAtomQueryAtom',
                                'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.IsotopeEqualsQueryAtom',
                                'rdkit.Chem.rdShapeHelpers.ComputeUnionBox', 'rdkit.Chem.CreateMolSubstanceGroup',
                                'rdkit.DataStructs.BulkBraunBlanquetSimilarity',
                                'rdkit.Chem.rdMolTransforms.SetDihedralDeg', 'rdkit.Chem.ReplaceSidechains',
                                'rdkit.Geometry.ProtrudeDistance', 'rdkit.Chem.AllChem.MolFromPNGString',
                                'rdkit.DataStructs.cDataStructs.KulczynskiSimilarityNeighbors',
                                'rdkit.Chem.rdChemReactions.GetDefaultAdjustParams',
                                'rdkit.Chem.rdChemReactions.ReactionFromPNGString',
                                'rdkit.Geometry.FindGridTerminalPoints', 'rdkit.Chem.rdchem.GetSupplementalSmilesLabel',
                                'rdkit.Chem.rdMolDescriptors.GetAtomFeatures',
                                'rdkit.Chem.rdqueries.NumRadicalElectronsEqualsQueryAtom',
                                'rdkit.Chem.rdmolfiles.MolFromMolFile', 'rdkit.DataStructs.RusselSimilarity',
                                'rdkit.Chem.JSONToMols', 'rdkit.DataStructs.cDataStructs.DiceSimilarity',
                                'rdkit.rdBase.AttachFileToLog', 'rdkit.Chem.FragmentOnSomeBonds',
                                'rdkit.Chem.rdmolfiles.CreateAtomIntPropertyList',
                                'rdkit.Chem.rdChemReactions.MatchOnlyAtRgroupsAdjustParams',
                                'rdkit.Chem.rdSLNParse.MolFromSLN', 'rdkit.Chem.rdmolops.FindAtomEnvironmentOfRadiusN',
                                'rdkit.Chem.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters',
                                'rdkit.Chem.CreateAtomDoublePropertyList',
                                'rdkit.Chem.rdqueries.ExplicitValenceEqualsQueryAtom',
                                'rdkit.Chem.rdMolDescriptors.GetUSRDistributions',
                                'rdkit.DataStructs.cDataStructs.KulczynskiSimilarityNeighbors_sparse',
                                'rdkit.Chem.rdDistGeom.ETKDGv3', 'rdkit.Chem.rdShapeHelpers.ShapeTverskyIndex',
                                'rdkit.Chem.AllChem.ComputeConfDimsAndOffset', 'rdkit.Chem.SetAtomValue',
                                'rdkit.Chem.AllChem.ParseMolQueryDefFile', 'rdkit.Geometry.WriteGridToFile',
                                'rdkit.DataStructs.BitVectToFPSText', 'rdkit.Chem.AllChem.WedgeBond',
                                'rdkit.Chem.SetTerminalAtomCoords', 'rdkit.DataStructs.AsymmetricSimilarity',
                                'rdkit.Chem.AllChem.IsotopeLessQueryAtom',
                                'rdkit.DataStructs.SokalSimilarityNeighbors_sparse',
                                'rdkit.Geometry.ComputeGridCentroid', 'rdkit.DataStructs.OnBitProjSimilarity',
                                'rdkit.Chem.rdMolDescriptors.CalcChiNn', 'rdkit.DataStructs.OffBitProjSimilarity',
                                'rdkit.Chem.rdqueries.HasDoublePropWithValueQueryAtom', 'rdkit.Chem.GetAtomValue',
                                'rdkit.Chem.AllChem.FormalChargeLessQueryAtom',
                                'rdkit.DataStructs.InitFromDaylightString',
                                'rdkit.Chem.rdqueries.HasBoolPropWithValueQueryAtom',
                                'rdkit.Geometry.rdGeometry.WriteGridToFile',
                                'rdkit.Chem.rdChemReactions.ReactionFromPNGFile', 'rdkit.DataStructs.CreateFromFPSText',
                                'rdkit.Chem.AllChem.TotalDegreeGreaterQueryAtom',
                                'rdkit.DataStructs.cDataStructs.McConnaugheySimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetPeriodicTable', 'rdkit.Chem.rdMolAlign.AlignMol',
                                'rdkit.Chem.EState.GetPrincipleQuantumNumber', 'rdkit.Chem.AllChem.SetAtomAlias',
                                'rdkit.DataStructs.TanimotoSimilarity', 'rdkit.Chem.AllChem.InchiToInchiKey',
                                'rdkit.Chem.AllChem.NumHeteroatomNeighborsGreaterQueryAtom',
                                'rdkit.Chem.AllChem.HasPropQueryAtom',
                                'rdkit.Chem.rdChemReactions.ReactionMetadataToPNGString',
                                'rdkit.DataStructs.FoldFingerprint',
                                'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarityNeighbors',
                                'rdkit.Chem.MolFromPDBFile', 'rdkit.Geometry.TanimotoDistance',
                                'rdkit.DataStructs.cDataStructs.BulkOnBitSimilarity', 'rdkit.Chem.LogWarningMsg',
                                'rdkit.Chem.rdqueries.MassLessQueryAtom', 'rdkit.Chem.rdmolops.FindAllPathsOfLengthN',
                                'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsLessQueryAtom',
                                'rdkit.Chem.rdchem.SetDefaultPickleProperties',
                                'rdkit.Chem.rdmolfiles.MolFragmentToCXSmiles', 'rdkit.Chem.AllChem.MolFromTPLFile',
                                'rdkit.Chem.AllChem.BondFromSmiles', 'rdkit.Chem.MolFromInchi',
                                'rdkit.Chem.rdMolAlign.GetBestRMS', 'rdkit.Chem.AllChem.ExplicitDegreeGreaterQueryAtom',
                                'rdkit.Chem.AllChem.HybridizationGreaterQueryAtom',
                                'rdkit.Chem.rdMolChemicalFeatures.BuildFeatureFactoryFromString',
                                'rdkit.DataStructs.CosineSimilarityNeighbors_sparse', 'rdkit.Chem.WrapLogs',
                                'rdkit.Chem.rdqueries.HasBoolPropWithValueQueryBond',
                                'rdkit.DataStructs.cDataStructs.CreateFromFPSText',
                                'rdkit.Chem.AllChem.SmilesMolSupplierFromText',
                                'rdkit.DataStructs.cDataStructs.BulkCosineSimilarity',
                                'rdkit.Chem.rdqueries.RingBondCountLessQueryAtom',
                                'rdkit.Geometry.rdGeometry.ProtrudeDistance',
                                'rdkit.DataStructs.cDataStructs.InitFromDaylightString',
                                'rdkit.Chem.AllChem.MolBlockToInchi', 'rdkit.Chem.SetAtomAlias',
                                'rdkit.Chem.rdmolfiles.SmilesMolSupplierFromText',
                                'rdkit.DataStructs.cDataStructs.KulczynskiSimilarity', 'rdkit.Chem.AllChem.GetUSRScore',
                                'rdkit.Chem.AllChem.CreateStructuralFingerprintForReaction',
                                'rdkit.DataStructs.ConvertToExplicit',
                                'rdkit.Chem.AllChem.MatchOnlyAtRgroupsAdjustParams',
                                'rdkit.Chem.rdmolfiles.MolToPDBFile', 'rdkit.Chem.ReplaceCore',
                                'rdkit.Chem.MolFromMolFile', 'rdkit.Chem.AllChem.HasStringPropWithValueQueryAtom',
                                'rdkit.ML.InfoTheory.rdInfoTheory.InfoEntropy',
                                'rdkit.Chem.rdmolops.DetectBondStereoChemistry',
                                'rdkit.DataStructs.cDataStructs.OffBitProjSimilarity',
                                'rdkit.Chem.AllChem.SetBondLength', 'rdkit.Chem.rdDistGeom.srETKDGv3',
                                'rdkit.DataStructs.CosineSimilarityNeighbors', 'rdkit.Chem.AllChem.GetDihedralRad',
                                'rdkit.Chem.AllChem.ComputeConfBox',
                                'rdkit.DataStructs.cDataStructs.CosineSimilarityNeighbors_sparse',
                                'rdkit.Chem.rdchem.SetAtomValue',
                                'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarityNeighbors',
                                'rdkit.Chem.rdmolfiles.MolFromSmarts', 'rdkit.Chem.AllChem.MolBlockToInchiAndAuxInfo',
                                'rdkit.Chem.rdMolTransforms.GetBondLength', 'rdkit.Chem.CreateAtomIntPropertyList',
                                'rdkit.Chem.AllChem.ExplicitDegreeEqualsQueryAtom',
                                'rdkit.DataStructs.cDataStructs.ConvertToExplicit', 'rdkit.Chem.AllChem.CalcChiNn',
                                'rdkit.Chem.rdChemReactions.GetChemDrawRxnAdjustParams',
                                'rdkit.Chem.AllChem.FindAtomEnvironmentOfRadiusN',
                                'rdkit.Chem.rdmolops.GetShortestPath', 'rdkit.Chem.AllChem.GetDihedralDeg',
                                'rdkit.Chem.BondFromSmiles', 'rdkit.Chem.AllChem.GetAngleDeg',
                                'rdkit.Chem.rdinchi.MolBlockToInchi', 'rdkit.DataStructs.McConnaugheySimilarity',
                                'rdkit.Chem.rdChemReactions.CreateDifferenceFingerprintForReaction',
                                'rdkit.Chem.rdMolTransforms.ComputePrincipalAxesAndMoments',
                                'rdkit.Chem.AllChem.TotalDegreeLessQueryAtom',
                                'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsLessQueryAtom',
                                'rdkit.Chem.AllChem.IsUnsaturatedQueryAtom',
                                'rdkit.Chem.rdqueries.RingBondCountGreaterQueryAtom',
                                'rdkit.Chem.AllChem.BuildFeatureFactory',
                                'rdkit.DataStructs.cDataStructs.RusselSimilarity',
                                'rdkit.Chem.rdShapeHelpers.ComputeConfBox',
                                'rdkit.DataStructs.BraunBlanquetSimilarityNeighbors',
                                'rdkit.Chem.rdqueries.AtomNumGreaterQueryAtom',
                                'rdkit.Chem.EState.AtomTypes.BuildPatts', 'rdkit.Chem.MetadataFromPNGString',
                                'rdkit.Chem.AllChem.MolsFromPNGFile', 'rdkit.Chem.rdmolops.SetTerminalAtomCoords',
                                'rdkit.DataStructs.ConvertToNumpyArray',
                                'rdkit.Chem.AllChem.FormalChargeEqualsQueryAtom', 'rdkit.Chem.GetAtomRLabel',
                                'rdkit.Chem.MolMetadataToPNGString',
                                'rdkit.DataStructs.cDataStructs.BulkTanimotoSimilarity',
                                'rdkit.Chem.AllChem.TotalDegreeEqualsQueryAtom',
                                'rdkit.DataStructs.TanimotoSimilarityNeighbors',
                                'rdkit.Chem.rdChemReactions.ReactionMetadataToPNGFile',
                                'rdkit.Chem.AllChem.MetadataFromPNGString',
                                'rdkit.DataStructs.cDataStructs.AsymmetricSimilarityNeighbors',
                                'rdkit.Chem.AllChem.MolsToJSON', 'rdkit.Chem.AllChem.SetDihedralRad',
                                'rdkit.DataStructs.cDataStructs.BulkDiceSimilarity',
                                'rdkit.DataStructs.BulkTverskySimilarity', 'rdkit.Geometry.TverskyIndex',
                                'rdkit.Chem.AllChem.CreateAtomIntPropertyList', 'rdkit.Chem.AllChem.MolToMolFile',
                                'rdkit.Chem.AllChem.GetHashedMorganFingerprint',
                                'rdkit.Chem.rdmolfiles.MolFromPNGString',
                                'rdkit.Chem.AllChem.TotalValenceGreaterQueryAtom', 'rdkit.Chem.AllChem.SetAtomRLabel',
                                'rdkit.DataStructs.BulkAsymmetricSimilarity',
                                'rdkit.DataStructs.cDataStructs.BulkTverskySimilarity',
                                'rdkit.Chem.AllChem.NumRadicalElectronsGreaterQueryAtom',
                                'rdkit.DataStructs.DiceSimilarity',
                                'rdkit.Chem.AllChem.HasProductTemplateSubstructMatch',
                                'rdkit.DataStructs.cDataStructs.OffBitsInCommon', 'rdkit.Chem.rdmolops.PathToSubmol',
                                'rdkit.Chem.rdmolops.FindAllSubgraphsOfLengthN',
                                'rdkit.DataStructs.cDataStructs.OnBitsInCommon',
                                'rdkit.Chem.AllChem.HasStringPropWithValueQueryBond',
                                'rdkit.DataStructs.RogotGoldbergSimilarityNeighbors',
                                'rdkit.Chem.AllChem.RemoveMappingNumbersFromReactions',
                                'rdkit.Chem.GetMolSubstanceGroupWithIdx',
                                'rdkit.Chem.AllChem.HasBoolPropWithValueQueryBond',
                                'rdkit.Chem.AllChem.AddMolSubstanceGroup', 'rdkit.ML.InfoTheory.ChiSquare',
                                'rdkit.Chem.rdqueries.AtomNumEqualsQueryAtom',
                                'rdkit.Chem.AllChem.HCountGreaterQueryAtom', 'rdkit.Chem.rdchem.SetAtomRLabel',
                                'rdkit.Chem.tossit', 'rdkit.Chem.AllChem.HasPropQueryBond',
                                'rdkit.Chem.AllChem.GetAtomMatch', 'rdkit.Chem.AllChem.HasIntPropWithValueQueryBond',
                                'rdkit.Chem.rdMolAlign.GetCrippenO3A', 'rdkit.Chem.rdmolops.AddRecursiveQuery',
                                'rdkit.Chem.rdMolTransforms.GetDihedralRad',
                                'rdkit.Chem.AllChem.MolMetadataToPNGString', 'rdkit.Chem.rdchem.GetAtomAlias',
                                'rdkit.Chem.AllChem.ReactionFromSmarts',
                                'rdkit.Chem.rdqueries.TotalValenceLessQueryAtom',
                                'rdkit.DataStructs.BitVectToBinaryText', 'rdkit.Chem.AllChem.MolFromMol2Block',
                                'rdkit.Chem.AddMetadataToPNGString', 'rdkit.rdBase.SeedRandomNumberGenerator',
                                'rdkit.Chem.rdChemReactions.RemoveMappingNumbersFromReactions',
                                'rdkit.Chem.rdForceFieldHelpers.GetUFFVdWParams',
                                'rdkit.DataStructs.RusselSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.ReactionFromPNGString', 'rdkit.Chem.AllChem.ShapeProtrudeDist',
                                'rdkit.Chem.rdMolInterchange.MolsToJSON', 'rdkit.DataStructs.SokalSimilarity',
                                'rdkit.Chem.AllChem.GenerateErGFingerprintForReducedGraph',
                                'rdkit.Chem.rdDistGeom.ETKDG', 'rdkit.Chem.AllChem.ShapeTverskyIndex',
                                'rdkit.Chem.rdChemReactions.PreprocessReaction', 'rdkit.Chem.AllChem.MolFromTPLBlock',
                                'rdkit.Chem.rdqueries.MinRingSizeLessQueryAtom',
                                'rdkit.Chem.AllChem.InNRingsGreaterQueryAtom', 'rdkit.Chem.AllChem.MolToTPLFile',
                                'rdkit.Chem.AllChem.MetadataFromPNGFile',
                                'rdkit.DataStructs.McConnaugheySimilarityNeighbors',
                                'rdkit.Geometry.ComputeDihedralAngle',
                                'rdkit.DataStructs.cDataStructs.BulkRogotGoldbergSimilarity',
                                'rdkit.DataStructs.AllProbeBitsMatch', 'rdkit.Chem.MolFromSmarts',
                                'rdkit.DataStructs.AsymmetricSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.FindUniqueSubgraphsOfLengthN', 'rdkit.Chem.AllChem.GetAtomAlias',
                                'rdkit.Chem.rdmolops.FindUniqueSubgraphsOfLengthN',
                                'rdkit.DataStructs.AsymmetricSimilarityNeighbors',
                                'rdkit.Chem.DetectBondStereoChemistry',
                                'rdkit.Chem.rdForceFieldHelpers.GetUFFTorsionParams',
                                'rdkit.Chem.rdqueries.HasIntPropWithValueQueryAtom',
                                'rdkit.Chem.rdqueries.HybridizationEqualsQueryAtom',
                                'rdkit.Chem.AllChem.MolToV3KMolFile', 'rdkit.Chem.inchi.MolBlockToInchi',
                                'rdkit.Chem.rdchem.GetPeriodicTable', 'rdkit.DataStructs.RogotGoldbergSimilarity',
                                'rdkit.Chem.rdqueries.HasStringPropWithValueQueryAtom',
                                'rdkit.Chem.AllChem.NumHeteroatomNeighborsEqualsQueryAtom',
                                'rdkit.Chem.rdMolAlign.GetCrippenO3AForProbeConfs',
                                'rdkit.Chem.rdMolAlign.GetAlignmentTransform', 'rdkit.DataStructs.BulkSokalSimilarity',
                                'rdkit.Chem.AllChem.MolFromRDKitSVG', 'rdkit.Chem.EState.BuildPatts',
                                'rdkit.Chem.AllChem.RingBondCountEqualsQueryAtom',
                                'rdkit.DataStructs.cDataStructs.DiceSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetSupplementalSmilesLabel', 'rdkit.DataStructs.TverskySimilarity',
                                'rdkit.Chem.AllChem.HasReactionSubstructMatch',
                                'rdkit.Chem.rdmolfiles.MolsFromPNGString', 'rdkit.Chem.AllChem.IsotopeGreaterQueryAtom',
                                'rdkit.DataStructs.cDataStructs.CreateFromBitString', 'rdkit.Chem.AllChem.TransformMol',
                                'rdkit.Geometry.rdGeometry.UniformGrid3D', 'rdkit.Chem.AllChem.RenumberAtoms',
                                'rdkit.Chem.ChemicalFeatures.BuildFeatureFactory',
                                'rdkit.Chem.rdChemReactions.Compute2DCoordsForReaction',
                                'rdkit.Chem.rdChemReactions.EnumerateLibraryCanSerialize',
                                'rdkit.Chem.AllChem.SortMatchesByDegreeOfCoreSubstitution',
                                'rdkit.Chem.rdqueries.IsInRingQueryAtom', 'rdkit.Chem.rdSLNParse.MolFromQuerySLN',
                                'rdkit.Chem.AllChem.GenerateDepictionMatching3DStructure',
                                'rdkit.Chem.AllChem.MinRingSizeEqualsQueryAtom',
                                'rdkit.Chem.rdqueries.TotalDegreeGreaterQueryAtom',
                                'rdkit.Chem.AllChem.ComputeCentroid',
                                'rdkit.Chem.rdmolops.FindAllSubgraphsOfLengthMToN',
                                'rdkit.DataStructs.cDataStructs.BulkBraunBlanquetSimilarity',
                                'rdkit.Chem.rdmolfiles.MolFromMol2Block',
                                'rdkit.Chem.rdMolTransforms.ComputePrincipalAxesAndMomentsFromGyrationMatrix',
                                'rdkit.Chem.rdqueries.TotalDegreeEqualsQueryAtom', 'rdkit.Chem.MolFromMolBlock',
                                'rdkit.Chem.rdmolops.FragmentOnBonds', 'rdkit.Chem.AllChem.MolFromSequence',
                                'rdkit.Chem.MolsFromPNGString', 'rdkit.Chem.FindAllSubgraphsOfLengthMToN',
                                'rdkit.DataStructs.cDataStructs.BulkAsymmetricSimilarity',
                                'rdkit.Chem.rdmolfiles.MolFragmentToSmarts',
                                'rdkit.Chem.AllChem.HasIntPropWithValueQueryAtom',
                                'rdkit.Chem.AllChem.HasChiralTagQueryAtom', 'rdkit.Chem.ParseMolQueryDefFile',
                                'rdkit.DataStructs.cDataStructs.CreateFromBinaryText',
                                'rdkit.Chem.rdqueries.TotalValenceEqualsQueryAtom',
                                'rdkit.Chem.rdChemReactions.CreateStructuralFingerprintForReaction',
                                'rdkit.Chem.inchi.InchiToInchiKey', 'rdkit.DataStructs.cDataStructs.TanimotoSimilarity',
                                'rdkit.Chem.rdqueries.MassEqualsQueryAtom', 'rdkit.Chem.rdchem.CreateStereoGroup',
                                'rdkit.Chem.rdmolops.SortMatchesByDegreeOfCoreSubstitution',
                                'rdkit.Chem.AllChem.CreateMolSubstanceGroup',
                                'rdkit.DataStructs.cDataStructs.OnBitSimilarity',
                                'rdkit.Chem.rdmolops.ParseMolQueryDefFile', 'rdkit.Chem.AllChem.MolFromMolBlock',
                                'rdkit.Chem.AllChem.MolToXYZFile', 'rdkit.Chem.AllChem.GetAlignmentTransform',
                                'rdkit.Chem.FragmentOnBonds',
                                'rdkit.Chem.rdChemReactions.HasReactantTemplateSubstructMatch',
                                'rdkit.Chem.rdqueries.AtomNumLessQueryAtom',
                                'rdkit.DataStructs.RogotGoldbergSimilarityNeighbors_sparse',
                                'rdkit.Chem.FindAllSubgraphsOfLengthN', 'rdkit.Geometry.rdGeometry.ComputeGridCentroid',
                                'rdkit.Chem.rdmolfiles.MolToMolFile',
                                'rdkit.Chem.rdqueries.NumHeteroatomNeighborsGreaterQueryAtom',
                                'rdkit.Chem.rdqueries.InNRingsLessQueryAtom', 'rdkit.Chem.AllChem.PathToSubmol',
                                'rdkit.Geometry.rdGeometry.TverskyIndex', 'rdkit.Chem.AllChem.ReactionFromRxnFile',
                                'rdkit.DataStructs.McConnaugheySimilarityNeighbors_sparse',
                                'rdkit.DataStructs.cDataStructs.ComputeL1Norm', 'rdkit.Chem.AllChem.ReplaceSubstructs',
                                'rdkit.Chem.rdmolops.ReplaceCore', 'rdkit.Chem.rdqueries.MAtomQueryAtom',
                                'rdkit.Chem.MolFromPNGString', 'rdkit.Chem.rdqueries.XHAtomQueryAtom',
                                'rdkit.Chem.AllChem.GetMolSubstanceGroupWithIdx',
                                'rdkit.Chem.rdchem.CreateMolSubstanceGroup', 'rdkit.Chem.rdmolops.WedgeMolBonds',
                                'rdkit.DataStructs.SokalSimilarityNeighbors', 'rdkit.Chem.AllChem.AtomFromSmiles',
                                'rdkit.Chem.rdinchi.InchiToMol', 'rdkit.Chem.GetSupplementalSmilesLabel',
                                'rdkit.Chem.AllChem.SupplierFromFilename', 'rdkit.ML.InfoTheory.rdInfoTheory.ChiSquare',
                                'rdkit.Chem.AllChem.ComputeCanonicalTransform',
                                'rdkit.Chem.rdChemReactions.IsReactionTemplateMoleculeAgent',
                                'rdkit.Chem.AllChem.WrapLogs', 'rdkit.Chem.rdqueries.MHAtomQueryAtom',
                                'rdkit.Chem.rdDepictor.SetPreferCoordGen',
                                'rdkit.Chem.rdqueries.IsotopeGreaterQueryAtom', 'rdkit.Chem.MolToRandomSmilesVect',
                                'rdkit.DataStructs.OnBitSimilarity', 'rdkit.Chem.WedgeBond',
                                'rdkit.Chem.AllChem.GetO3AForProbeConfs', 'rdkit.Chem.RenumberAtoms',
                                'rdkit.Chem.rdmolfiles.AtomFromSmiles',
                                'rdkit.Chem.AllChem.RingBondCountGreaterQueryAtom',
                                'rdkit.Chem.rdmolfiles.MolFromFASTA', 'rdkit.Chem.AllChem.EncodeShape',
                                'rdkit.Chem.AllChem.EnumerateLibraryCanSerialize',
                                'rdkit.DataStructs.cDataStructs.TanimotoSimilarityNeighbors',
                                'rdkit.DataStructs.BitVectToText', 'rdkit.Chem.rdmolfiles.AddMetadataToPNGFile',
                                'rdkit.Chem.rdShapeHelpers.ShapeTanimotoDist', 'rdkit.Chem.rdMolInterchange.JSONToMols',
                                'rdkit.Chem.AllChem.BondFromSmarts',
                                'rdkit.Chem.rdqueries.HasIntPropWithValueQueryBond',
                                'rdkit.Chem.AllChem.QHAtomQueryAtom', 'rdkit.Chem.ReplaceSubstructs',
                                'rdkit.Chem.AllChem.Compute2DCoordsMimicDistmat',
                                'rdkit.Chem.AllChem.CreateAtomStringPropertyList',
                                'rdkit.Chem.AllChem.HCountEqualsQueryAtom',
                                'rdkit.Chem.EState.EState.GetPrincipleQuantumNumber',
                                'rdkit.Chem.rdmolfiles.MolFromTPLFile', 'rdkit.Chem.rdqueries.MassGreaterQueryAtom',
                                'rdkit.Chem.SetDefaultPickleProperties',
                                'rdkit.Chem.rdCoordGen.SetDefaultTemplateFileDir',
                                'rdkit.DataStructs.cDataStructs.BulkSokalSimilarity',
                                'rdkit.Chem.CanonicalRankAtomsInFragment', 'rdkit.RDLogger.LogMessage',
                                'rdkit.Chem.rdMolTransforms.SetDihedralRad', 'rdkit.Chem.rdShapeHelpers.EncodeShape',
                                'rdkit.Chem.AddMetadataToPNGFile', 'rdkit.Chem.rdmolfiles.MolMetadataToPNGString',
                                'rdkit.Chem.AllChem.GetDefaultPickleProperties',
                                'rdkit.Chem.AllChem.SetDefaultPickleProperties', 'rdkit.Chem.SetAtomRLabel',
                                'rdkit.DataStructs.cDataStructs.McConnaugheySimilarityNeighbors',
                                'rdkit.Chem.rdDistGeom.KDG', 'rdkit.Chem.AllChem.HasDoublePropWithValueQueryBond',
                                'rdkit.DataStructs.RusselSimilarityNeighbors', 'rdkit.Chem.AddRecursiveQuery',
                                'rdkit.Chem.AllChem.HasReactionAtomMapping', 'rdkit.Chem.AllChem.tossit',
                                'rdkit.DataStructs.cDataStructs.SokalSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetUFFInversionParams',
                                'rdkit.DataStructs.cDataStructs.ConvertToNumpyArray',
                                'rdkit.Geometry.ComputeSignedDihedralAngle', 'rdkit.Chem.rdmolfiles.MolsFromPNGFile',
                                'rdkit.Chem.AllChem.FragmentOnBonds',
                                'rdkit.DataStructs.cDataStructs.SokalSimilarityNeighbors',
                                'rdkit.Chem.rdqueries.IsotopeEqualsQueryAtom',
                                'rdkit.Chem.rdqueries.IsotopeLessQueryAtom', 'rdkit.Chem.AllChem.MolFromMolFile',
                                'rdkit.Chem.rdForceFieldHelpers.OptimizeMolecule', 'rdkit.Chem.rdchem.GetAtomValue',
                                'rdkit.Chem.AllChem.HasAgentTemplateSubstructMatch',
                                'rdkit.Chem.rdqueries.HasPropQueryBond',
                                'rdkit.Chem.rdChemReactions.ReactionToRxnBlock',
                                'rdkit.Chem.AllChem.ExplicitValenceGreaterQueryAtom',
                                'rdkit.Chem.rdqueries.QAtomQueryAtom', 'rdkit.Chem.rdChemReactions.ReactionToSmiles',
                                'rdkit.Chem.MolFragmentToSmiles', 'rdkit.Chem.inchi.MolFromInchi',
                                'rdkit.Chem.AllChem.CalcRMS', 'rdkit.Chem.rdForceFieldHelpers.OptimizeMoleculeConfs',
                                'rdkit.Chem.AllChem.MolFromPDBFile', 'rdkit.Chem.MolFromTPLBlock',
                                'rdkit.Chem.AllChem.AssignBondOrdersFromTemplate',
                                'rdkit.DataStructs.BulkTanimotoSimilarity',
                                'rdkit.Chem.AllChem.ReactionMetadataToPNGString',
                                'rdkit.Chem.rdMolDescriptors.GetUSRDistributionsFromPoints',
                                'rdkit.Chem.AllChem.InNRingsEqualsQueryAtom',
                                'rdkit.Chem.AllChem.CreateDifferenceFingerprintForReaction',
                                'rdkit.Chem.BondFromSmarts', 'rdkit.Chem.CreateStereoGroup',
                                'rdkit.Chem.rdmolfiles.MolFromPDBBlock', 'rdkit.Chem.rdMolAlign.GetO3AForProbeConfs',
                                'rdkit.Chem.AllChem.MolFromPNGFile', 'rdkit.Chem.MolFromPNGFile',
                                'rdkit.DataStructs.AllBitSimilarity', 'rdkit.DataStructs.cDataStructs.FoldFingerprint',
                                'rdkit.Chem.AllChem.AtomNumEqualsQueryAtom', 'rdkit.Chem.rdmolops.ReplaceSidechains',
                                'rdkit.Chem.ChemicalFeatures.MCFF_GetFeaturesForMol',
                                'rdkit.Chem.AllChem.ComputePrincipalAxesAndMoments',
                                'rdkit.DataStructs.FingerprintSimilarity', 'rdkit.DataStructs.OnBitsInCommon',
                                'rdkit.Chem.rdDepictor.GenerateDepictionMatching2DStructure',
                                'rdkit.Chem.MolFromMol2File', 'rdkit.Chem.rdqueries.FormalChargeEqualsQueryAtom',
                                'rdkit.Chem.AllChem.AtomFromSmarts', 'rdkit.Chem.AllChem.MassGreaterQueryAtom',
                                'rdkit.Chem.MolToMolFile', 'rdkit.Chem.rdqueries.ExplicitDegreeGreaterQueryAtom',
                                'rdkit.Chem.AllChem.MolFromFASTA', 'rdkit.Chem.SmilesMolSupplierFromText',
                                'rdkit.Chem.rdMolTransforms.CanonicalizeConformer', 'rdkit.Chem.InchiToInchiKey',
                                'rdkit.DataStructs.cDataStructs.RusselSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetUFFBondStretchParams',
                                'rdkit.Chem.rdqueries.InNRingsEqualsQueryAtom',
                                'rdkit.Chem.rdqueries.MinRingSizeEqualsQueryAtom', 'rdkit.Chem.AllChem.ETKDG',
                                'rdkit.Chem.AllChem.RingBondCountLessQueryAtom',
                                'rdkit.Chem.AllChem.FindAllSubgraphsOfLengthMToN', 'rdkit.ML.InfoTheory.InfoEntropy',
                                'rdkit.DataStructs.CreateFromBinaryText',
                                'rdkit.Chem.rdForceFieldHelpers.GetUFFBondStretchParams',
                                'rdkit.Chem.AllChem.MassEqualsQueryAtom', 'rdkit.Chem.MolFromMol2Block',
                                'rdkit.Chem.rdChemReactions.ReactionToSmarts',
                                'rdkit.Chem.AllChem.HybridizationEqualsQueryAtom',
                                'rdkit.Chem.CreateAtomStringPropertyList',
                                'rdkit.Chem.rdqueries.ExplicitDegreeEqualsQueryAtom',
                                'rdkit.Chem.AllChem.GetMorganFingerprint',
                                'rdkit.DataStructs.DiceSimilarityNeighbors_sparse',
                                'rdkit.Chem.rdmolops.ReplaceSubstructs',
                                'rdkit.DataStructs.cDataStructs.BulkMcConnaugheySimilarity',
                                'rdkit.Chem.AllChem.TransformConformer', 'rdkit.Chem.inchi.MolBlockToInchiAndAuxInfo',
                                'rdkit.DataStructs.cDataStructs.BulkAllBitSimilarity',
                                'rdkit.DataStructs.cDataStructs.AllProbeBitsMatch',
                                'rdkit.Chem.Descriptors.setupAUTOCorrDescriptors',
                                'rdkit.Chem.rdqueries.RingBondCountEqualsQueryAtom',
                                'rdkit.Chem.rdmolfiles.MolToV3KMolFile', 'rdkit.Chem.CreateAtomBoolPropertyList',
                                'rdkit.DataStructs.cDataStructs.OnBitProjSimilarity',
                                'rdkit.Chem.rdmolfiles.CreateAtomStringPropertyList',
                                'rdkit.Chem.AllChem.ReactionFromRxnBlock', 'rdkit.Chem.AllChem.WedgeMolBonds',
                                'rdkit.DataStructs.cDataStructs.BulkRusselSimilarity',
                                'rdkit.DataStructs.FoldToTargetDensity', 'rdkit.Chem.MolFromSmiles',
                                'rdkit.Chem.LogErrorMsg', 'rdkit.Chem.AllChem.LogErrorMsg',
                                'rdkit.Chem.rdchem.GetAtomRLabel',
                                'rdkit.Chem.AllChem.CalcNumUnspecifiedAtomStereoCenters', 'rdkit.Chem.rdDistGeom.ETDG',
                                'rdkit.Chem.AtomFromSmarts', 'rdkit.Chem.rdShapeHelpers.ShapeProtrudeDist',
                                'rdkit.Chem.FindAtomEnvironmentOfRadiusN', 'rdkit.Chem.AllChem.ETDG',
                                'rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure',
                                'rdkit.Chem.AllChem.FindAllSubgraphsOfLengthN',
                                'rdkit.Chem.AllChem.ComputePrincipalAxesAndMomentsFromGyrationMatrix',
                                'rdkit.Chem.AllChem.XAtomQueryAtom',
                                'rdkit.Chem.rdShapeHelpers.ComputeConfDimsAndOffset',
                                'rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment',
                                'rdkit.DataStructs.cDataStructs.TanimotoSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetAtomFeatures', 'rdkit.Chem.AllChem.AAtomQueryAtom',
                                'rdkit.Chem.rdMolTransforms.GetDihedralDeg',
                                'rdkit.Chem.AllChem.AtomNumGreaterQueryAtom',
                                'rdkit.Chem.AllChem.FormalChargeGreaterQueryAtom', 'rdkit.RDLogger.DisableLog',
                                'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.GetDefaultAdjustParams', 'rdkit.Chem.WedgeMolBonds',
                                'rdkit.Chem.rdqueries.HasBitVectPropWithValueQueryAtom',
                                'rdkit.Chem.AllChem.ReactionToMolecule', 'rdkit.Chem.AllChem.ReactionMetadataToPNGFile',
                                'rdkit.Chem.AllChem.ExplicitValenceEqualsQueryAtom',
                                'rdkit.Chem.AllChem.XHAtomQueryAtom', 'rdkit.Chem.AllChem.SanitizeRxn',
                                'rdkit.Chem.AllChem.AHAtomQueryAtom', 'rdkit.Chem.AllChem.MolToRandomSmilesVect',
                                'rdkit.Chem.rdqueries.IsAromaticQueryAtom', 'rdkit.Chem.AllChem.CanonicalizeConformer',
                                'rdkit.DataStructs.cDataStructs.NumBitsInCommon', 'rdkit.Chem.rdmolfiles.MolToXYZFile',
                                'rdkit.Chem.rdMolChemicalFeatures.BuildFeatureFactory', 'rdkit.Chem.MolFromHELM',
                                'rdkit.Chem.rdqueries.FormalChargeLessQueryAtom',
                                'rdkit.Chem.rdqueries.HybridizationGreaterQueryAtom',
                                'rdkit.DataStructs.BraunBlanquetSimilarityNeighbors_sparse',
                                'rdkit.Chem.AllChem.ETKDGv2', 'rdkit.Chem.GetShortestPath',
                                'rdkit.Chem.rdqueries.FormalChargeGreaterQueryAtom',
                                'rdkit.DataStructs.BulkCosineSimilarity', 'rdkit.Chem.AllChem.MassLessQueryAtom',
                                'rdkit.Chem.AllChem.CreateAtomDoublePropertyList',
                                'rdkit.DataStructs.CreateFromBitString',
                                'rdkit.Chem.AllChem.HasBitVectPropWithValueQueryAtom', 'rdkit.Chem.AllChem.GetAngleRad',
                                'rdkit.Chem.rdMolTransforms.SetAngleDeg', 'rdkit.Chem.SupplierFromFilename',
                                'rdkit.Chem.rdqueries.HCountEqualsQueryAtom',
                                'rdkit.DataStructs.cDataStructs.AsymmetricSimilarityNeighbors_sparse',
                                'rdkit.Chem.GetPeriodicTable', 'rdkit.DataStructs.BulkRogotGoldbergSimilarity',
                                'rdkit.Chem.GetMostSubstitutedCoreMatch', 'rdkit.rdBase.DisableLog',
                                'rdkit.Chem.rdMolAlign.CalcRMS', 'rdkit.Chem.AllChem.ReactionToSmiles',
                                'rdkit.Chem.AllChem.MissingChiralTagQueryAtom', 'rdkit.Chem.AllChem.SetAngleDeg',
                                'rdkit.Chem.rdMolDescriptors.CalcNumAtomStereoCenters',
                                'rdkit.Chem.rdqueries.ExplicitValenceLessQueryAtom',
                                'rdkit.Chem.rdMolTransforms.GetAngleDeg', 'rdkit.Chem.AllChem.GetUFFTorsionParams',
                                'rdkit.Chem.MolFragmentToSmarts', 'rdkit.Chem.AllChem.ComputeUnionBox',
                                'rdkit.Chem.rdmolops.DeleteSubstructs', 'rdkit.Chem.AllChem.MHAtomQueryAtom',
                                'rdkit.Chem.rdqueries.XAtomQueryAtom', 'rdkit.DataStructs.CosineSimilarity',
                                'rdkit.Chem.MolBlockToInchiAndAuxInfo', 'rdkit.Chem.rdmolfiles.MetadataFromPNGFile',
                                'rdkit.Chem.rdchem.tossit', 'rdkit.Chem.MetadataFromPNGFile',
                                'rdkit.Chem.rdMolDescriptors.GetAtomPairCode', 'rdkit.Chem.rdchem.AddMolSubstanceGroup',
                                'rdkit.Chem.rdmolfiles.AtomFromSmarts',
                                'rdkit.DataStructs.TanimotoSimilarityNeighbors_sparse',
                                'rdkit.Chem.rdmolfiles.MolFromTPLBlock',
                                'rdkit.Geometry.rdGeometry.FindGridTerminalPoints', 'rdkit.DataStructs.OffBitsInCommon',
                                'rdkit.RDLogger.EnableLog', 'rdkit.DataStructs.cDataStructs.TverskySimilarity',
                                'rdkit.Chem.AllChem.GetAtomPairAtomCode', 'rdkit.DataStructs.BulkAllBitSimilarity',
                                'rdkit.Chem.rdMolTransforms.SetBondLength',
                                'rdkit.DataStructs.KulczynskiSimilarityNeighbors_sparse', 'rdkit.Chem.AtomFromSmiles',
                                'rdkit.Chem.ChemicalFeatures.BuildFeatureFactoryFromString',
                                'rdkit.Chem.rdForceFieldHelpers.GetUFFAngleBendParams',
                                'rdkit.Chem.AllChem.ConstrainedEmbed',
                                'rdkit.Chem.rdChemReactions.ReactionFromRxnBlock', 'rdkit.Chem.AllChem.CalcChiNv',
                                'rdkit.Chem.rdmolfiles.BondFromSmiles', 'rdkit.Chem.AllChem.GetChemDrawRxnAdjustParams',
                                'rdkit.Chem.Graphs.CharacteristicPolynomial', 'rdkit.DataStructs.ComputeL1Norm',
                                'rdkit.Chem.AllChem.DetectBondStereoChemistry', 'rdkit.Chem.AllChem.GetShortestPath',
                                'rdkit.Chem.AllChem.MolsFromPNGString', 'rdkit.Chem.AllChem.InNRingsLessQueryAtom',
                                'rdkit.rdBase.LogMessage', 'rdkit.Chem.QED.ads',
                                'rdkit.Chem.rdmolfiles.MolFromSequence', 'rdkit.Chem.MolFromRDKitSVG',
                                'rdkit.Chem.rdmolops.WedgeBond', 'rdkit.Chem.AllChem.UpdateProductsStereochemistry',
                                'rdkit.Chem.AllChem.OptimizeMoleculeConfs', 'rdkit.rdBase.LogStatus',
                                'rdkit.Chem.MolToXYZFile', 'rdkit.Chem.AllChem.MolFromSmarts',
                                'rdkit.Chem.rdqueries.HasDoublePropWithValueQueryBond',
                                'rdkit.Chem.AllChem.MolFragmentToCXSmiles', 'rdkit.rdBase.EnableLog',
                                'rdkit.Chem.rdqueries.HCountLessQueryAtom', 'rdkit.Chem.AllChem.AlignMol',
                                'rdkit.Chem.AllChem.GetUSRDistributions',
                                'rdkit.Chem.rdMolDescriptors.MakePropertyRangeQuery',
                                'rdkit.Chem.rdmolfiles.MolFromRDKitSVG', 'rdkit.Chem.rdchem.LogWarningMsg',
                                'rdkit.DataStructs.NumBitsInCommon', 'rdkit.DataStructs.BulkMcConnaugheySimilarity',
                                'rdkit.Chem.MolFromFASTA', 'rdkit.Chem.AllChem.MakePropertyRangeQuery',
                                'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsEqualsQueryAtom',
                                'rdkit.Chem.rdChemReactions.SanitizeRxn', 'rdkit.Chem.rdmolfiles.MolToRandomSmilesVect',
                                'rdkit.Chem.rdqueries.InNRingsGreaterQueryAtom',
                                'rdkit.Chem.rdChemReactions.ReactionFromRxnFile',
                                'rdkit.DataStructs.cDataStructs.SokalSimilarity', 'rdkit.Chem.DeleteSubstructs',
                                'rdkit.Chem.AllChem.LogWarningMsg',
                                'rdkit.Chem.rdqueries.NumHeteroatomNeighborsLessQueryAtom',
                                'rdkit.Chem.AllChem.TotalValenceLessQueryAtom', 'rdkit.Chem.rdmolfiles.MolFromHELM',
                                'rdkit.Chem.MolFromTPLFile', 'rdkit.Chem.AllChem.ExplicitDegreeLessQueryAtom',
                                'rdkit.Chem.AllChem.SetDihedralDeg', 'rdkit.Chem.rdMolTransforms.SetAngleRad',
                                'rdkit.Chem.rdChemReactions.ReactionToMolecule',
                                'rdkit.Chem.AllChem.FragmentOnSomeBonds', 'rdkit.Chem.AllChem.FindAllPathsOfLengthN',
                                'rdkit.Chem.AllChem.MolAddRecursiveQueries',
                                'rdkit.Chem.AllChem.MCFF_GetFeaturesForMol', 'rdkit.Chem.rdmolfiles.MolFromSmiles',
                                'rdkit.Chem.rdqueries.ExplicitDegreeLessQueryAtom', 'rdkit.Chem.AllChem.srETKDGv3',
                                'rdkit.Chem.AllChem.MolFragmentToSmarts', 'rdkit.Chem.AllChem.SetTerminalAtomCoords',
                                'rdkit.Chem.rdqueries.TotalValenceGreaterQueryAtom', 'rdkit.Chem.AllChem.MolFromHELM',
                                'rdkit.Chem.QuickSmartsMatch', 'rdkit.Chem.AllChem.EnumerateLibraryFromReaction',
                                'rdkit.Chem.rdMolDescriptors.GetUSRScore', 'rdkit.Chem.rdMolTransforms.GetAngleRad',
                                'rdkit.Chem.AllChem.MMFFGetMoleculeForceField',
                                'rdkit.Chem.rdMolTransforms.TransformConformer', 'rdkit.Chem.MolFromPDBBlock',
                                'rdkit.Chem.rdchem.GetDefaultPickleProperties',
                                'rdkit.DataStructs.cDataStructs.McConnaugheySimilarity', 'rdkit.Chem.QED.namedtuple',
                                'rdkit.Chem.PathToSubmol', 'rdkit.Geometry.rdGeometry.ComputeDihedralAngle',
                                'rdkit.ML.InfoTheory.rdInfoTheory.InfoGain', 'rdkit.Chem.AllChem.JSONToMols',
                                'rdkit.Chem.AllChem.AddMetadataToPNGString',
                                'rdkit.Chem.rdDepictor.GenerateDepictionMatching3DStructure',
                                'rdkit.Chem.rdMolDescriptors.GetMorganFingerprint', 'rdkit.Chem.AllChem.GetAtomValue',
                                'rdkit.Chem.rdmolops.MolAddRecursiveQueries', 'rdkit.Chem.MolFromSequence',
                                'rdkit.Chem.rdChemReactions.HasAgentTemplateSubstructMatch',
                                'rdkit.DataStructs.BulkOnBitSimilarity', 'rdkit.Chem.AllChem.IsInRingQueryAtom',
                                'rdkit.Chem.AllChem.GetUSRFromDistributions', 'rdkit.Chem.rdmolfiles.MolFromPDBFile',
                                'rdkit.Chem.rdChemReactions.UpdateProductsStereochemistry',
                                'rdkit.Chem.AllChem.CanonicalRankAtomsInFragment',
                                'rdkit.Chem.AllChem.IsAromaticQueryAtom', 'rdkit.Chem.SetSupplementalSmilesLabel',
                                'rdkit.Chem.rdChemReactions.HasReactionSubstructMatch',
                                'rdkit.Chem.rdChemReactions.HasReactionAtomMapping',
                                'rdkit.Chem.AllChem.HasBoolPropWithValueQueryAtom',
                                'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsGreaterQueryAtom',
                                'rdkit.DataStructs.cDataStructs.AsymmetricSimilarity',
                                'rdkit.Chem.AllChem.AtomNumLessQueryAtom',
                                'rdkit.Chem.rdqueries.ExplicitValenceGreaterQueryAtom',
                                'rdkit.Chem.rdmolfiles.MolFragmentToSmiles', 'rdkit.Chem.rdchem.SetAtomAlias',
                                'rdkit.Chem.AllChem.GetAtomPairCode', 'rdkit.Chem.AllChem.MinRingSizeLessQueryAtom',
                                'rdkit.DataStructs.cDataStructs.CosineSimilarity', 'rdkit.Chem.AllChem.GetAtomRLabel',
                                'rdkit.Chem.AllChem.GetUFFVdWParams', 'rdkit.ML.InfoTheory.entropy.PyInfoGain',
                                'rdkit.Chem.rdmolfiles.BondFromSmarts', 'rdkit.DataStructs.BulkRusselSimilarity',
                                'rdkit.Chem.rdMolTransforms.ComputeCentroid', 'rdkit.Chem.AllChem.GetBondLength',
                                'rdkit.Chem.AddMolSubstanceGroup', 'rdkit.Chem.AllChem.MolFromSLN',
                                'rdkit.Chem.AllChem.HasReactantTemplateSubstructMatch',
                                'rdkit.Chem.AllChem.OptimizeMolecule', 'rdkit.Chem.AllChem.GetConformerRMS',
                                'rdkit.Chem.SortMatchesByDegreeOfCoreSubstitution',
                                'rdkit.Chem.AllChem.CalcNumAtomStereoCenters', 'rdkit.Chem.AllChem.AddRecursiveQuery',
                                'rdkit.Chem.AllChem.CanonSmiles', 'rdkit.Chem.AllChem.NumRadicalElectronsLessQueryAtom',
                                'rdkit.Chem.rdmolfiles.CreateAtomDoublePropertyList',
                                'rdkit.Chem.rdmolfiles.MolFromMol2File',
                                'rdkit.Chem.rdMolDescriptors.GetUSRFromDistributions',
                                'rdkit.Chem.rdqueries.AHAtomQueryAtom',
                                'rdkit.DataStructs.KulczynskiSimilarityNeighbors',
                                'rdkit.Chem.rdqueries.HasChiralTagQueryAtom',
                                'rdkit.Chem.AllChem.CreateAtomBoolPropertyList',
                                'rdkit.Chem.rdForceFieldHelpers.GetUFFInversionParams',
                                'rdkit.Chem.rdmolops.RenumberAtoms',
                                'rdkit.Chem.rdMolTransforms.ComputeCanonicalTransform',
                                'rdkit.DataStructs.BraunBlanquetSimilarity', 'rdkit.Chem.rdchem.WrapLogs',
                                'rdkit.Chem.GetAtomAlias', 'rdkit.Chem.rdmolfiles.MetadataFromPNGString',
                                'rdkit.ML.InfoTheory.entropy.PyInfoEntropy',
                                'rdkit.Chem.AllChem.GetCrippenO3AForProbeConfs',
                                'rdkit.Chem.rdqueries.HybridizationLessQueryAtom', 'rdkit.Chem.AllChem.namedtuple',
                                'rdkit.Chem.AllChem.QAtomQueryAtom', 'rdkit.Chem.MolAddRecursiveQueries',
                                'rdkit.Chem.ChemicalFeatures.GetAtomMatch', 'rdkit.Chem.MolsToJSON',
                                'rdkit.Chem.CombineMols', 'rdkit.DataStructs.cDataStructs.BitVectToFPSText',
                                'rdkit.Chem.AllChem.GetCrippenO3A', 'rdkit.Chem.AllChem.MolFragmentToSmiles',
                                'rdkit.Chem.CanonSmiles', 'rdkit.Chem.AllChem.ETKDGv3',
                                'rdkit.Chem.AllChem.GetUFFAngleBendParams', 'rdkit.Chem.rdMolDescriptors.CalcChiNv',
                                'rdkit.Chem.rdmolfiles.AddMetadataToPNGString', 'rdkit.Geometry.UniformGrid3D',
                                'rdkit.Chem.AllChem.GetMostSubstitutedCoreMatch',
                                'rdkit.DataStructs.cDataStructs.BitVectToBinaryText',
                                'rdkit.Chem.rdqueries.HCountGreaterQueryAtom',
                                'rdkit.Geometry.rdGeometry.ComputeSignedDihedralAngle',
                                'rdkit.DataStructs.KulczynskiSimilarity',
                                'rdkit.Chem.AllChem.MinRingSizeGreaterQueryAtom', 'rdkit.Chem.rdDistGeom.ETKDGv2',
                                'rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField',
                                'rdkit.Chem.AllChem.ReplaceSidechains', 'rdkit.Chem.AllChem.ShapeTanimotoDist',
                                'rdkit.Chem.AllChem.HCountLessQueryAtom', 'rdkit.ML.InfoTheory.entropy.InfoGain',
                                'rdkit.Chem.AllChem.MolFromMol2File', 'rdkit.DataStructs.cDataStructs.BitVectToText',
                                'rdkit.Chem.rdmolfiles.CreateAtomBoolPropertyList',
                                'rdkit.Chem.rdmolfiles.MolMetadataToPNGFile', 'rdkit.Chem.AllChem.MolFromPDBBlock',
                                'rdkit.Chem.AllChem.SetSupplementalSmilesLabel',
                                'rdkit.DataStructs.DiceSimilarityNeighbors',
                                'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarity',
                                'rdkit.Chem.AllChem.HasDoublePropWithValueQueryAtom', 'rdkit.Chem.AllChem.GetO3A',
                                'rdkit.Chem.AllChem.QuickSmartsMatch', 'rdkit.Chem.MolMetadataToPNGFile',
                                'rdkit.Chem.rdqueries.MinRingSizeGreaterQueryAtom', 'rdkit.Chem.AllChem.MolFromSmiles',
                                'rdkit.Chem.rdqueries.HasStringPropWithValueQueryBond',
                                'rdkit.DataStructs.cDataStructs.DiceSimilarityNeighbors',
                                'rdkit.RDLogger.AttachFileToLog', 'rdkit.Chem.AllChem.ReplaceCore',
                                'rdkit.Chem.AllChem.AddMetadataToPNGFile',
                                'rdkit.Chem.rdMolDescriptors.GetHashedMorganFingerprint',
                                'rdkit.Chem.rdmolops.CombineMols', 'rdkit.Chem.rdmolfiles.MolFromMolBlock',
                                'rdkit.Chem.rdqueries.TotalDegreeLessQueryAtom', 'rdkit.Chem.AllChem.MAtomQueryAtom',
                                'rdkit.ML.InfoTheory.InfoGain', 'rdkit.Chem.rdMolAlign.GetO3A',
                                'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarity',
                                'rdkit.Chem.rdqueries.NumHeteroatomNeighborsEqualsQueryAtom',
                                'rdkit.Chem.rdqueries.MissingChiralTagQueryAtom',
                                'rdkit.Chem.AllChem.ReactionFromPNGFile', 'rdkit.Chem.AllChem.ReactionToSmarts',
                                'rdkit.Chem.AllChem.SetAtomValue', 'rdkit.Chem.AllChem.CombineMols',
                                'rdkit.Chem.rdMolDescriptors.GetAtomPairAtomCode', 'rdkit.Chem.rdinchi.InchiToInchiKey',
                                'rdkit.Chem.AllChem.NumHeteroatomNeighborsLessQueryAtom',
                                'rdkit.Chem.rdmolops.GetMostSubstitutedCoreMatch',
                                'rdkit.Chem.GetDefaultPickleProperties', 'rdkit.Chem.AllChem.IsAliphaticQueryAtom',
                                'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsEqualsQueryAtom',
                                'rdkit.Chem.rdMolChemicalFeatures.GetAtomMatch', 'rdkit.Chem.AllChem.MolFromQuerySLN',
                                'rdkit.Chem.FindUniqueSubgraphsOfLengthN', 'rdkit.Chem.MolFragmentToCXSmiles',
                                'rdkit.Chem.rdqueries.HasPropQueryAtom', 'rdkit.Chem.rdmolops.FragmentOnSomeBonds',
                                'rdkit.DataStructs.cDataStructs.RusselSimilarityNeighbors',
                                'rdkit.Chem.AllChem.MolToPDBFile', 'rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect',
                                'rdkit.Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom',
                                'rdkit.Chem.rdchem.LogErrorMsg', 'rdkit.Chem.MolsFromPNGFile',
                                'rdkit.DataStructs.BulkDiceSimilarity',
                                'rdkit.Chem.rdDepictor.Compute2DCoordsMimicDistmat',
                                'rdkit.Chem.AllChem.TotalValenceEqualsQueryAtom',
                                'rdkit.Chem.AllChem.SetPreferCoordGen', 'rdkit.Chem.rdchem.GetMolSubstanceGroupWithIdx',
                                'rdkit.Chem.AllChem.CreateStereoGroup',
                                'rdkit.Chem.rdqueries.NumRadicalElectronsLessQueryAtom',
                                'rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect',
                                'rdkit.Chem.MolToV3KMolFile', 'rdkit.Chem.AllChem.SetAngleRad',
                                'rdkit.Chem.AllChem.KDG', 'rdkit.Chem.MolBlockToInchi',
                                'rdkit.Chem.rdchem.SetSupplementalSmilesLabel', 'rdkit.Chem.AllChem.PreprocessReaction',
                                'rdkit.DataStructs.cDataStructs.CosineSimilarityNeighbors',
                                'rdkit.Chem.rdqueries.IsUnsaturatedQueryAtom',
                                'rdkit.Chem.rdReducedGraphs.GenerateErGFingerprintForReducedGraph',
                                'rdkit.Chem.rdmolfiles.MolFromPNGFile', 'rdkit.DataStructs.BulkKulczynskiSimilarity',
                                'rdkit.Chem.AllChem.DeleteSubstructs', 'rdkit.Chem.rdChemReactions.ReactionFromSmarts',
                                'rdkit.DataStructs.cDataStructs.AllBitSimilarity',
                                'rdkit.Chem.AllChem.Compute2DCoordsForReaction',
                                'rdkit.Chem.AllChem.MolMetadataToPNGFile', 'rdkit.Chem.AllChem.MolFromInchi',
                                'rdkit.Chem.AllChem.HybridizationLessQueryAtom', 'rdkit.Chem.MolToPDBFile',
                                'rdkit.Chem.rdqueries.AAtomQueryAtom',
                                'rdkit.Chem.AllChem.NumRadicalElectronsEqualsQueryAtom',
                                'rdkit.Chem.FindAllPathsOfLengthN', 'rdkit.Chem.AllChem.ReactionToRxnBlock',
                                'rdkit.Chem.rdmolfiles.MolToTPLFile', 'rdkit.ML.InfoTheory.entropy.InfoEntropy',
                                'rdkit.Chem.rdqueries.IsAliphaticQueryAtom',
                                'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsGreaterQueryAtom',
                                'rdkit.Chem.AllChem.IsReactionTemplateMoleculeAgent',
                                'rdkit.Chem.AllChem.BuildFeatureFactoryFromString',
                                'rdkit.DataStructs.cDataStructs.BulkKulczynskiSimilarity',
                                ]
_MOL_FUNCTIONS_MANUALLY_REM = ['rdkit.Chem.EnumerateStereoisomers.EnumerateStereoisomers',
                               'rdkit.Chem.AllChem.GetSymmSSSR',
                               'rdkit.Chem.rdChemReactions.ReduceProductToSideChains',
                               'rdkit.Chem.RemoveAllHs',
                               'rdkit.Chem.RemoveAllHs',
                               'rdkit.Chem.MurckoDecompose',
                               'rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs',
                               'rdkit.Chem.EState.TypeAtoms',
                               'rdkit.Chem.AllChem.AddHs',
                               'rdkit.Chem.MergeQueryHs',
                               'rdkit.Chem.AllChem.ComputeMolShape',
                               'rdkit.Chem.rdMolDescriptors.CalcCoulombMat ',
                               'rdkit.Chem.rdDistGeom.GetMoleculeBoundsMatrix',
                               'rdkit.Chem.rdmolfiles.CanonicalRankAtoms',
                               'rdkit.Chem.AllChem.UFFGetMoleculeForceField',
                               'rdkit.Chem.AllChem.EmbedMultipleConfs',
                               'rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties',
                               'rdkit.Chem.rdmolops.FindPotentialStereo',
                               'rdkit.Chem.AllChem.UFFOptimizeMoleculeConfs',
                               'rdkit.Chem.rdinchi.MolToInchi',
                               'rdkit.Chem.GetMolFrags',
                               'rdkit.Chem.AllChem.MMFFOptimizeMolecule',
                               "rdkit.Chem.AllChem.UFFOptimizeMolecule",
                               "rdkit.Chem.AllChem.MMFFHasAllMoleculeParams",
                               "rdkit.Chem.AllChem.UFFHasAllMoleculeParams"
                               ]
MAX_LENGTH = 4096
IGNORED_MOL_FUNCTIONS = ["Debug",
                         "ClearComputedProps",
                         "ComputeGasteigerCharges",
                         "RemoveAllConformers",
                         "UpdatePropertyCache",
                         'Compute2DCoords',
                         'GetNumConformers',
                         'GetPropNames',
                         'GetPropsAsDict',
                         'NeedsUpdatePropertyCache',
                         'AddConformer',
                         'ClearProp',
                         'GetAromaticAtoms',
                         'GetAtomWithIdx',
                         'GetAtoms',
                         'GetAtomsMatchingQuery',
                         'GetBondBetweenAtoms',
                         'GetBondWithIdx',
                         'GetBonds',
                         'GetBoolProp',
                         'GetConformer',
                         'GetConformers',
                         'GetDoubleProp',
                         'GetIntProp',
                         'GetProp',
                         'GetRingInfo',
                         'GetStereoGroups',
                         'GetSubstructMatch',
                         'GetSubstructMatches',
                         'GetUnsignedProp',
                         'HasProp',
                         'HasSubstructMatch',
                         'RemoveConformer',
                         'SetBoolProp',
                         'SetDoubleProp',
                         'SetIntProp',
                         'SetProp',
                         'SetUnsignedProp',
                         'ToBinary',
                         ] + _MOL_FUNCTIONS_PRERUN_FAILED + _MOL_FUNCTIONS_MANUALLY_REM

_ATOM_FUNCTIONS_PRERUN_FAILED = ['rdkit.Chem.MolToTPLFile', 'rdkit.Chem.AllChem.GetBestRMS',
                                 'rdkit.Chem.AllChem.GetUSRDistributionsFromPoints',
                                 'rdkit.Geometry.rdGeometry.TanimotoDistance',
                                 'rdkit.Chem.AllChem.ExplicitValenceLessQueryAtom',
                                 'rdkit.Chem.rdmolops.MurckoDecompose', 'rdkit.Chem.MolToJSON',
                                 'rdkit.Chem.inchi.MolToInchi', 'rdkit.Chem.AllChem.CalcHallKierAlpha',
                                 'rdkit.Chem.AllChem.MolToSmiles', 'rdkit.Chem.MolSurf.SlogP_VSA1',
                                 'rdkit.Chem.Descriptors.Chi0n', 'rdkit.Chem.Lipinski.NumSaturatedCarbocycles',
                                 'rdkit.Chem.QED.weights_max', 'rdkit.Chem.Descriptors.SlogP_VSA10',
                                 'rdkit.Chem.rdChemReactions.HasProductTemplateSubstructMatch',
                                 'rdkit.Chem.Descriptors.PEOE_VSA6', 'rdkit.Chem.AllChem.MolToMolBlock',
                                 'rdkit.Chem.Descriptors.SlogP_VSA4', 'rdkit.Chem.rdqueries.QHAtomQueryAtom',
                                 'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarityNeighbors_sparse',
                                 'rdkit.Chem.GraphDescriptors.BertzCT', 'rdkit.Chem.AllChem.IsotopeEqualsQueryAtom',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA2',
                                 'rdkit.Chem.rdShapeHelpers.ComputeUnionBox', 'rdkit.Chem.Fragments.fr_benzodiazepine',
                                 'rdkit.Chem.Descriptors.EState_VSA5', 'rdkit.Chem.CreateMolSubstanceGroup',
                                 'rdkit.Chem.AllChem.GetMACCSKeysFingerprint',
                                 'rdkit.Chem.rdMolDescriptors.CalcEEMcharges', 'rdkit.Chem.MergeQueryHs',
                                 'rdkit.Chem.MolToSequence', 'rdkit.DataStructs.BulkBraunBlanquetSimilarity',
                                 'rdkit.Chem.Fragments.fr_azo', 'rdkit.Chem.rdMolDescriptors.CalcNumSaturatedRings',
                                 'rdkit.Chem.rdMolTransforms.SetDihedralDeg', 'rdkit.Chem.ReplaceSidechains',
                                 'rdkit.Geometry.ProtrudeDistance', 'rdkit.Chem.AllChem.MolFromPNGString',
                                 'rdkit.DataStructs.cDataStructs.KulczynskiSimilarityNeighbors',
                                 'rdkit.Chem.FindMolChiralCenters', 'rdkit.Chem.rdChemReactions.GetDefaultAdjustParams',
                                 'rdkit.Chem.Lipinski.NumAromaticRings', 'rdkit.Chem.Descriptors.SlogP_VSA6',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState3', 'rdkit.Chem.MolSurf.SMR_VSA1',
                                 'rdkit.Chem.rdChemReactions.ReactionFromPNGString',
                                 'rdkit.Geometry.FindGridTerminalPoints', 'rdkit.Chem.Descriptors.fr_oxime',
                                 'rdkit.Chem.AllChem.GetConnectivityInvariants', 'rdkit.Chem.AllChem.RemoveHs',
                                 'rdkit.Chem.Descriptors.MolWt', 'rdkit.Chem.SetHybridization',
                                 'rdkit.Chem.rdmolops.AssignAtomChiralTagsFromMolParity',
                                 'rdkit.Chem.rdMolDescriptors.GetAtomFeatures', 'rdkit.Chem.Descriptors.fr_halogen',
                                 'rdkit.Chem.rdqueries.NumRadicalElectronsEqualsQueryAtom',
                                 'rdkit.Chem.Descriptors.fr_nitro_arom_nonortho',
                                 'rdkit.Chem.rdmolfiles.MolFromMolFile', 'rdkit.Chem.AllChem.CalcCrippenDescriptors',
                                 'rdkit.Chem.AssignCIPLabels', 'rdkit.Chem.Descriptors.VSA_EState10',
                                 'rdkit.Chem.rdmolops.SetHybridization', 'rdkit.Chem.MolSurf.LabuteASA',
                                 'rdkit.Chem.Descriptors.EState_VSA4', 'rdkit.DataStructs.RusselSimilarity',
                                 'rdkit.Chem.JSONToMols', 'rdkit.Chem.rdmolops.RemoveHs',
                                 'rdkit.DataStructs.cDataStructs.DiceSimilarity', 'rdkit.rdBase.AttachFileToLog',
                                 'rdkit.Chem.Descriptors.MinAbsEStateIndex', 'rdkit.Chem.FragmentOnSomeBonds',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi0v',
                                 'rdkit.Chem.rdmolfiles.CreateAtomIntPropertyList',
                                 'rdkit.Chem.rdChemReactions.MatchOnlyAtRgroupsAdjustParams',
                                 'rdkit.Chem.Descriptors.fr_priamide', 'rdkit.Chem.Descriptors.VSA_EState2',
                                 'rdkit.Chem.rdSLNParse.MolFromSLN', 'rdkit.Chem.rdmolops.FindAtomEnvironmentOfRadiusN',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters',
                                 'rdkit.Chem.AllChem.MolToHELM', 'rdkit.Chem.Descriptors.fr_Ar_NH',
                                 'rdkit.Chem.CreateAtomDoublePropertyList',
                                 'rdkit.Chem.Descriptors.NumValenceElectrons',
                                 'rdkit.Chem.rdqueries.ExplicitValenceEqualsQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.GetUSRDistributions',
                                 'rdkit.DataStructs.cDataStructs.KulczynskiSimilarityNeighbors_sparse',
                                 'rdkit.Chem.Descriptors.fr_amide', 'rdkit.Chem.rdDistGeom.ETKDGv3',
                                 'rdkit.Chem.Descriptors.NHOHCount', 'rdkit.Chem.rdShapeHelpers.ShapeTverskyIndex',
                                 'rdkit.Chem.MolSurf.PEOE_VSA8', 'rdkit.Chem.Cleanup',
                                 'rdkit.Chem.Fragments.fr_morpholine', 'rdkit.Chem.MolSurf.SMR_VSA10',
                                 'rdkit.Chem.AllChem.ComputeConfDimsAndOffset', 'rdkit.Chem.SetAtomValue',
                                 'rdkit.Chem.AllChem.ParseMolQueryDefFile', 'rdkit.Chem.rdmolops.MergeQueryHs',
                                 'rdkit.Chem.AllChem.GetMolFrags', 'rdkit.Geometry.WriteGridToFile',
                                 'rdkit.DataStructs.BitVectToFPSText', 'rdkit.Chem.Fragments.fr_thiophene',
                                 'rdkit.Chem.AllChem.AdjustQueryProperties', 'rdkit.Chem.Descriptors.MaxAbsEStateIndex',
                                 'rdkit.Chem.AllChem.WedgeBond', 'rdkit.Chem.SetTerminalAtomCoords',
                                 'rdkit.DataStructs.AsymmetricSimilarity', 'rdkit.Chem.AllChem.IsotopeLessQueryAtom',
                                 'rdkit.DataStructs.SokalSimilarityNeighbors_sparse',
                                 'rdkit.Geometry.ComputeGridCentroid', 'rdkit.DataStructs.OnBitProjSimilarity',
                                 'rdkit.Chem.Fragments.fr_lactone', 'rdkit.Chem.rdmolops.molzip',
                                 'rdkit.Chem.AllChem.ComputeMolVolume', 'rdkit.Chem.MolSurf.SlogP_VSA10',
                                 'rdkit.Chem.Descriptors.MaxEStateIndex', 'rdkit.Chem.AllChem.CalcNumHeterocycles',
                                 'rdkit.Chem.rdMolDescriptors.CalcChiNn', 'rdkit.Chem.AllChem.CalcWHIM',
                                 'rdkit.Chem.rdmolops.AssignChiralTypesFromBondDirs',
                                 'rdkit.DataStructs.OffBitProjSimilarity', 'rdkit.Chem.rdmolfiles.MolToSmarts',
                                 'rdkit.Chem.rdqueries.HasDoublePropWithValueQueryAtom',
                                 'rdkit.Chem.AllChem.FormalChargeLessQueryAtom',
                                 'rdkit.DataStructs.InitFromDaylightString', 'rdkit.Chem.Descriptors.PEOE_VSA7',
                                 'rdkit.Chem.EState.EState.MaxEStateIndex',
                                 'rdkit.Chem.rdqueries.HasBoolPropWithValueQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.CalcGETAWAY', 'rdkit.Chem.MolSurf.PEOE_VSA7',
                                 'rdkit.Geometry.rdGeometry.WriteGridToFile',
                                 'rdkit.Chem.rdChemReactions.ReactionFromPNGFile',
                                 'rdkit.DataStructs.CreateFromFPSText',
                                 'rdkit.Chem.AllChem.TotalDegreeGreaterQueryAtom',
                                 'rdkit.DataStructs.cDataStructs.McConnaugheySimilarityNeighbors_sparse',
                                 'rdkit.Chem.MurckoDecompose', 'rdkit.Chem.AllChem.GetPeriodicTable',
                                 'rdkit.Chem.AllChem.CalcInertialShapeFactor', 'rdkit.Chem.rdMolAlign.AlignMol',
                                 'rdkit.Chem.QED.default', 'rdkit.Chem.Fragments.fr_phos_ester',
                                 'rdkit.Chem.Descriptors.VSA_EState6', 'rdkit.Chem.QED.weights_mean',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA7',
                                 'rdkit.Chem.EState.GetPrincipleQuantumNumber', 'rdkit.Chem.AllChem.SetAtomAlias',
                                 'rdkit.Chem.AllChem.CalcNumRotatableBonds', 'rdkit.Chem.MolToSmiles',
                                 'rdkit.Chem.Fragments.fr_amide', 'rdkit.DataStructs.TanimotoSimilarity',
                                 'rdkit.Chem.AllChem.InchiToInchiKey',
                                 'rdkit.Chem.AllChem.NumHeteroatomNeighborsGreaterQueryAtom',
                                 'rdkit.Chem.Fragments.fr_NH0', 'rdkit.Chem.AllChem.HasPropQueryAtom',
                                 'rdkit.Chem.rdChemReactions.ReactionMetadataToPNGString',
                                 'rdkit.DataStructs.FoldFingerprint',
                                 'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarityNeighbors',
                                 'rdkit.Chem.MolFromPDBFile', 'rdkit.Chem.FindRingFamilies',
                                 'rdkit.Geometry.TanimotoDistance',
                                 'rdkit.DataStructs.cDataStructs.BulkOnBitSimilarity', 'rdkit.Chem.LogWarningMsg',
                                 'rdkit.Chem.rdmolops.UnfoldedRDKFingerprintCountBased',
                                 'rdkit.Chem.rdqueries.MassLessQueryAtom', 'rdkit.Chem.Fragments.fr_phenol',
                                 'rdkit.Chem.rdmolops.FindAllPathsOfLengthN',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumLipinskiHBD',
                                 'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsLessQueryAtom',
                                 'rdkit.Chem.rdchem.SetDefaultPickleProperties', 'rdkit.Chem.AllChem.MolToPDBBlock',
                                 'rdkit.Chem.GetFormalCharge', 'rdkit.Chem.rdmolfiles.MolFragmentToCXSmiles',
                                 'rdkit.Chem.Descriptors.PEOE_VSA10', 'rdkit.Chem.AllChem.MolFromTPLFile',
                                 'rdkit.Chem.Descriptors.EState_VSA9', 'rdkit.Chem.AllChem.BondFromSmiles',
                                 'rdkit.Chem.Descriptors.VSA_EState9', 'rdkit.Chem.rdDistGeom.EmbedMolecule',
                                 'rdkit.Chem.MolFromInchi', 'rdkit.Chem.rdMolAlign.GetBestRMS',
                                 'rdkit.Chem.rdMolDescriptors.CalcAsphericity',
                                 'rdkit.Chem.rdMolDescriptors.CalcRadiusOfGyration', 'rdkit.Chem.AllChem.MolToInchiKey',
                                 'rdkit.Chem.AllChem.ExplicitDegreeGreaterQueryAtom', 'rdkit.Chem.Descriptors.fr_furan',
                                 'rdkit.Chem.Descriptors.fr_lactam', 'rdkit.Chem.rdChemReactions.ReactionFromMolecule',
                                 'rdkit.Chem.AllChem.GetMoleculeBoundsMatrix',
                                 'rdkit.Chem.AllChem.HybridizationGreaterQueryAtom',
                                 'rdkit.Chem.rdmolfiles.MolToCXSmiles',
                                 'rdkit.Chem.rdMolChemicalFeatures.BuildFeatureFactoryFromString',
                                 'rdkit.Chem.rdMolDescriptors.CalcTPSA', 'rdkit.Chem.EState.EState_VSA.EState_VSA11',
                                 'rdkit.DataStructs.CosineSimilarityNeighbors_sparse', 'rdkit.Chem.WrapLogs',
                                 'rdkit.Chem.Descriptors.SMR_VSA8',
                                 'rdkit.Chem.rdqueries.HasBoolPropWithValueQueryBond',
                                 'rdkit.DataStructs.cDataStructs.CreateFromFPSText', 'rdkit.Chem.Descriptors.BalabanJ',
                                 'rdkit.Chem.Descriptors.NumAromaticHeterocycles',
                                 'rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint',
                                 'rdkit.Chem.EState.EState.MinAbsEStateIndex',
                                 'rdkit.Chem.AllChem.SmilesMolSupplierFromText', 'rdkit.Chem.AllChem.RDKFingerprint',
                                 'rdkit.Chem.Fragments.fr_hdrzone', 'rdkit.Chem.MolToInchiAndAuxInfo',
                                 'rdkit.Chem.EState.EState.MaxAbsEStateIndex',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA9', 'rdkit.Chem.Fragments.fr_nitroso',
                                 'rdkit.Chem.rdMolDescriptors.CalcFractionCSP3',
                                 'rdkit.DataStructs.cDataStructs.BulkCosineSimilarity',
                                 'rdkit.Chem.rdqueries.RingBondCountLessQueryAtom',
                                 'rdkit.Geometry.rdGeometry.ProtrudeDistance',
                                 'rdkit.DataStructs.cDataStructs.InitFromDaylightString',
                                 'rdkit.Chem.AllChem.MolBlockToInchi', 'rdkit.Chem.SetAtomAlias',
                                 'rdkit.Chem.rdmolfiles.SmilesMolSupplierFromText',
                                 'rdkit.Chem.AllChem.ComputeGasteigerCharges',
                                 'rdkit.Chem.AllChem.GetFeatureInvariants',
                                 'rdkit.DataStructs.cDataStructs.KulczynskiSimilarity',
                                 'rdkit.Chem.AllChem.GetUSRScore',
                                 'rdkit.Chem.AllChem.CreateStructuralFingerprintForReaction',
                                 'rdkit.Chem.rdMolDescriptors.CalcAUTOCORR3D', 'rdkit.Chem.rdmolops.FindRingFamilies',
                                 'rdkit.Chem.Descriptors.VSA_EState4', 'rdkit.DataStructs.ConvertToExplicit',
                                 'rdkit.Chem.AllChem.CalcChi3n', 'rdkit.Chem.AllChem.MatchOnlyAtRgroupsAdjustParams',
                                 'rdkit.Chem.Descriptors3D.Asphericity', 'rdkit.Chem.MolToXYZBlock',
                                 'rdkit.Chem.rdmolfiles.MolToPDBFile', 'rdkit.Chem.Descriptors.PEOE_VSA5',
                                 'rdkit.Chem.Descriptors.FractionCSP3', 'rdkit.Chem.ReplaceCore',
                                 'rdkit.Chem.MolFromMolFile', 'rdkit.Chem.rdMolAlign.RandomTransform',
                                 'rdkit.Chem.AllChem.HasStringPropWithValueQueryAtom',
                                 'rdkit.ML.InfoTheory.rdInfoTheory.InfoEntropy',
                                 'rdkit.Chem.rdmolops.DetectBondStereoChemistry',
                                 'rdkit.DataStructs.cDataStructs.OffBitProjSimilarity', 'rdkit.Chem.AllChem.MolToInchi',
                                 'rdkit.Chem.AllChem.SetBondLength', 'rdkit.Chem.MolSurf.TPSA',
                                 'rdkit.Chem.rdMolDescriptors.BCUT2D', 'rdkit.Chem.Descriptors.fr_N_O',
                                 'rdkit.Chem.Descriptors.MinEStateIndex', 'rdkit.Chem.rdDistGeom.srETKDGv3',
                                 'rdkit.DataStructs.CosineSimilarityNeighbors',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState7', 'rdkit.Chem.AllChem.GetDihedralRad',
                                 'rdkit.Chem.AllChem.ComputeConfBox',
                                 'rdkit.DataStructs.cDataStructs.CosineSimilarityNeighbors_sparse',
                                 'rdkit.Chem.rdchem.SetAtomValue',
                                 'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarityNeighbors',
                                 'rdkit.Chem.rdmolfiles.MolFromSmarts', 'rdkit.Chem.AllChem.MolBlockToInchiAndAuxInfo',
                                 'rdkit.Chem.Descriptors.fr_C_S', 'rdkit.Chem.Fragments.fr_ArN',
                                 'rdkit.Chem.rdMolTransforms.GetBondLength', 'rdkit.Chem.CreateAtomIntPropertyList',
                                 'rdkit.Chem.AllChem.Get3DDistanceMatrix',
                                 'rdkit.Chem.AllChem.ExplicitDegreeEqualsQueryAtom',
                                 'rdkit.DataStructs.cDataStructs.ConvertToExplicit', 'rdkit.Chem.AllChem.CalcChiNn',
                                 'rdkit.Chem.Descriptors.NumHeteroatoms',
                                 'rdkit.Chem.rdChemReactions.GetChemDrawRxnAdjustParams',
                                 'rdkit.Chem.AllChem.FindAtomEnvironmentOfRadiusN',
                                 'rdkit.Chem.rdmolops.GetShortestPath', 'rdkit.Chem.AllChem.UFFGetMoleculeForceField',
                                 'rdkit.Chem.AllChem.GetDihedralDeg', 'rdkit.Chem.Descriptors.fr_lactone',
                                 'rdkit.Chem.BondFromSmiles', 'rdkit.Chem.AllChem.GetAngleDeg',
                                 'rdkit.Chem.AllChem.GetAdjacencyMatrix', 'rdkit.Chem.rdinchi.MolBlockToInchi',
                                 'rdkit.Chem.Descriptors.VSA_EState5', 'rdkit.DataStructs.McConnaugheySimilarity',
                                 'rdkit.Chem.rdChemReactions.CreateDifferenceFingerprintForReaction',
                                 'rdkit.Chem.AllChem.AssignChiralTypesFromBondDirs',
                                 'rdkit.Chem.Fragments.fr_prisulfonamd',
                                 'rdkit.Chem.rdMolTransforms.ComputePrincipalAxesAndMoments',
                                 'rdkit.Chem.AllChem.TotalDegreeLessQueryAtom',
                                 'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsLessQueryAtom',
                                 'rdkit.Chem.GraphDescriptors.Chi4n', 'rdkit.Chem.AllChem.IsUnsaturatedQueryAtom',
                                 'rdkit.Chem.AllChem.MMFFGetMoleculeProperties', 'rdkit.Chem.Fragments.fr_Al_OH_noTert',
                                 'rdkit.Chem.rdqueries.RingBondCountGreaterQueryAtom',
                                 'rdkit.Chem.AllChem.BuildFeatureFactory',
                                 'rdkit.DataStructs.cDataStructs.RusselSimilarity',
                                 'rdkit.Chem.Descriptors.NumAromaticRings', 'rdkit.Chem.EState.EState_VSA.EState_VSA6',
                                 'rdkit.Chem.rdShapeHelpers.ComputeConfBox',
                                 'rdkit.DataStructs.BraunBlanquetSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.CalcNumSpiroAtoms', 'rdkit.Chem.AllChem.PatternFingerprint',
                                 'rdkit.Chem.MolSurf.PEOE_VSA2', 'rdkit.Chem.rdqueries.AtomNumGreaterQueryAtom',
                                 'rdkit.Chem.Descriptors.fr_ketone', 'rdkit.Chem.EState.AtomTypes.BuildPatts',
                                 'rdkit.Chem.GraphDescriptors.Kappa1', 'rdkit.Chem.MetadataFromPNGString',
                                 'rdkit.Chem.Fragments.fr_dihydropyridine',
                                 'rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMoleculeConfs',
                                 'rdkit.Chem.AllChem.MolsFromPNGFile', 'rdkit.Chem.rdmolops.DetectBondStereochemistry',
                                 'rdkit.Chem.rdDistGeom.GetMoleculeBoundsMatrix',
                                 'rdkit.Chem.rdmolops.SetTerminalAtomCoords', 'rdkit.Chem.AllChem.MergeQueryHs',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles',
                                 'rdkit.DataStructs.ConvertToNumpyArray', 'rdkit.Chem.Descriptors.FpDensityMorgan1',
                                 'rdkit.Chem.AllChem.FormalChargeEqualsQueryAtom', 'rdkit.Chem.Fragments.fr_Ar_COO',
                                 'rdkit.Chem.MolMetadataToPNGString', 'rdkit.Chem.rdMolDescriptors.CalcChi4n',
                                 'rdkit.DataStructs.cDataStructs.BulkTanimotoSimilarity', 'rdkit.Chem.SetConjugation',
                                 'rdkit.Chem.MolSurf.SlogP_VSA9', 'rdkit.Chem.AllChem.TotalDegreeEqualsQueryAtom',
                                 'rdkit.Chem.RemoveHs', 'rdkit.DataStructs.TanimotoSimilarityNeighbors',
                                 'rdkit.Chem.Fragments.fr_azide',
                                 'rdkit.Chem.rdChemReactions.ReactionMetadataToPNGFile',
                                 'rdkit.Chem.AllChem.MetadataFromPNGString',
                                 'rdkit.Chem.SetDoubleBondNeighborDirections',
                                 'rdkit.DataStructs.cDataStructs.AsymmetricSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.CalcChi1v', 'rdkit.Chem.AllChem.MolToXYZBlock',
                                 'rdkit.Chem.Descriptors.fr_phenol_noOrthoHbond',
                                 'rdkit.Chem.Lipinski.NumSaturatedHeterocycles', 'rdkit.Chem.AllChem.MolsToJSON',
                                 'rdkit.Chem.AllChem.SetDihedralRad', 'rdkit.Chem.Fragments.fr_pyridine',
                                 'rdkit.DataStructs.cDataStructs.BulkDiceSimilarity', 'rdkit.Chem.Descriptors.Chi2n',
                                 'rdkit.Chem.Descriptors.SlogP_VSA2', 'rdkit.DataStructs.BulkTverskySimilarity',
                                 'rdkit.Chem.Fragments.fr_thiazole', 'rdkit.Chem.AllChem.CanonicalRankAtoms',
                                 'rdkit.Chem.rdmolops.FragmentOnBRICSBonds', 'rdkit.Chem.DetectChemistryProblems',
                                 'rdkit.Chem.AllChem.CalcNumSaturatedRings', 'rdkit.Chem.AssignStereochemistry',
                                 'rdkit.Geometry.TverskyIndex', 'rdkit.Chem.AllChem.CreateAtomIntPropertyList',
                                 'rdkit.Chem.AllChem.MolToMolFile',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumBridgeheadAtoms',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumHBD',
                                 'rdkit.Chem.AllChem.GetHashedMorganFingerprint',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA1', 'rdkit.Chem.rdmolfiles.MolFromPNGString',
                                 'rdkit.Chem.Fragments.fr_piperdine', 'rdkit.Chem.rdMolDescriptors.CalcKappa3',
                                 'rdkit.Chem.Descriptors.LabuteASA', 'rdkit.Chem.MolSurf.SMR_VSA4',
                                 'rdkit.Chem.AllChem.GetTopologicalTorsionFingerprint',
                                 'rdkit.Chem.AllChem.TotalValenceGreaterQueryAtom',
                                 'rdkit.Chem.AllChem.CalcSpherocityIndex', 'rdkit.Chem.rdinchi.MolToInchi',
                                 'rdkit.Chem.Descriptors.fr_benzodiazepine', 'rdkit.Chem.AllChem.SetAtomRLabel',
                                 'rdkit.DataStructs.BulkAsymmetricSimilarity',
                                 'rdkit.Chem.AllChem.AssignAtomChiralTagsFromStructure',
                                 'rdkit.DataStructs.cDataStructs.BulkTverskySimilarity', 'rdkit.Chem.Descriptors.qed',
                                 'rdkit.Chem.AllChem.NumRadicalElectronsGreaterQueryAtom',
                                 'rdkit.Chem.MolSurf.PEOE_VSA6', 'rdkit.Chem.rdmolfiles.MolToMolBlock',
                                 'rdkit.Chem.Descriptors.fr_nitrile', 'rdkit.DataStructs.DiceSimilarity',
                                 'rdkit.Chem.AllChem.CalcKappa1', 'rdkit.Chem.AllChem.FragmentOnBRICSBonds',
                                 'rdkit.Chem.Fragments.fr_NH1', 'rdkit.Chem.AllChem.HasProductTemplateSubstructMatch',
                                 'rdkit.Chem.rdmolfiles.MolToXYZBlock', 'rdkit.Chem.AssignAtomChiralTagsFromStructure',
                                 'rdkit.Chem.MolSurf.pyLabuteASA', 'rdkit.DataStructs.cDataStructs.OffBitsInCommon',
                                 'rdkit.Chem.Fragments.fr_phos_acid', 'rdkit.Chem.rdmolops.PathToSubmol',
                                 'rdkit.Chem.Descriptors.FpDensityMorgan3',
                                 'rdkit.Chem.rdmolops.FindAllSubgraphsOfLengthN',
                                 'rdkit.DataStructs.cDataStructs.OnBitsInCommon', 'rdkit.Chem.MolSurf.SMR_VSA2',
                                 'rdkit.Chem.Descriptors.fr_Ar_OH',
                                 'rdkit.Chem.AllChem.HasStringPropWithValueQueryBond',
                                 'rdkit.DataStructs.RogotGoldbergSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.RemoveMappingNumbersFromReactions',
                                 'rdkit.Chem.GetMolSubstanceGroupWithIdx',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds',
                                 'rdkit.Chem.AllChem.HasBoolPropWithValueQueryBond', 'rdkit.Chem.Fragments.fr_imide',
                                 'rdkit.Chem.AllChem.AddMolSubstanceGroup', 'rdkit.ML.InfoTheory.ChiSquare',
                                 'rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect',
                                 'rdkit.Chem.AllChem.ClearMolSubstanceGroups',
                                 'rdkit.Chem.rdqueries.AtomNumEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.HCountGreaterQueryAtom',
                                 'rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs',
                                 'rdkit.Chem.rdchem.SetAtomRLabel', 'rdkit.Chem.tossit',
                                 'rdkit.Chem.AllChem.HasPropQueryBond', 'rdkit.Chem.AllChem.GetAtomMatch',
                                 'rdkit.Chem.AllChem.HasIntPropWithValueQueryBond', 'rdkit.Chem.MolSurf.PEOE_VSA12',
                                 'rdkit.Chem.rdMolAlign.GetCrippenO3A', 'rdkit.Chem.AllChem.CalcNumLipinskiHBA',
                                 'rdkit.Chem.Descriptors.fr_NH1', 'rdkit.Chem.rdmolops.GetSymmSSSR',
                                 'rdkit.Chem.PatternFingerprint', 'rdkit.Chem.rdmolops.AddRecursiveQuery',
                                 'rdkit.Chem.Descriptors.SlogP_VSA9', 'rdkit.Chem.AllChem.GetDistanceMatrix',
                                 'rdkit.Chem.rdmolops.SetBondStereoFromDirections',
                                 'rdkit.Chem.rdMolDescriptors.CalcLabuteASA', 'rdkit.Chem.Descriptors.Kappa2',
                                 'rdkit.Chem.AssignAtomChiralTagsFromMolParity',
                                 'rdkit.Chem.rdMolTransforms.GetDihedralRad',
                                 'rdkit.Chem.rdMolTransforms.CanonicalizeMol',
                                 'rdkit.Chem.AllChem.MolMetadataToPNGString', 'rdkit.Chem.Fragments.fr_alkyl_halide',
                                 'rdkit.Chem.rdmolops.AssignRadicals', 'rdkit.Chem.AllChem.CalcEccentricity',
                                 'rdkit.Chem.AllChem.CalcTPSA', 'rdkit.Chem.Fragments.fr_Ndealkylation1',
                                 'rdkit.Chem.AllChem.ReactionFromSmarts',
                                 'rdkit.Chem.AllChem.UnfoldedRDKFingerprintCountBased',
                                 'rdkit.Chem.rdqueries.TotalValenceLessQueryAtom',
                                 'rdkit.DataStructs.BitVectToBinaryText', 'rdkit.Chem.AllChem.MolFromMol2Block',
                                 'rdkit.Chem.Fragments.fr_Ndealkylation2', 'rdkit.Chem.QED.properties',
                                 'rdkit.Chem.AllChem.CalcExactMolWt', 'rdkit.Chem.AddMetadataToPNGString',
                                 'rdkit.rdBase.SeedRandomNumberGenerator',
                                 'rdkit.Chem.rdChemReactions.RemoveMappingNumbersFromReactions',
                                 'rdkit.Chem.rdForceFieldHelpers.GetUFFVdWParams',
                                 'rdkit.DataStructs.RusselSimilarityNeighbors_sparse', 'rdkit.Chem.Descriptors.Chi4n',
                                 'rdkit.Chem.GraphDescriptors.Ipc', 'rdkit.Chem.AllChem.ReactionFromPNGString',
                                 'rdkit.Chem.AllChem.ShapeProtrudeDist', 'rdkit.Chem.rdMolInterchange.MolsToJSON',
                                 'rdkit.Chem.EState.TypeAtoms',
                                 'rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect',
                                 'rdkit.DataStructs.SokalSimilarity',
                                 'rdkit.Chem.AllChem.GenerateErGFingerprintForReducedGraph',
                                 'rdkit.Chem.rdmolops.SetDoubleBondNeighborDirections',
                                 'rdkit.Chem.GraphDescriptors.Chi2v', 'rdkit.Chem.rdDistGeom.ETKDG',
                                 'rdkit.Chem.Descriptors.HallKierAlpha', 'rdkit.Chem.Descriptors.fr_Al_OH',
                                 'rdkit.Chem.Fragments.fr_SH', 'rdkit.Chem.Fragments.fr_bicyclic',
                                 'rdkit.Chem.AllChem.ShapeTverskyIndex', 'rdkit.Chem.Descriptors.Chi3n',
                                 'rdkit.Chem.Fragments.fr_para_hydroxylation', 'rdkit.Chem.AllChem.UFFOptimizeMolecule',
                                 'rdkit.Chem.RemoveAllHs', 'rdkit.Chem.Descriptors.fr_ether',
                                 'rdkit.Chem.AssignRadicals', 'rdkit.Chem.Descriptors.fr_C_O',
                                 'rdkit.Chem.rdChemReactions.PreprocessReaction', 'rdkit.Chem.MolSurf.SMR_VSA7',
                                 'rdkit.Chem.EState.EState.MinEStateIndex', 'rdkit.Chem.AllChem.MolFromTPLBlock',
                                 'rdkit.Chem.rdqueries.MinRingSizeLessQueryAtom', 'rdkit.Chem.Fragments.fr_methoxy',
                                 'rdkit.Chem.MolSurf.PEOE_VSA1', 'rdkit.Chem.AllChem.InNRingsGreaterQueryAtom',
                                 'rdkit.Chem.AllChem.MolToTPLFile', 'rdkit.Chem.AllChem.MetadataFromPNGFile',
                                 'rdkit.Chem.Descriptors.fr_ketone_Topliss', 'rdkit.Chem.Descriptors.SMR_VSA9',
                                 'rdkit.Chem.Fragments.fr_unbrch_alkane',
                                 'rdkit.DataStructs.McConnaugheySimilarityNeighbors',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA10', 'rdkit.Geometry.ComputeDihedralAngle',
                                 'rdkit.DataStructs.cDataStructs.BulkRogotGoldbergSimilarity',
                                 'rdkit.DataStructs.AllProbeBitsMatch', 'rdkit.Chem.GraphDescriptors.Chi0v',
                                 'rdkit.Chem.rdmolfiles.MolToV3KMolBlock', 'rdkit.Chem.MolSurf.SlogP_VSA11',
                                 'rdkit.Chem.Descriptors.fr_azide', 'rdkit.Chem.MolFromSmarts',
                                 'rdkit.DataStructs.AsymmetricSimilarityNeighbors_sparse',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi4v', 'rdkit.Chem.AllChem.GetUSRCAT',
                                 'rdkit.Chem.Descriptors.fr_HOCCN',
                                 'rdkit.Chem.rdForceFieldHelpers.UFFGetMoleculeForceField',
                                 'rdkit.Chem.AllChem.FindUniqueSubgraphsOfLengthN', 'rdkit.Chem.Fragments.fr_ether',
                                 'rdkit.Chem.rdmolops.FindUniqueSubgraphsOfLengthN',
                                 'rdkit.Chem.Fragments.fr_nitro_arom_nonortho',
                                 'rdkit.DataStructs.AsymmetricSimilarityNeighbors',
                                 'rdkit.Chem.DetectBondStereoChemistry', 'rdkit.Chem.MolToMolBlock',
                                 'rdkit.Chem.rdForceFieldHelpers.GetUFFTorsionParams',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles',
                                 'rdkit.Chem.rdqueries.HasIntPropWithValueQueryAtom',
                                 'rdkit.Chem.rdqueries.HybridizationEqualsQueryAtom', 'rdkit.Chem.GetMolFrags',
                                 'rdkit.Chem.AllChem.MolToV3KMolFile', 'rdkit.Chem.Descriptors.VSA_EState3',
                                 'rdkit.Chem.Descriptors.SMR_VSA2', 'rdkit.Chem.Descriptors.PEOE_VSA11',
                                 'rdkit.Chem.rdMolAlign.AlignMolConformers', 'rdkit.Chem.inchi.MolBlockToInchi',
                                 'rdkit.Chem.rdchem.GetPeriodicTable', 'rdkit.Chem.AllChem.MolToSequence',
                                 'rdkit.DataStructs.RogotGoldbergSimilarity',
                                 'rdkit.Chem.rdqueries.HasStringPropWithValueQueryAtom',
                                 'rdkit.Chem.AllChem.NumHeteroatomNeighborsEqualsQueryAtom',
                                 'rdkit.Chem.Descriptors.fr_urea', 'rdkit.Chem.rdMolAlign.GetCrippenO3AForProbeConfs',
                                 'rdkit.Chem.rdMolDescriptors.CalcPMI1', 'rdkit.Chem.rdmolops.FastFindRings',
                                 'rdkit.Chem.rdMolAlign.GetAlignmentTransform', 'rdkit.DataStructs.BulkSokalSimilarity',
                                 'rdkit.Chem.AllChem.MolFromRDKitSVG', 'rdkit.Chem.EState.BuildPatts',
                                 'rdkit.Chem.AllChem.RingBondCountEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.MMFFHasAllMoleculeParams', 'rdkit.Chem.Descriptors.SMR_VSA4',
                                 'rdkit.Chem.Fragments.fr_oxazole',
                                 'rdkit.DataStructs.cDataStructs.DiceSimilarityNeighbors_sparse',
                                 'rdkit.Chem.Lipinski.NumAliphaticHeterocycles', 'rdkit.DataStructs.TverskySimilarity',
                                 'rdkit.Chem.AllChem.HasReactionSubstructMatch',
                                 'rdkit.Chem.AllChem.MolToInchiAndAuxInfo', 'rdkit.Chem.AllChem.CalcNumAliphaticRings',
                                 'rdkit.Chem.rdmolfiles.MolsFromPNGString',
                                 'rdkit.Chem.AllChem.IsotopeGreaterQueryAtom', 'rdkit.Chem.AllChem.CalcNPR1',
                                 'rdkit.DataStructs.cDataStructs.CreateFromBitString',
                                 'rdkit.Chem.AllChem.TransformMol', 'rdkit.Chem.AllChem.AssignStereochemistry',
                                 'rdkit.Chem.Descriptors.MaxAbsPartialCharge',
                                 'rdkit.Geometry.rdGeometry.UniformGrid3D',
                                 'rdkit.Chem.EnumerateStereoisomers.GetStereoisomerCount',
                                 'rdkit.Chem.Lipinski.NumAliphaticCarbocycles', 'rdkit.Chem.AllChem.RenumberAtoms',
                                 'rdkit.Chem.ChemicalFeatures.BuildFeatureFactory', 'rdkit.Chem.Descriptors.Chi2v',
                                 'rdkit.Chem.AllChem.CalcMolFormula',
                                 'rdkit.Chem.rdChemReactions.Compute2DCoordsForReaction',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings',
                                 'rdkit.Chem.rdChemReactions.EnumerateLibraryCanSerialize',
                                 'rdkit.Chem.AllChem.SortMatchesByDegreeOfCoreSubstitution',
                                 'rdkit.Chem.rdqueries.IsInRingQueryAtom',
                                 'rdkit.Chem.EnumerateStereoisomers.EmbedMolecule', 'rdkit.Chem.MolToTPLBlock',
                                 'rdkit.Chem.Descriptors.ExactMolWt', 'rdkit.Chem.AllChem.RandomTransform',
                                 'rdkit.Chem.rdMolInterchange.MolToJSON', 'rdkit.Chem.rdSLNParse.MolFromQuerySLN',
                                 'rdkit.Chem.Descriptors.HeavyAtomCount', 'rdkit.Chem.rdinchi.MolToInchiKey',
                                 'rdkit.Chem.Fragments.fr_C_O_noCOO',
                                 'rdkit.Chem.AllChem.GenerateDepictionMatching3DStructure',
                                 'rdkit.Chem.AllChem.MinRingSizeEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.CalcNumBridgeheadAtoms',
                                 'rdkit.Chem.rdqueries.TotalDegreeGreaterQueryAtom',
                                 'rdkit.Chem.AllChem.ComputeCentroid', 'rdkit.Chem.Descriptors.PEOE_VSA13',
                                 'rdkit.Chem.rdmolops.FindAllSubgraphsOfLengthMToN',
                                 'rdkit.DataStructs.cDataStructs.BulkBraunBlanquetSimilarity',
                                 'rdkit.Chem.rdmolfiles.MolFromMol2Block',
                                 'rdkit.Chem.rdMolTransforms.ComputePrincipalAxesAndMomentsFromGyrationMatrix',
                                 'rdkit.Chem.rdqueries.TotalDegreeEqualsQueryAtom', 'rdkit.Chem.MolFromMolBlock',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles',
                                 'rdkit.Chem.rdmolops.FragmentOnBonds', 'rdkit.Chem.AllChem.MolFromSequence',
                                 'rdkit.Chem.MolsFromPNGString', 'rdkit.Chem.FindAllSubgraphsOfLengthMToN',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState4', 'rdkit.Chem.Fragments.fr_Nhpyrrole',
                                 'rdkit.Chem.AllChem.CalcPMI1',
                                 'rdkit.DataStructs.cDataStructs.BulkAsymmetricSimilarity',
                                 'rdkit.Chem.Descriptors.EState_VSA2',
                                 'rdkit.Chem.rdMolDescriptors.GetConnectivityInvariants',
                                 'rdkit.Chem.Descriptors.fr_nitro', 'rdkit.Chem.rdmolfiles.MolFragmentToSmarts',
                                 'rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs',
                                 'rdkit.Chem.AllChem.HasIntPropWithValueQueryAtom',
                                 'rdkit.Chem.AllChem.HasChiralTagQueryAtom', 'rdkit.Chem.ParseMolQueryDefFile',
                                 'rdkit.DataStructs.cDataStructs.CreateFromBinaryText',
                                 'rdkit.Chem.Descriptors.EState_VSA6', 'rdkit.Chem.DetectBondStereochemistry',
                                 'rdkit.Chem.rdqueries.TotalValenceEqualsQueryAtom',
                                 'rdkit.Chem.rdChemReactions.CreateStructuralFingerprintForReaction',
                                 'rdkit.Chem.AllChem.CalcRadiusOfGyration', 'rdkit.Chem.Descriptors.Chi1v',
                                 'rdkit.Chem.inchi.InchiToInchiKey',
                                 'rdkit.DataStructs.cDataStructs.TanimotoSimilarity', 'rdkit.Chem.AllChem.CalcChi0v',
                                 'rdkit.Chem.Descriptors.Chi1n', 'rdkit.Chem.rdqueries.MassEqualsQueryAtom',
                                 'rdkit.Chem.Lipinski.NHOHCount', 'rdkit.Chem.AllChem.GetSSSR',
                                 'rdkit.Chem.rdchem.CreateStereoGroup',
                                 'rdkit.Chem.rdmolops.SortMatchesByDegreeOfCoreSubstitution',
                                 'rdkit.Chem.AllChem.CreateMolSubstanceGroup',
                                 'rdkit.DataStructs.cDataStructs.OnBitSimilarity',
                                 'rdkit.Chem.AllChem.CalcNumLipinskiHBD', 'rdkit.Chem.rdmolops.ParseMolQueryDefFile',
                                 'rdkit.Chem.AllChem.MolFromMolBlock', 'rdkit.Chem.AllChem.MolToXYZFile',
                                 'rdkit.Chem.Descriptors3D.NPR2',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAromaticHeterocycles',
                                 'rdkit.Chem.AllChem.GetAlignmentTransform',
                                 'rdkit.Chem.Descriptors.fr_para_hydroxylation', 'rdkit.Chem.Descriptors.fr_thiophene',
                                 'rdkit.Chem.MolSurf.PEOE_VSA10', 'rdkit.Chem.CanonicalRankAtoms',
                                 'rdkit.Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams', 'rdkit.Chem.FragmentOnBonds',
                                 'rdkit.Chem.rdMolDescriptors.CalcPBF', 'rdkit.Chem.Descriptors3D.RadiusOfGyration',
                                 'rdkit.Chem.QED.qed', 'rdkit.Chem.rdCoordGen.AddCoords',
                                 'rdkit.Chem.rdmolops.FindPotentialStereoBonds',
                                 'rdkit.Chem.rdmolops.PatternFingerprint',
                                 'rdkit.Chem.rdChemReactions.HasReactantTemplateSubstructMatch',
                                 'rdkit.Chem.rdqueries.AtomNumLessQueryAtom', 'rdkit.Chem.MolToSmarts',
                                 'rdkit.Chem.rdmolops.GetFormalCharge',
                                 'rdkit.Chem.Descriptors.NumSaturatedHeterocycles',
                                 'rdkit.Chem.Descriptors.fr_C_O_noCOO',
                                 'rdkit.DataStructs.RogotGoldbergSimilarityNeighbors_sparse',
                                 'rdkit.Chem.FindAllSubgraphsOfLengthN', 'rdkit.Chem.Descriptors.PEOE_VSA1',
                                 'rdkit.Geometry.rdGeometry.ComputeGridCentroid', 'rdkit.Chem.rdmolfiles.MolToMolFile',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState6',
                                 'rdkit.Chem.rdqueries.NumHeteroatomNeighborsGreaterQueryAtom',
                                 'rdkit.Chem.rdqueries.InNRingsLessQueryAtom', 'rdkit.Chem.Descriptors.fr_nitro_arom',
                                 'rdkit.Chem.AllChem.PathToSubmol',
                                 'rdkit.Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams',
                                 'rdkit.Geometry.rdGeometry.TverskyIndex', 'rdkit.Chem.rdMolDescriptors.CalcChi3v',
                                 'rdkit.Chem.AllChem.ReactionFromRxnFile',
                                 'rdkit.DataStructs.McConnaugheySimilarityNeighbors_sparse',
                                 'rdkit.Chem.AllChem.MolToTPLBlock', 'rdkit.DataStructs.cDataStructs.ComputeL1Norm',
                                 'rdkit.Chem.rdchem.GetMolSubstanceGroups', 'rdkit.Chem.AllChem.ReplaceSubstructs',
                                 'rdkit.Chem.rdmolops.ReplaceCore', 'rdkit.Chem.rdqueries.MAtomQueryAtom',
                                 'rdkit.Chem.Fragments.fr_C_O', 'rdkit.Chem.MolFromPNGString',
                                 'rdkit.Chem.Descriptors.fr_quatN', 'rdkit.Chem.Lipinski.NumAromaticHeterocycles',
                                 'rdkit.Chem.rdqueries.XHAtomQueryAtom', 'rdkit.Chem.rdmolops.SetAromaticity',
                                 'rdkit.Chem.AllChem.GetMolSubstanceGroupWithIdx',
                                 'rdkit.Chem.rdchem.CreateMolSubstanceGroup', 'rdkit.Chem.Descriptors.PEOE_VSA14',
                                 'rdkit.Chem.Fragments.fr_halogen', 'rdkit.Chem.rdmolops.WedgeMolBonds',
                                 'rdkit.Chem.AdjustQueryProperties', 'rdkit.DataStructs.SokalSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.SetAromaticity', 'rdkit.Chem.Descriptors.fr_piperdine',
                                 'rdkit.Chem.Descriptors.fr_methoxy', 'rdkit.Chem.Descriptors.fr_Nhpyrrole',
                                 'rdkit.Chem.Fragments.fr_Ar_OH', 'rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint',
                                 'rdkit.Chem.rdmolops.FindPotentialStereo', 'rdkit.Chem.AllChem.AtomFromSmiles',
                                 'rdkit.Chem.rdMolDescriptors.CalcInertialShapeFactor', 'rdkit.Chem.rdinchi.InchiToMol',
                                 'rdkit.Chem.Descriptors3D.PMI1', 'rdkit.Chem.Descriptors.MolLogP',
                                 'rdkit.Chem.AllChem.SupplierFromFilename', 'rdkit.Chem.Fragments.fr_nitro_arom',
                                 'rdkit.ML.InfoTheory.rdInfoTheory.ChiSquare',
                                 'rdkit.Chem.AllChem.ComputeCanonicalTransform',
                                 'rdkit.Chem.rdChemReactions.IsReactionTemplateMoleculeAgent',
                                 'rdkit.Chem.Descriptors.NumAliphaticHeterocycles', 'rdkit.Chem.Descriptors.fr_amidine',
                                 'rdkit.Chem.Descriptors.fr_morpholine',
                                 'rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprint',
                                 'rdkit.Chem.EState.MaxAbsEStateIndex', 'rdkit.Chem.GraphDescriptors.Chi2n',
                                 'rdkit.Chem.AllChem.WrapLogs', 'rdkit.Chem.rdqueries.MHAtomQueryAtom',
                                 'rdkit.Chem.AllChem.CalcRDF', 'rdkit.Chem.AssignChiralTypesFromBondDirs',
                                 'rdkit.Chem.rdDepictor.SetPreferCoordGen',
                                 'rdkit.Chem.rdqueries.IsotopeGreaterQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAromaticCarbocycles',
                                 'rdkit.Chem.MolToRandomSmilesVect', 'rdkit.Chem.inchi.MolToInchiAndAuxInfo',
                                 'rdkit.DataStructs.OnBitSimilarity', 'rdkit.Chem.AllChem.EmbedMolecule',
                                 'rdkit.Chem.WedgeBond', 'rdkit.Chem.AllChem.GetO3AForProbeConfs',
                                 'rdkit.Chem.Lipinski.NumHeteroatoms', 'rdkit.Chem.RenumberAtoms',
                                 'rdkit.Chem.MolSurf.SMR_VSA3', 'rdkit.Chem.rdMolDescriptors.CalcNumRings',
                                 'rdkit.Chem.rdmolfiles.AtomFromSmiles', 'rdkit.Chem.Fragments.fr_urea',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi2v',
                                 'rdkit.Chem.AllChem.RingBondCountGreaterQueryAtom',
                                 'rdkit.Chem.rdmolfiles.MolFromFASTA', 'rdkit.Chem.MolSurf.SlogP_VSA7',
                                 'rdkit.Chem.Descriptors.fr_Al_COO', 'rdkit.Chem.Get3DDistanceMatrix',
                                 'rdkit.Chem.AllChem.EncodeShape', 'rdkit.Chem.AllChem.EnumerateLibraryCanSerialize',
                                 'rdkit.DataStructs.cDataStructs.TanimotoSimilarityNeighbors',
                                 'rdkit.DataStructs.BitVectToText', 'rdkit.Chem.rdmolfiles.AddMetadataToPNGFile',
                                 'rdkit.Chem.Descriptors.NumHAcceptors', 'rdkit.Chem.Descriptors.EState_VSA3',
                                 'rdkit.Chem.AllChem.SetBondStereoFromDirections',
                                 'rdkit.Chem.rdShapeHelpers.ShapeTanimotoDist',
                                 'rdkit.Chem.rdMolInterchange.JSONToMols', 'rdkit.Chem.AllChem.CalcFractionCSP3',
                                 'rdkit.Chem.rdMolDescriptors.CalcKappa2', 'rdkit.Chem.Descriptors.Chi3v',
                                 'rdkit.Chem.AllChem.BondFromSmarts', 'rdkit.Chem.rdMolDescriptors.CalcMORSE',
                                 'rdkit.Chem.rdChemReactions.ReduceProductToSideChains',
                                 'rdkit.Chem.rdqueries.HasIntPropWithValueQueryBond',
                                 'rdkit.Chem.AllChem.QHAtomQueryAtom', 'rdkit.Chem.ReplaceSubstructs',
                                 'rdkit.Chem.rdMolDescriptors.CalcCoulombMat',
                                 'rdkit.Chem.AllChem.Compute2DCoordsMimicDistmat',
                                 'rdkit.Chem.Descriptors.fr_phos_ester',
                                 'rdkit.Chem.AllChem.CreateAtomStringPropertyList',
                                 'rdkit.Chem.AllChem.HCountEqualsQueryAtom',
                                 'rdkit.Chem.EState.EState.GetPrincipleQuantumNumber',
                                 'rdkit.Chem.rdmolfiles.MolFromTPLFile', 'rdkit.Chem.rdqueries.MassGreaterQueryAtom',
                                 'rdkit.Chem.SetDefaultPickleProperties',
                                 'rdkit.Chem.Descriptors.NumAliphaticCarbocycles',
                                 'rdkit.Chem.rdCoordGen.SetDefaultTemplateFileDir',
                                 'rdkit.DataStructs.cDataStructs.BulkSokalSimilarity', 'rdkit.Chem.AllChem.CalcChi3v',
                                 'rdkit.Chem.CanonicalRankAtomsInFragment', 'rdkit.RDLogger.LogMessage',
                                 'rdkit.Chem.rdMolTransforms.SetDihedralRad', 'rdkit.Chem.Descriptors.Kappa3',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState8', 'rdkit.Chem.rdShapeHelpers.EncodeShape',
                                 'rdkit.Chem.AddMetadataToPNGFile', 'rdkit.Chem.rdmolfiles.MolMetadataToPNGString',
                                 'rdkit.Chem.Descriptors.fr_thiocyan', 'rdkit.Chem.AllChem.GetDefaultPickleProperties',
                                 'rdkit.Chem.AllChem.SetDefaultPickleProperties', 'rdkit.Chem.SetAtomRLabel',
                                 'rdkit.DataStructs.cDataStructs.McConnaugheySimilarityNeighbors',
                                 'rdkit.Chem.rdDistGeom.KDG', 'rdkit.Chem.AllChem.HasDoublePropWithValueQueryBond',
                                 'rdkit.DataStructs.RusselSimilarityNeighbors', 'rdkit.Chem.Fragments.fr_allylic_oxid',
                                 'rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges', 'rdkit.Chem.AllChem.CalcMORSE',
                                 'rdkit.Chem.Fragments.fr_nitro', 'rdkit.Chem.AssignStereochemistryFrom3D',
                                 'rdkit.Chem.AddRecursiveQuery', 'rdkit.Chem.AllChem.HasReactionAtomMapping',
                                 'rdkit.Chem.AllChem.tossit',
                                 'rdkit.DataStructs.cDataStructs.SokalSimilarityNeighbors_sparse',
                                 'rdkit.Chem.Fragments.fr_Ar_NH', 'rdkit.Chem.Descriptors.fr_hdrzine',
                                 'rdkit.Chem.AllChem.GetUFFInversionParams', 'rdkit.Chem.GraphDescriptors.Chi0n',
                                 'rdkit.DataStructs.cDataStructs.ConvertToNumpyArray',
                                 'rdkit.Geometry.ComputeSignedDihedralAngle', 'rdkit.Chem.AllChem.CalcNumHBA',
                                 'rdkit.Chem.rdmolfiles.MolsFromPNGFile', 'rdkit.Chem.rdMolDescriptors.CalcRDF',
                                 'rdkit.Chem.AllChem.FragmentOnBonds',
                                 'rdkit.DataStructs.cDataStructs.SokalSimilarityNeighbors',
                                 'rdkit.Chem.Descriptors.NumSaturatedRings', 'rdkit.Chem.Descriptors.fr_NH0',
                                 'rdkit.Chem.rdqueries.IsotopeEqualsQueryAtom',
                                 'rdkit.Chem.rdqueries.IsotopeLessQueryAtom', 'rdkit.Chem.Descriptors.NOCount',
                                 'rdkit.Chem.AllChem.MolFromMolFile', 'rdkit.Chem.rdForceFieldHelpers.OptimizeMolecule',
                                 'rdkit.Chem.Descriptors.fr_piperzine', 'rdkit.Chem.AllChem.CalcPBF',
                                 'rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprint',
                                 'rdkit.Chem.Fragments.fr_thiocyan',
                                 'rdkit.Chem.AllChem.HasAgentTemplateSubstructMatch',
                                 'rdkit.Chem.rdqueries.HasPropQueryBond',
                                 'rdkit.Chem.rdMolDescriptors.GetAtomPairFingerprint',
                                 'rdkit.Chem.rdChemReactions.ReactionToRxnBlock',
                                 'rdkit.Chem.AllChem.ExplicitValenceGreaterQueryAtom',
                                 'rdkit.Chem.rdqueries.QAtomQueryAtom', 'rdkit.Chem.rdChemReactions.ReactionToSmiles',
                                 'rdkit.Chem.MolFragmentToSmiles', 'rdkit.Chem.inchi.MolFromInchi',
                                 'rdkit.Chem.MolToInchi', 'rdkit.Chem.AllChem.GetMolSubstanceGroups',
                                 'rdkit.Chem.Descriptors.SlogP_VSA7', 'rdkit.Chem.AllChem.CalcRMS',
                                 'rdkit.Chem.rdForceFieldHelpers.OptimizeMoleculeConfs',
                                 'rdkit.Chem.AllChem.MolFromPDBFile', 'rdkit.Chem.MolFromTPLBlock',
                                 'rdkit.Chem.rdmolops.AddHs', 'rdkit.Chem.EState.EState_VSA.VSA_EState9',
                                 'rdkit.Chem.AllChem.AssignBondOrdersFromTemplate',
                                 'rdkit.DataStructs.BulkTanimotoSimilarity',
                                 'rdkit.Chem.AllChem.ReactionMetadataToPNGString',
                                 'rdkit.Chem.Descriptors.fr_sulfonamd', 'rdkit.Chem.AllChem.CalcNumAromaticRings',
                                 'rdkit.Chem.MolToHELM', 'rdkit.Chem.Descriptors.fr_guanido',
                                 'rdkit.Chem.AllChem.MolToCXSmiles',
                                 'rdkit.Chem.rdMolDescriptors.GetUSRDistributionsFromPoints',
                                 'rdkit.Chem.AllChem.InNRingsEqualsQueryAtom', 'rdkit.Chem.MolSurf.SMR_VSA8',
                                 'rdkit.Chem.AllChem.CreateDifferenceFingerprintForReaction',
                                 'rdkit.Chem.AllChem.ComputeMolShape', 'rdkit.Chem.BondFromSmarts',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAliphaticRings',
                                 'rdkit.Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect',
                                 'rdkit.Chem.CreateStereoGroup', 'rdkit.Chem.AllChem.AlignMolConformers',
                                 'rdkit.Chem.rdmolfiles.MolFromPDBBlock', 'rdkit.Chem.rdMolAlign.GetO3AForProbeConfs',
                                 'rdkit.Chem.AllChem.MolFromPNGFile', 'rdkit.Chem.MolFromPNGFile',
                                 'rdkit.DataStructs.AllBitSimilarity', 'rdkit.DataStructs.cDataStructs.FoldFingerprint',
                                 'rdkit.Chem.AllChem.AtomNumEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.ReduceProductToSideChains',
                                 'rdkit.Chem.rdmolops.ReplaceSidechains',
                                 'rdkit.Chem.ChemicalFeatures.MCFF_GetFeaturesForMol',
                                 'rdkit.Chem.AllChem.ComputePrincipalAxesAndMoments',
                                 'rdkit.DataStructs.FingerprintSimilarity', 'rdkit.DataStructs.OnBitsInCommon',
                                 'rdkit.Chem.rdDepictor.GenerateDepictionMatching2DStructure',
                                 'rdkit.Chem.MolFromMol2File', 'rdkit.Chem.AllChem.GetHashedAtomPairFingerprint',
                                 'rdkit.Chem.rdqueries.FormalChargeEqualsQueryAtom',
                                 'rdkit.Chem.Descriptors.EState_VSA7', 'rdkit.Chem.GetSymmSSSR',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi1n', 'rdkit.Chem.AllChem.Cleanup',
                                 'rdkit.Chem.AllChem.LayeredFingerprint', 'rdkit.Chem.AllChem.AtomFromSmarts',
                                 'rdkit.Chem.AllChem.MassGreaterQueryAtom', 'rdkit.Chem.MolToMolFile',
                                 'rdkit.Chem.rdCIPLabeler.AssignCIPLabels',
                                 'rdkit.Chem.rdqueries.ExplicitDegreeGreaterQueryAtom',
                                 'rdkit.Chem.rdmolops.AssignStereochemistry', 'rdkit.Chem.Descriptors.fr_tetrazole',
                                 'rdkit.Chem.AllChem.MolFromFASTA', 'rdkit.Chem.rdmolops.RemoveStereochemistry',
                                 'rdkit.Chem.SmilesMolSupplierFromText', 'rdkit.Chem.Descriptors.fr_alkyl_carbamate',
                                 'rdkit.Chem.rdMolTransforms.CanonicalizeConformer', 'rdkit.Chem.InchiToInchiKey',
                                 'rdkit.Chem.Descriptors.fr_Imine', 'rdkit.Chem.Descriptors.fr_imide',
                                 'rdkit.Chem.AllChem.CalcChi1n',
                                 'rdkit.DataStructs.cDataStructs.RusselSimilarityNeighbors_sparse',
                                 'rdkit.Chem.AllChem.GetUFFBondStretchParams',
                                 'rdkit.Chem.rdqueries.InNRingsEqualsQueryAtom', 'rdkit.Chem.GraphDescriptors.Chi3v',
                                 'rdkit.Chem.rdqueries.MinRingSizeEqualsQueryAtom',
                                 'rdkit.Chem.rdReducedGraphs.GenerateMolExtendedReducedGraph',
                                 'rdkit.Chem.AllChem.ETKDG', 'rdkit.Chem.rdmolfiles.MolToFASTA',
                                 'rdkit.Chem.AllChem.RingBondCountLessQueryAtom',
                                 'rdkit.Chem.Descriptors.MaxPartialCharge', 'rdkit.Chem.Descriptors.BertzCT',
                                 'rdkit.Chem.AllChem.FindAllSubgraphsOfLengthMToN', 'rdkit.ML.InfoTheory.InfoEntropy',
                                 'rdkit.Chem.Fragments.fr_barbitur', 'rdkit.DataStructs.CreateFromBinaryText',
                                 'rdkit.Chem.GraphDescriptors.Chi1n', 'rdkit.Chem.AllChem.CalcChi0n',
                                 'rdkit.Chem.rdForceFieldHelpers.GetUFFBondStretchParams',
                                 'rdkit.Chem.AllChem.MassEqualsQueryAtom', 'rdkit.Chem.MolFromMol2Block',
                                 'rdkit.Chem.Descriptors.fr_SH', 'rdkit.Chem.EState.EState_VSA.VSA_EState5',
                                 'rdkit.Chem.Descriptors.fr_benzene', 'rdkit.Chem.Fragments.fr_ketone_Topliss',
                                 'rdkit.Chem.rdChemReactions.ReactionToSmarts',
                                 'rdkit.Chem.AllChem.HybridizationEqualsQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.CalcNPR1', 'rdkit.Chem.CreateAtomStringPropertyList',
                                 'rdkit.Chem.Descriptors.RingCount', 'rdkit.Chem.rdmolfiles.MolToSequence',
                                 'rdkit.Chem.rdqueries.ExplicitDegreeEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.GetMorganFingerprint', 'rdkit.Chem.Fragments.fr_COO2',
                                 'rdkit.Chem.AllChem.molzip', 'rdkit.DataStructs.DiceSimilarityNeighbors_sparse',
                                 'rdkit.Chem.rdmolops.ReplaceSubstructs',
                                 'rdkit.DataStructs.cDataStructs.BulkMcConnaugheySimilarity', 'rdkit.Chem.Kekulize',
                                 'rdkit.Chem.AllChem.TransformConformer', 'rdkit.Chem.inchi.MolBlockToInchiAndAuxInfo',
                                 'rdkit.Chem.AllChem.GenerateMolExtendedReducedGraph', 'rdkit.Chem.Fragments.fr_HOCCN',
                                 'rdkit.Chem.rdMolEnumerator.Enumerate', 'rdkit.Chem.Descriptors.MinPartialCharge',
                                 'rdkit.Chem.GraphDescriptors.HallKierAlpha',
                                 'rdkit.DataStructs.cDataStructs.BulkAllBitSimilarity', 'rdkit.Chem.MolSurf.SlogP_VSA4',
                                 'rdkit.Chem.AllChem.UFFHasAllMoleculeParams', 'rdkit.Chem.Descriptors.VSA_EState8',
                                 'rdkit.Chem.FindPotentialStereo', 'rdkit.Chem.Descriptors.Chi0',
                                 'rdkit.Chem.Descriptors.fr_barbitur',
                                 'rdkit.DataStructs.cDataStructs.AllProbeBitsMatch',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA3', 'rdkit.Chem.Fragments.fr_amidine',
                                 'rdkit.Chem.AllChem.MurckoDecompose', 'rdkit.Chem.GraphDescriptors.BalabanJ',
                                 'rdkit.Chem.Descriptors.setupAUTOCorrDescriptors', 'rdkit.Chem.Fragments.fr_benzene',
                                 'rdkit.Chem.Descriptors.fr_dihydropyridine', 'rdkit.Chem.Descriptors.SMR_VSA1',
                                 'rdkit.Chem.rdqueries.RingBondCountEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.FindRingFamilies', 'rdkit.Chem.Descriptors.fr_COO',
                                 'rdkit.Chem.rdmolfiles.MolToV3KMolFile', 'rdkit.Chem.CreateAtomBoolPropertyList',
                                 'rdkit.DataStructs.cDataStructs.OnBitProjSimilarity',
                                 'rdkit.Chem.AllChem.DetectChemistryProblems', 'rdkit.Chem.rdmolfiles.MolToTPLBlock',
                                 'rdkit.Chem.Descriptors.EState_VSA8',
                                 'rdkit.Chem.rdmolfiles.CreateAtomStringPropertyList', 'rdkit.Chem.Descriptors.fr_Ar_N',
                                 'rdkit.Chem.Lipinski.FractionCSP3', 'rdkit.Chem.Fragments.fr_isocyan',
                                 'rdkit.Chem.AllChem.ReactionFromRxnBlock', 'rdkit.Chem.AllChem.WedgeMolBonds',
                                 'rdkit.DataStructs.cDataStructs.BulkRusselSimilarity',
                                 'rdkit.DataStructs.FoldToTargetDensity', 'rdkit.Chem.MolFromSmiles',
                                 'rdkit.Chem.LogErrorMsg', 'rdkit.Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles',
                                 'rdkit.Chem.AllChem.CalcNPR2', 'rdkit.Chem.Fragments.fr_quatN',
                                 'rdkit.Chem.AllChem.LogErrorMsg', 'rdkit.Chem.GraphDescriptors.Kappa2',
                                 'rdkit.Chem.AllChem.CalcNumUnspecifiedAtomStereoCenters', 'rdkit.Chem.rdDistGeom.ETDG',
                                 'rdkit.Chem.rdmolops.LayeredFingerprint', 'rdkit.Chem.Descriptors.NumRotatableBonds',
                                 'rdkit.Chem.AtomFromSmarts', 'rdkit.Chem.rdShapeHelpers.ShapeProtrudeDist',
                                 'rdkit.Chem.FindAtomEnvironmentOfRadiusN',
                                 'rdkit.Chem.AllChem.AssignStereochemistryFrom3D', 'rdkit.Chem.MolSurf.SMR_VSA5',
                                 'rdkit.Chem.AllChem.ETDG', 'rdkit.Chem.rdmolops.GetAdjacencyMatrix',
                                 'rdkit.Chem.SetAromaticity', 'rdkit.Chem.Descriptors.EState_VSA11',
                                 'rdkit.Chem.MolSurf.SlogP_VSA2',
                                 'rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure',
                                 'rdkit.Chem.AllChem.FindAllSubgraphsOfLengthN', 'rdkit.Chem.Fragments.fr_imidazole',
                                 'rdkit.Chem.AllChem.ComputePrincipalAxesAndMomentsFromGyrationMatrix',
                                 'rdkit.Chem.MolSurf.SMR_VSA6', 'rdkit.Chem.AllChem.XAtomQueryAtom',
                                 'rdkit.Chem.rdShapeHelpers.ComputeConfDimsAndOffset',
                                 'rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment',
                                 'rdkit.DataStructs.cDataStructs.TanimotoSimilarityNeighbors_sparse',
                                 'rdkit.Chem.AllChem.GetAtomFeatures', 'rdkit.Chem.AllChem.UFFOptimizeMoleculeConfs',
                                 'rdkit.Chem.EState.EStateIndices', 'rdkit.Chem.EState.MinEStateIndex',
                                 'rdkit.Chem.AllChem.AAtomQueryAtom', 'rdkit.Chem.AllChem.Compute2DCoords',
                                 'rdkit.Chem.rdMolTransforms.GetDihedralDeg', 'rdkit.Chem.Descriptors.fr_Ar_COO',
                                 'rdkit.Chem.AllChem.AtomNumGreaterQueryAtom', 'rdkit.Chem.Fragments.fr_COO',
                                 'rdkit.Chem.rdMolDescriptors.GetUSRCAT', 'rdkit.Chem.Fragments.fr_C_S',
                                 'rdkit.Chem.FragmentOnBRICSBonds', 'rdkit.Chem.MolSurf.PEOE_VSA3',
                                 'rdkit.Chem.AllChem.FormalChargeGreaterQueryAtom', 'rdkit.RDLogger.DisableLog',
                                 'rdkit.Chem.rdmolops.Kekulize',
                                 'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarityNeighbors_sparse',
                                 'rdkit.Chem.MolSurf.PEOE_VSA5', 'rdkit.Chem.AllChem.GetDefaultAdjustParams',
                                 'rdkit.Chem.WedgeMolBonds', 'rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMolecule',
                                 'rdkit.Chem.AllChem.AddHs', 'rdkit.Chem.rdqueries.HasBitVectPropWithValueQueryAtom',
                                 'rdkit.Chem.ClearMolSubstanceGroups', 'rdkit.Chem.AllChem.ReactionToMolecule',
                                 'rdkit.Chem.rdmolops.AssignStereochemistryFrom3D',
                                 'rdkit.Chem.AllChem.ReactionMetadataToPNGFile',
                                 'rdkit.Chem.AllChem.ExplicitValenceEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.XHAtomQueryAtom', 'rdkit.Chem.Descriptors.fr_Ndealkylation2',
                                 'rdkit.Chem.Descriptors.SlogP_VSA11', 'rdkit.Chem.AllChem.CalcNumAmideBonds',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAmideBonds', 'rdkit.Chem.Descriptors.SMR_VSA6',
                                 'rdkit.Chem.rdMolDescriptors.CalcKappa1', 'rdkit.Chem.AllChem.SanitizeRxn',
                                 'rdkit.Chem.Descriptors.fr_ArN', 'rdkit.Chem.Descriptors.SMR_VSA3',
                                 'rdkit.Chem.AllChem.AHAtomQueryAtom', 'rdkit.Chem.AllChem.MolToRandomSmilesVect',
                                 'rdkit.Chem.rdmolfiles.MolToSmiles', 'rdkit.Chem.rdqueries.IsAromaticQueryAtom',
                                 'rdkit.Chem.Fragments.fr_ester', 'rdkit.Chem.AllChem.SetHybridization',
                                 'rdkit.Chem.AllChem.CalcNumSaturatedCarbocycles',
                                 'rdkit.Chem.rdMolDescriptors.CalcNPR2', 'rdkit.Chem.AllChem.CanonicalizeConformer',
                                 'rdkit.DataStructs.cDataStructs.NumBitsInCommon', 'rdkit.Chem.rdmolfiles.MolToXYZFile',
                                 'rdkit.Chem.AllChem.CalcCoulombMat', 'rdkit.Chem.Fragments.fr_epoxide',
                                 'rdkit.Chem.UnfoldedRDKFingerprintCountBased', 'rdkit.Chem.MolSurf.PEOE_VSA4',
                                 'rdkit.Chem.rdMolChemicalFeatures.BuildFeatureFactory', 'rdkit.Chem.MolFromHELM',
                                 'rdkit.Chem.rdqueries.FormalChargeLessQueryAtom', 'rdkit.Chem.Fragments.fr_lactam',
                                 'rdkit.Chem.rdqueries.HybridizationGreaterQueryAtom',
                                 'rdkit.DataStructs.BraunBlanquetSimilarityNeighbors_sparse',
                                 'rdkit.Chem.Descriptors.TPSA', 'rdkit.Chem.AllChem.ETKDGv2',
                                 'rdkit.Chem.GetShortestPath', 'rdkit.Chem.MolSurf.SMR_VSA9',
                                 'rdkit.Chem.rdqueries.FormalChargeGreaterQueryAtom', 'rdkit.Chem.AllChem.Enumerate',
                                 'rdkit.DataStructs.BulkCosineSimilarity', 'rdkit.Chem.AllChem.MassLessQueryAtom',
                                 'rdkit.Chem.Descriptors.fr_COO2', 'rdkit.Chem.AllChem.CreateAtomDoublePropertyList',
                                 'rdkit.DataStructs.CreateFromBitString',
                                 'rdkit.Chem.AllChem.HasBitVectPropWithValueQueryAtom',
                                 'rdkit.Chem.AllChem.GetAngleRad', 'rdkit.Chem.rdMolTransforms.SetAngleDeg',
                                 'rdkit.Chem.SupplierFromFilename', 'rdkit.Chem.EState.MaxEStateIndex',
                                 'rdkit.Chem.rdqueries.HCountEqualsQueryAtom',
                                 'rdkit.DataStructs.cDataStructs.AsymmetricSimilarityNeighbors_sparse',
                                 'rdkit.Chem.GetPeriodicTable', 'rdkit.DataStructs.BulkRogotGoldbergSimilarity',
                                 'rdkit.Chem.Fragments.fr_aryl_methyl', 'rdkit.Chem.GetMostSubstitutedCoreMatch',
                                 'rdkit.Chem.AllChem.DetectBondStereochemistry', 'rdkit.rdBase.DisableLog',
                                 'rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect',
                                 'rdkit.Chem.rdMolAlign.CalcRMS', 'rdkit.Chem.AllChem.ReactionToSmiles',
                                 'rdkit.Chem.AllChem.CanonicalizeMol', 'rdkit.Chem.AllChem.MissingChiralTagQueryAtom',
                                 'rdkit.Chem.Descriptors3D.Eccentricity', 'rdkit.Chem.AllChem.SetAngleDeg',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumAtomStereoCenters',
                                 'rdkit.Chem.Descriptors.PEOE_VSA4', 'rdkit.Chem.MolSurf.SlogP_VSA8',
                                 'rdkit.Chem.Descriptors.PEOE_VSA8', 'rdkit.Chem.FastFindRings',
                                 'rdkit.Chem.rdMolDescriptors.CalcWHIM',
                                 'rdkit.Chem.rdqueries.ExplicitValenceLessQueryAtom',
                                 'rdkit.Chem.AllChem.MMFFOptimizeMolecule', 'rdkit.Chem.rdmolfiles.CanonicalRankAtoms',
                                 'rdkit.Chem.rdMolTransforms.GetAngleDeg', 'rdkit.Chem.AllChem.GetUFFTorsionParams',
                                 'rdkit.Chem.Lipinski.NumSaturatedRings', 'rdkit.Chem.MolFragmentToSmarts',
                                 'rdkit.Chem.Descriptors.NumAromaticCarbocycles', 'rdkit.Chem.AllChem.ComputeUnionBox',
                                 'rdkit.Chem.AllChem.CalcAUTOCORR2D', 'rdkit.Chem.Descriptors3D.PMI3',
                                 'rdkit.Chem.GetMolSubstanceGroups', 'rdkit.Chem.rdmolops.DeleteSubstructs',
                                 'rdkit.Chem.AllChem.MHAtomQueryAtom', 'rdkit.Chem.rdqueries.XAtomQueryAtom',
                                 'rdkit.DataStructs.CosineSimilarity', 'rdkit.Chem.MolBlockToInchiAndAuxInfo',
                                 'rdkit.Chem.rdmolfiles.MetadataFromPNGFile', 'rdkit.Chem.rdchem.tossit',
                                 'rdkit.Chem.MetadataFromPNGFile', 'rdkit.Chem.Descriptors.fr_pyridine',
                                 'rdkit.Chem.rdMolDescriptors.GetAtomPairCode',
                                 'rdkit.Chem.rdchem.AddMolSubstanceGroup', 'rdkit.Chem.rdmolfiles.AtomFromSmarts',
                                 'rdkit.DataStructs.TanimotoSimilarityNeighbors_sparse',
                                 'rdkit.Chem.Descriptors.fr_imidazole', 'rdkit.Chem.Descriptors.VSA_EState1',
                                 'rdkit.Chem.rdmolfiles.MolFromTPLBlock', 'rdkit.Chem.Descriptors.Kappa1',
                                 'rdkit.Chem.EState.MinAbsEStateIndex',
                                 'rdkit.Chem.AllChem.CalcNumAliphaticCarbocycles',
                                 'rdkit.Geometry.rdGeometry.FindGridTerminalPoints',
                                 'rdkit.DataStructs.OffBitsInCommon', 'rdkit.RDLogger.EnableLog',
                                 'rdkit.Chem.Descriptors.fr_isocyan',
                                 'rdkit.DataStructs.cDataStructs.TverskySimilarity',
                                 'rdkit.DataStructs.BulkAllBitSimilarity', 'rdkit.Chem.rdMolTransforms.SetBondLength',
                                 'rdkit.Chem.MolSurf.PEOE_VSA11',
                                 'rdkit.DataStructs.KulczynskiSimilarityNeighbors_sparse', 'rdkit.Chem.AtomFromSmiles',
                                 'rdkit.Chem.ChemicalFeatures.BuildFeatureFactoryFromString',
                                 'rdkit.Chem.rdForceFieldHelpers.GetUFFAngleBendParams',
                                 'rdkit.Chem.AllChem.FindMolChiralCenters', 'rdkit.Chem.MolToV3KMolBlock',
                                 'rdkit.Chem.AllChem.ConstrainedEmbed', 'rdkit.Chem.rdMolDescriptors.CalcAUTOCORR2D',
                                 'rdkit.Chem.rdChemReactions.ReactionFromRxnBlock', 'rdkit.Chem.Fragments.fr_furan',
                                 'rdkit.Chem.AllChem.CalcPMI2', 'rdkit.Chem.AllChem.CalcChiNv',
                                 'rdkit.Chem.rdmolfiles.BondFromSmiles', 'rdkit.Chem.AllChem.CalcAsphericity',
                                 'rdkit.Chem.Descriptors.fr_phos_acid', 'rdkit.Chem.AllChem.GetChemDrawRxnAdjustParams',
                                 'rdkit.Chem.Graphs.CharacteristicPolynomial', 'rdkit.Chem.Descriptors.PEOE_VSA9',
                                 'rdkit.DataStructs.ComputeL1Norm', 'rdkit.Chem.AllChem.DetectBondStereoChemistry',
                                 'rdkit.Chem.AllChem.GetShortestPath', 'rdkit.Chem.Descriptors.fr_nitroso',
                                 'rdkit.Chem.GraphDescriptors.Chi3n', 'rdkit.Chem.AllChem.MolsFromPNGString',
                                 'rdkit.Chem.AllChem.InNRingsLessQueryAtom', 'rdkit.rdBase.LogMessage',
                                 'rdkit.Chem.AllChem.GetConformerRMSMatrix', 'rdkit.Chem.RemoveStereochemistry',
                                 'rdkit.Chem.rdchem.ClearMolSubstanceGroups', 'rdkit.Chem.QED.weights_none',
                                 'rdkit.Chem.QED.ads', 'rdkit.Chem.rdmolfiles.MolFromSequence',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumHeteroatoms', 'rdkit.Chem.MolFromRDKitSVG',
                                 'rdkit.Chem.rdmolops.WedgeBond', 'rdkit.Chem.AllChem.Kekulize',
                                 'rdkit.Chem.Fragments.fr_N_O', 'rdkit.Chem.AllChem.UpdateProductsStereochemistry',
                                 'rdkit.Chem.AllChem.OptimizeMoleculeConfs', 'rdkit.rdBase.LogStatus',
                                 'rdkit.Chem.AllChem.GetErGFingerprint', 'rdkit.Chem.EState.EState_VSA.EState_VSA4',
                                 'rdkit.Chem.Descriptors.fr_bicyclic', 'rdkit.Chem.MolToXYZFile',
                                 'rdkit.Chem.AllChem.MolFromSmarts', 'rdkit.Chem.Descriptors.fr_prisulfonamd',
                                 'rdkit.Chem.rdqueries.HasDoublePropWithValueQueryBond',
                                 'rdkit.Chem.Descriptors.SlogP_VSA8', 'rdkit.Chem.AllChem.MolFragmentToCXSmiles',
                                 'rdkit.Chem.AllChem.EmbedMultipleConfs', 'rdkit.rdBase.EnableLog',
                                 'rdkit.Chem.Fragments.fr_diazo', 'rdkit.Chem.rdqueries.HCountLessQueryAtom',
                                 'rdkit.Chem.rdDepictor.Compute2DCoords', 'rdkit.Chem.Descriptors.fr_Ndealkylation1',
                                 'rdkit.Chem.Descriptors.fr_phenol', 'rdkit.Chem.AllChem.AlignMol',
                                 'rdkit.Chem.AllChem.GetUSRDistributions',
                                 'rdkit.Chem.rdMolDescriptors.MakePropertyRangeQuery',
                                 'rdkit.Chem.Descriptors.fr_allylic_oxid', 'rdkit.Chem.Descriptors.fr_epoxide',
                                 'rdkit.Chem.rdmolops.GetMolFrags', 'rdkit.Chem.rdmolfiles.MolFromRDKitSVG',
                                 'rdkit.Chem.Descriptors.SMR_VSA5', 'rdkit.Chem.GraphDescriptors.Chi1v',
                                 'rdkit.Chem.GraphDescriptors.Kappa3', 'rdkit.Chem.GraphDescriptors.Chi4v',
                                 'rdkit.Chem.rdchem.LogWarningMsg', 'rdkit.DataStructs.NumBitsInCommon',
                                 'rdkit.DataStructs.BulkMcConnaugheySimilarity', 'rdkit.Chem.rdmolfiles.MolToPDBBlock',
                                 'rdkit.Chem.MolFromFASTA', 'rdkit.Chem.AllChem.MakePropertyRangeQuery',
                                 'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsEqualsQueryAtom',
                                 'rdkit.Chem.Descriptors.fr_aryl_methyl', 'rdkit.Chem.rdChemReactions.SanitizeRxn',
                                 'rdkit.Chem.Lipinski.NumHAcceptors', 'rdkit.Chem.rdmolfiles.MolToRandomSmilesVect',
                                 'rdkit.Chem.rdqueries.InNRingsGreaterQueryAtom',
                                 'rdkit.Chem.rdChemReactions.ReactionFromRxnFile',
                                 'rdkit.Chem.rdMolDescriptors.CalcEccentricity',
                                 'rdkit.DataStructs.cDataStructs.SokalSimilarity', 'rdkit.Chem.AllChem.CalcPMI3',
                                 'rdkit.Chem.rdmolops.RDKFingerprint', 'rdkit.Chem.DeleteSubstructs',
                                 'rdkit.Chem.AllChem.LogWarningMsg',
                                 'rdkit.Chem.rdqueries.NumHeteroatomNeighborsLessQueryAtom',
                                 'rdkit.Chem.AllChem.TotalValenceLessQueryAtom',
                                 'rdkit.Chem.AllChem.CalcNumSaturatedHeterocycles',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA5', 'rdkit.Chem.MolSurf.PEOE_VSA9',
                                 'rdkit.Chem.rdmolfiles.MolFromHELM',
                                 'rdkit.Chem.rdmolops.AssignAtomChiralTagsFromStructure',
                                 'rdkit.Chem.FindPotentialStereoBonds', 'rdkit.Chem.Descriptors.fr_diazo',
                                 'rdkit.Chem.AllChem.CalcChi2n', 'rdkit.Chem.MolToInchiKey',
                                 'rdkit.Chem.Descriptors.fr_thiazole', 'rdkit.Chem.MolFromTPLFile',
                                 'rdkit.Chem.AllChem.ExplicitDegreeLessQueryAtom', 'rdkit.Chem.AllChem.SetDihedralDeg',
                                 'rdkit.Chem.rdMolTransforms.SetAngleRad',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumLipinskiHBA',
                                 'rdkit.Chem.rdChemReactions.ReactionToMolecule',
                                 'rdkit.Chem.AllChem.FragmentOnSomeBonds',
                                 'rdkit.Chem.AllChem.FindPotentialStereoBonds',
                                 'rdkit.Chem.AllChem.FindAllPathsOfLengthN',
                                 'rdkit.Chem.rdmolops.AdjustQueryProperties', 'rdkit.Chem.Fragments.fr_sulfone',
                                 'rdkit.Chem.AddHs', 'rdkit.Chem.AllChem.MolAddRecursiveQueries',
                                 'rdkit.Chem.Descriptors.EState_VSA1', 'rdkit.Chem.rdMolDescriptors.CalcNumSpiroAtoms',
                                 'rdkit.Chem.AllChem.MCFF_GetFeaturesForMol', 'rdkit.Chem.rdmolfiles.MolFromSmiles',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi3n', 'rdkit.Chem.AllChem.MolToJSON',
                                 'rdkit.Chem.rdMolDescriptors.CalcHallKierAlpha',
                                 'rdkit.Chem.rdqueries.ExplicitDegreeLessQueryAtom',
                                 'rdkit.Chem.AllChem.CalcNumAromaticHeterocycles', 'rdkit.Chem.AllChem.srETKDGv3',
                                 'rdkit.Chem.AllChem.MolFragmentToSmarts', 'rdkit.Chem.Fragments.fr_sulfonamd',
                                 'rdkit.Chem.Fragments.fr_tetrazole', 'rdkit.Chem.MolToFASTA',
                                 'rdkit.Chem.MolSurf.PEOE_VSA13', 'rdkit.Chem.rdMolDescriptors.CalcChi1v',
                                 'rdkit.Chem.Descriptors.SlogP_VSA12', 'rdkit.Chem.Descriptors.SlogP_VSA1',
                                 'rdkit.Chem.AllChem.SetTerminalAtomCoords',
                                 'rdkit.Chem.rdqueries.TotalValenceGreaterQueryAtom', 'rdkit.Chem.AllChem.MolFromHELM',
                                 'rdkit.Chem.QuickSmartsMatch', 'rdkit.Chem.AllChem.EnumerateLibraryFromReaction',
                                 'rdkit.Chem.rdMolDescriptors.GetUSRScore', 'rdkit.Chem.Descriptors.fr_aniline',
                                 'rdkit.Chem.rdMolTransforms.GetAngleRad',
                                 'rdkit.Chem.AllChem.MMFFGetMoleculeForceField',
                                 'rdkit.Chem.rdReducedGraphs.GetErGFingerprint',
                                 'rdkit.Chem.rdMolTransforms.TransformConformer', 'rdkit.Chem.MolFromPDBBlock',
                                 'rdkit.Chem.rdchem.GetDefaultPickleProperties', 'rdkit.Chem.Descriptors.Ipc',
                                 'rdkit.DataStructs.cDataStructs.McConnaugheySimilarity',
                                 'rdkit.Chem.Descriptors.VSA_EState7', 'rdkit.Chem.QED.namedtuple',
                                 'rdkit.Chem.PathToSubmol', 'rdkit.Geometry.rdGeometry.ComputeDihedralAngle',
                                 'rdkit.ML.InfoTheory.rdInfoTheory.InfoGain', 'rdkit.Chem.rdmolops.GetSSSR',
                                 'rdkit.Chem.AllChem.JSONToMols', 'rdkit.Chem.MolToCXSmiles',
                                 'rdkit.Chem.AllChem.AddMetadataToPNGString', 'rdkit.Chem.Descriptors.Chi4v',
                                 'rdkit.Chem.GetSSSR', 'rdkit.Chem.rdmolops.Get3DDistanceMatrix',
                                 'rdkit.Chem.MolSurf.SlogP_VSA3',
                                 'rdkit.Chem.rdDepictor.GenerateDepictionMatching3DStructure',
                                 'rdkit.Chem.AllChem.FastFindRings', 'rdkit.Chem.rdmolops.SetConjugation',
                                 'rdkit.Chem.AllChem.CalcNumHeteroatoms',
                                 'rdkit.Chem.rdMolDescriptors.GetMorganFingerprint', 'rdkit.Chem.Descriptors.SMR_VSA7',
                                 'rdkit.Chem.rdmolops.MolAddRecursiveQueries',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState1', 'rdkit.Chem.MolFromSequence',
                                 'rdkit.Chem.Descriptors3D.PMI2', 'rdkit.Chem.Descriptors.fr_hdrzone',
                                 'rdkit.Chem.AllChem.MolToFASTA',
                                 'rdkit.Chem.AllChem.AssignAtomChiralTagsFromMolParity',
                                 'rdkit.Chem.rdChemReactions.HasAgentTemplateSubstructMatch',
                                 'rdkit.Chem.AllChem.RemoveStereochemistry', 'rdkit.Chem.Descriptors.NumAliphaticRings',
                                 'rdkit.DataStructs.BulkOnBitSimilarity', 'rdkit.Chem.AllChem.IsInRingQueryAtom',
                                 'rdkit.Chem.AllChem.GetUSRFromDistributions', 'rdkit.Chem.rdmolfiles.MolFromPDBFile',
                                 'rdkit.Chem.rdChemReactions.UpdateProductsStereochemistry',
                                 'rdkit.Chem.Fragments.fr_term_acetylene',
                                 'rdkit.Chem.AllChem.CanonicalRankAtomsInFragment', 'rdkit.Chem.Fragments.fr_sulfide',
                                 'rdkit.Chem.rdForceFieldHelpers.UFFOptimizeMolecule',
                                 'rdkit.Chem.AllChem.IsAromaticQueryAtom', 'rdkit.Chem.SetSupplementalSmilesLabel',
                                 'rdkit.Chem.MolSurf.SlogP_VSA5',
                                 'rdkit.Chem.rdChemReactions.HasReactionSubstructMatch',
                                 'rdkit.Chem.rdChemReactions.HasReactionAtomMapping',
                                 'rdkit.Chem.AllChem.HasBoolPropWithValueQueryAtom',
                                 'rdkit.Chem.rdqueries.NumAliphaticHeteroatomNeighborsGreaterQueryAtom',
                                 'rdkit.DataStructs.cDataStructs.AsymmetricSimilarity',
                                 'rdkit.Chem.AllChem.AtomNumLessQueryAtom',
                                 'rdkit.Chem.rdqueries.ExplicitValenceGreaterQueryAtom',
                                 'rdkit.Chem.rdmolfiles.MolFragmentToSmiles', 'rdkit.Chem.AllChem.CalcLabuteASA',
                                 'rdkit.Chem.rdchem.SetAtomAlias', 'rdkit.Chem.AllChem.CalcChi4v',
                                 'rdkit.Chem.AllChem.GetAtomPairCode', 'rdkit.Chem.AllChem.MinRingSizeLessQueryAtom',
                                 'rdkit.Chem.Descriptors.MinAbsPartialCharge', 'rdkit.Chem.MolToPDBBlock',
                                 'rdkit.DataStructs.cDataStructs.CosineSimilarity', 'rdkit.Chem.Descriptors.SMR_VSA10',
                                 'rdkit.Chem.Fragments.fr_Al_OH', 'rdkit.Chem.Descriptors.PEOE_VSA12',
                                 'rdkit.Chem.Fragments.fr_priamide', 'rdkit.Chem.AllChem.GetUFFVdWParams',
                                 'rdkit.ML.InfoTheory.entropy.PyInfoGain',
                                 'rdkit.Chem.Fragments.fr_phenol_noOrthoHbond', 'rdkit.Chem.rdmolfiles.BondFromSmarts',
                                 'rdkit.Chem.MolSurf.PEOE_VSA14', 'rdkit.Chem.RDKFingerprint',
                                 'rdkit.Chem.Descriptors3D.NPR1', 'rdkit.DataStructs.BulkRusselSimilarity',
                                 'rdkit.Chem.rdMolTransforms.ComputeCentroid', 'rdkit.Chem.AllChem.MolToV3KMolBlock',
                                 'rdkit.Chem.AllChem.GetBondLength', 'rdkit.Chem.rdMolDescriptors.CalcPMI2',
                                 'rdkit.Chem.AddMolSubstanceGroup', 'rdkit.Chem.AllChem.MolFromSLN',
                                 'rdkit.Chem.AllChem.HasReactantTemplateSubstructMatch',
                                 'rdkit.Chem.AllChem.OptimizeMolecule', 'rdkit.Chem.AllChem.GetConformerRMS',
                                 'rdkit.Chem.Descriptors.fr_NH2', 'rdkit.Chem.SortMatchesByDegreeOfCoreSubstitution',
                                 'rdkit.Chem.rdMolDescriptors.CalcCrippenDescriptors',
                                 'rdkit.Chem.AllChem.CalcNumAtomStereoCenters', 'rdkit.Chem.Descriptors.PEOE_VSA2',
                                 'rdkit.Chem.AllChem.AddRecursiveQuery', 'rdkit.Chem.rdmolops.RemoveAllHs',
                                 'rdkit.Chem.AllChem.CanonSmiles', 'rdkit.Chem.Descriptors.fr_azo',
                                 'rdkit.Chem.AllChem.NumRadicalElectronsLessQueryAtom',
                                 'rdkit.Chem.rdmolfiles.CreateAtomDoublePropertyList',
                                 'rdkit.Chem.rdmolfiles.MolFromMol2File',
                                 'rdkit.Chem.rdMolDescriptors.GetUSRFromDistributions', 'rdkit.Chem.AllChem.CalcChi2v',
                                 'rdkit.Chem.rdqueries.AHAtomQueryAtom',
                                 'rdkit.DataStructs.KulczynskiSimilarityNeighbors',
                                 'rdkit.Chem.rdDistGeom.EmbedMultipleConfs',
                                 'rdkit.Chem.rdqueries.HasChiralTagQueryAtom', 'rdkit.Chem.rdMolDescriptors.CalcChi2n',
                                 'rdkit.Chem.Descriptors.fr_Al_OH_noTert', 'rdkit.Chem.Descriptors.fr_sulfone',
                                 'rdkit.Chem.AllChem.CreateAtomBoolPropertyList', 'rdkit.Chem.rdmolops.Cleanup',
                                 'rdkit.Chem.rdForceFieldHelpers.GetUFFInversionParams',
                                 'rdkit.Chem.rdmolops.RenumberAtoms',
                                 'rdkit.Chem.rdMolTransforms.ComputeCanonicalTransform', 'rdkit.Chem.Descriptors.Chi0v',
                                 'rdkit.DataStructs.BraunBlanquetSimilarity', 'rdkit.Chem.Fragments.fr_NH2',
                                 'rdkit.Chem.rdMolDescriptors.CalcChi0n', 'rdkit.Chem.rdchem.WrapLogs',
                                 'rdkit.Chem.rdmolfiles.MetadataFromPNGString',
                                 'rdkit.ML.InfoTheory.entropy.PyInfoEntropy',
                                 'rdkit.Chem.AllChem.GetCrippenO3AForProbeConfs',
                                 'rdkit.Chem.Descriptors.fr_unbrch_alkane',
                                 'rdkit.Chem.rdqueries.HybridizationLessQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.GetUSR', 'rdkit.Chem.Lipinski.RingCount',
                                 'rdkit.Chem.AllChem.namedtuple', 'rdkit.Chem.AllChem.QAtomQueryAtom',
                                 'rdkit.Chem.AllChem.CalcNumHBD', 'rdkit.Chem.Descriptors.NumHDonors',
                                 'rdkit.Chem.MolAddRecursiveQueries', 'rdkit.Chem.AllChem.CalcGETAWAY',
                                 'rdkit.Chem.ChemicalFeatures.GetAtomMatch', 'rdkit.Chem.MolsToJSON',
                                 'rdkit.Chem.MolSurf.SlogP_VSA12', 'rdkit.Chem.CombineMols',
                                 'rdkit.Chem.Descriptors.PEOE_VSA3', 'rdkit.DataStructs.cDataStructs.BitVectToFPSText',
                                 'rdkit.Chem.AllChem.GetCrippenO3A', 'rdkit.Chem.AllChem.MolFragmentToSmiles',
                                 'rdkit.Chem.MolSurf.SlogP_VSA6', 'rdkit.Chem.Descriptors3D.SpherocityIndex',
                                 'rdkit.Chem.CanonSmiles', 'rdkit.Chem.GraphDescriptors.Chi0',
                                 'rdkit.Chem.AllChem.ETKDGv3', 'rdkit.Chem.AllChem.GetUFFAngleBendParams',
                                 'rdkit.Chem.Descriptors.FpDensityMorgan2', 'rdkit.Chem.rdMolDescriptors.CalcChiNv',
                                 'rdkit.Chem.Descriptors.fr_alkyl_halide',
                                 'rdkit.Chem.rdmolfiles.AddMetadataToPNGString', 'rdkit.Geometry.UniformGrid3D',
                                 'rdkit.Chem.AllChem.GetMostSubstitutedCoreMatch',
                                 'rdkit.DataStructs.cDataStructs.BitVectToBinaryText',
                                 'rdkit.Chem.rdqueries.HCountGreaterQueryAtom',
                                 'rdkit.Geometry.rdGeometry.ComputeSignedDihedralAngle', 'rdkit.Chem.molzip',
                                 'rdkit.Chem.EState.AtomTypes.TypeAtoms', 'rdkit.Chem.AllChem.MolToSmarts',
                                 'rdkit.DataStructs.KulczynskiSimilarity', 'rdkit.Chem.AllChem.BCUT2D',
                                 'rdkit.Chem.AllChem.MinRingSizeGreaterQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.CalcMolFormula', 'rdkit.Chem.AllChem.GetUSR',
                                 'rdkit.Chem.rdDistGeom.ETKDGv2', 'rdkit.Chem.Crippen.MolLogP',
                                 'rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField',
                                 'rdkit.Chem.AllChem.AssignRadicals', 'rdkit.Chem.AllChem.ReplaceSidechains',
                                 'rdkit.Chem.AllChem.ShapeTanimotoDist', 'rdkit.Chem.AllChem.HCountLessQueryAtom',
                                 'rdkit.Chem.rdmolfiles.MolToHELM', 'rdkit.Chem.Descriptors.EState_VSA10',
                                 'rdkit.ML.InfoTheory.entropy.InfoGain', 'rdkit.Chem.Descriptors.fr_term_acetylene',
                                 'rdkit.Chem.AllChem.MolFromMol2File', 'rdkit.Chem.Fragments.fr_Imine',
                                 'rdkit.DataStructs.cDataStructs.BitVectToText', 'rdkit.Chem.rdMolDescriptors.CalcPhi',
                                 'rdkit.Chem.rdmolfiles.CreateAtomBoolPropertyList',
                                 'rdkit.Chem.Descriptors.fr_sulfide', 'rdkit.Chem.Lipinski.NumRotatableBonds',
                                 'rdkit.Chem.Lipinski.NumAliphaticRings', 'rdkit.Chem.EState.EState.EStateIndices',
                                 'rdkit.Chem.rdmolfiles.MolMetadataToPNGFile', 'rdkit.Chem.AllChem.MolFromPDBBlock',
                                 'rdkit.Chem.AllChem.SetSupplementalSmilesLabel',
                                 'rdkit.DataStructs.DiceSimilarityNeighbors',
                                 'rdkit.DataStructs.cDataStructs.BraunBlanquetSimilarity',
                                 'rdkit.Chem.Fragments.fr_guanido',
                                 'rdkit.Chem.AllChem.HasDoublePropWithValueQueryAtom', 'rdkit.Chem.Descriptors.MolMR',
                                 'rdkit.Chem.Descriptors.fr_isothiocyan', 'rdkit.Chem.Fragments.fr_Ar_N',
                                 'rdkit.Chem.LayeredFingerprint', 'rdkit.Chem.Descriptors.SlogP_VSA5',
                                 'rdkit.Chem.Fragments.fr_aniline', 'rdkit.Chem.AllChem.GetO3A',
                                 'rdkit.Chem.AllChem.QuickSmartsMatch', 'rdkit.Chem.Descriptors3D.InertialShapeFactor',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState10', 'rdkit.Chem.MolMetadataToPNGFile',
                                 'rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprint',
                                 'rdkit.Chem.rdqueries.MinRingSizeGreaterQueryAtom', 'rdkit.Chem.GetAdjacencyMatrix',
                                 'rdkit.Chem.AllChem.MolFromSmiles', 'rdkit.Chem.Lipinski.NOCount',
                                 'rdkit.Chem.rdqueries.HasStringPropWithValueQueryBond',
                                 'rdkit.Chem.Fragments.fr_alkyl_carbamate', 'rdkit.Chem.AllChem.AssignCIPLabels',
                                 'rdkit.DataStructs.cDataStructs.DiceSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.RemoveAllHs', 'rdkit.RDLogger.AttachFileToLog',
                                 'rdkit.Chem.AllChem.ReplaceCore', 'rdkit.Chem.rdMolDescriptors.GetFeatureInvariants',
                                 'rdkit.Chem.Fragments.fr_oxime', 'rdkit.Chem.AllChem.AddMetadataToPNGFile',
                                 'rdkit.Chem.SetBondStereoFromDirections', 'rdkit.Chem.Fragments.fr_Al_COO',
                                 'rdkit.Chem.rdMolDescriptors.GetHashedMorganFingerprint',
                                 'rdkit.Chem.rdmolops.CombineMols', 'rdkit.Chem.rdmolfiles.MolFromMolBlock',
                                 'rdkit.Chem.Fragments.fr_aldehyde', 'rdkit.Chem.AllChem.CalcEEMcharges',
                                 'rdkit.Chem.rdqueries.TotalDegreeLessQueryAtom', 'rdkit.Chem.AllChem.MAtomQueryAtom',
                                 'rdkit.Chem.AllChem.GetFormalCharge', 'rdkit.ML.InfoTheory.InfoGain',
                                 'rdkit.Chem.rdMolAlign.GetO3A',
                                 'rdkit.DataStructs.cDataStructs.RogotGoldbergSimilarity',
                                 'rdkit.Chem.rdqueries.NumHeteroatomNeighborsEqualsQueryAtom',
                                 'rdkit.Chem.rdqueries.MissingChiralTagQueryAtom', 'rdkit.Chem.AllChem.GetSymmSSSR',
                                 'rdkit.Chem.AllChem.ReactionFromPNGFile', 'rdkit.Chem.AllChem.ReactionToSmarts',
                                 'rdkit.Chem.AllChem.SetAtomValue', 'rdkit.Chem.AllChem.CombineMols',
                                 'rdkit.Chem.Lipinski.NumAromaticCarbocycles', 'rdkit.Chem.Fragments.fr_ketone',
                                 'rdkit.Chem.Lipinski.HeavyAtomCount', 'rdkit.Chem.rdinchi.InchiToInchiKey',
                                 'rdkit.Chem.Descriptors.HeavyAtomMolWt', 'rdkit.Chem.Fragments.fr_nitrile',
                                 'rdkit.Chem.AllChem.NumHeteroatomNeighborsLessQueryAtom',
                                 'rdkit.Chem.rdmolops.GetMostSubstitutedCoreMatch', 'rdkit.Chem.Descriptors.fr_ester',
                                 'rdkit.Chem.GetDefaultPickleProperties',
                                 'rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex',
                                 'rdkit.Chem.AllChem.IsAliphaticQueryAtom',
                                 'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.CalcNumRings', 'rdkit.Chem.rdMolChemicalFeatures.GetAtomMatch',
                                 'rdkit.Chem.AllChem.CalcAUTOCORR3D', 'rdkit.Chem.Descriptors.NumSaturatedCarbocycles',
                                 'rdkit.Chem.AllChem.MolFromQuerySLN', 'rdkit.Chem.AllChem.CalcKappa2',
                                 'rdkit.Chem.FindUniqueSubgraphsOfLengthN', 'rdkit.Chem.MolFragmentToCXSmiles',
                                 'rdkit.Chem.EState.EState_VSA.VSA_EState2', 'rdkit.Chem.rdqueries.HasPropQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumHBA', 'rdkit.Chem.rdmolops.FragmentOnSomeBonds',
                                 'rdkit.Chem.AllChem.CalcChi4n',
                                 'rdkit.DataStructs.cDataStructs.RusselSimilarityNeighbors',
                                 'rdkit.Chem.AllChem.GetAtomPairFingerprint', 'rdkit.Chem.AllChem.ReactionFromMolecule',
                                 'rdkit.Chem.rdMolDescriptors.CalcNumHeterocycles', 'rdkit.Chem.AllChem.MolToPDBFile',
                                 'rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect',
                                 'rdkit.Chem.rdqueries.NumRadicalElectronsGreaterQueryAtom',
                                 'rdkit.Chem.rdchem.LogErrorMsg', 'rdkit.Chem.AllChem.CalcNumAliphaticHeterocycles',
                                 'rdkit.Chem.MolsFromPNGFile', 'rdkit.DataStructs.BulkDiceSimilarity',
                                 'rdkit.Chem.rdDepictor.Compute2DCoordsMimicDistmat',
                                 'rdkit.Chem.AllChem.TotalValenceEqualsQueryAtom',
                                 'rdkit.Chem.AllChem.SetPreferCoordGen', 'rdkit.Chem.inchi.MolToInchiKey',
                                 'rdkit.Chem.rdchem.GetMolSubstanceGroupWithIdx',
                                 'rdkit.Chem.AllChem.CreateStereoGroup', 'rdkit.Chem.Crippen.MolMR',
                                 'rdkit.Chem.EState.EState_VSA.EState_VSA8',
                                 'rdkit.Chem.rdqueries.NumRadicalElectronsLessQueryAtom',
                                 'rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect',
                                 'rdkit.Chem.Fragments.fr_hdrzine', 'rdkit.Chem.Descriptors.NumRadicalElectrons',
                                 'rdkit.Chem.MolToV3KMolFile', 'rdkit.Chem.AllChem.SetAngleRad',
                                 'rdkit.Chem.Descriptors.fr_oxazole', 'rdkit.Chem.AllChem.KDG',
                                 'rdkit.Chem.rdmolops.GetDistanceMatrix', 'rdkit.Chem.rdmolops.DetectChemistryProblems',
                                 'rdkit.Chem.MolBlockToInchi',
                                 'rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties',
                                 'rdkit.Chem.rdchem.SetSupplementalSmilesLabel',
                                 'rdkit.Chem.AllChem.PreprocessReaction',
                                 'rdkit.DataStructs.cDataStructs.CosineSimilarityNeighbors',
                                 'rdkit.Chem.rdqueries.IsUnsaturatedQueryAtom',
                                 'rdkit.Chem.rdReducedGraphs.GenerateErGFingerprintForReducedGraph',
                                 'rdkit.Chem.rdmolfiles.MolFromPNGFile', 'rdkit.Chem.AllChem.CalcKappa3',
                                 'rdkit.DataStructs.BulkKulczynskiSimilarity', 'rdkit.Chem.AllChem.DeleteSubstructs',
                                 'rdkit.Chem.rdChemReactions.ReactionFromSmarts',
                                 'rdkit.DataStructs.cDataStructs.AllBitSimilarity',
                                 'rdkit.Chem.AllChem.Compute2DCoordsForReaction',
                                 'rdkit.Chem.AllChem.MolMetadataToPNGFile', 'rdkit.Chem.AllChem.MolFromInchi',
                                 'rdkit.Chem.AllChem.HybridizationLessQueryAtom', 'rdkit.Chem.Lipinski.NumHDonors',
                                 'rdkit.Chem.Fragments.fr_isothiocyan', 'rdkit.Chem.AllChem.FindPotentialStereo',
                                 'rdkit.Chem.MolToPDBFile', 'rdkit.Chem.rdqueries.AAtomQueryAtom',
                                 'rdkit.Chem.AllChem.NumRadicalElectronsEqualsQueryAtom',
                                 'rdkit.Chem.FindAllPathsOfLengthN', 'rdkit.Chem.AllChem.ReactionToRxnBlock',
                                 'rdkit.Chem.AllChem.CalcPhi', 'rdkit.Chem.Fragments.fr_piperzine',
                                 'rdkit.Chem.rdmolfiles.MolToTPLFile', 'rdkit.Chem.Descriptors.SlogP_VSA3',
                                 'rdkit.Chem.Descriptors.fr_aldehyde',
                                 'rdkit.Chem.AllChem.SetDoubleBondNeighborDirections',
                                 'rdkit.ML.InfoTheory.entropy.InfoEntropy', 'rdkit.Chem.GetDistanceMatrix',
                                 'rdkit.Chem.AllChem.SetConjugation', 'rdkit.Chem.rdqueries.IsAliphaticQueryAtom',
                                 'rdkit.Chem.AllChem.NumAliphaticHeteroatomNeighborsGreaterQueryAtom',
                                 'rdkit.Chem.AllChem.IsReactionTemplateMoleculeAgent',
                                 'rdkit.Chem.AllChem.BuildFeatureFactoryFromString',
                                 'rdkit.Chem.rdMolDescriptors.CalcExactMolWt', 'rdkit.Chem.rdMolDescriptors.CalcPMI3',
                                 'rdkit.DataStructs.cDataStructs.BulkKulczynskiSimilarity',
                                 'rdkit.Chem.AllChem.CalcNumAromaticCarbocycles',
                                 ]

IGNORED_ATOM_FUNCTIONS = ["GetMonomerInfo",
                          "GetPDBResidueInfo",
                          "InvertChirality",
                          "UpdatePropertyCache",
                          'DescribeQuery',
                          'GetNeighbors',
                          'GetBonds',
                          'GetIdx',
                          'GetOwningMol',
                          'NeedsUpdatePropertyCache',
                          'HasQuery',
                          'HasOwningMol',
                          'GetPropsAsDict',
                          'GetPropNames',
                          # unnessesary prefiltered
                          'GetAtomMapNum',
                          'rdkit.Chem.rdchem.GetSupplementalSmilesLabel',
                          'rdkit.Chem.rdchem.GetAtomValue',
                          'rdkit.Chem.rdchem.GetAtomRLabel',
                          'rdkit.Chem.rdchem.GetAtomAlias',
                          'rdkit.Chem.GetSupplementalSmilesLabel',
                          'rdkit.Chem.GetAtomValue',
                          'rdkit.Chem.GetAtomRLabel',
                          'rdkit.Chem.GetAtomAlias',
                          'rdkit.Chem.AllChem.GetAtomPairAtomCode',
                          ] + _ATOM_FUNCTIONS_PRERUN_FAILED

TESTMOLS = []
TESTATOMS = []
MOL_FUNCS = {}
ATOM_FUNCS = {}


def reduce_name(n):
    red_name = n
    if red_name.startswith("Calc") and red_name[4:].strip("_")[0].isalpha():
        red_name = red_name[4:]

    if red_name.startswith("Get") and red_name[3:].strip("_")[0].isalpha():
        red_name = red_name[3:]

    return red_name


def test_mol(mol=None):
    if mol is None:
        mol = TESTMOLS[0]
    return Chem.Mol(mol)


def generate_all_testmols():
    for mol in TESTMOLS:
        yield test_mol(mol)


def generate_all_test_atoms():
    for mol in generate_all_testmols():
        for atom in mol.GetAtoms():
            yield atom


def test_atom(atom=None, mol=None):
    if mol is None and atom is not None:
        mol = atom.GetOwningMol()
    mol = test_mol(mol)
    if atom is not None:
        atom = mol.GetAtomWithIdx(atom.GetIdx())
    else:
        atom = mol.GetAtomWithIdx(0)
    return atom


def check_enum(value):
    if issubclass(value.__class__, enum.Enum):
        return True
    if isinstance(value, enum.Enum):
        return True

    for bc in value.__class__.__bases__:
        if bc.__name__ == "enum":
            return True
    return False


def split_enum(value):
    try:
        names = value.__class__._member_names_
    except AttributeError:
        names = list(value.names.keys())

    dtype = np.array([getattr(value.__class__, n) for n in names]).dtype
    names = [f"{value.__class__.__name__}.{n}" for n in names]

    return dtype, names


def get_type(value):
    if check_enum(value):
        return enum.Enum, value.__class__

    if value is None:
        return None, None
    if isinstance(value, str):
        return str, None
    if isinstance(value, bool):
        return bool, None
    if isinstance(value, int):
        return int, None
    if isinstance(value, float):
        return float, None
    if isinstance(value, Mol):
        return Mol, None

    if isinstance(value, dict):
        return dict, {k: get_type(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return list, [get_type(v) for v in value]
    if value.__class__.__name__.startswith("_vectclass"):
        return list, value.__class__
    if isinstance(value, np.ndarray):
        return np.ndarray, value.dtype
    if isinstance(value, ExplicitBitVect):
        return ExplicitBitVect, None
    if isinstance(value, _vectdouble):
        return _vectdouble, None
    if isinstance(value, _vectint):
        return _vectint, None
    if isinstance(value, IntSparseIntVect):
        return IntSparseIntVect, None
    if isinstance(value, ULongSparseIntVect):
        return ULongSparseIntVect, None
    if isinstance(value, LongSparseIntVect):
        return LongSparseIntVect, None
    if isinstance(value, LongSparseIntVect):
        return LongSparseIntVect, None
    if isinstance(value, ForceField):
        return ForceField, None
    if isinstance(value, SubstanceGroup_VECT):
        return SubstanceGroup_VECT, None
    if isinstance(value, UniformGrid3D_):
        return UniformGrid3D_, None
    if isinstance(value, ChemicalReaction):
        return ChemicalReaction, None
    if isinstance(value, MMFFMolProperties):
        return MMFFMolProperties, None
    if isinstance(value, MolBundle):
        return MolBundle, None
    if isgenerator(value):
        return None, None
    if isfunction(value):
        return None, None

    if value.__class__.__name__ == "_vectunsigned int":
        return "check", None
    if value.__class__.__name__ == "_vectstruct RDKit::Chirality::StereoInfo":
        return "check", None

    raise NotImplementedError(f"{value.__class__.__name__}, {value}")


def parse_mod_for_func(mod, funcs=None, ignore_modules=None):
    if ignore_modules is None:
        ignore_modules = set()

    if funcs is None:
        funcs = set()
    prefix = mod.__name__
    ignore_modules.add(prefix)
    for n in dir(mod):
        if any([r.search(n) is not None for r in BAD_LIST]):
            continue

        attr = getattr(mod, n)
        # print(n,attr,attr.__class__.__name__)
        if isfunction(attr) or attr.__class__.__name__ == "builtin_function_or_method":
            funcs.add(f"{prefix}.{n}")
        if ismodule(attr) and attr.__name__.startswith("rdkit") and attr.__name__ not in ignore_modules:
            parse_mod_for_func(attr, funcs=funcs, ignore_modules=ignore_modules)
    return funcs, ignore_modules


def find_duple_funcs(funcdict):
    duplicates = []
    single_duplicates = set()
    for n1, fd1 in funcdict.items():
        for n2, fd2 in funcdict.items():
            if fd1 == fd2:
                continue
            if n1 in single_duplicates or n2 in single_duplicates:
                continue
            if fd1.function != fd2.function:
                continue
            duplicates.append((n1, n2))
            single_duplicates.add(n1)
            single_duplicates.add(n2)
    return duplicates


def remove_duplicate_funcs(funcdict):
    to_del = set()
    for n1, n2 in find_duple_funcs(funcdict):
        # add function with les module depth
        if len(n1.split(".")) != len(n2.split(".")):
            if len(n1.split(".")) < len(n2.split(".")):
                to_del.add(n2)
            else:
                to_del.add(n1)
        # or with shorter module path
        else:
            if len(n1) <= len(n2):
                to_del.add(n2)
            else:
                to_del.add(n1)
    for d in to_del:
        del funcdict[d]


def generate_single_atom_mols():
    mols = []
    for symbol, number in ATOMIC_SYMBOL_NUMBERS.items():
        em = Chem.RWMol()
        em.AddAtom(Chem.Atom(number))
        mol = em.GetMol()
        SanitizeMol(mol)
        mol = prepare_mol_for_featurization(mol)
        # mol.calcExplicitValence()
        mols.append(mol)
        print(MolToSmiles(mol))
    return mols


class Rtypes(enum.Enum):
    SINGLE_VAL = 0
    ONE_HOT = 1
    STRING = 2
    NONE = 3
    ARRAY = 4
    RDVEC = 5
    MOL = 6


def derive_return_type(values, verbose=False, verbose_prefix=""):
    l = len(values)
    if verbose: print(verbose_prefix, "check moltype")
    if isinstance(values[0], Mol):
        if verbose: print(verbose_prefix, "found moltype")
        return None, Rtypes.MOL, -1
    if all([v is None for v in values]) or isgenerator(values[0]) or isfunction(values[0]) or isinstance(values[0],
                                                                                                         ChemicalReaction):
        if verbose: print(verbose_prefix, "unwanted return type")
        return None, Rtypes.NONE, 0
    ar = np.array(values)

    if verbose: print(verbose_prefix, "check vec")
    if values[0].__class__.__name__.endswith("Vect"):
        if verbose: print(verbose_prefix, "found vec")
        if isinstance(values[0], ExplicitBitVect):
            dtype = bool
        elif isinstance(values[0], IntSparseIntVect):
            dtype = np.int32
        elif isinstance(values[0], LongSparseIntVect):
            dtype = np.int64
        elif isinstance(values[0], ULongSparseIntVect):
            dtype = np.uint64
        else:
            raise NotImplementedError(values[0])
        if verbose: print(verbose_prefix, f"found vec, with dtype {dtype}")
        inconsistend_length = False
        try:
            l = values[0].GetLength()
        except:
            l = len(values[0])
        if l > MAX_LENGTH:
            if verbose: print(verbose_prefix, f"found vec too long with {l} entries")
            return None, Rtypes.RDVEC, l
        for v in values:
            try:
                _l = v.GetLength()
            except:
                _l = len(v)
            if l != _l:
                inconsistend_length = True
            break
        if inconsistend_length:
            raise NotImplementedError(values)
        else:
            a = np.zeros(l, dtype=dtype)
            ConvertToNumpyArray(values[0], a)
            dtype, _, _ = derive_return_type(a, verbose=verbose, verbose_prefix="\t" + verbose_prefix)
            return dtype, Rtypes.RDVEC, l

    # print("ar.shape",ar.shape,sep="",end=" ")
    if verbose: print(verbose_prefix, "check enum")
    if check_enum(values[0]):
        if verbose: print(verbose_prefix, "found enum")
        dtype, rtype, l = derive_return_type(ar, verbose=verbose, verbose_prefix="\t" + verbose_prefix)
        _dtype, names = split_enum(values[0])
        return dtype, Rtypes.ONE_HOT, len(names)

    if verbose: print(verbose_prefix, "check ar type")
    if ar.dtype in [bool, ]:
        if verbose: print(verbose_prefix, "found bool array")
        if ar.shape == (l,):
            return 'bool', Rtypes.SINGLE_VAL, 1
        elif len(ar.shape) == 2:
            return 'bool', Rtypes.ARRAY, ar.shape[1]
        raise NotImplementedError()

    if ar.dtype in [int, np.int32, ]:
        if verbose: print(verbose_prefix, "found int array")
        if ar.shape == (l,):
            return 'np.int32', Rtypes.SINGLE_VAL, 1
        elif len(ar.shape) == 2:
            return 'np.int32', Rtypes.ARRAY, ar.shape[1]
        raise NotImplementedError()
    if ar.dtype in [np.int64, ]:
        if verbose: print(verbose_prefix, "found int array")
        if ar.shape == (l,):
            return 'np.int64', Rtypes.SINGLE_VAL, 1
        elif len(ar.shape) == 2:
            return 'np.int64', Rtypes.ARRAY, ar.shape[1]
        raise NotImplementedError()
    if ar.dtype in [float, np.float64, ]:
        if verbose: print(verbose_prefix, "found float array")
        if ar.shape == (l,):
            return 'np.float32', Rtypes.SINGLE_VAL, 1
        elif len(ar.shape) == 2:
            return 'np.float32', Rtypes.ARRAY, ar.shape[1]
        raise NotImplementedError()
    if ar.dtype in [np.float64, ]:
        if verbose: print(verbose_prefix, "found float array")
        if ar.shape == (l,):
            return 'np.float64', Rtypes.SINGLE_VAL, 1
        elif len(ar.shape) == 2:
            return 'np.float64', Rtypes.ARRAY, ar.shape[1]
    if ar.dtype.char == "U":
        if verbose: print(verbose_prefix, "found string array")
        if ar.shape == (l,):
            return 'str', Rtypes.STRING, 1
    if ar.dtype in [bool, ]:
        if verbose: print(verbose_prefix, "found bool array")
        if ar.shape == (l,):
            return 'bool', Rtypes.SINGLE_VAL, 1

    if verbose: print(verbose_prefix, "check ndarray")
    if isinstance(values[0], np.ndarray):
        if verbose: print(verbose_prefix, "found arrays")
        d0 = values[0].dtype
        s0 = values[0].shape
        inconsistend_shape = False
        inconsistend_datatype = False
        for v in values:
            s = v.shape
            if s != s0:
                inconsistend_shape = True
                break
        for v in values:
            d = v.dtype
            if d != d0:
                inconsistend_datatype = True
                break
        if inconsistend_datatype:
            raise NotImplementedError()
        dtype, _, _ = derive_return_type(values[0], verbose=verbose, verbose_prefix="\t" + verbose_prefix)
        if inconsistend_shape:
            return dtype, Rtypes.ARRAY, -1
        return dtype, Rtypes.ARRAY, s0

    if verbose: print(verbose_prefix, "check list")
    if isinstance(values[0], (list, tuple)):
        if verbose: print(verbose_prefix, "found list array")
        l0 = len(values[0])
        inconsistend_length = False
        for v in values:
            l = len(v)
            if l != l0:
                inconsistend_length = True
                break
        d0, _, _ = derive_return_type(values[0], verbose=verbose, verbose_prefix="\t" + verbose_prefix)
        inconsistend_datatype = False
        for v in values:
            d, _, _ = derive_return_type(v, verbose=verbose, verbose_prefix="\t" + verbose_prefix)
            if d != d0:
                if d0 == "np.int32" and d == "np.int64":
                    d0 = "np.int64"
                    continue
                if d == "np.int32" and d0 == "np.int64":
                    continue
                if d0 == "np.float32" and d == "np.float64":
                    d0 = "np.float64"
                    continue
                if d == "np.float32" and d0 == "np.float64":
                    continue
                inconsistend_datatype = True
                break

        if inconsistend_length:
            l = -1
        else:
            l = l0

        if inconsistend_datatype:
            raise NotImplementedError()

        if inconsistend_length:
            return d0, Rtypes.ARRAY, -1
        return d0, Rtypes.ARRAY, l

    print(type(values[0]), ar.shape, ar.dtype, ar.dtype.char, l)
    raise NotImplementedError(values)


@dataclass
class ModFunction():
    name: str
    module: str
    function: Callable
    values: Any
    data_type: Any
    rtype: Rtypes
    length: int = -1
    estimated_class = None
    # in case
    one_hot: bool = False
    one_hot_values: Any = field(default_factory=list)
    ignored: bool = False

    @classmethod
    def from_data(cls, name, module, function, values):
        # print(name,end=" ")
        dtype, rtype, l = derive_return_type(values, verbose_prefix=name)
        assert isinstance(dtype, str) or dtype is None, derive_return_type(values, verbose=True, verbose_prefix=name)
        # print(dtype,rtype,l)
        return cls(name=name, module=module, function=function, values=values, data_type=dtype, rtype=rtype, length=l)


def func_to_modfunction(func, funcname, mod_name, test_obect, func_dict, data_generator):
    try:
        v = func(test_obect)
    except Exception as e:
        return False
    # ty, sty = get_type(v)
    values = []
    for d in data_generator():
        try:
            values.append(func(d))
        except Exception:
            pass
    if len(values) == 0:
        raise ValueError(funcname)
    func_dict[funcname] = ModFunction.from_data(
        name=funcname,
        module=mod_name,
        function=func,
        values=values,
    )
    return True


def remove_unwanted_returns(funcdict):
    to_del = set()
    unwanted = [None]
    for name, modfunc in funcdict.items():
        if modfunc.data_type in unwanted:
            to_del.add(name)

        if modfunc.data_type in ['np.float32', 'np.float64']:
            if modfunc.length == -1:
                allnan = True
                for v in modfunc.values:
                    if not np.all(np.isnan(np.array(v))):
                        allnan = False
                        break
                if allnan:
                    to_del.add(name)
            else:
                if np.all(np.isnan(np.array(modfunc.values))):
                    to_del.add(name)

    for n in to_del:
        del funcdict[n]
    pass


# def derive_dtype(dtype):

def _update_dtype(mf):
    if mf.data_type in [int, np.int32]:
        mf.data_type = np.int32(mf.values[0]).dtype
        return
    if mf.data_type in [float, np.float32]:
        mf.data_type = np.float32(mf.values[0]).dtype
        return

    if mf.data_type in [bool]:
        mf.data_type = np.array([mf.values[0]], dtype=bool).dtype
        return

    if mf.data_type in [str]:
        mf.data_type = str
        return

    if mf.data_type == enum.Enum:
        dtype, names = split_enum(mf.values[0])
        mf.data_type = dtype
        mf.one_hot = True
        mf.one_hot_values = names
        return

    if mf.data_type == list:
        if mf.length == -1:
            print("uneven list", mf.special_type)
            print(mf.name, mf.data_type)
        else:
            print("even list", mf.length, mf.special_type)
            print(mf.name, mf.data_type)
    print(mf.name, mf.data_type)
    # raise NotImplementedError(mf.data_type)


def update_dtype(functict):
    for n, f in functict.items():
        _update_dtype(f)


def _create_code(modfunc):
    # print(modfunc)
    if modfunc.feat_target == "mol":
        target_type = "Molecule"
    elif modfunc.feat_target == "atom":
        target_type = "Atom"
    else:
        raise NotImplementedError()
    parentclass_subfix = target_type + "Featurizer"
    modfunc_split=modfunc.name.split(".")

    modfunc.modfunc_classname = target_type + "_"
    if len(modfunc_split)>1:
        modfunc.modfunc_classname+=modfunc_split[-2] + "_"
    modfunc.modfunc_classname+=reduce_name(modfunc_split[-1]) + "_Featurizer"
    class_attributes = {}
    class_functions = {}
    # if modfunc.module=="self":
    #    class_attributes
    class_attributes["dtype"] = modfunc.data_type
    class_attributes["featurize"] = f"staticmethod({modfunc_split[-1]})"
    if modfunc.module == "self":
        del class_attributes["featurize"]
        class_functions["featurize"] = f"def featurize(self,{modfunc.feat_target}):\n" \
                                       f"    return {modfunc.feat_target}.{modfunc.name}()"

    if modfunc.rtype == Rtypes.SINGLE_VAL:
        modfunc.parentclass = "SingleValue" + parentclass_subfix

    elif modfunc.rtype == Rtypes.ONE_HOT:
        modfunc.parentclass = "OneHot" + parentclass_subfix
        _dtype, names = split_enum(modfunc.values[0])
        class_attributes["POSSIBLE_VALUES"] = "[" + ",".join(names) + "]"
        del class_attributes["dtype"]
    elif modfunc.rtype == Rtypes.STRING or modfunc.data_type == 'str':
        modfunc.parentclass = "String" + parentclass_subfix

    elif modfunc.rtype == Rtypes.ARRAY:
        if modfunc.length == -1:
            modfunc.parentclass = "VarSize" + parentclass_subfix
        else:
            modfunc.parentclass = "FixedSize" + parentclass_subfix
            class_attributes["LENGTH"] = modfunc.length
    elif modfunc.rtype == Rtypes.RDVEC:
        if modfunc.length == -1:
            modfunc.parentclass = "VarSize" + parentclass_subfix
        else:
            modfunc.parentclass = "FixedSize" + parentclass_subfix
            class_attributes["LENGTH"] = modfunc.length

        del class_attributes["featurize"]
        class_functions["featurize"] = f"def featurize(self,{modfunc.feat_target}):\n" \
                                       f"    a = np.zeros(len(self), dtype=self.dtype)\n" \
                                       f"    ConvertToNumpyArray({modfunc_split[-1]}({modfunc.feat_target}), a)\n" \
                                       f"    return a"

    else:
        raise NotImplementedError()

    code = f"class {modfunc.modfunc_classname}({modfunc.parentclass}):\n"
    code += f"   # _rdfunc={modfunc.name}\n"
    for k, v in class_attributes.items():
        code += f"    {k} = {v}\n"
    for k, v in class_functions.items():
        code += "\n".join(["    " + f for f in v.split("\n")]) + "\n"

    modfunc.code = code


def create_code_file(funcdict):
    files = {}
    funcdict = {k: v for k, v in sorted(funcdict.items(), key=lambda v: v[1].modfunc_classname)}
    for n, modfunc in funcdict.items():
        if modfunc.feat_target == "mol":
            target_type = "molecule"
        elif modfunc.feat_target == "atom":
            target_type = "atom"
        else:
            raise NotImplementedError()

        if modfunc.rtype == Rtypes.SINGLE_VAL:
            filename = f"_autogen_rdkit_feats_numeric_{target_type}_featurizer"
        elif modfunc.rtype == Rtypes.ONE_HOT:
            filename = f"_autogen_rdkit_feats_one_hot_{target_type}_featurizer"
        elif modfunc.rtype == Rtypes.STRING or modfunc.data_type == 'str':
            filename = f"_autogen_rdkit_feats_str_{target_type}_featurizer"
        elif modfunc.rtype == Rtypes.ARRAY:
            filename = f"_autogen_rdkit_feats_array_{target_type}_featurizer"
        elif modfunc.rtype == Rtypes.RDVEC:
            filename = f"_autogen_rdkit_feats_vec_{target_type}_featurizer"
        else:
            raise NotImplementedError(modfunc.rtype)

        if filename not in files:
            files[filename] = {
                'imports': {},
                'codes': [],
                'instances': [],
                'available_featurizer': [],
                'all': [],
                'type': None
            }
        if files[filename]["type"] is None:
            files[filename]["type"] = modfunc.feat_target
        if files[filename]["type"] != modfunc.feat_target:
            raise ValueError

        def add_import(_from, _what):
            if _from not in files[filename]['imports']:
                files[filename]['imports'][_from] = set()
            files[filename]['imports'][_from].add(_what)

        if modfunc.rtype == Rtypes.SINGLE_VAL:
            if target_type == "molecule":
                add_import("molNet.featurizer._molecule_featurizer", "SingleValueMoleculeFeaturizer")
            elif target_type == "atom":
                add_import("molNet.featurizer._atom_featurizer", "SingleValueAtomFeaturizer")
        elif modfunc.rtype == Rtypes.ONE_HOT:
            if target_type == "molecule":
                add_import("molNet.featurizer._molecule_featurizer", "OneHotMoleculeFeaturizer")
            elif target_type == "atom":
                add_import("molNet.featurizer._atom_featurizer", "OneHotAtomFeaturizer")
            add_import(modfunc.values[0].__class__.__module__, modfunc.values[0].__class__.__name__)
        elif modfunc.rtype == Rtypes.ARRAY or modfunc.rtype == Rtypes.RDVEC:
            if target_type == "molecule":
                add_import("molNet.featurizer._molecule_featurizer", "FixedSizeMoleculeFeaturizer")
                add_import("molNet.featurizer._molecule_featurizer", "VarSizeMoleculeFeaturizer")
            elif target_type == "atom":
                add_import("molNet.featurizer._atom_featurizer", "FixedSizeAtomeFeaturizer")
                add_import("molNet.featurizer._atom_featurizer", "VarSizeAtomFeaturizer")
            if modfunc.rtype == Rtypes.RDVEC:
                add_import("rdkit.DataStructs", "ConvertToNumpyArray")
        elif modfunc.rtype == Rtypes.STRING or modfunc.data_type == 'str':
            if target_type == "molecule":
                add_import("molNet.featurizer._molecule_featurizer", "StringMoleculeFeaturizer")
            elif target_type == "atom":
                add_import("molNet.featurizer._atom_featurizer", "StringAtomFeaturizer")

        modfunc.instance_name = modfunc.modfunc_classname.split("_", 1)[0].lower() + "_" + \
                                modfunc.modfunc_classname.split("_", 1)[1].rsplit("_", 1)[0] + \
                                "_" + modfunc.modfunc_classname.rsplit("_", 1)[-1].lower()
        files[filename]['codes'].append(modfunc.code)
        files[filename]['instances'].append(f"{modfunc.instance_name} = {modfunc.modfunc_classname}()")
        files[filename]['available_featurizer'].append(f"'{modfunc.instance_name}':{modfunc.instance_name},")
        files[filename]['all'].append(f"'{modfunc.modfunc_classname}',")
        files[filename]['all'].append(f"'{modfunc.instance_name}',")
        if modfunc.module != "self":
            add_import(modfunc.module, modfunc.name.rsplit(".", 1)[-1])

    for f, d in files.items():
        code = "import numpy as np\n"
        for i, l in d['imports'].items():
            if len(l) > 0:
                code += f"from {i} import ({','.join(l)})\n"
        for c in d['codes']:
            code += c + "\n"
        code += "\n".join(d['instances']) + "\n"
        code += "_available_featurizer = {\n" + "\n    ".join(d['available_featurizer']) + "\n}\n"
        code += "__all__ = [\n" + "\n    ".join(d['all']) + "\n]\n"

        code += "def get_available_featurizer():\n" \
                "    return _available_featurizer\n" \
                "def main():\n" \
                "    from rdkit import Chem\n" \
                "    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization\n"
        if d["type"] == "mol":
            code += "    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))\n"
        elif d["type"] == "atom":
            code += "    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1')).GetAtoms()[-1]\n"
        else:
            raise NotImplementedError(d["type"])
        code += "    for n, f in get_available_featurizer().items():\n" \
                "        print(n,end=' ')\n" \
                "        print(f(testdata))\n" \
                "    print(len(get_available_featurizer()))" \
                "\n" \
                "if __name__ == '__main__':\n" \
                "    main()\n"

        code = black.format_str(code, mode=black.FileMode())
        with open(f + ".py", "w+") as f:
            f.write(code)


def create_code(funcdict):
    for n, f in funcdict.items():
        _create_code(f)

    classnames = {}
    to_del = set()
    for n, f in funcdict.items():
        if f.modfunc_classname in classnames:
            # print(n)
            # print("\t",f)
            # print("\t",classnames[f.modfunc_classname])

            if not compare_arrays(f.values, classnames[f.modfunc_classname].values):
                print("\t", f)
                print("\t", classnames[f.modfunc_classname])
                print("\t", f.values)
                print("\t", classnames[f.modfunc_classname].values)
                print(np.array(f.values).dtype.char)
                raise ValueError()

            n1 = f.name
            n2 = classnames[f.modfunc_classname].name
            if len(n1.split(".")) != len(n2.split(".")):
                if len(n1.split(".")) < len(n2.split(".")):
                    to_del.add(n2)
                else:
                    to_del.add(n1)
            # or with shorter module path
            else:
                if len(n1) <= len(n2):
                    to_del.add(n2)
                else:
                    to_del.add(n1)
        classnames[f.modfunc_classname] = f
    for d in to_del:
        del funcdict[d]


# create_code(ATOM_FUNCS)
# create_code_file(ATOM_FUNCS)
# create_code(MOL_FUNCS)
# create_code_file(MOL_FUNCS)


def compare_arrays(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    if len(a1) != len(a2):
        return False
    if a1.dtype.char == "O":
        return np.all([compare_arrays(a1[i], a2[i]) for i in range(len(a1))])
    else:
        return np.array_equal(a1, a2)


def main():
    initial_modules = [rdMolDescriptors, Descriptors3D, GraphDescriptors, Descriptors, rdkit, Chem, rdmolops,
                       ]

    smiles = [
        "C1=CC=C(C=C1)C2=CC(=CC(=C2)C3=CC(=CC4=C3SC5=CC=CC=C54)N(C6=CC=CC=C6)C7=CC=CC=C7)C8=CC(=CC9=C8SC1=CC=CC=C19)N(C1=CC=CC=C1)C1=CC=CC=C1",
        "C" * 20,
        "CC(C(=O)NCCN1C(=O)C=CC(=N1)N2C=CC=N2)OC3=CC(=CC=C3)Cl",
        "CC1=C(C(NC(=S)N1)C2=CC=C(C=C2)OCC3=CC=CC=C3)C(=O)C4=CC=CC=C4",
        "CC1=CC(=CC=C1)OC(=O)C23CC4CC(C2)CC(C4)(C3)Cl",
        "COC(=O)CN1C2C(NC(=O)N2)NC1=O",
        "CC1=C(SC2=CC=CC=C12)C(=O)C[NH+]3CCN(CC3)C4=[NH+]C=C(C=C4)C(F)(F)F",
    ]  # + generate_n_random_hetero_carbon_lattice(n=60, max_c=15)

    if len(TESTMOLS) == 0:
        print("generate testmoles")
        for smile in smiles:
            try:
                TESTMOLS.append(prepare_mol_for_featurization(Chem.MolFromSmiles(smile)))
            except ConformerError:
                pass

        TESTMOLS.extend(generate_single_atom_mols())

    print("generate testatoms")

    if len(TESTATOMS) == 0:
        for mol in TESTMOLS:
            for atom in mol.GetAtoms():
                TESTATOMS.append(atom)

    test_atom_class = test_atom().__class__
    test_mol_class = test_mol().__class__

    print("find attribute functions")
    for attr in dir(test_atom()):
        if attr.startswith("_") or attr in IGNORED_ATOM_FUNCTIONS:
            continue
        func_to_modfunction(getattr(test_atom_class, attr), attr, "self", test_atom(), ATOM_FUNCS,
                            generate_all_test_atoms)

    for attr in dir(test_mol()):
        if attr.startswith("__") or attr in IGNORED_MOL_FUNCTIONS:
            continue
        func_to_modfunction(getattr(test_mol_class, attr), attr, "self", test_mol(), MOL_FUNCS, generate_all_testmols)

    print("find module functions")
    funcs = set()
    ignore_modules = set()
    for mod in initial_modules:
        parse_mod_for_func(mod, funcs=funcs, ignore_modules=ignore_modules)

    print("module functions to modfunctions")
    print("\tdetect ignored functions")
    modules = {}
    _ignored_mol_functions = []
    _ignored_atom_functions = []
    for f in funcs:
        if f in IGNORED_ATOM_FUNCTIONS or f in IGNORED_MOL_FUNCTIONS:
            mod_name, func_name = f.rsplit(".", 1)
            if mod_name not in modules:
                modules[mod_name] = importlib.import_module(mod_name)
            func = getattr(modules[mod_name], func_name)
            if f in IGNORED_MOL_FUNCTIONS:
                _ignored_mol_functions.append(func)
            if f in IGNORED_ATOM_FUNCTIONS:
                _ignored_atom_functions.append(func)

    print("\tfill modfunctions")
    invalid_mol_funcs = []
    invalid_atom_funcs = []
    for f in funcs:
        mod_name, func_name = f.rsplit(".", 1)
        if mod_name not in modules:
            modules[mod_name] = importlib.import_module(mod_name)
        func = getattr(modules[mod_name], func_name)
        if f not in IGNORED_MOL_FUNCTIONS and func not in _ignored_mol_functions and f not in MOL_FUNCS:
            if not func_to_modfunction(func, f, mod_name, test_mol(), MOL_FUNCS, generate_all_testmols):
                invalid_mol_funcs.append(f)
                # print(f"'{f}',") # used to fill manually ignored
        if f not in IGNORED_ATOM_FUNCTIONS and func not in _ignored_atom_functions and f not in ATOM_FUNCS:
            if not func_to_modfunction(func, f, mod_name, test_atom(), ATOM_FUNCS, generate_all_test_atoms):
                invalid_atom_funcs.append(f)
                # (f"'{f}',") # used to fill manually ignored
    print(invalid_mol_funcs)
    print(invalid_atom_funcs)

    print("remove dupicates")
    remove_duplicate_funcs(ATOM_FUNCS)
    remove_duplicate_funcs(MOL_FUNCS)

    print("remove unwanted returns")
    remove_unwanted_returns(ATOM_FUNCS)
    remove_unwanted_returns(MOL_FUNCS)

    for n, f in ATOM_FUNCS.items():
        f.feat_target = "atom"
    for n, f in MOL_FUNCS.items():
        f.feat_target = "mol"
    # print("update_dtype")
    # update_dtype(ATOM_FUNCS)
    # update_dtype(MOL_FUNCS)

    # pprint(MOL_FUNCS)
    # pprint(ATOM_FUNCS)
    create_code(ATOM_FUNCS)
    create_code_file(ATOM_FUNCS)
    create_code(MOL_FUNCS)
    create_code_file(MOL_FUNCS)


if __name__ == '__main__':
    main()
