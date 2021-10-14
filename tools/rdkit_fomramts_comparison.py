import gzip
import os
import tempfile
import timeit
import pickle

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.PropertyMol import PropertyMol

from molNet.utils.mol.generator import generate_n_random_hetero_carbon_lattice
from molNet.utils.mol.properties import assert_conformers

n_mols=100
mols = generate_n_random_hetero_carbon_lattice(n=n_mols,max_c=10)
mols = [PropertyMol(assert_conformers(mol)) for mol in mols]
for mol in mols:
    mol.SetProp('IntVal',1)

with tempfile.TemporaryDirectory() as tmpdirname:
    print("storing tspeed")

    def store_as_mols():
        for i in range(n_mols):
            with open(os.path.join(tmpdirname,f"{i}.mol"), "w+b") as f:
                pickle.dump(mols[i],f)
    reps=100
    mols_time = timeit.timeit(store_as_mols,number=reps)
    print(f"\t\tpickle as single mols took {mols_time} s ({1000*mols_time/(reps*n_mols)} ms/mol)")

    def store_as_gzipmols():
        for i in range(n_mols):
            with gzip.open(os.path.join(tmpdirname,f"{i}.mol.gz"), "w+b") as f:
                pickle.dump(mols[i],f)
    reps=100
    mols_time = timeit.timeit(store_as_gzipmols,number=reps)
    print(f"\t\tpickle as single gzip mols took {mols_time} s ({1000*mols_time/(reps*n_mols)} ms/mol)")

    def store_as_sdf():
        with Chem.SDWriter(os.path.join(tmpdirname,f"all.sdf")) as w:
            for m in mols:
                w.write(m)
    reps=100
    sdf_time = timeit.timeit(store_as_sdf,number=reps)
    print(f"\t\tpickle as merged sdf took {sdf_time} s ({1000*sdf_time/(reps*n_mols)} ms/mol)")

    def store_as_gzsdf():
        with gzip.open(os.path.join(tmpdirname,f"all.sdf.gz"),"wt+") as f:
            with Chem.SDWriter(f) as w:
                for m in mols:
                    w.write(m)
    reps=100
    sdf_time = timeit.timeit(store_as_gzsdf,number=reps)
    print(f"\t\tpickle as gzipped merged sdf took {sdf_time} s ({1000*sdf_time/(reps*n_mols)} ms/mol)")


    print("reading speed")

    def read_mols():
        for i in range(n_mols):
            with open(os.path.join(tmpdirname,f"{i}.mol"), "rb") as f:
                m = pickle.load(f)
                m.GetProp('IntVal')
    store_as_mols()
    read_mols()
    mols_time = timeit.timeit(store_as_mols,number=reps)
    print(f"\t\treading as single mols took {mols_time} s ({1000*mols_time/(reps*n_mols)} ms/mol)")

    def read_gzmols():
        for i in range(n_mols):
            with gzip.open(os.path.join(tmpdirname,f"{i}.mol.gz"), "rb") as f:
                m = pickle.load(f)
                m.GetProp('IntVal')
    store_as_mols()
    read_mols()
    mols_time = timeit.timeit(read_gzmols,number=reps)
    print(f"\t\treading as single gzipped mols took {mols_time} s ({1000*mols_time/(reps*n_mols)} ms/mol)")

    def read_sdf():
        with gzip.open(os.path.join(tmpdirname,f"all.sdf.gz"),"rb") as f:
            for mol in Chem.ForwardSDMolSupplier(f):
                k=mol.GetProp('IntVal')
    store_as_sdf()
    read_sdf()
    sdf_time = timeit.timeit(read_sdf,number=reps)
    print(f"\t\treading as merged took {sdf_time} s ({1000*sdf_time/(reps*n_mols)} ms/mol)")

    def read_gzsdf():
        for mol in Chem.ForwardSDMolSupplier(os.path.join(tmpdirname,f"all.sdf")):
            k=mol.GetProp('IntVal')
    store_as_sdf()
    read_sdf()
    sdf_time = timeit.timeit(read_gzsdf,number=reps)
    print(f"\t\treading as zipped merged sdf took {sdf_time} s ({1000*sdf_time/(reps*n_mols)} ms/mol)")




    print("filesize")

    store_as_mols()
    tot_size=0
    for i in range(n_mols):
        tot_size += os.path.getsize(os.path.join(tmpdirname,f"{i}.mol"))

    print(f"\t\tsaving as single mols took {tot_size} b ({tot_size/n_mols} b/mol)")

    store_as_gzipmols()
    tot_size=0
    for i in range(n_mols):
        tot_size += os.path.getsize(os.path.join(tmpdirname,f"{i}.mol.gz"))

    print(f"\t\tsaving as single gz mols took {tot_size} b ({tot_size/n_mols} b/mol)")

    store_as_sdf()
    tot_size=os.path.getsize(os.path.join(tmpdirname,f"all.sdf"))
    print(f"\t\tsaving as merged sdf took {tot_size} b ({tot_size/n_mols} b/mol)")

    store_as_gzsdf()
    tot_size=os.path.getsize(os.path.join(tmpdirname,f"all.sdf.gz"))
    print(f"\t\tsaving as gz merged sdf took {tot_size} b ({tot_size/n_mols} b/mol)")

