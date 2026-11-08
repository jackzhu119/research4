"""Recompute graph checks and formula-only basis sizing; no force field or QM."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = json.loads((HERE/'uploaded_identity_overlay.json').read_text())
    checks = []
    candidates = []
    for row in overlay['records']:
        sdf = ROOT/row['uploaded_sdf_path']
        assert sha(sdf) == row['uploaded_sdf_sha256']
        source = Chem.MolFromSmiles(row['source_smiles'])
        molecules = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False) if m is not None]
        assert len(molecules) == 1
        upload = molecules[0]
        assert Chem.MolToInchiKey(source) == Chem.MolToInchiKey(upload) == row['source_inchikey']
        assert Chem.MolToSmiles(source, isomericSmiles=True) == Chem.MolToSmiles(Chem.RemoveHs(upload), isomericSmiles=True)
        assert rdMolDescriptors.CalcMolFormula(upload) == row['formula']
        assert upload.GetNumAtoms() == row['explicit_atom_count']
        assert Chem.GetFormalCharge(upload) == 0
        assert sum(a.GetNumRadicalElectrons() for a in upload.GetAtoms()) == 0
        checks.append(dict(name=row['name'], status='PASS', exact_sdf_sha256=sha(sdf),
                           full_stereo_key=row['source_inchikey'], quantum_calls=0))
        candidates.append(dict(name=row['name'], cid=row['pubchem_cid'],
                               smiles=row['source_smiles'],
                               graph_source=row['source_url'],
                               coordinate_status='USER_SDF_COMPUTED_3D_NOT_DFT_OPTIMIZED',
                               priority=0 if row['name']=='Spinosin' else 2,
                               role='PRIMARY_KEEP_OLD_C001' if row['name']=='Spinosin' else 'AUXILIARY_HUANGQI_COMPOUND'))
    candidates.append(dict(name='Protocatechuic acid', cid=72,
        smiles='O=C(O)c1ccc(O)c(O)c1',
        graph_source='https://www.ebi.ac.uk/chebi/CHEBI:36062',
        coordinate_status='NOT_GENERATED', priority=1,
        role='SMALL_MOLECULE_METHOD_AND_HBOND_EXPLORATION'))
    for name, path in [('Jujuboside A', 'jujuboside_a_source_record.json'),
                       ('Jujuboside B', 'jujuboside_b_source_record.json')]:
        previous = ROOT/'reports/pyscf-spinosin-pilot-20261031/structures'/path
        saved = json.loads(previous.read_text())
        candidates.append(dict(name=name, cid=saved['pubchem_cid'],
            smiles=saved['source_isomeric_smiles'], graph_source=saved['source_url'],
            coordinate_status='NOT_GENERATED', priority=3,
            role='GRAPH_AND_RESOURCE_REVIEW_ONLY',
            retained_source_record_sha256=sha(previous)))
    output_rows = []
    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate['smiles'])
        assert mol is not None
        explicit = Chem.AddHs(mol)
        atoms = Counter(a.GetSymbol() for a in explicit.GetAtoms())
        assert set(atoms) <= {'C','H','O'}
        # Spherical def2-SVP C/O14, H5; formula-only sizing, no integrals/SCF.
        basis = sum(atoms[k]*v for k,v in {'C':14,'O':14,'H':5}.items())
        out = dict(candidate,
            formula=rdMolDescriptors.CalcMolFormula(mol),
            total_atoms_including_H=explicit.GetNumAtoms(),
            heavy_atoms=mol.GetNumHeavyAtoms(),
            neutral_electrons=sum(a.GetAtomicNum() for a in explicit.GetAtoms()),
            full_inchikey=Chem.MolToInchiKey(mol),
            formula_only_spherical_def2svp_AO_count=basis,
            expected_nonlinear_vibration_count=3*explicit.GetNumAtoms()-6,
            expected_count_is_not_a_frequency_calculation=True,
            estimated_wall_time='NOT_ESTIMATED_OR_BENCHMARKED',
            new_project_THZ_training_records=0,
            quantum_calculations_in_this_audit=0)
        output_rows.append(out)
    pa = next(x for x in output_rows if x['name']=='Protocatechuic acid')
    assert pa['full_inchikey']=='YQUVCSBJEUQKSH-UHFFFAOYSA-N'
    assert pa['total_atoms_including_H']==17 and pa['neutral_electrons']==80
    result = dict(status='PASS', source_script_sha256=sha(__file__),
        source_overlay_sha256=sha(HERE/'uploaded_identity_overlay.json'),
        uploaded_sdf_checks=checks, candidates=output_rows,
        no_conformer_generation=True, no_force_field=True, no_QM=True,
        no_classifier_training=True, no_old_files_modified=True)
    with (args.output/'graph_and_triage_replay.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    with (args.output/'chemical_candidates.csv').open('x',newline='') as f:
        fields=sorted(set().union(*(set(r) for r in output_rows)))
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(output_rows)
    print(json.dumps(dict(status='PASS', uploaded_sdfs=len(checks),
                         candidate_graphs=len(output_rows), quantum_calls=0)))


if __name__=='__main__':
    main()
