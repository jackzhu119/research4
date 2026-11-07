"""Read-only actual-export validation and uncropped model-geometry supplement.

No quantum calculations. Run from the repository root. Only the specified output
file(s) are created; sealed analysis products are checked without modification.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'analysis'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(name):
    with (ANALYSIS / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

index = json.loads((ANALYSIS / 'files_sha256.json').read_text())
for name, expected in index.items():
    path = ANALYSIS / name
    assert digest(path) == expected['sha256'], name
    assert path.stat().st_size == expected['bytes'], name
assert len(index) == 19
summary = json.loads((ANALYSIS / 'summary.json').read_text())
assert summary['new_SCF_gradient_Hessian_jobs'] == 0
assert summary['IR_intensity_status'] == 'NOT_COMPUTED'
phase_rows = rows('phase_status_and_resources.csv')
assert len(phase_rows) == 4
assert math.isclose(sum(float(row['actual_cpu_seconds']) for row in phase_rows), 2666.234148, abs_tol=1e-6)
water = rows('water_validated_frequencies.csv')
assert len(water) == 6
assert collections.Counter(row['phase'] for row in water) == {'water-optfreq': 3, 'water-cached-optfreq': 3}
for row in water:
    assert row['identity'] == 'water' and row['IR_intensity_status'] == 'NOT_COMPUTED'
    assert math.isclose(float(row['frequency_cm1']) * 0.0299792458, float(row['frequency_thz']), abs_tol=1e-10)
for name in ('spinosin_computed_frequencies_including_unvalidated.csv', 'spinosin_validated_all_frequencies.csv', 'spinosin_validated_0p1_1p5_thz_modes.csv'):
    assert rows(name) == [], name
experimental = rows('experimental_candidate_positions.csv')
pairs = rows('experiment_theory_pairs.csv')
assert len(experimental) == len(pairs) == 20
for row in pairs:
    assert row['comparison_status'] == 'BLOCKED_NO_VALIDATED_SPINOSIN_MODES_IN_RANGE'
    assert row['chemical_attribution_verified'] == 'False'
    for key in ('theoretical_mode', 'theoretical_frequency_thz', 'theoretical_wavenumber_cm1', 'absolute_difference_thz', 'absolute_difference_cm1'):
        assert row[key] == '', key
steps = rows('actual_optimization_steps.csv')
assert len(steps) == 4
for row in steps:
    step = ROOT / row['phase'] / 'geometry_steps' / f"step_{int(row['cycle']):04d}"
    raw = json.loads(step.with_suffix('.json').read_text())
    gradient = np.load(step.with_suffix('.gradient.npy'))
    assert gradient.shape == (75, 3) and np.isfinite(gradient).all()
    assert float(row['energy_hartree']) == raw['energy_hartree']
    assert math.isclose(float(row['atom_rms_hartree_per_bohr']), np.sqrt(np.mean(np.sum(gradient**2, axis=1))), abs_tol=1e-14)
    assert math.isclose(float(row['atom_max_hartree_per_bohr']), np.linalg.norm(gradient, axis=1).max(), abs_tol=1e-14)
provenance = json.loads((ANALYSIS / 'displayed_geometry_provenance.json').read_text())
geometry = Path(provenance['source'])
assert digest(geometry) == provenance['sha256']
xyz_lines = geometry.read_text().splitlines()
atoms = [line.split()[0] for line in xyz_lines[2:] if line.strip()]
coords = np.array([[float(v) for v in line.split()[1:4]] for line in xyz_lines[2:] if line.strip()])
assert collections.Counter(atoms) == {'C': 28, 'H': 32, 'O': 15}
assert coords.shape == (75, 3) and not provenance['geometry_optimized']
for png in ANALYSIS.glob('*.png'):
    with Image.open(png) as im:
        im.verify()
    assert png.with_suffix('.pdf').read_bytes().startswith(b'%PDF-')

# A separate rendering fixes a cropped axis label in the immutable first export.
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
for symbol, color, size in (('C', '#333333', 40), ('H', '#dddddd', 16), ('O', '#cc3333', 50)):
    xyz = coords[np.array(atoms) == symbol]
    ax.scatter(*xyz.T, c=color, s=size, label=symbol, edgecolors='#777777', linewidths=0.3)
mid = (coords.max(axis=0) + coords.min(axis=0)) / 2
radius = max(np.ptp(coords, axis=0)) / 2 + .5
for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
    setter(center-radius, center+radius)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel('x / Angstrom', labelpad=8)
ax.set_ylabel('y / Angstrom', labelpad=8)
# Fixed figure-level label keeps all dimensions readable across 3D view backends.
fig.text(.03, .48, 'z / Angstrom', rotation=90, va='center', fontsize=11)
ax.legend(loc='upper right', frameon=False)
fig.suptitle('Spinosin C28H32O15: last actually evaluated geometry\nPARTIAL optimization; not a verified stationary minimum', y=.97, fontsize=12)
fig.subplots_adjust(left=.12, right=.91, bottom=.08, top=.88)
for suffix in ('png', 'pdf'):
    fig.savefig(OUT / f'spinosin_model_geometry_uncropped.{suffix}', dpi=180, bbox_inches='tight', pad_inches=.15)
plt.close(fig)
result = {'status': 'PASS', 'new_quantum_jobs': 0, 'analysis_index_entries_verified': len(index), 'analysis_index_sha256': digest(ANALYSIS / 'files_sha256.json'), 'complete_paired_geometry_gradient_points': len(steps), 'spinosin_hessian_status': 'NOT_RUN', 'spinosin_low_frequency_result': 'NOT_COMPUTED_NOT_AN_ABSENCE_RESULT', 'water_independent_molecules': 1, 'water_modes_per_backend': 3, 'water_backend_runs': 2, 'geometry_sha256': provenance['sha256'], 'supplement_png_sha256': digest(OUT / 'spinosin_model_geometry_uncropped.png'), 'supplement_pdf_sha256': digest(OUT / 'spinosin_model_geometry_uncropped.pdf')}
(OUT / 'validation_replay.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
