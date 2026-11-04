"""Build the manually source-checked chemical evidence register; no network/QM.

URLs were actually searched/opened on 2026-11-04 using the web connector.
This script writes only to a new output directory and does not modify spectra,
classifiers, existing audits, or any quantum calculation.
"""
import argparse
import csv
import json
from pathlib import Path


def record(key, title, url, kind, access, licence, scope, role, limitation):
    return dict(source_id=key, title=title, url=url, source_type=kind,
                access_status=access, accessed_date_utc='2026-11-04',
                licence_or_reuse_scope=licence, verified_scope=scope,
                permitted_research_role=role, direct_855_training_eligible=False,
                limitations=limitation, primary_source=True)


def records():
    database = 'PRIMARY_DATABASE'
    index = 'PRIMARY_DATABASE_INDEX'
    paper = 'PRIMARY_RESEARCH_ARTICLE'
    return [
        record('CHEBI_SPINOSIN', 'spinosin (CHEBI:81360)',
               'https://www.ebi.ac.uk/chebi/CHEBI:81360', database,
               'WEB_OPEN_SUCCESS', 'CC BY 4.0',
               'Full stereo graph, formula and neutral charge; cited species occurrence.',
               'DFT_INPUT_IDENTITY', 'No assay of our specimens or measured THz spectrum.'),
        record('PUBCHEM_SPINOSIN', 'Spinosin CID155692',
               'https://pubchem.ncbi.nlm.nih.gov/compound/155692', index,
               'INDEX_VISIBLE_DIRECT_OPEN_CAPTCHA', 'Contributor-specific; identity facts summarized only',
               'Official indexed CID/name/formula/key agree with ChEBI and uploaded SDF.',
               'IDENTITY_CORROBORATION', 'No independent byte-exact 3D download succeeded.'),
        record('CHEBI_CALYCOSIN', 'calycosin (CHEBI:17793)',
               'https://www.ebi.ac.uk/chebi/CHEBI:17793', database,
               'WEB_OPEN_SUCCESS', 'CC BY 4.0',
               'Neutral graph/key; Astragalus membranaceus root occurrence reference.',
               'DFT_INPUT_IDENTITY', 'Does not measure calycosin in the project Huangqi samples.'),
        record('PUBCHEM_CALYCOSIN', 'Calycosin CID5280448',
               'https://pubchem.ncbi.nlm.nih.gov/compound/5280448', index,
               'INDEX_VISIBLE_DIRECT_OPEN_CAPTCHA', 'Contributor-specific; identity facts summarized only',
               'CID/name/formula/key corroboration; linked crystal deposition CCDC242382.',
               'IDENTITY_AND_CRYSTAL_LEAD', 'Third-party NMR/MS spectra are different modalities and were not copied.'),
        record('CHEBI_CALYCOSIN_ANION', 'calycosin(1-) (CHEBI:77992)',
               'https://www.ebi.ac.uk/chebi/CHEBI:77992', database,
               'WEB_OPEN_SUCCESS', 'CC BY 4.0',
               'C16H11O5, charge -1 and key suffix M differ from the neutral upload.',
               'EXCLUDED_DIFFERENT_PROTONATION', 'Do not replace the neutral molecule by this anion.'),
        record('CHEBI_PA', '3,4-dihydroxybenzoic acid (CHEBI:36062)',
               'https://www.ebi.ac.uk/chebi/CHEBI:36062', database,
               'WEB_OPEN_SUCCESS', 'CC BY 4.0',
               'Protocatechuic-acid neutral graph C7H6O4 and identifier.',
               'SMALL_MOLECULE_CANDIDATE', 'Graph data are not computed frequencies or THz records.'),
        record('CHEBI_LICENSE', 'About ChEBI: licensing',
               'https://www.ebi.ac.uk/chebi/about', 'PRIMARY_PROVIDER_POLICY',
               'WEB_OPEN_SUCCESS', 'CC BY 4.0 with EMBL-EBI terms',
               'The resource expressly states CC BY 4.0.',
               'SOURCE_LICENCE_EVIDENCE', 'Do not generalize the institute preferred CC0 policy to ChEBI.'),
        record('SPINOSIN_ROTAMERS_2000', 'Flavonoids from Ziziphus jujuba Mill var. spinosa',
               'https://doi.org/10.1016/S0040-4020(00)00842-5', paper,
               'PUBLISHER_INDEX_ABSTRACT_READ', 'Elsevier copyright; paraphrase/bibliographic facts only',
               'Seed-isolation study includes spinosin and reports its rotamers.',
               'CONFORMER_JUSTIFICATION', 'No optimized XYZ or THz raw data acquired.'),
        record('ZSS_DECOCTION_2019',
               'Quality control of Semen Ziziphi Spinosae standard decoction based on determination of multi-components using TOF-MS/MS and UPLC-PDA technology',
               'https://doi.org/10.1016/j.jpha.2019.01.001', paper,
               'PRIMARY_TEXT_AND_TABLES_VERIFIED', 'CC BY-NC-ND 4.0; short factual summary and links only',
               'Ten decoction batches, nine quantified markers; standard-confirmed protocatechuic acid.',
               'CHEMICAL_OCCURRENCE_AND_ASSAY_DESIGN', 'Hot-water decoction is not untreated seed or paired chemistry for our855 spectra.'),
        record('ZSS_ENVIRONMENT_2025',
               'Plant traits and environment affect the contents of medicinal components in Ziziphi spinosae semen in Lincheng County of Hebei Province',
               'https://doi.org/10.1186/s12870-025-06946-4', paper,
               'PRIMARY_TEXT_AND_DATA_AVAILABILITY_VERIFIED', 'CC BY-NC-ND 4.0; short factual summary and links only',
               '24 habitats × four sampled wild plants; four components assayed with HPLC-DAD/ELSD.',
               'ENVIRONMENT_COMPOSITION_RATIONALE', 'Underlying individual data author-request-only; published aggregates are not an individual matrix.'),
        record('PA_THZ_2021',
               'Terahertz and Raman Spectroscopic Investigation of Monohydrate Cocrystal of Antitubercular Isoniazid with Protocatechuic Acid',
               'https://doi.org/10.3390/pharmaceutics13081303', paper,
               'PRIMARY_TEXT_METHODS_AND_SI_LINK_READ', 'CC BY 4.0',
               'Pure PA, INH, mixture and hydrate-cocrystal THz; molecular DFT; three repeats averaged.',
               'MECHANISM_AND_SMALL_MOLECULE_REFERENCE', 'No raw vectors acquired; not acid-jujube spectra or independent origin-labelled plants.'),
        record('CALYCOSIN_DFT_2023',
               'Radical Scavenging Capability and Mechanism of Three Isoflavonoids Extracted from Radix Astragali: A Theoretical Study',
               'https://doi.org/10.3390/molecules28135039', paper,
               'PRIMARY_INDEX_METHOD_AND_SI_DESCRIPTION_READ', 'CC BY 4.0',
               'Molclus search, M06-2X/6-311+G(d,p); SI TableS2 lists optimized XYZ.',
               'POTENTIAL_EXTERNAL_OPTIMIZED_GEOMETRIES', 'SI coordinates not retrieved; antioxidant thermochemistry is not THz origin classification.'),
        record('CCDC_CALYCOSIN_242382', 'CCDC242382 linked to Calycosin',
               'https://doi.org/10.5517/cc846sc', 'PRIMARY_DATABASE_LINK_FROM_PUBCHEM',
               'LISTED_DIRECT_DEPOSITION_ACCESS_BLOCKED', 'NOT_VERIFIED_FOR_THIS_DEPOSITION',
               'Potential experimental crystal geometry.',
               'CRYSTAL_STRUCTURE_LEAD', 'No CIF, unit cell, solvent, quality or licence independently inspected.'),
        record('ZSSG_FIGSHARE_2024', 'Additional file1: list of85 ingredients from ZSSG',
               'https://doi.org/10.6084/m9.figshare.26610144.v1', 'PRIMARY_RESEARCH_DATA_REPOSITORY',
               'LANDING_PAGE_READ_DOWNLOAD_BLOCKED', 'Landing page: CC BY + CC0; per-file scope still to verify',
               '41.5kB XLS compound list from ZSS plus GABA preparation.',
               'CANDIDATE_LIST_ONLY', '85ingredients are not85 independent seeds or paired THz/HPLC observations.'),
        record('PUBCHEM_JUJUBA_FULL', 'Jujuboside A CID51346169',
               'https://pubchem.ncbi.nlm.nih.gov/compound/51346169', index,
               'INDEX_FORMULA_READ_EXISTING_GRAPH_EVIDENCE_REUSED', 'Chemical identity facts summarized only',
               'C58H94O26; retained fully specified local candidate.',
               'LARGE_MOLECULE_PREPARATION_ONLY', 'Actual reference-standard identity remains to be reconciled.'),
        record('PUBCHEM_JUJUBA_PARTIAL', 'Jujuboside-A CID171446',
               'https://pubchem.ncbi.nlm.nih.gov/compound/171446', index,
               'INDEX_INCHI_WITH_UNSPECIFIED_STEREO_READ', 'Chemical identity facts summarized only',
               'Same formula/connectivity; indexed InChI retains unspecified stereo.',
               'IDENTITY_AMBIGUITY_EVIDENCE', 'Not interchangeable with51346169 merely because the names/formula match.'),
        record('PUBCHEM_JUJUBB', 'Jujuboside B CID24721031',
               'https://pubchem.ncbi.nlm.nih.gov/compound/24721031', index,
               'INDEX_FORMULA_AND_STEREO_INCHI_READ', 'Chemical identity facts summarized only',
               'C52H84O21; retained existing graph evidence.',
               'LARGE_MOLECULE_PREPARATION_ONLY', 'Do not substitute B1 CID73156987; no new full DFT in this audit.'),
    ]


def assets():
    return [
        ('calycosin_optimized_xyz_s2', 'CALYCOSIN_DFT_2023',
         'https://www.mdpi.com/article/10.3390/molecules28135039/s1',
         'TableS2 optimized XYZ for formononetin, calycosin and calycosin-7-glucoside',
         'Publisher429/tool inaccessible; no coordinate bytes acquired'),
        ('calycosin_crystal', 'CCDC_CALYCOSIN_242382',
         'https://www.ccdc.cam.ac.uk/structures/StructureSummary/form?pid=242382',
         'CCDC242382 / DOI10.5517/cc846sc',
         'Deposition access blocked; licence/CIF details unverified'),
        ('pa_thz_supporting', 'PA_THZ_2021',
         'https://pmc.ncbi.nlm.nih.gov/articles/instance/8400006/bin/pharmaceutics-13-01303-s001.zip',
         'Supporting figures and characterization, including a THz spectra figure',
         'Web retrieval CAPTCHA; no numeric raw spectral data established'),
        ('zssg_85_ingredients_xls', 'ZSSG_FIGSHARE_2024',
         'https://springernature.figshare.com/ndownloader/files/48438234',
         '13020_2023_814_MOESM1_ESM.xls; landing page41.5kB/list85ingredients',
         'Cloud HTTP CONNECT403; web redirect inaccessible'),
        ('lincheng_supplement_docx', 'ZSS_ENVIRONMENT_2025',
         'https://static-content.springer.com/esm/art%3A10.1186%2Fs12870-025-06946-4/MediaObjects/12870_2025_6946_MOESM1_ESM.docx',
         'SupplementaryMaterial1; validation and supplemental figures described',
         'DOCX parser unsupported/directHTTP403; individual raw data author-request-only'),
    ]


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    values = records()
    save(a.output/'source_registry.json', {
        'schema': 'research4-external-chemical-source-registry-v1',
        'new_quantum_calculations': 0, 'new_classifier_training': 0,
        'new_independent_855_task_samples': 0, 'records': values,
        'bounded_negative_search': 'No open numeric Spinosin THz or full vibration dataset acquired. Not proof none exists.'})
    with (a.output/'source_registry.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)
    save(a.output/'external_assets.json', {
        'downloaded_external_files_count': 0,
        'uploaded_user_sdfs_already_archived_count': 2,
        'assets': [dict(asset_id=k, source_id=s, url=u, listed_content=c,
                        status='LISTED_NOT_RETRIEVED', blocker=b, sha256=None)
                   for k, s, u, c, b in assets()]})
    with (a.output/'literature_peak_reference.csv').open('x', newline='') as f:
        fields = ['source_id', 'material', 'peak_thz', 'location', 'data_kind',
                  'repeat_scans_averaged', 'independent_sample_count',
                  'training_eligible', 'limitation']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for frequency in [1.20, 1.40, 1.49]:
            writer.writerow(dict(source_id='PA_THZ_2021',
                material='pure_protocatechuic_acid_solid', peak_thz=frequency,
                location='section3.1 Figure4 narrative',
                data_kind='published_peak_position_not_raw_spectrum',
                repeat_scans_averaged=3, independent_sample_count='UNKNOWN',
                training_eligible=False,
                limitation='Reference only; no correspondence with project855specimens'))
    save(a.output/'method_cautions.json', {
        'source_id': 'PA_THZ_2021', 'gaussian_version_reported': 'Gaussian09',
        'functional_reported': 'B3LYP',
        'basis_set': 'NOT_IDENTIFIED_IN_INSPECTED_MAIN_METHOD',
        'model': 'Three candidate construction units, including water-containing clusters; not a periodic solid',
        'lorentzian_fwhm_reported_cm1': 4.0,
        'cocrystal_frequency_reporting_difference': {
            'section3_1_text_thz': 1.22, 'Table1_comparison_thz': 1.20,
            'handling': 'Preserve discrepancy; do not choose whichever matches our data.'},
        'raw_ir_intensities_retrieved': False,
        'pyscf_ir_intensity_not_computed': True})
    print(json.dumps({'source_records': len(values), 'listed_assets': len(assets()),
                      'quantum_calculations': 0, 'classifier_fits': 0}))


if __name__ == '__main__':
    main()
