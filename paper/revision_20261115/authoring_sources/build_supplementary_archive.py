"""Package current corrected data and accepted molecular outputs for submission."""
from pathlib import Path
import gzip,hashlib,json,tempfile,shutil,zipfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/active_workbooks.json').exists())
BASE=ROOT/'paper/revision_20261115'
CURRENT=ROOT/'paper/current'
EVIDENCE=BASE/'Supplementary_Data_S1'
source_manifest=json.loads((CURRENT/'manifest.json').read_text())
allow={x['path']:x for x in source_manifest['files']}
records=[]
with tempfile.TemporaryDirectory() as scratch:
    stage=Path(scratch)/'Supplementary_Data_S1';stage.mkdir()
    def copy(source,target,current_rel=None):
        data=source.read_bytes()
        if current_rel:
            assert hashlib.sha256(data).hexdigest()==allow[current_rel]['sha256'],current_rel
        path=stage/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        records.append({'path':str(target),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':str(source.relative_to(ROOT))})
    for path in sorted((CURRENT/'tables').rglob('*')):
        if path.is_file():
            rel=str(path.relative_to(CURRENT));copy(path,Path('corrected_tables')/path.relative_to(CURRENT/'tables'),rel)
    for name in ['metadata.csv','split_manifest.csv','active_workbooks.json','feature_audit.json']:
        rel='inputs/'+name;copy(CURRENT/rel,Path('specimen_inputs')/name,rel)
    for name in ['dataset_by_species.csv','dataset_by_workbook.csv','record_role_assignment.csv','nominal_thickness_by_workbook.csv','task_role_counts.csv']:
        copy(CURRENT/'methods'/name,Path('specimen_inputs')/name)
    for path in sorted((CURRENT/'protocols').glob('*.json')):
        rel=str(path.relative_to(CURRENT));copy(path,Path('protocols')/path.name,rel)
    for i in range(1,6):
        rel=f'results/B{i}/results_summary.json';copy(CURRENT/rel,Path('mechanism_summaries')/f'B{i}.json',rel)
    for item in source_manifest['files']:
        if item['path'].startswith('results/kind_lowo_20261115/'):
            copy(CURRENT/item['path'],Path('kind_lowo')/Path(item['path']).relative_to('results/kind_lowo_20261115'),item['path'])
    derived=EVIDENCE/'derived_checks'
    for path in sorted(derived.glob('*')):
        if path.is_file() and path.suffix in ['.csv','.gz','.json'] and not path.name.startswith('withdrawn_'):
            copy(path,Path('derived_checks')/path.name)
    for path in sorted((derived/'verified_v4_sources').rglob('*')):
        if path.is_file():copy(path,Path('verified_v4_sources')/path.relative_to(derived/'verified_v4_sources'))
    molecular=EVIDENCE/'molecular_sources'
    for folder in ['acid_d2f61d35','acid_f626f77f']:
        for path in sorted((molecular/folder).rglob('*')):
            if path.is_file():copy(path,Path('accepted_molecular')/folder/path.relative_to(molecular/folder))
    for name in ['molecular_table_verification.json','verify_molecular_tables.py']:
        copy(molecular/name,Path('accepted_molecular')/name)
    receipt={'schema':'research4-submission-data-v1','classifier_manifest_sha256':hashlib.sha256((CURRENT/'manifest.json').read_bytes()).hexdigest(),'canonical_raw_spectra':'https://github.com/jackzhu119/research4/tree/main/data','experimental_data_provider':'Yin Shen','first_author_orcid':'https://orcid.org/0009-0005-1153-4926','corresponding_orcid':'https://orcid.org/0000-0002-3221-7875','author_orcids':{'Ziyuan Zhu':'https://orcid.org/0009-0005-1153-4926','Yin Shen':'https://orcid.org/0000-0002-3221-7875'},'dataset_use':'Shared research resource also used in other supervised studies; no exclusive-use claim','new_model_training':True,'new_model_fit_scope':'Only material LOWO:10 binary and18 four-material fixed-pipeline fits','files':records}
    (stage/'SOURCE_MANIFEST.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (stage/'README.txt').write_text('Supplementary Data S1\n\n'
      'Manuscript: Terahertz fingerprints and machine learning for complementary discrimination of medicinal materials\n'
      'Authors: Ziyuan Zhu, Jiayi Ren, Shuyu Zhou, Mengting Zhu and Yin Shen\n'
      'Corresponding author: Yin Shen; shenyin285819016@126.com\n'
      'ORCID (Ziyuan Zhu, first author): https://orcid.org/0009-0005-1153-4926\n'
      'ORCID (Yin Shen, corresponding author): https://orcid.org/0000-0002-3221-7875\n\n'
      'The measured dataset was supplied by the corresponding author and is a shared resource used in other supervised research. This archive supports the analyses reported in the present manuscript.\n\n'
      'corrected_tables: all included corrected positive and negative settings, task definitions, model settings, paired effects and B1-B5 results.\n'
      'specimen_inputs: recorded labels, frozen roles, workbook registry and nominal header thickness annotations.\n'
      'protocols and mechanism_summaries: the locked analysis definitions and mechanism endpoints.\n'
      'derived_checks and verified_v4_sources: archived predictions/features and the feature-budget reanalysis.\n'
      'kind_lowo: the fixed-pipeline workbook-held-out material protocol, complete grouped partitions and out-of-workbook predictions.\n'
      'accepted_molecular: the two qualified protocatechuic-acid mode and infrared calculations and numerical checks.\n\n'
      'The 18 original spectral workbooks are available from the repository data registry and are not duplicated in this archive. Four workbook views describe the same specimens. Task counts and repeated predictions do not increase physical specimen counts. Only the material-discrimination LOWO analysis adds classifier fits (10 binary and18 four-material folds); all source-group, representation and mechanism results remain archived, and no new quantum calculations were run.\n\n'
      'Canonical repository: https://github.com/jackzhu119/research4\n'
      'SOURCE_MANIFEST.json identifies copied data by SHA-256 and records the source paths.\n')
    target=BASE/'Supplementary_Data_S1.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(stage)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(f'{target}: {len(records)} verified data files, {target.stat().st_size} bytes')
