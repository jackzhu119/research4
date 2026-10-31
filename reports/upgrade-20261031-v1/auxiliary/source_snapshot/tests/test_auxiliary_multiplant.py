from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from herb_spectra.auxiliary_multiplant import (FUSIONS,PROFILES,fit_candidate,grids,
    read_pack,save_fitted,load_fitted,metrics,default_protocol,write_json,write_predictions,
    sha,manifest,score,FAMILIES,DATASETS,trial)
from herb_spectra.leakage_guard import LeakageError,audit_pointwise_inference
from herb_spectra.upgrade_neural import state_hash


class AuxiliaryMultiplantTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.X=np.random.default_rng(7).normal(size=(32,40))
        self.ids=np.asarray([f'unit-source-{i:02d}' for i in range(32)])
        self.labels=np.asarray([f'HQ{i%4+1:02d}' for i in range(32)])
    def tearDown(self):self.tmp.cleanup()
    def fit(self,candidate,**kwargs):
        return fit_candidate({'alpha':self.X},self.labels,self.ids,'huangqi',candidate,42,'unit-split',self.folder/'scope.jsonl','unit-code',expected_source_ids=kwargs.get('expected',self.ids))

    def test_actual_source_ids_must_match_independent_context(self):
        with self.assertRaisesRegex(LeakageError,'fixed allowed source'):
            self.fit(grids()['full_svm'][0],expected=[*self.ids[:-1],'unit-forbidden-query'])

    def test_two_plants_cannot_share_one_class_space(self):
        labels=self.labels.copy();labels[0]='TM01'
        with self.assertRaisesRegex(LeakageError,'separate plant label'):
            fit_candidate({'alpha':self.X},labels,self.ids,'huangqi',grids()['full_svm'][0],42,'unit-split',None,'unit',expected_source_ids=self.ids)

    def test_inference_pack_rejects_labels_and_metadata_features(self):
        p=self.folder/'pack.npz';np.savez(p,record_ids=self.ids,X_alpha=self.X,source_labels=self.labels)
        with self.assertRaisesRegex(LeakageError,'pack schema'):read_pack(p)
        np.savez(p,record_ids=self.ids,X_alpha=np.asarray([['filename.xlsx']]*32))
        with self.assertRaisesRegex(LeakageError,'paired spectra'):read_pack(p)

    def test_original_mcuve_pls_calls_are_source_scoped(self):
        fitted=self.fit(grids()['mcuve_svm'][0])
        lines=[json.loads(v) for v in (self.folder/'scope.jsonl').read_text().splitlines()]
        pls=[r for r in lines if r['event']=='fit_start' and r['component']=='MC-UVE.PLSRegression.fit']
        # Legacy classification MCUVE fits one PLS response per source class.
        self.assertEqual(len(pls),10*4)
        self.assertTrue(all(set(r['fit_sample_ids'])<=set(self.ids) for r in pls))
        self.assertEqual(len(fitted.selector.selected_indices_),4)

    def test_cached_weight_refuses_another_fold_or_source(self):
        fitted=self.fit(grids()['full_svm'][0]);p=self.folder/'pipeline.joblib';save_fitted(fitted,p)
        self.assertEqual(load_fitted(p,expected_split='unit-split',expected_ids=self.ids).classes,tuple(f'HQ{i:02d}' for i in range(1,5)))
        with self.assertRaisesRegex(LeakageError,'Cross-fold'):load_fitted(p,expected_split='other-fold')
        with self.assertRaisesRegex(LeakageError,'Cross-source'):load_fitted(p,expected_ids=self.ids[::-1])

    def test_four_class_nn_batchnorm_inference_is_pointwise_and_frozen(self):
        candidate={**PROFILES[0],'id':'unit-only','kind':'se','width':4,'blocks':1,'epochs':1}
        fitted=self.fit(candidate);before=state_hash(fitted.estimator)
        values=fitted.transform({'alpha':self.X[:7]})
        result=audit_pointwise_inference(fitted.estimator,values,batch_sizes=(1,3),atol=1e-5,rtol=1e-5)
        self.assertTrue(result['parameters_and_buffers_unchanged']);self.assertEqual(before,state_hash(fitted.estimator))
        self.assertEqual(fitted.predict_scores({'alpha':self.X[:7]}).shape,(7,4))

    def test_missing_modality_routes_are_explicit_not_silent_class_deletion(self):
        self.assertEqual(FUSIONS['huangqi'],('time','frequency'))
        self.assertEqual(FUSIONS['tianma'],('alpha','n'))
        self.assertEqual(len(grids()['full_svm']),4);self.assertEqual(len(grids()['mcuve_svm']),8)

    def test_scoring_rejects_repeated_sample_denominator(self):
        truth={'a':'HQ01','b':'HQ02'};rows=[{'record_id':'a','y_predicted':'HQ01'},{'record_id':'a','y_predicted':'HQ02'}]
        with self.assertRaisesRegex(LeakageError,'denominator'):metrics(rows,truth,('HQ01','HQ02','HQ03','HQ04'))

    def test_svm_trial_tail_batches_use_exact_inference_population(self):
        # v2 failed before scoring its first real trial because7 checked rows
        # batched by3 silently included rows8/9 from the larger query pack.
        for query_count,outer in [(1,False),(2,False),(3,False),(7,False),(13,False),(13,True)]:
            with self.subTest(query_count=query_count,outer=outer):
                folder=self.folder/f'tail{query_count}-{outer}';folder.mkdir()
                source=folder/'source.npz';query=folder/'inference.npz'
                np.savez(source,record_ids=self.ids,X_alpha=self.X,source_labels=self.labels)
                ids=np.asarray([f'unit-query-{i}' for i in range(query_count)])
                np.savez(query,record_ids=ids,X_alpha=np.random.default_rng(19).normal(size=(query_count,40)))
                context={'dataset_id':'huangqi','split_id':f'unit-tail-{query_count}-{outer}','outer_fold':0,'inner_fold':None if outer else 0,
                    'source_pack':str(source),'source_pack_sha256':sha(source),'inference_pack':str(query),'inference_pack_sha256':sha(query),
                    'source_ids':self.ids.tolist(),'inference_ids':ids.tolist()}
                record,rows=trial(context,'full_svm',grids()['full_svm'][0],42,folder,'unit-code',outer=outer)
                self.assertEqual(len(rows),query_count)
                self.assertEqual(record['inference_audit']['samples_checked'],query_count if outer else min(7,query_count))
                self.assertEqual(set(record['inference_audit']['batch_maximum_difference']),{'1','2','3','7'})
                self.assertTrue(all(v<1e-12 for v in record['inference_audit']['batch_maximum_difference'].values()))

    def boundary_fixture(self):
        """Tiny prehashed files exercise the score boundary; no model is fitted."""
        out=self.folder/'boundary';out.mkdir();protocol=default_protocol();protocol['sample_counts']={d:5 for d in DATASETS}
        pp=out/'protocol.json';write_json(pp,protocol);write_json(out/'protocol_snapshot.json',protocol)
        contexts=[];choices=[];records=[];predictions=[]
        for dataset in DATASETS:
            classes=protocol['class_orders'][dataset];all_ids=[f'{dataset}:unit{i}' for i in range(5)]
            for fold,rid in enumerate(all_ids):
                base=out/f'{dataset}-{fold}';base.mkdir()
                sp=base/'source.npz';ip=base/'inference.npz';tp=base/'truth.csv'
                sp.write_bytes(b'fixture source bytes');ip.write_bytes(b'fixture inference bytes')
                tp.write_text(f'record_id,label\n{rid},{classes[fold%4]}\n')
                context={'dataset_id':dataset,'outer_fold':fold,'stage':'outer','source_pack':str(sp),'source_pack_sha256':sha(sp),
                    'inference_pack':str(ip),'inference_pack_sha256':sha(ip),'source_ids':[v for v in all_ids if v!=rid],
                    'inference_ids':[rid],'scoring_truth':str(tp),'scoring_truth_sha256':sha(tp)};contexts.append(context)
                selected={f:{'selected':grids()[f][0]} for f in FAMILIES}
                choices.append({'dataset_id':dataset,'outer_fold':fold,'choices':selected,
                    'inner_selected_neural':{'family':'cnn','candidate':selected['cnn']['selected']}})
                for family in FAMILIES:
                    for seed in protocol['seeds']:
                        candidate=selected[family]['selected'];pipe=base/f'{family}-{seed}.joblib';meta=pipe.with_suffix('.json');pred=pipe.with_suffix('.csv.gz')
                        pipe.write_bytes(b'fixture model bytes');meta.write_text('{}\n');pred.write_bytes(b'fixture per-trial prediction bytes')
                        records.append({'dataset_id':dataset,'outer_fold':fold,'family':family,'seed':seed,'candidate':candidate,
                            'pipeline':str(pipe),'pipeline_sha256':sha(pipe),'pipeline_metadata':str(meta),'pipeline_metadata_sha256':sha(meta),
                            'predictions':str(pred),'predictions_sha256':sha(pred),'source_ids':context['source_ids'],'inference_ids':[rid],
                            'source_pack_sha256':sha(sp),'inference_pack_sha256':sha(ip)})
                        row={'dataset_id':dataset,'outer_fold':fold,'family':family,'seed':seed,'record_id':rid,
                            'candidate_id':candidate['id'],'architecture_family':family,'y_predicted':classes[fold%4],
                            **{'score_'+c:float(i==fold%4) for i,c in enumerate(classes)}};predictions.append(row)
                        if family=='cnn':predictions.append({**row,'family':'inner_selected_neural'})
        write_json(out/'preparation.json',{'contexts':contexts});write_json(out/'split_manifest.json',{'fixture':True})
        current=manifest(pp,out);write_json(out/'run_manifest.json',current)
        write_json(out/'source_release_snapshot.json',{'allow_source_training':True,**current})
        write_json(out/'inner_metrics.json',[]);write_predictions(out/'inner_predictions.csv.gz',[])
        write_json(out/'selection_lock.json',{'created_at_utc':'2026-10-31T00:00:00+00:00','manifest_sha256':sha(out/'run_manifest.json'),
            'protocol_sha256':sha(pp),'preparation_sha256':sha(out/'preparation.json'),'source_release_sha256':sha(out/'source_release_snapshot.json'),
            'inner_metrics_sha256':sha(out/'inner_metrics.json'),'inner_predictions_sha256':sha(out/'inner_predictions.csv.gz'),
            'inner_trials':1200,'outer_test_used_to_select':False,'outer_folds':choices})
        write_json(out/'outer_release_snapshot.json',{'allow_outer_evaluation':True,'released_at_utc':'2026-10-31T00:01:00+00:00',
            'selection_lock_sha256':sha(out/'selection_lock.json'),'protocol_sha256':sha(pp),'manifest_sha256':sha(out/'run_manifest.json')})
        write_json(out/'outer_model_records.json',records);write_predictions(out/'outer_predictions.csv.gz',predictions)
        write_json(out/'prediction_lock.json',{'created_at_utc':'2026-10-31T00:02:00+00:00','outer_truth_parsed':False,
            'prediction_sha256':sha(out/'outer_predictions.csv.gz'),'outer_model_records_sha256':sha(out/'outer_model_records.json'),
            'manifest_sha256':sha(out/'run_manifest.json'),'protocol_sha256':sha(pp),'preparation_sha256':sha(out/'preparation.json'),
            'selection_lock_sha256':sha(out/'selection_lock.json'),'root_outer_release_sha256':sha(out/'outer_release_snapshot.json'),
            'source_sha256':current['source_sha256'],'class_orders':protocol['class_orders'],'methods':[*FAMILIES,'inner_selected_neural'],
            'seeds':protocol['seeds'],'samples_per_seed':protocol['sample_counts'],'prediction_rows':210,'fitted_model_count':180,
            'pipeline_metadata_sha256':{r['pipeline_metadata']:r['pipeline_metadata_sha256'] for r in records}})
        return pp,out,contexts

    def test_locked_boundary_scores_only_intact_fixture(self):
        pp,out,_=self.boundary_fixture();result=score(pp,out)
        self.assertEqual(result['scored_rows'],210)
        self.assertEqual(json.loads((out/'pooled_metrics.json').read_text())['huangqi']['full_svm']['42']['correct'],5)

    def test_tampered_prediction_rejected_before_any_truth_parse(self):
        pp,out,_=self.boundary_fixture()
        with (out/'outer_predictions.csv.gz').open('ab') as h:h.write(b'tampered')
        with patch('herb_spectra.auxiliary_multiplant.read_truth',side_effect=AssertionError('Truth parsed before integrity rejection')) as read:
            with self.assertRaisesRegex(LeakageError,'Prediction lock'):score(pp,out)
            read.assert_not_called()
        self.assertFalse((out/'pooled_metrics.json').exists())

    def test_last_tampered_truth_rejected_before_even_first_truth_parse(self):
        pp,out,contexts=self.boundary_fixture()
        Path(contexts[-1]['scoring_truth']).write_text('record_id,label\nwrong,TM01\n')
        with patch('herb_spectra.auxiliary_multiplant.read_truth',side_effect=AssertionError('Truth parsed before integrity rejection')) as read:
            with self.assertRaisesRegex(LeakageError,'outer truth changed before parsing'):score(pp,out)
            read.assert_not_called()
        self.assertFalse((out/'pooled_metrics.json').exists())

    def test_tampered_protocol_rejected_before_any_truth_parse(self):
        pp,out,_=self.boundary_fixture();p=json.loads(pp.read_text());p['seeds']=[42];write_json(pp,p)
        with patch('herb_spectra.auxiliary_multiplant.read_truth') as read:
            with self.assertRaisesRegex(LeakageError,'implementation or preparation'):score(pp,out)
            read.assert_not_called()


if __name__=='__main__':unittest.main()
