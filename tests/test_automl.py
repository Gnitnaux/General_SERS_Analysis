import csv
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import joblib
import numpy as np

from general_sers_analysis.automl import (Candidate, build_workflow, candidate_space,
                                          run_automl, split_holdout, subset, cv_score)
from general_sers_analysis.core.data import load_dataset as load_raw_dataset
from general_sers_analysis.automl.representation import prepare_group_means


def load_dataset(root, data_id, **options):
    return prepare_group_means(load_raw_dataset(root, data_id), **options)

from general_sers_analysis.core import Spectra
from general_sers_analysis.workflow import Workflow
from general_sers_analysis.tools.model_selection import TargetScaledRegressor, RidgeRegressor, IndependentPresence
from general_sers_analysis.tools.preprocessing import Standardize
from general_sers_analysis.tools.classification import LogisticClassifier
from general_sers_analysis.prediction import predict_directory


def database(root, classification=False):
    root = Path(root)
    info = dict(Data_id='T001', task='Classification' if classification else 'Single-Component Quantification',
                modality='Raman', n_classes=2 if classification else 1,
                molecules='A_B' if classification else 'A', quant_unit='null' if classification else 'uM',
                n_group=16, n_spectra=32)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'content.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(info)); writer.writeheader(); writer.writerow(info)
    groups = []
    axis = np.arange(600, 641, 2.)
    rng = np.random.default_rng(42)
    for i in range(16):
        labels = [i % 2, (i // 2) % 2] if classification else [i * 1e-9]
        name = ('Class_' if classification else 'Quan_') + '_'.join(map(str, labels)) + f'_{i:06}'
        folder = root / 'data' / 'T001' / name
        folder.mkdir(parents=True)
        for repeat in range(2):
            signal = (1 + np.sin(axis / 15)) * (i % 2 if classification else i)
            if classification:
                signal += labels[1] * np.cos(axis / 5)
            signal += rng.normal(0, .001, len(axis))
            with (folder / f'{repeat}.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['RamanShift', 'intensity']); writer.writerows(zip(axis, signal))
        groups.append({'folder': name, 'labels': labels, 'shared_sample_id': f'sample{i // 2}'})
    (root / 'data' / 'T001' / 'metadata.json').write_text(json.dumps({'groups': groups}), encoding='utf-8')
    return groups


class AutoMLTests(unittest.TestCase):
    def test_shared_reader_preserves_individual_spectra_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            database(tmp)
            ds = load_raw_dataset(tmp, 'T001')
            self.assertEqual(len(ds.spectra), 32)
            self.assertEqual(ds.y.shape, (32, 1))
            self.assertEqual(len(set(ds.group_ids)), 16)
            self.assertEqual(len(set(ds.split_blocks)), 8)
            self.assertEqual(len(set(ds.sample_ids)), 32)
            self.assertEqual(ds.spectra[0].intensity.shape, (1, 21))
            self.assertNotIn('grid', ds.audit)
            prepared = prepare_group_means(ds, lower=600, upper=640)
            self.assertEqual(prepared.X.intensity.shape, (16, 21))
            from general_sers_analysis.tools.aggregation import MeanAggregator
            self.assertIsInstance(prepared.representation_workflow.specs['group_mean'].tool, MeanAggregator)

    def test_shared_label_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            groups = database(tmp)
            groups[0]['label_review_status'] = 'conflict_requires_source_clarification'
            p = Path(tmp) / 'data/T001/metadata.json'
            p.write_text(json.dumps({'groups': groups}), encoding='utf-8')
            self.assertEqual(len(load_raw_dataset(tmp, 'T001').spectra), 30)
            included = load_raw_dataset(tmp, 'T001', label_policy='include_flagged')
            self.assertEqual(len(included.spectra), 32)
            self.assertIn('conflict', included.audit['quality_flags'][groups[0]['folder']])
            with self.assertRaisesRegex(ValueError, 'review flag'):
                load_raw_dataset(tmp, 'T001', label_policy='error')

    def test_block_holdout_and_cv_are_disjoint(self):
        g = np.repeat(np.arange(12), 3)
        a, b, _ = split_holdout(g, .25, 0)
        self.assertFalse(set(g[a]) & set(g[b]))
        a2, b2, _ = split_holdout(g, .25, 0)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)

    def test_author_test_partition_expands_to_blocks(self):
        g = np.repeat(np.arange(8), 2)
        types = ['train'] * 16; types[-1] = 'test'
        a, b, kind = split_holdout(g, .25, 0, types)
        np.testing.assert_array_equal(b, [14, 15])
        self.assertIn('author', kind)

    def test_tiny_unit_targets_are_actually_learned_and_restored(self):
        X = np.arange(20.)[:, None]
        y = X * 1e-10
        m = TargetScaledRegressor(RidgeRegressor({'alpha': 1e-8})).fit(X, y)
        np.testing.assert_allclose(m.predict(X), y, atol=1e-18, rtol=1e-8)

    def test_nested_node_json_roundtrip_and_allowlist(self):
        X = Spectra(np.arange(100.).reshape(10, 10), np.arange(10.))
        y = np.arange(10.)[:, None]
        w = build_workflow(Candidate(model='ridge', parameter=1)).fit(X, y)
        spec = json.loads(json.dumps(w.to_dict()))
        registry = {c.__name__: c for c in (TargetScaledRegressor, RidgeRegressor, Standardize)}
        restored = Workflow.from_dict(spec, registry).fit(X, y)
        np.testing.assert_allclose(w.predict(X), restored.predict(X))
        del registry['RidgeRegressor']
        with self.assertRaisesRegex(ValueError, 'not in registry'):
            Workflow.from_dict(spec, registry)

    def test_loader_means_shared_ids_conflict_and_window_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); groups = database(root)
            groups[0]['label_consistent_with_checked_SI_table'] = False
            (root / 'data/T001/metadata.json').write_text(json.dumps({'groups': groups}))
            ds = load_dataset(root, 'T001', lower=600, upper=640)
            self.assertEqual(len(ds.y), 15)
            self.assertEqual(len(set(ds.groups)), 8)
            self.assertEqual(ds.audit['counts'][0]['used'], 2)
            self.assertEqual(ds.audit['exclusions'][0]['reason'], 'unresolved source-label review flag')

    def test_rare_component_constant_training_fold(self):
        X = np.arange(20.).reshape(10, 2)
        y = np.column_stack([np.zeros(10), np.arange(10) % 2])
        m = IndependentPresence(LogisticClassifier({'max_iter': 1000})).fit(X, y)
        self.assertEqual(m.predict(X).shape, y.shape)
        np.testing.assert_array_equal(m.predict(X)[:, 0], 0)

    def test_exact_duplicates_join_otherwise_separate_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); groups = database(root)
            directory = root / 'data' / 'T001'
            shutil.copyfile(directory / groups[0]['folder'] / '0.csv', directory / groups[6]['folder'] / '0.csv')
            ds = load_dataset(root, 'T001', lower=600, upper=640)
            blocks = dict(zip(ds.X.sample_ids, ds.groups))
            self.assertEqual(blocks[groups[0]['folder']], blocks[groups[6]['folder']])
            self.assertEqual(blocks[groups[1]['folder']], blocks[groups[7]['folder']])

    def test_single_presence_target_f1_counts_positive_class(self):
        y = np.array([[0], [0], [1]])
        self.assertEqual(cv_score(y, np.zeros_like(y), y, True), 0.)

    def test_no_eligible_window_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            database(tmp)
            with self.assertRaisesRegex(ValueError, 'fewer than four eligible'):
                load_dataset(tmp, 'T001', lower=700, upper=800)

    def test_search_deterministic_and_bounded(self):
        self.assertEqual(candidate_space(False, 42, 0), candidate_space(False, 42, 0))
        self.assertEqual(len(set(candidate_space(False, 42, 0))), 42)

    def test_end_to_end_artifacts_and_fold_local_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); database(root / 'db')
            output = root / 'run'
            s = run_automl(root / 'db', 'T001', output, root / 'plots', n_trials=3,
                           lower=600, upper=640, cv=3)
            ds = load_dataset(root / 'db', 'T001', lower=600, upper=640)
            split = json.loads((output / 'splits.json').read_text())
            for a, b in [(split['train'], split['test'])] + [(f['train'], f['validation']) for f in split['cv']]:
                self.assertFalse(set(ds.groups[a]) & set(ds.groups[b]))
            saved = joblib.load(output / 'workflow.joblib')['workflow']
            np.testing.assert_allclose(saved.nodes_['feature_scale'].model_.mean_, ds.X.intensity[split['train']].mean(axis=0))
            pred = saved.predict(subset(ds.X, split['test']))
            self.assertEqual(pred.shape, (len(split['test']), 1))
            self.assertTrue(Path(s['figures']['holdout']).exists())
            self.assertTrue((output / 'workflow_spec.json').exists())
            self.assertTrue((output / 'baseline_fixed_pls2.joblib').exists())
            input_folder = root / 'db' / 'data' / 'T001' / ds.X.sample_ids[split['test'][0]]
            predictions_csv = predict_directory(output / 'workflow.joblib', input_folder, root / 'new_predictions.csv')
            with predictions_csv.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]['pred_A']), pred[0, 0], places=18)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                predict_directory(output / 'workflow.joblib', input_folder, predictions_csv)
            with self.assertRaisesRegex(ValueError, 'not empty'):
                run_automl(root / 'db', 'T001', output, root / 'plots', n_trials=1)

    def test_multilabel_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); database(root / 'db', True)
            s = run_automl(root / 'db', 'T001', root / 'run', root / 'plots', n_trials=2,
                           lower=600, upper=640)
            self.assertGreaterEqual(s['holdout_metrics']['macro_f1'], .8)


if __name__ == '__main__':
    unittest.main()
