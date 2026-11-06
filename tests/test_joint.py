import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import joblib
import numpy as np
from test_automl import database
from general_sers_analysis.core.data import load_dataset, Spectra
from general_sers_analysis.core.batch import DataBatch, RowSelection
from general_sers_analysis.automl.joint import JointCandidate, build_joint, group_result, run_joint
from general_sers_analysis.automl import Candidate
from general_sers_analysis.workflow import Workflow, NodeSpec
from general_sers_analysis.core import Node, Edge
from general_sers_analysis.prediction import predict_directory


class Selector(Node):
    def transform(self, X): return RowSelection([2, 0])


class JointTests(unittest.TestCase):
    def test_batch_runner_continues_after_failure_and_cli_reports_exit(self):
        from general_sers_analysis.automl.runner import run_experiment
        from general_sers_analysis.cli import build_parser, run_automl_command
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('general_sers_analysis.automl.runner.read_dataset_index',
                       return_value=[{'Data_id':'A'}, {'Data_id':'B'}]), \
                 patch('general_sers_analysis.automl.runner.run_joint',
                       side_effect=[ValueError('bad data'), {'best_candidate': {}}]) as run:
                records = run_experiment(root, 'all', root/'batch', root/'plots', n_trials=2)
                self.assertEqual([r['status'] for r in records], ['failed','ok'])
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[1].args[2], root/'batch/B')
                self.assertEqual(json.loads((root/'batch/runs.json').read_text()), records)
            args = build_parser(root).parse_args(['automl','--data-id','all','--output-dir',str(root/'another')])
            self.assertFalse(hasattr(args, 'search_mode'))
            with patch('general_sers_analysis.automl.runner.run_experiment', return_value=records):
                self.assertEqual(run_automl_command(args), 1)

    def test_selection_maps_labels_and_lineage(self):
        b=DataBatch(np.arange(6).reshape(3,2),('a','b','c'),('g1','g2','g3'),np.arange(3)[:,None])
        w=Workflow({'select':NodeSpec(Selector(),row_action='select')},[Edge('input','select',source_port='X')],'select')
        result=w.fit(b).predict(b)
        self.assertEqual(result.row_ids,('c','a'))
        np.testing.assert_array_equal(result.y[:,0],[2,0])
        self.assertEqual(result.lineage,(('c',),('a',)))
        self.assertEqual(w.to_dict()['nodes']['select']['row_action'],'select')

    def test_all_aggregation_positions_have_identical_scoring_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            database(tmp); ds=load_dataset(tmp,'T001')
            batch=DataBatch.from_dataset(ds,np.arange(32))
            for position in ('none','before_preprocessing','before_model','after_model'):
                c=JointCandidate(Candidate(model='ridge',parameter=1),position,lower=600,upper=640)
                w=build_joint(c).fit(batch)
                result,p=group_result(w.predict(batch))
                self.assertEqual(p.shape,(16,1))
                self.assertEqual(sum(map(len,result.lineage)),32)
                if position!='none': self.assertTrue(any(t['output_rows']<t['input_rows'] for t in w.fit_trace_))
            branched=build_joint(JointCandidate(Candidate(model='ridge',parameter=1),lower=600,upper=640,feature_branch=True)).fit(batch)
            self.assertIn('feature_union',branched.nodes_)
            self.assertEqual(group_result(branched.predict(batch))[1].shape,(16,1))
            # Inconsistent labels must never silently average into a fabricated target.
            batch.y[1]=123
            with self.assertRaisesRegex(ValueError,'inconsistent'):
                build_joint(JointCandidate(Candidate(model='ridge',parameter=1),lower=600,upper=640)).fit(batch)

    def test_joint_end_to_end_and_raw_saved_inference(self):
        for classification in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); database(root/'db',classification)
                result=run_joint(root/'db','T001',root/'run',root/'figures',n_trials=4,lower=600,upper=640)
                self.assertEqual(result['trials_succeeded'],4)
                ds=load_dataset(root/'db','T001'); split=json.loads((root/'run/splits.json').read_text())
                for a,b in [(split['train'],split['test'])]+[(f['train'],f['validation']) for f in split['cv']]:
                    self.assertFalse(set(ds.split_blocks[a]) & set(ds.split_blocks[b]))
                saved=joblib.load(root/'run/workflow.joblib')
                batch=DataBatch.from_dataset(ds,np.asarray(split['test']))
                expected,_=group_result(saved['workflow'].predict(batch),classification)
                output=predict_directory(root/'run/workflow.joblib',root/'db/data/T001'/expected.row_ids[0],root/'new.csv')
                self.assertTrue(output.exists())
