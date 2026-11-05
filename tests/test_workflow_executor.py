import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from general_sers_analysis import Edge
from general_sers_analysis.workflow import Workflow, NodeSpec
from general_sers_analysis.tools.preprocessing import Standardize
from general_sers_analysis.tools.regression import LinearRegressor, PLSRRegressor
from general_sers_analysis.tools.classification import LogisticClassifier
from general_sers_analysis.tools.visualization import PredictionPlot

class WorkflowTests(unittest.TestCase):
    def make(self, model=None, plot=None):
        specs={'scale':NodeSpec(Standardize()),'model':NodeSpec(model or LinearRegressor(),method='predict')}
        edges=[Edge('input','scale',source_port='X'),Edge('scale','model'),
               Edge('input','model',source_port='y',target_port='fit:y')]
        if plot is not None:
            specs['plot']=NodeSpec(PredictionPlot(plot),stages=('evaluate',))
            edges += [Edge('model','plot'),Edge('input','plot',source_port='y',target_port='y_true')]
        return Workflow(specs,edges,'model')

    def test_fit_only_training_predict_no_labels(self):
        x=np.array([[0,1],[2,3],[4,2],[5,8]],dtype=float); y=x[:,0]*3
        w=self.make()
        with self.assertRaises(NotFittedError): w.predict(x)
        w.fit(x,y)
        mean=w.nodes_['scale'].model_.mean_.copy()
        w.predict(x+100)
        np.testing.assert_array_equal(mean,w.nodes_['scale'].model_.mean_)
        np.testing.assert_allclose(w.predict(x)[:,0],y,atol=1e-10)
        self.assertTrue(any(t['fit'] for t in w.fit_trace_))
        self.assertFalse(any(t['fit'] for t in w.last_trace_))
        self.assertFalse(hasattr(w.specs['model'].tool,'model_'))

    def test_plsr_and_classification_share_executor(self):
        x=np.random.default_rng(8).normal(size=(20,3))
        for model,y in [(PLSRRegressor({'n_components':2}),x[:,:2]),
                        (LogisticClassifier(),np.where(x[:,0]>0,'yes','no'))]:
            w=self.make(model).fit(x,y)
            self.assertEqual(len(w.predict(x)),20)

    def test_plot_is_evaluation_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plot.png'; x=np.arange(12.).reshape(6,2); y=x[:,0,None]
            w=self.make(plot=p).fit(x,y)
            w.predict(x)
            self.assertFalse(p.exists())
            self.assertEqual(set(w.evaluate(x,y)),{'scale','model','plot'})
            self.assertTrue(p.exists())

    def test_serialization_allowlist_and_refit(self):
        x=np.arange(12.).reshape(6,2); y=x[:,0,None]
        w=self.make().fit(x,y)
        spec=json.loads(json.dumps(w.to_dict()))
        v=Workflow.from_dict(spec,{'Standardize':Standardize,'LinearRegressor':LinearRegressor}).fit(x,y)
        np.testing.assert_allclose(w.predict(x),v.predict(x))
        with self.assertRaises(ValueError): Workflow.from_dict(spec,{})
        w.fit(x+100,y)
        np.testing.assert_allclose(w.nodes_['scale'].model_.mean_, x.mean(axis=0)+100)

    def test_invalid_graphs_and_ports(self):
        specs={'a':NodeSpec(Standardize()),'b':NodeSpec(Standardize())}
        for edges in ([Edge('a','b'),Edge('b','a')],
                      [Edge('missing','a'),Edge('a','b')],
                      [Edge('input','a',source_port='X'),Edge('input','a',source_port='y'),Edge('a','b')],
                      [Edge('input','a',source_port='X'),Edge('a','b',target_port='unknown')]):
            with self.assertRaises(ValueError): Workflow(specs,edges,'b')
        w=self.make()
        with self.assertRaises(ValueError): w.fit(np.ones((4,2)))
        with self.assertRaises(NotFittedError): w.predict(np.ones((4,2)))

    def test_supervised_model_cascade_rejected(self):
        specs={k:NodeSpec(LinearRegressor(),method='predict') for k in ('a','b')}
        edges=[Edge('input','a',source_port='X'),Edge('a','b')]
        edges += [Edge('input',k,source_port='y',target_port='fit:y') for k in specs]
        with self.assertRaisesRegex(ValueError,'out-of-fold'): Workflow(specs,edges,'b')

    def test_parallel_branches_and_named_merge_ports(self):
        from general_sers_analysis import Node
        class SumNode(Node):
            def transform(self,X,*,other): return X+other
        specs={'a':NodeSpec(Standardize()),'b':NodeSpec(Standardize()),'sum':NodeSpec(SumNode())}
        edges=[Edge('input',k,source_port='X') for k in ('a','b')]
        edges += [Edge('a','sum'),Edge('b','sum',target_port='other')]
        x=np.arange(8.).reshape(4,2)
        w=Workflow(specs,edges,'sum').fit(x)
        np.testing.assert_allclose(w.predict(x),2*w.nodes_['a'].transform(x))
