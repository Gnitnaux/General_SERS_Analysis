import unittest
import numpy as np
from general_sers_analysis import Spectra, Edge
from general_sers_analysis.workflow import Workflow, NodeSpec
from general_sers_analysis.tools.preprocessing import SpectralCut, Interpolate

class ExampleGraphTests(unittest.TestCase):
    def test_sample_mapping_and_stack(self):
        specs={'cut':NodeSpec(SpectralCut(401,403),map_samples=True),
               'align':NodeSpec(Interpolate([401,401.5,402]),map_samples=True,stack_output=True)}
        edges=[Edge('input','cut',source_port='X',data_kind='spectra'),Edge('cut','align',data_kind='spectra')]
        x=[Spectra([[1,3,5,7]],[400,401,402,403],('sample',))]
        w=Workflow(specs,edges,'align').fit(x)
        out=w.predict(x)
        np.testing.assert_allclose(out.intensity,[[3,4,5]])
        self.assertEqual(out.sample_ids,('sample',))
        with self.assertRaises(ValueError): Workflow(specs,edges[:1],'align')

    def test_no_extrapolation(self):
        w=Workflow({'align':NodeSpec(Interpolate([399,400]))},
                   [Edge('input','align',source_port='X')],'align')
        with self.assertRaisesRegex(ValueError,'fit node align'):
            w.fit(Spectra([[1,2]],[400,401]))
