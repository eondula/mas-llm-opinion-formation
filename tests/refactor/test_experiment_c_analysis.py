import copy
import json
import tempfile
import unittest
from pathlib import Path

from llm_mas.experiment_c_analysis import execute, select_cases, switches, replay_row, heatmaps
from test_interventions import Tiny, Block


def record(schedule='synchronous'):
    rows=[]
    for t in range(8):
        for agent,strategy in [('Agent A',1),('Agent B',2 if t<2 else 1)]:
            score={'margin':1 if strategy==1 else -1}
            rows.append(dict(agent=agent,model='toy',round=t,strategy=strategy,forced=t==0,
                             before=score,decision=None if t==0 else score,post=score))
    return dict(scenario={'id':'sensor'},config={'rounds':7,'schedule':schedule,'models':['toy','toy'],'initialization':[1,2]},rows=rows)


class AnalysisTests(unittest.TestCase):
    def test_selection_and_forced_switch(self):
        a=record();b=record('sequential')
        for r in b['rows']:
            if r['round']>0:r['strategy']=2
        cases, selection=select_cases({'a':a,'b':b})
        self.assertTrue(any(c['reason']=='schedule_sensitivity' for c in cases))
        self.assertTrue(any(s['reason']=='repeated_switching' and s['status']=='absent' for s in selection))
        self.assertEqual(switches([r for r in a['rows'] if r['agent']=='Agent B']),[2])
        forced=[{'round':0,'strategy':2,'decision':None},{'round':1,'strategy':1,'decision':{'margin':1}}]
        self.assertEqual(switches(forced),[])

    def test_outputs_preserve_source(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'conversations').mkdir();(root/'figures').mkdir()
            source=root/'conversations/a.json';source.write_text(json.dumps(record()))
            original=source.read_bytes();existing=root/'figures/existing.png';existing.write_bytes(b'untouched')
            cases=execute(root)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(existing.read_bytes(),b'untouched')
            self.assertEqual(len(list((root/'figures').glob('analysis*.png'))),len(cases))
            for f in (root/'figures').glob('analysis*.png'):
                with Image.open(f) as im:im.verify()
            self.assertFalse(list(root.rglob('*.pdf')))
            incomplete=record();incomplete['rows'].pop();source.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError,'Incomplete'):execute(root)

    def test_all_layers_patch_and_cleanup(self):
        runtime=Tiny();runtime.model.model.layers.append(Block())
        history=[{'role':'user','content':'Choose.'},{'role':'user','content':'Peer says A.'}]
        scenario={'strategy_1':{'candidate':'AB'},'strategy_2':{'candidate':'BA'}}
        measurement=runtime.measure(history,scenario,{'stance_probe':'A or B?'})
        row=dict(agent='Agent A',round=1,decision_history=history,deliveries=[{}],decision=measurement)
        results=replay_row(runtime,row,scenario,1e-5)
        self.assertEqual([r['layer'] for r in results],[0,1])
        self.assertTrue(all(abs(r['self_patch_error'])<1e-5 for r in results))
        self.assertTrue(all(r['activation_change']>=0 for r in results))
        for module in runtime.model.modules():self.assertFalse(module._forward_hooks)
        broken=copy.deepcopy(row);broken['decision']['margin']+=1
        with self.assertRaisesRegex(ValueError,'Replay baseline'):replay_row(runtime,broken,scenario,1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'figures').mkdir()
            heatmaps(root,dict(id='analysis_case01',tasks=['a'],reason='test'),{'a':record()},[dict(task='a',**r) for r in results])
            self.assertEqual(len(list((root/'figures').glob('*.png'))),2)
