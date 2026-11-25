import json
from pathlib import Path
import tempfile
import unittest

from dashboard import Monitor


class DashboardTest(unittest.TestCase):
    def test_partial_lines_restart_dedup_and_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); wave=root/'remote'/'trial';wave.mkdir(parents=True)
            m=Monitor(root); row=dict(manifest='abc',id=1,event='done',subsets=2**70,anchor_pass=42,two_pass=2)
            line=json.dumps(row)+'\n'; log=wave/'a.jsonl';log.write_text(line[:-1]);self.assertEqual(m.update()['receipts'],0)
            log.write_text(line);(wave/'duplicate.jsonl').write_text(line)
            d=m.update();self.assertEqual(d['counts']['subsets'],str(2**70));self.assertEqual(d['receipts'],1);self.assertEqual(d['verdicts'],{})
            with log.open('a') as f:
                f.write(json.dumps(dict(manifest='abc',id=2,event='candidate',factors=[11,13]))+'\n')
            d=m.update();self.assertEqual(d['verdicts'],{'candidate':1});self.assertEqual(d['events'][0]['factors'],'11 × 13');self.assertEqual(m.update()['verdicts'],{'candidate':1})
            (root/'released.txt').touch();self.assertTrue(m.update()['released'])

    def test_gcd_batches_do_not_count_as_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wave=root/'remote'/'gcd-v1';wave.mkdir(parents=True); (wave/'a.jsonl').write_text(json.dumps({'event':'done','batch':0,'params':[50,0,5,1,2],'prefixes':16})+'\n'); d=Monitor(root).update();self.assertEqual(d['counts']['prefixes'],'16');self.assertEqual(d['counts']['subsets'],'0')
