import unittest
from bigcpu import ROOT,native
from campaign import scan


@unittest.skipUnless((ROOT/'join').is_file(),'compile join.cpp first')
class NativeTest(unittest.TestCase):
    def test_collision_and_parity(self):
        # All rows satisfy q=1 mod c, q=-1 mod d; plenty of planted hits.
        core=[(3+4*i,2,4) for i in range(26,39)]
        core+=[(157,3,2),(163,3,4)]
        for anchor in [(211,2,4),(211,3,4)]:
            job=dict(anchor=anchor,core=core); expected=[]
            ref=scan(job,lambda fs:expected.append(list(fs)))
            counts,hits=native(job); self.assertTrue(hits); self.assertEqual(counts,ref);self.assertCountEqual(hits,expected)
