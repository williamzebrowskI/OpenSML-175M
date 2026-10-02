import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sml_v1.batch_transition import change_batches
from sml_v1.common import fingerprint,file_sha256
from sml_v1.recipe import learning_rate,tokens_per_update
from sml_v1.continuation import check_resume
ROOT=Path(__file__).resolve().parents[2]

class BatchChangeTests(unittest.TestCase):
    def parent(self):
        recipe=json.loads((ROOT/'configs/pretrain_20b.json').read_text())
        return dict(recipe=recipe,step=125,tokens=14336000,replica_hash='a'*64,
            v2_contract=dict(recipe=fingerprint(recipe),code='58f75d5d92576c268370e8f3dbb59f7cbc5ef58bcd1cf247a7cf6008c47e7e3c',data='data',tokenizer='tokenizer'))
    def contract(self,r):
        code=fingerprint({p.name:file_sha256(p) for p in (ROOT/'pretraining/sml_v1').glob('*.py')})
        return dict(recipe=fingerprint(r),code=code,data='data',tokenizer='tokenizer')
    def test_change_preserves_schedule_and_all_unrelated_settings(self):
        m=self.parent();r=change_batches(m,[6,4,2,2,2]);old=copy.deepcopy(m['recipe'])
        self.assertEqual(tokens_per_update(r),131072)
        self.assertEqual(learning_rate(m['tokens'],r),learning_rate(m['tokens'],old))
        base=copy.deepcopy(r);base.pop('batch_transition');base['batches']=old['batches'];self.assertEqual(base,old)
        self.assertEqual(check_resume(m,self.contract(r),r),'explicit-batch-transition')
        self.assertEqual(r['batch_transition']['start_tokens'],14336000)
    def test_rejects_unrelated_recipe_or_data_or_code_changes(self):
        m=self.parent();r=change_batches(m,[6,4,2,2,2]);r['peak_lr']*=2
        with self.assertRaises(ValueError):check_resume(m,self.contract(r),r)
        r=change_batches(m,[6,4,2,2,2]);c=self.contract(r);c['data']='changed'
        with self.assertRaises(ValueError):check_resume(m,c,r)
        c=self.contract(r);c['code']='unreviewed'
        with self.assertRaises(ValueError):check_resume(m,c,r)
    def test_new_checkpoint_can_resume_unchanged(self):
        m=self.parent();r=change_batches(m,[6,4,2,2,2]);c=self.contract(r)
        m.update(recipe=r,v2_contract=c,step=126,tokens=14467072)
        self.assertEqual(check_resume(m,c,r),'unchanged')
        self.assertEqual(change_batches(m,[6,4,2,2,2]),r)
        with self.assertRaises(ValueError):change_batches(m,[8,4,2,2,2])
    def test_rejects_invalid_or_untrained_parent(self):
        for batches in ([6,4,2,2], [6,4,2,2,0], [4,4,2,2,2]):
            with self.assertRaises(ValueError):change_batches(self.parent(),batches)
        m=self.parent();m['tokens']=0
        with self.assertRaises(ValueError):change_batches(m,[6,4,2,2,2])
