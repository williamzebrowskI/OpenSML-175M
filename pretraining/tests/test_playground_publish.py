import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sml_v1.common import atomic_json, read_json
from sml_v1.playground_publish import publish_best,publish_safely
from completion_playground import create_app, ModelSource
from fastapi.testclient import TestClient


def checkpoint(root,step):
    name=f'step_{step:07d}_abcdef';p=root/name;p.mkdir()
    (p/'model.safetensors').write_bytes(b'weights '+str(step).encode())
    atomic_json(p/'model.safetensors.json',dict(step=step,tokens=step*1000,best_val_loss=3.,
                    v2_contract={'tokenizer':'fixture'},recipe={'model':{'max_seq_len':2048}}))
    records={}
    for path in p.iterdir():
        raw=path.read_bytes();records[path.name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    atomic_json(p/'manifest.json',dict(step=step,files=records))
    atomic_json(root/'best.json',dict(bundle=name))
    return p


class PublishTests(unittest.TestCase):
    def test_best_snapshot_survives_training_checkpoint_prune_and_same_best_is_reused(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=checkpoint(root,5)
            result=publish_best(root,event_step=10,event_tokens=100_000_000)
            preview=Path(result['bundle']);self.assertEqual(len(list(preview.iterdir())),3)
            publish_best(root,event_step=20,event_tokens=200_000_000)
            self.assertEqual(len(list((root/'playground').glob('step_*'))),1)
            shutil.rmtree(source)
            c=ModelSource('research','Best',root/'playground').checkpoint()
            self.assertEqual(c.metadata['step'],5)
            self.assertEqual(read_json(root/'playground/best.json')['event_tokens'],200_000_000)

    def test_corrupt_best_cannot_replace_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint(root,5);publish_best(root,event_step=10,event_tokens=100_000_000)
            old=(root/'playground/best.json').read_bytes()
            bad=checkpoint(root,15);(bad/'model.safetensors').write_bytes(b'corrupt')
            self.assertIsNone(publish_safely(root,event_step=20,event_tokens=200_000_000))
            self.assertEqual((root/'playground/best.json').read_bytes(),old)
            self.assertFalse(list((root/'playground').glob('.publish_*')))

    def test_server_waits_then_follows_new_published_best_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);app=create_app(root/'playground')
            try:
                with TestClient(app) as client:
                    self.assertFalse(client.get('/api/status').json()['available'])
                    checkpoint(root,5);publish_best(root,event_step=10,event_tokens=100_000_000)
                    self.assertEqual(client.get('/api/status').json()['step'],5)
                    checkpoint(root,15);publish_best(root,event_step=20,event_tokens=200_000_000)
                    self.assertEqual(client.get('/api/status').json()['step'],15)
                    self.assertEqual(client.get('/').status_code,200)
            finally:app.state.service.close()

    def test_disk_error_reporting_does_not_fail_training(self):
        with patch('sml_v1.playground_publish.publish_best',side_effect=OSError('disk')), \
             patch('sml_v1.playground_publish.atomic_json',side_effect=OSError('disk')):
            self.assertIsNone(publish_safely('/unused',event_step=1,event_tokens=1))

if __name__=='__main__':unittest.main()
