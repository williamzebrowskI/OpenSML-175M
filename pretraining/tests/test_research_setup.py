import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sml_v1.common import ROOT, read_json, fingerprint
from sml_v1.recipe import validate, tokens_per_update, learning_rate
from sml_v1.sample_prompts import sample_event
from sml_v1.stream import stream_manifest
from launch_research import main
from five_mac_network import existing_plan, plan
from configure_five_mac_network import read_plan
from test_five_mac_network import network_fixture


class LivePromptTests(unittest.TestCase):
    def test_edits_are_loaded_between_events_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); prompts = p/'prompts.json'
            prompts.write_text(json.dumps({'prompts':['First prompt']}))
            calls = []
            def generate(text):
                calls.append(text)
                return 'answer '+text
            first = sample_event(1, 100, generate, p, prompts_file=prompts)
            prompts.write_text(json.dumps({'prompts':['Replacement','Another prompt']}))
            second = sample_event(2, 200, generate, p, prompts_file=prompts)
            self.assertEqual(calls, ['First prompt','Replacement','Another prompt'])
            self.assertNotEqual(first['prompts_sha256'], second['prompts_sha256'])
            self.assertEqual(read_json(p/'samples/step_0000002.json'), second)

    def test_partial_deleted_or_invalid_edit_skips_without_stale_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); prompts=p/'prompts.json'
            for text in ('{', '{"prompts": []}', '{"prompts": [123]}', '{"prompts": [""]}'):
                prompts.write_text(text)
                with patch('builtins.print'), patch('builtins.input'):
                    result=sample_event(1,100,lambda _:self.fail('should not generate'),p,
                                        prompts_file=prompts,legacy_prompts=['stale'])
                self.assertEqual(result['status'],'skipped')
            prompts.unlink()
            self.assertEqual(sample_event(2,200,lambda _:self.fail('should not generate'),p,
                                          prompts_file=prompts)['status'],'skipped')

    def test_overlong_token_context_skips_only_that_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); prompts=p/'prompts.json'
            prompts.write_text(json.dumps({'prompts':['too long','okay']}))
            def generate(text):
                if text=='too long':raise ValueError('context exceeded')
                return 'answer'
            result=sample_event(1,100,generate,p,prompts_file=prompts)
            self.assertEqual([r['status'] for r in result['results']],['skipped','generated'])


class ResearchSetupTests(unittest.TestCase):
    def test_budget_schedule_and_five_rank_plan_without_side_effects(self):
        recipe=validate(read_json(ROOT.parent/'configs/pretrain_20b.json'))
        self.assertEqual(tokens_per_update(recipe),114688)
        self.assertEqual(recipe['target_tokens'],20_000_000_000)
        self.assertAlmostEqual(learning_rate(200_000_000,recipe),3e-4)
        self.assertAlmostEqual(learning_rate(16_000_000_000,recipe),3e-4)
        self.assertAlmostEqual(learning_rate(20_000_000_000,recipe),3e-5)
        with patch('builtins.print'), patch('subprocess.run',side_effect=AssertionError('no subprocess')):
            main(['--output',str(ROOT/'runs/test-plan-only')])
        self.assertNotIn('prompts',recipe)

    def test_frozen_tokenizer_reuse_requires_explicit_policy_and_tracks_corpus(self):
        corpus=read_json(ROOT.parent/'configs/corpus_20b.json')
        tokenizer=SimpleNamespace(manifest={'corpus_fingerprint':'old-corpus'}, fingerprint='frozen')
        with self.assertRaisesRegex(ValueError,'differs'):
            stream_manifest(corpus,tokenizer,{})
        settings={'tokenizer_corpus_policy':'reuse-frozen'}
        original=stream_manifest(corpus,tokenizer,settings)
        changed=copy.deepcopy(corpus);changed['sources'][0]['weight']=49
        self.assertNotEqual(fingerprint(original), fingerprint(stream_manifest(changed,tokenizer,settings)))

    def test_adopted_network_is_read_only_verified_and_reconstructed(self):
        reports=network_fixture(); planned=plan(reports,[0,1,2,4,3])
        for report,node in zip(reports,planned['nodes']):
            text='en0: flags=1<UP>\n inet 192.168.12.10 netmask 0xffffff00\n'
            for e in node['endpoints']:
                text+=f'{e["interface"]}: flags=1<UP>\n inet {e["ip"]} netmask 0xfffffffc\n'
            report['probes']['interfaces']['stdout']=text+'bridge0: flags=0<BROADCAST>\n'
        value=existing_plan(reports,[0,1,2,4,3])
        self.assertTrue(all(not n['commands'] and not n['rollback'] for n in value['nodes']))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'inventories.json').write_text(json.dumps(reports))
            (p/'plan.json').write_text(json.dumps(value))
            self.assertEqual(read_plan(p)[1],json.loads(json.dumps(value)))
            value['nodes'][0]['commands']=[['ifconfig','en0','down']]
            (p/'plan.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):read_plan(p)
        reports[0]['probes']['interfaces']['stdout']=reports[0]['probes']['interfaces']['stdout'].replace('0xfffffffc','0xffffff00')
        with self.assertRaises(ValueError):existing_plan(reports,[0,1,2,4,3])


if __name__=='__main__': unittest.main()
