import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import config
import control
import fetch
import patch_decoder
import privacy_check
import validate


class RecipeTests(unittest.TestCase):
    def test_manifests(self):
        self.assertEqual(fetch.manifest('source')['file_count'],35)
        data=fetch.manifest('model')
        self.assertEqual(data['file_count'],83)
        self.assertEqual(data['total_bytes'],510311045295)

    def test_model_manifest_contains_license(self):
        self.assertIn('LICENSE',{item['file'] for item in fetch.manifest('model')['files']})

    def test_invalid_gpu_selection(self):
        for gpus in ('0,1,2','0,0,1,2','a,1,2,3'):
            with self.subTest(gpus=gpus),patch.dict(os.environ,{'MODEL_DIR':'/srv/model',
                'RUNTIME_CACHE_DIR':'/srv/cache','GPU_IDS':gpus}):
                with self.assertRaises(RuntimeError):config.settings()

    def test_overlapping_paths_rejected(self):
        with patch.dict(os.environ,{'MODEL_DIR':'/srv/model','RUNTIME_CACHE_DIR':'/srv/model/cache'}):
            with self.assertRaises(RuntimeError):config.settings()

    def test_port_bounds(self):
        with patch.dict(os.environ,{'MODEL_DIR':'/srv/model','RUNTIME_CACHE_DIR':'/srv/cache','HOST_PORT':'65536'}):
            with self.assertRaises(RuntimeError):config.settings()

    def test_private_patterns(self):
        for value in ('192.'+'168.0.1','100.'+'100.0.1','/ho'+'me/'+'example/'+'file',
                      'hf_'+'x'*40,'GPU-'+'12345678-'+'1234-'*3+'123456789abc'):
            with self.subTest(value=value):self.assertTrue(privacy_check.findings(value))
        self.assertFalse(privacy_check.findings('http://127.0.0.1:8000 and 0.0.0.0'))

    def test_sha256_and_git_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data'
            path.write_bytes(b'abc')
            fetch.verify_file(Path(directory),{'file':'data','bytes':3,'algorithm':'sha256',
                'hash':hashlib.sha256(b'abc').hexdigest()})
            fetch.verify_file(Path(directory),{'file':'data','bytes':3,'algorithm':'git-blob-sha1',
                'hash':hashlib.sha1(b'blob 3\0abc').hexdigest()})

    def test_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'data').write_bytes(b'abc')
            with self.assertRaises(RuntimeError):
                fetch.verify_file(Path(directory),{'file':'data','bytes':3,'algorithm':'sha256','hash':'0'*64})

    def test_patch_rejects_unexpected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'decoder.py').write_text('unexpected\n')
            with self.assertRaises(RuntimeError):
                patch_decoder.apply(Path(directory),ROOT/'patches/hybrid-dispatch.patch',config.PINS)

    def test_patch_exact_reference(self):
        source=ROOT/'.build/upstream/src/brrr/decoder.py'
        if not source.exists():self.skipTest('Fetch pinned upstream source before the integration patch test')
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'decoder.py'
            shutil.copyfile(source,target)
            patch_decoder.apply(Path(directory),ROOT/'patches/hybrid-dispatch.patch',config.PINS)
            self.assertEqual(config.digest(target),config.PINS['hybrid_decoder_sha256'])

    def cell(self):
        return {'context_tokens':0,'concurrency':1,'underfilled':False,'warmup_timed_out':False,
            'capacity_limited':False,'loop_detected':False,'num_errors':0,
            'aggregate_source':'openai_continuous_usage','avg_running_reqs':1,'max_running_reqs':1,
            'effective_concurrency':1,'avg_queue_reqs':0,'max_queue_reqs':0,'queue_fraction':0,
            'aggregate_tps':350.0,'server_gen_throughput':350.0,'request_count':1}

    def test_valid_decode(self):
        self.assertTrue(validate.decode_summary({'results':[self.cell()]},True)['all_cells_valid'])

    def test_loop_and_renewal_not_published_as_rates(self):
        for key,value in (('loop_detected',True),('request_count',2),('max_running_reqs',2),
                          ('avg_queue_reqs',1),('server_gen_throughput',100),('aggregate_tps',float('nan'))):
            row=self.cell();row[key]=value
            result=validate.decode_summary({'results':[row]},True)
            self.assertFalse(result['all_cells_valid'])
            self.assertIsNone(result['cells'][0]['aggregate_tok_s'])

    def test_renewal_allowed_and_labeled_in_sweep(self):
        row=self.cell();row['request_count']=2
        result=validate.decode_summary({'results':[row]},False)
        self.assertTrue(result['all_cells_valid'])
        self.assertTrue(result['cells'][0]['renewing_streams'])

    def test_foreign_gpus_never_stopped(self):
        with patch.object(control.subprocess,'check_output',side_effect=['', 'GPU-placeholder, 123\n']):
            with self.assertRaises(RuntimeError):control.assert_idle(['GPU-placeholder'],8000)

    def test_quality_rejects_truncation_and_token_caps(self):
        summary=dict(attempted=10,completed=10,correct=10,errors=0,truncated=0,
                     stalled=0,timeout=0,incomplete=0,hit_max_tokens=0)
        self.assertTrue(validate.quality_summary({'all_summary':summary},10)['passed'])
        for field in ('errors','truncated','stalled','timeout','incomplete','hit_max_tokens'):
            with self.subTest(field=field):
                bad={**summary,field:1}
                self.assertFalse(validate.quality_summary({'all_summary':bad},10)['passed'])


if __name__=='__main__':unittest.main()
