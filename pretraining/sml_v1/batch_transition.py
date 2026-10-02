"""Explicit microbatch change with preserved checkpoint and token schedule."""
import copy
from .common import fingerprint
FORMAT='sml-research-batch-transition-v1'


def validate_transition(recipe):
    t=recipe['batch_transition']
    keys={'format','parent_step','start_tokens','parent_contract','parent_replica_hash','parent_batches'}
    if not isinstance(t,dict) or set(t)!=keys or t['format']!=FORMAT:
        raise ValueError('Invalid batch transition')
    old=t['parent_batches'];new=recipe['batches']
    if (not isinstance(old,list) or len(old)!=5 or len(new)!=5
            or any(type(b)is not int or b<1 for b in old) or old==new):
        raise ValueError('Batch transition requires changed positive five-rank batches')
    if any(type(t[k])is not int or t[k]<=0 for k in ('parent_step','start_tokens')):
        raise ValueError('Invalid batch transition counters')
    if set(t['parent_contract'])!={'code','recipe','data','tokenizer'}:
        raise ValueError('Missing batch parent contract')
    if not isinstance(t['parent_replica_hash'],str) or len(t['parent_replica_hash'])!=64:
        raise ValueError('Missing parent replica hash')


def change_batches(metadata,batches):
    from .recipe import validate
    previous=validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous)!=metadata['v2_contract']['recipe']:
        raise ValueError('Parent recipe fingerprint differs')
    if (not isinstance(batches,list) or len(batches)!=5 or len(previous['batches'])!=5
            or any(type(b)is not int or b<1 for b in batches)):
        raise ValueError('Require five positive microbatches')
    if 'batch_transition' in previous:
        if batches!=previous['batches']:raise ValueError('Further batch transition requires separate review')
        return previous
    if batches==previous['batches'] or not 0<metadata['tokens']<previous['target_tokens']:
        raise ValueError('Require changed batches and an unfinished trained checkpoint')
    r=copy.deepcopy(previous);r['batches']=batches.copy()
    r['batch_transition']=dict(format=FORMAT,parent_step=metadata['step'],start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']),parent_replica_hash=metadata['replica_hash'],
        parent_batches=previous['batches'].copy())
    return validate(r)
