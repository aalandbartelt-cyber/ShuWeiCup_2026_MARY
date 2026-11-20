"""Training/inference feature parity and exclusion of undisclosed future input."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from types import SimpleNamespace
from shuwei_c.q1v3_state import pack_records
from shuwei_c.q1v3_prefix import causal_pack


def test_disclosed_prefix_exactly_matches_batch_training_features():
    rng=np.random.default_rng(12);t=np.arange(65)/30
    c=np.stack([np.c_[.2*t,.1*t*t,np.sin(t)],np.c_[-.1*t,.03*t*t,np.cos(t)]],axis=1)
    p=c[:,:,None,:]+rng.normal(0,.01,(1,2,21,3));bad=np.zeros((len(t),2),bool);bad[15:19,0]=True
    head=np.c_[.01*t,.02*t,np.ones_like(t)]
    rec=SimpleNamespace(episode='synthetic',environment='one',dataset='EgoPro',split='train',streams={'head':{'time':t,'position':head}},hands=lambda:(t,p,bad))
    config={'warmup_s':.4,'target_tolerance_steps':.55};scale=.04
    batch=pack_records([rec],.1,config,{'EgoPro':{'hand_scale':scale}})
    for origin in [20,31,47]:
        cut=origin+1;query=t[origin]+.1
        a=causal_pack(t[:cut],p[:cut],bad[:cut],t,head,'EgoPro',scale,query,.1)
        assert a['feature_names']==batch['feature_names']
        for hand in [0,1]:
            row=np.flatnonzero((batch['meta'].origin==origin)&(batch['meta'].hand==hand))[0]
            np.testing.assert_allclose(a['features'][hand],batch['features'][row],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(a['experts'][hand],batch['experts'][row],rtol=1e-6,atol=1e-6)
        altered=head.copy();altered[t>t[origin]]+=1000
        b=causal_pack(t[:cut],p[:cut],bad[:cut],t,altered,'EgoPro',scale,query,.1)
        np.testing.assert_array_equal(a['features'],b['features'])


def test_sparse_pack_uses_only_the_disclosed_samples():
    t=np.array([0.,.03,.07,.2,.4]);p=np.zeros((5,2,21,3));p[:,:,:,0]=t[:,None,None]
    a=causal_pack(t,p,np.zeros((5,2),bool),[0.],np.zeros((1,3)),'EgoDemo',.04,.5,.1)
    np.testing.assert_allclose(a['experts'][:,:,0],np.tile([2.5,0,0],(2,1)),rtol=1e-6)
    assert a['origin_s']==.4 and a['query_s']==.5
