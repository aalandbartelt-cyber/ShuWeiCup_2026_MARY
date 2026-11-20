"""A future or another recording's frame cannot enter a causal sequence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_sequence import history


def test_history_is_asof_and_resets_at_recording_boundary():
    f=pd.DataFrame(dict(episode=['a']*5+['b']*5,time_s=[0.,.1,.2,.3,.4]*2))
    raw=np.arange(10,dtype=np.float32)[:,None,None]*np.ones((1,2,3),np.float32)
    query=pd.DataFrame(dict(episode=['b','a'],time_s=[.15,.25]));a,known=history(f,query,raw,[.2,.1,0.])
    np.testing.assert_array_equal(a[0,:,0],0);np.testing.assert_array_equal(known[0,:,0],0)
    np.testing.assert_array_equal(a[0,:,1],5);np.testing.assert_array_equal(a[0,:,2],6)
    altered=raw.copy();altered[[3,4,7,8,9]]=1000;b,bknown=history(f,query,altered,[.2,.1,0.])
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(known,bknown)
