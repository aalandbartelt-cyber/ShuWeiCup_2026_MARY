"""Geometry, causality and exact derivatives for the next v3 experiments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_landmarks import ray_midpoint,align_stereo,descriptor
from shuwei_c.q1v3_refine import distance_objective


def test_stereo_rays_recover_points_and_reject_parallel_lines():
    p=np.array([[.1,.1,.7],[-.1,.2,1.1]]);left=np.tile([-.03,0,0],(2,1));right=np.tile([.03,0,0],(2,1))
    result,valid,angle,gap=ray_midpoint(left,p-left,right,p-right)
    assert valid.all();np.testing.assert_allclose(result,p,atol=1e-12);assert max(gap)<1e-12
    result,valid,_,_=ray_midpoint(left,np.tile([0,0,1.],(2,1)),right,np.tile([0,0,1.],(2,1)))
    assert not valid.any() and np.isfinite(result).all()


def test_stereo_anchor_is_unavailable_until_both_images_arrive(tmp_path):
    file=tmp_path/'a__stereo.npz';data=dict(time_s=np.array([.201,.3]),position=np.ones((2,2,3)),valid=np.ones((2,2),bool),parallax=np.ones((2,2)),ray_gap=np.ones((2,2)),time_gap=np.array([.001,0.]))
    np.savez(file,**data);frame=pd.DataFrame(dict(episode=['a'],hand=[0],origin_s=[.2]));p,q=align_stereo(frame,tmp_path)
    assert np.all(p==0) and np.all(q==0)
    data['position']*=99;data['parallax']*=99;np.savez(file,**data);a,b=align_stereo(frame,tmp_path)
    np.testing.assert_array_equal(a,p);np.testing.assert_array_equal(b,q)


def test_canonical_hand_shape_is_invariant_to_image_scaling():
    rng=np.random.default_rng(32);points=np.c_[rng.uniform(.3,.7,(21,2)),np.zeros(21)]
    a,_=descriptor(points,[500,400],1000,800,.8);b,_=descriptor(points,[1000,800],2000,1600,.8)
    np.testing.assert_allclose(a[12:],b[12:],atol=1e-9)


def test_exact_distance_gradient_and_hessian_match_finite_differences():
    rng=np.random.default_rng(6);x=rng.normal(size=(12,3,4));y=rng.normal(size=(12,3));w=np.ones(12)/12;code=np.arange(12)%3;c=rng.normal(size=(3,4));direction=rng.normal(size=c.shape);eps=1e-5
    value,g,h=distance_objective(c,x,y,w,code,.01,True)
    plus,gp,_=distance_objective(c+eps*direction,x,y,w,code,.01)
    minus,gm,_=distance_objective(c-eps*direction,x,y,w,code,.01)
    np.testing.assert_allclose((plus-minus)/(2*eps),g@direction.ravel(),rtol=1e-6)
    np.testing.assert_allclose((gp-gm)/(2*eps),h@direction.ravel(),rtol=1e-5,atol=1e-8)
