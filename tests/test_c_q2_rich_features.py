import numpy as np
from shuwei_c.q2_replay import History
from shuwei_c.q2_features import CausalExperts
from shuwei_c.q2_rich_features import RichExperts


def history(points=None):
    t=np.linspace(0,1,31)
    if points is None:
        velocity=np.array([[.2,.1,-.3],[-.1,.5,.2]])
        points=np.repeat((t[:,None,None]*velocity[None,:,:])[:,:,None,:],21,axis=2)
    return History(t,points,np.zeros((len(t),2),bool))


def test_rich_velocity_and_polynomial_on_exact_linear_motion():
    h=history();p=RichExperts('EgoDemo').at(h,1.2)
    expected=np.array([[.2,.1,-.3],[-.1,.5,.2]])*.2
    np.testing.assert_allclose(p['experts'][:,:,16:],np.repeat(expected[:,:,None],7,axis=2),rtol=1e-6,atol=1e-8)


def test_rich_direct_observation_and_rigid_rotation():
    h=history();r=RichExperts('EgoDemo');p=r.at(h,1.2);direct=r.at(h,1.)
    assert np.all(direct['experts']==0)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    h2=history(h.points@rotation.T+np.array([4.,-2.,7.]))
    q=RichExperts('EgoDemo').at(h2,1.2)
    np.testing.assert_allclose(q['experts'],np.einsum('ij,hjk->hik',rotation,p['experts']),atol=1e-7)
    # Head is deliberately absent; all scalar geometric/temporal features agree.
    np.testing.assert_allclose(q['features'],p['features'],atol=1e-6)


def test_original_feature_and_expert_prefix_unchanged():
    h=history();h=History(h.time,h.points+np.random.default_rng(12).normal(0,.01,h.points.shape),h.bad)
    base=CausalExperts('EgoStandard').at(h,1.4);rich=RichExperts('EgoStandard').at(h,1.4)
    np.testing.assert_array_equal(rich['experts'][:,:,:16],base['experts'])
    np.testing.assert_array_equal(rich['features'][:,:len(base['feature_names'])],base['features'])
