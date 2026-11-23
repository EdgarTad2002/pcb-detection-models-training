import numpy as np
from tools.paper_results_data import ap101, confusion, greedy_pairs, pooled, split_name_overlap


def test_pooled_precision_uses_counts_not_unweighted_class_average():
    values={'a':dict(tp=1,fp=0,fn=0),'b':dict(tp=0,fp=9,fn=1)}
    metrics=pooled(values)
    assert metrics['precision']==.1
    assert metrics['recall']==.5


def test_ap_endpoint_does_not_invent_precision_after_maximum_recall():
    ap,_,_=ap101(np.array([.5]),np.array([1.]))
    assert np.isclose(ap,.495)
    assert ap101(np.array([0.]),np.array([0.]))[0]==0.


def test_matching_does_not_reuse_one_ground_truth_for_two_detections():
    pairs=greedy_pairs(np.array([[.9,.8]]))
    assert pairs.tolist()==[[0,0]]


def test_confusion_retains_wrong_class_as_off_diagonal_and_counts_misses():
    gt=[dict(boxes=np.array([[0,0,10,10],[20,20,30,30]]),cls=np.array([0,3]))]
    pred=[np.array([[0,0,10,10,.9,1],[50,50,60,60,.8,2]])]
    cm=confusion(gt,pred)
    assert cm[1,0]==1 and cm[4,3]==1 and cm[2,4]==1
    assert cm.sum()==3


def test_view_suffix_aliases_cannot_be_called_a_disjoint_split():
    data={'splits':{
        'train':{'records':[dict(name='DuetWIFI_Top_jpg.rf.a.jpg')]},
        'test':{'records':[dict(name='DuetWIFI_Top_png_jpg.rf.b.jpg')]}}}
    assert split_name_overlap(data)=={'duetwifi_top':{'train':1,'test':1}}
