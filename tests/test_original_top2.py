"""Guard against silently disabling the Exp7 NWD and misrouting capacitor masks."""
import pytest
import torch
from torch import nn
from tools.scale_adaptive_nwd_loss import ScaleAdaptiveNWDBboxLoss, ScaleDecoupledMultiStageLoss


class ConstantBoxLoss(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.25), torch.tensor(0.75)


@pytest.mark.parametrize('denominator', [1, torch.tensor(1.)])
def test_nwd_stays_active_with_ultralytics_scalar_denominator(denominator):
    prediction = torch.tensor([[[0., 0., .8, .8]]], requires_grad=True)
    target = torch.tensor([[[0., 0., 1., 1.]]])
    loss, dfl = ScaleAdaptiveNWDBboxLoss(ConstantBoxLoss())(
        None, prediction, None, target, torch.ones(1, 1, 4) / 4,
        denominator, torch.tensor([[True]]))
    assert loss > .25
    assert dfl == .75
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_fourclass_mapping_routes_capacitors_and_ics_to_distinct_masks():
    batch = dict(img=torch.zeros(2, 3, 64, 64),
                 bboxes=torch.tensor([[.5, .5, .001, .001], [.5, .5, .4, .4]]),
                 batch_idx=torch.tensor([0, 1]), cls=torch.tensor([[0], [3]]))
    masks = ScaleDecoupledMultiStageLoss.build_decoupled_gt_masks(batch, (16, 16), torch.device('cpu'))
    assert masks[0, 0].sum() >= 4
    assert masks[0, 2].sum() == 0
    assert masks[1, 0].sum() == 0
    assert masks[1, 2].sum() > 0
    assert masks[:, 1].sum() > 0
