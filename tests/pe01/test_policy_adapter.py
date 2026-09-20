import torch

from unilab.adapters.pe01_legacy import PE01EncoderPolicy


def test_pe01_encoder_actor_and_critic_dimensions():
    policy = PE01EncoderPolicy()
    history = torch.zeros(2, 300)
    commands = torch.zeros(2, 3)
    assert policy.action_mean(history, history[:, -30:], commands).shape == (2, 6)
    assert policy.value(history, torch.zeros(2, 33), commands).shape == (2,)
