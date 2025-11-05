import re
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import POLICY_SIZE


class Residual(nn.Module):
    """Residual block: Conv-BN-ReLU → Conv-BN, then add skip and ReLU."""
    def __init__(self, ch: int):
        super().__init__()
        # Two 3×3 convolutions that keep channel count the same (padding=1 preserves 8×8).
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convs with BN, add the skip, then ReLU."""
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    """
    Policy–value network for 8×8 chess.
    - Stem      : 3×3 conv to lift 18 planes → C channels.
    - Tower     : res_blocks of Residual(C).
    - Policy head: 1×1 conv → BN → Linear → logits over POLICY_SIZE.
    - Value head : 1×1 conv → BN → Linear → Linear → tanh scalar in [-1, 1].
    """
    def __init__(self, in_planes: int = 18, channels: int = 192,
                 res_blocks: int = 10, policy_out: int = POLICY_SIZE):
        super().__init__()
        # Stem: bring inputs to 'channels' with a 3×3 conv.
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Residual tower.
        self.res = nn.Sequential(*[Residual(channels) for _ in range(res_blocks)])

        # Policy head: 1×1 conv → BN → flatten → linear to POLICY_SIZE.
        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(32)
        self.p_fc   = nn.Linear(32 * 8 * 8, policy_out)

        # Value head: 1×1 conv → BN → flatten → 256 → 1 → tanh.
        self.v_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(32)
        self.v_fc1  = nn.Linear(32 * 8 * 8, 256)
        self.v_fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, value) for a batch of boards."""
        x = self.stem(x)
        x = self.res(x)

        # Policy path.
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.flatten(1))  # [N, POLICY_SIZE]

        # Value path.
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)  # [N]
        return p, v


def load_policy_value_net(path: str, device: str = "cpu") -> ChessNet:
    """
    Load a policy–value network from a .pth checkpoint.

    - Handles DataParallel ('module.') key prefixes.
    - Verifies the policy head output size matches POLICY_SIZE.
    - Infers channel width and residual depth from the checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip DataParallel prefix if present.
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Basic shape checks for the policy head.
    p_w = next((v for k, v in state.items() if k.endswith("p_fc.weight")), None)
    if p_w is None:
        raise RuntimeError("Checkpoint missing policy head (p_fc.weight).")
    if p_w.shape[0] != POLICY_SIZE:
        raise RuntimeError(f"Expected {POLICY_SIZE} policy logits, got {p_w.shape[0]}.")

    # Infer channel width from stem conv.
    stem_w = state.get("stem.0.weight")
    channels = int(stem_w.shape[0]) if stem_w is not None else 192

    # Infer residual depth by scanning block indices (res.0.*, res.1.*, ...).
    max_block = -1
    for k in state:
        m = re.match(r"res\.(\d+)\.", k)
        if m:
            max_block = max(max_block, int(m.group(1)))
    res_blocks = (max_block + 1) if max_block >= 0 else 10

    # Build and load.
    net = ChessNet(channels=channels, res_blocks=res_blocks).to(device).eval()
    net.load_state_dict(state, strict=True)
    return net
