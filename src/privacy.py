"""
Stage 5: differential privacy (DP) for federated model updates.

The mechanism is client-level DP on the *update* a client sends:

    delta = w_local - w_global
    delta <- delta * min(1, C / ||delta||_2)      # clip L2 norm to C
    delta <- delta + N(0, (sigma * C)^2)          # Gaussian noise per coordinate

The server averages the noisy deltas (weighted by the number of training
windows, as in FedAvg) and adds that average to the global weights.

Everything here works on flat float32 CPU vectors (the "flat vector over all
parameters" from the build plan) plus small helpers to go back and forth
between a flat vector and a state dict.

Privacy accounting is the simple RDP bound for the Gaussian mechanism with
full participation and no subsampling. For order alpha > 1 the RDP of one
round is alpha / (2 * sigma^2), and after T rounds it is
T * alpha / (2 * sigma^2). Converting to (epsilon, delta):

    epsilon = min_alpha [ T * alpha / (2 * sigma^2)
                          + log(1 / delta) / (alpha - 1) ]

with alpha searched on a fine grid in (1.01, 256]. This assumes the
add/remove adjacency convention and sensitivity = C.
"""

import math
from typing import Dict, Optional

import numpy as np
import torch


def flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single 1D float32 CPU vector."""
    return torch.cat([state[k].detach().float().reshape(-1).cpu() for k in state.keys()])


def unflatten_like(flat: torch.Tensor,
                   reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into a state dict with the keys and shapes of `reference`."""
    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for k, v in reference.items():
        num = v.numel()
        chunk = flat[offset:offset + num].reshape(v.shape)
        out[k] = chunk.to(dtype=torch.float32, device=v.device).clone()
        offset += num
    return out


def subtract_states(a: Dict[str, torch.Tensor],
                    b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Element-wise a - b (float32)."""
    return {k: a[k].float() - b[k].float() for k in a}


def add_states(a: Dict[str, torch.Tensor],
               b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Element-wise a + b (float32)."""
    return {k: a[k].float() + b[k].float() for k in a}


def clip_update(delta: torch.Tensor, clip_norm: float) -> torch.Tensor:
    """Clip a flat update so that its L2 norm is at most `clip_norm`."""
    norm = delta.float().norm()
    scale = min(1.0, float(clip_norm) / (float(norm) + 1e-12))
    return delta.float() * scale


def add_noise(delta: torch.Tensor,
              sigma: float,
              clip_norm: float,
              generator: torch.Generator) -> torch.Tensor:
    """Add Gaussian noise with std `sigma * clip_norm` to every coordinate."""
    if sigma == 0:
        return delta.clone()
    noise = torch.randn(delta.shape, generator=generator,
                        dtype=delta.dtype, device=delta.device)
    return delta + noise * (float(sigma) * float(clip_norm))


def dp_seed(seed: int, round_index: int, client_id) -> int:
    """Deterministic integer seed from (seed, round, client_id).

    Uses integer mixing (not Python's hash(), which is process-salted) so the
    noise stream is reproducible across runs and independent of the global RNG.
    """
    base = int(seed) * 1000003 + int(round_index) * 9176
    cid = sum((i + 1) * ord(ch) for i, ch in enumerate(str(client_id)))
    return (base ^ (cid * 2654435761)) % (2 ** 31 - 1)


def make_generator(seed: int, round_index: int, client_id) -> torch.Generator:
    """A dedicated torch Generator for DP noise, seeded from (seed, round, client_id)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(dp_seed(seed, round_index, client_id))
    return gen


def privatize_delta(local_state: Dict[str, torch.Tensor],
                    global_state: Dict[str, torch.Tensor],
                    clip_norm: float,
                    sigma: float,
                    generator: torch.Generator) -> torch.Tensor:
    """delta = local - global, clipped to C, plus N(0, (sigma*C)^2). Returns a flat vector."""
    delta = flatten_state(subtract_states(local_state, global_state))
    delta = clip_update(delta, clip_norm)
    return add_noise(delta, sigma, clip_norm, generator)


def compute_epsilon(sigma: Optional[float],
                    rounds: int,
                    delta: float = 1e-5,
                    alpha_max: float = 256.0,
                    grid_size: int = 100000) -> float:
    """Simple RDP -> (epsilon, delta) bound for the Gaussian mechanism.

    Full participation, no subsampling, sensitivity = clip norm C. Returns +inf
    when sigma <= 0 (no noise => no privacy guarantee).
    """
    if sigma is None or sigma <= 0:
        return float("inf")
    alphas = np.linspace(1.01, alpha_max, grid_size)
    rdp = rounds * alphas / (2.0 * sigma ** 2)
    bound = rdp + math.log(1.0 / delta) / (alphas - 1.0)
    return float(np.min(bound))


def round_sig(x: float, sig: int = 2) -> float:
    """Round to `sig` significant digits."""
    if x == 0:
        return 0.0
    return round(float(x), -int(math.floor(math.log10(abs(float(x))))) + (sig - 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    delta = torch.randn(1000)
    C = 1.0
    clipped = clip_update(delta, C)
    assert float(clipped.norm()) <= C + 1e-6
    print(f"clip ok: ||clipped|| = {float(clipped.norm()):.6f} <= {C}")

    gen = torch.Generator().manual_seed(0)
    assert torch.equal(add_noise(clipped, 0.0, C, gen), clipped)
    print("sigma=0 returns the clipped delta")

    gen = torch.Generator().manual_seed(0)
    noisy = add_noise(torch.zeros(200000), 0.5, C, gen)
    print(f"noise std = {float(noisy.std()):.4f} (expected {0.5 * C})")

    eps = [compute_epsilon(s, 30) for s in [0.01, 0.1, 0.3, 1.0, 3.0]]
    print("epsilon:", [f"{e:.2f}" for e in eps])
    assert all(eps[i] > eps[i + 1] for i in range(len(eps) - 1))
    print("epsilon decreases as sigma increases: OK")
