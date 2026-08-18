# ---------------------------------------------------------------------
# bev_corruption.py — Structured BEV Corruption Module (BCM)
#
# Hypothesis (from your 4-run evidence): input perturbation helps
# (NCM > baseline), loss reweighting doesn't (ext < baseline). So model
# *realistic camera-BEV* corruption instead of global affines / iid noise:
#
#   warp      — smooth local grid deformation  (calibration / ego-pose drift)
#   ray_gain  — per-azimuth multiplicative noise, constant along each ray
#               from ego (depth-estimation error smears along camera rays)
#   lowfreq   — spatially correlated additive noise (structured, not white)
#   dropout   — smooth region zeroing            (occlusion / range falloff)
#   identity  — vanilla BEVDiffuser path (exact baseline)
#
# Restoration objective (cold-diffusion style):
#   x_t   = sqrt(a_bar_t) * C_t(x0) + sqrt(1 - a_bar_t) * eps
#   target = x0                      (the CLEAN latent, always)
# so the denoiser learns to restore, and the task loss needs NO inversion.
# Requires prediction_type == "sample" (BEVDiffuser's default).
#
# Drop-in: same construct() signature as NoiseConstructionModule; returns
# params=None so the existing `ncm_params is not None` branch is skipped.
# ---------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F


def _blur2d(x, sigma):
    """Separable Gaussian blur, per-channel. x: (B, C, H, W)."""
    if sigma <= 0:
        return x
    r = max(1, int(3.0 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / (2.0 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    x = F.conv2d(x, ky, padding=(r, 0), groups=C)
    return x


class BEVCorruptionModule:
    """Structured, timestep-scaled corruptions on BEV latents (train only).

    Per sample one type is drawn; intensity grows linearly with t/T.
    p_identity=1.0 reproduces vanilla BEVDiffuser exactly.
    """

    T_ID, T_WARP, T_RAY, T_LOWFREQ, T_DROP = 0, 1, 2, 3, 4

    def __init__(self,
                 num_train_timesteps=1000,
                 max_warp_cells=1.5,      # max flow magnitude (BEV cells) at t=T
                 warp_sigma=3.0,          # smoothness of the flow field
                 max_ray_gain=0.3,        # per-ray gain std at t=T
                 n_azimuth=64,            # azimuth bins for ray noise
                 lowfreq_sigma=2.0,       # blur of the structured additive noise
                 max_drop_frac=0.25,      # max fraction of cells zeroed at t=T
                 p_identity=0.2, p_warp=0.2, p_ray=0.2,
                 p_lowfreq=0.2, p_dropout=0.2):
        self.T = float(num_train_timesteps)
        self.max_warp_cells = float(max_warp_cells)
        self.warp_sigma = float(warp_sigma)
        self.max_ray_gain = float(max_ray_gain)
        self.n_az = int(n_azimuth)
        self.lowfreq_sigma = float(lowfreq_sigma)
        self.max_drop_frac = float(max_drop_frac)
        probs = torch.tensor(
            [p_identity, p_warp, p_ray, p_lowfreq, p_dropout],
            dtype=torch.float32)
        assert probs.sum() > 0
        self.type_probs = probs / probs.sum()
        self._az_cache = {}   # (H, W, device) -> azimuth-bin index map

    # -- keep the same interface the patched train loop already calls ----
    def sample_noise(self, latents):
        return torch.randn_like(latents)

    # ------------------------- corruptions ------------------------------
    def _warp(self, x, level):
        """Smooth random deformation; displacement = level*max_warp cells."""
        B, C, H, W = x.shape
        flow = torch.randn(B, 2, H, W, device=x.device, dtype=x.dtype)
        flow = _blur2d(flow, self.warp_sigma)
        std = flow.flatten(2).std(dim=2).view(B, 2, 1, 1).clamp_min(1e-6)
        flow = flow / std * (level.view(B, 1, 1, 1) * self.max_warp_cells)
        ys = torch.linspace(-1, 1, H, device=x.device,
                            dtype=x.dtype).view(H, 1).expand(H, W)
        xs = torch.linspace(-1, 1, W, device=x.device,
                            dtype=x.dtype).view(1, W).expand(H, W)
        base = torch.stack([xs, ys], dim=-1)                    # (H, W, 2)
        dx = flow[:, 0] * (2.0 / max(W - 1, 1))
        dy = flow[:, 1] * (2.0 / max(H - 1, 1))
        grid = base.unsqueeze(0) + torch.stack([dx, dy], dim=-1)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def _az_index(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._az_cache:
            yc = (torch.arange(H, device=device, dtype=torch.float32)
                  - (H - 1) / 2.0).view(H, 1).expand(H, W)
            xc = (torch.arange(W, device=device, dtype=torch.float32)
                  - (W - 1) / 2.0).view(1, W).expand(H, W)
            az = torch.atan2(yc, xc)                            # [-pi, pi)
            idx = ((az + math.pi) / (2 * math.pi) * self.n_az).long()
            self._az_cache[key] = idx.clamp_(0, self.n_az - 1)  # (H, W)
        return self._az_cache[key]

    def _ray_gain(self, x, level):
        """Per-azimuth gain, constant along each ray from the ego (center).

        Models per-ray depth/feature error of camera BEV lifting: every BEV
        cell on the same ray shares one camera ray's mistake.
        """
        B, C, H, W = x.shape
        v = torch.randn(B, 1, self.n_az, device=x.device, dtype=x.dtype)
        # circular smoothing over azimuth so adjacent rays correlate
        k = torch.tensor([0.25, 0.5, 0.25], device=x.device,
                         dtype=x.dtype).view(1, 1, 3)
        v = F.conv1d(F.pad(v, (1, 1), mode='circular'), k)
        v = v / v.std(dim=2, keepdim=True).clamp_min(1e-6)      # unit std
        idx = self._az_index(H, W, x.device)                    # (H, W)
        gain_map = v[:, 0, idx.reshape(-1)].reshape(B, 1, H, W)
        s = level.view(B, 1, 1, 1) * self.max_ray_gain
        return x * (1.0 + s * gain_map)

    def _lowfreq_noise(self, x, level):
        """Additive spatially-correlated noise, magnitude tied to x's scale."""
        B = x.shape[0]
        n = torch.randn_like(x[:, :1])                          # (B,1,H,W)
        n = _blur2d(n, self.lowfreq_sigma)
        n = n / n.flatten(1).std(dim=1).view(B, 1, 1, 1).clamp_min(1e-6)
        mag = x.flatten(1).std(dim=1).view(B, 1, 1, 1)
        s = level.view(B, 1, 1, 1)
        return x + s * mag * n                                  # broadcast over C

    def _dropout(self, x, level):
        """Zero a smooth random region covering ~level*max_drop_frac cells."""
        B, C, H, W = x.shape
        field = _blur2d(torch.rand(B, 1, H, W, device=x.device,
                                   dtype=x.dtype), 2.0)
        frac = (level * self.max_drop_frac).clamp(0.0, 0.9)     # (B,)
        q = torch.quantile(field.flatten(1), frac.double().to(field.dtype),
                           dim=1, keepdim=False)
        # torch.quantile with per-sample q: take diagonal of (B, B) result
        if q.dim() == 2:
            q = q.diagonal()
        mask = (field > q.view(B, 1, 1, 1)).to(x.dtype)
        return x * mask

    # --------------------------- main API -------------------------------
    def corrupt(self, x0, timesteps):
        types = torch.multinomial(
            self.type_probs.to(x0.device), x0.shape[0], replacement=True)
        level = (timesteps.float() + 1.0) / self.T              # (B,)
        out = x0.clone()
        for t_id, fn in ((self.T_WARP, self._warp),
                         (self.T_RAY, self._ray_gain),
                         (self.T_LOWFREQ, self._lowfreq_noise),
                         (self.T_DROP, self._dropout)):
            m = types == t_id
            if m.any():
                out[m] = fn(x0[m], level[m])
        return out

    def construct(self, latents, noise, timesteps, noise_scheduler,
                  prediction_type):
        """Returns (noisy_latents, target, params=None).

        target is ALWAYS the clean latents (restoration objective), so the
        task loss uses model_pred directly — no inversion.
        """
        if prediction_type != "sample":
            raise ValueError(
                "BEVCorruptionModule requires prediction_type='sample' "
                "(restoration objective).")
        x0c = self.corrupt(latents, timesteps)
        noisy_latents = noise_scheduler.add_noise(x0c, noise, timesteps)
        return noisy_latents, latents, None


# --------------------- CPU self-test: python bev_corruption.py ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    class Stub:
        class config:
            num_train_timesteps = 1000
            prediction_type = "sample"

        def __init__(self, T=1000):
            self.a = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, T), 0)

        def add_noise(self, x0, e, t):
            ab = self.a[t].view(-1, 1, 1, 1)
            return ab.sqrt() * x0 + (1 - ab).sqrt() * e

    B, C, H, W = 6, 256, 50, 50
    x0 = torch.randn(B, C, H, W)
    eps = torch.randn_like(x0)
    ts = torch.randint(0, 1000, (B,))
    s = Stub()

    # 1) identity-only == exact vanilla BEVDiffuser
    bcm_id = BEVCorruptionModule(p_identity=1, p_warp=0, p_ray=0,
                                 p_lowfreq=0, p_dropout=0)
    noisy, tgt, p = bcm_id.construct(x0, eps, ts, s, "sample")
    assert p is None and (tgt - x0).abs().max() == 0
    assert (noisy - s.add_noise(x0, eps, ts)).abs().max() < 1e-6
    print("[1/3] identity path == vanilla; target == clean x0; params None")

    # 2) each corruption runs, keeps shape, stays finite
    bcm = BEVCorruptionModule()
    lvl = torch.full((B,), 0.8)
    for name, fn in (("warp", bcm._warp), ("ray", bcm._ray_gain),
                     ("lowfreq", bcm._lowfreq_noise), ("drop", bcm._dropout)):
        y = fn(x0, lvl)
        assert y.shape == x0.shape and torch.isfinite(y).all(), name
        assert (y - x0).abs().max() > 1e-3, name + " must actually change x0"
    print("[2/3] warp / ray_gain / lowfreq / dropout: shape-safe, finite, active")

    # 3) dropout zeroes roughly the requested fraction
    y = bcm._dropout(x0, torch.full((B,), 1.0))
    frac = (y == 0).float().mean().item()
    assert 0.10 < frac < 0.40, frac
    print(f"[3/3] dropout fraction ~ {frac:.2f} (target ~0.25)")

    print("\nALL BCM TESTS PASSED")
