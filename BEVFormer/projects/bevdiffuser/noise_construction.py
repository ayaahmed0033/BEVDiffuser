# ---------------------------------------------------------------------
# Noise Construction Module (NCM) for BEVDiffuser
#
# Implements the multi-type / multi-level noise construction proposed in
# "Robust Single-Stage Fully Sparse 3D Object Detection via Detachable
#  Latent Diffusion" (RSDNet, arXiv:2508.03252), adapted to BEV feature
# latents used by BEVDiffuser (arXiv:2502.19694).
#
# Core idea (RSDNet Eq. 5/6, supplement Eqs. 17-19):
#   Standard DDPM noising:   x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * eps
#   Multi-type noising:      f_t(x_t) = sqrt(a_bar_t) * g_t(x0)
#                                       + sqrt(1-a_bar_t) * h_t(eps)
#   where g_t / h_t are the SAME invertible affine transform (translation,
#   scaling, or Givens channel rotation) whose intensity grows with t.
#
# Because g_t == h_t and the transform is affine, we can simply transform
# x0 and eps FIRST and then reuse the scheduler's add_noise() unchanged.
#
# Training targets (RSDNet supplement, Tab. 1 ablation):
#   prediction_type == "epsilon" -> target = h_t(eps)
#   prediction_type == "sample"  -> target = g_t(x0)
#     (for the task loss, invert:  x0_hat = g_t^{-1}(model_pred))
#
# NCM is TRAINING-TIME ONLY. Inference is untouched, preserving
# BEVDiffuser's plug-and-play property.
# ---------------------------------------------------------------------

import math
import torch


class NoiseConstructionModule:
    """Constructs multi-type, multi-level noise samples/targets for BEV latents.

    Per sample in the batch, one perturbation type is drawn:
        0: identity   (vanilla BEVDiffuser Gaussian noising)
        1: translation  g(x) = x - T_t
        2: scaling      g(x) = S_t * x
        3: rotation     g(x) = R_t x   (Givens rotations over channel pairs)

    Intensity scales linearly with the diffusion timestep t / T,
    following RSDNet ("intensity varying based on t").

    Args:
        num_train_timesteps: T of the DDPM scheduler (e.g. 1000).
        translation_range:   max |T_t| at t = T (latent-space units).
        scale_range:         (min, max) multiplicative scale at t = T.
                             s_t interpolates from 1.0 -> sampled endpoint.
        rotation_range:      max |theta| in radians at t = T.
        p_identity/p_translation/p_scaling/p_rotation:
                             per-sample probabilities of each type
                             (will be normalized).
        noise_dist:          distribution of eps: 'gaussian' | 'laplace'
                             | 'uniform' (RSDNet supplement Fig. 4 shows the
                             sample-fitting rule supports arbitrary eps).
    """

    TYPE_IDENTITY, TYPE_TRANSLATION, TYPE_SCALING, TYPE_ROTATION = 0, 1, 2, 3

    def __init__(self,
                 num_train_timesteps=1000,
                 translation_range=2.0,
                 scale_range=(0.5, 1.5),
                 rotation_range=math.pi / 2,
                 p_identity=0.25,
                 p_translation=0.25,
                 p_scaling=0.25,
                 p_rotation=0.25,
                 noise_dist='gaussian'):
        self.T = float(num_train_timesteps)
        self.translation_range = float(translation_range)
        self.scale_range = tuple(scale_range)
        self.rotation_range = float(rotation_range)
        probs = torch.tensor(
            [p_identity, p_translation, p_scaling, p_rotation],
            dtype=torch.float32)
        assert probs.sum() > 0, "at least one NCM type must have prob > 0"
        self.type_probs = probs / probs.sum()
        assert noise_dist in ('gaussian', 'laplace', 'uniform')
        self.noise_dist = noise_dist

    # ------------------------------------------------------------------
    # eps sampling (multi-distribution support, optional)
    # ------------------------------------------------------------------
    def sample_noise(self, latents):
        if self.noise_dist == 'gaussian':
            return torch.randn_like(latents)
        if self.noise_dist == 'laplace':
            # scale b = 1/sqrt(2) -> Var = 2*b^2 = 1 (unit variance for DDPM)
            d = torch.distributions.Laplace(
                torch.zeros_like(latents),
                torch.full_like(latents, 1.0 / math.sqrt(2.0)))
            return d.sample()
        # uniform on [-sqrt(3), sqrt(3)] -> zero mean, unit variance
        return (torch.rand_like(latents) * 2.0 - 1.0) * math.sqrt(3.0)

    # ------------------------------------------------------------------
    # per-batch parameter sampling
    # ------------------------------------------------------------------
    def sample_params(self, batch_size, timesteps, device, generator=None):
        """Draw a transform type + intensity for each sample.

        Returns a dict of per-sample tensors used by apply()/invert().
        """
        types = torch.multinomial(
            self.type_probs.to(device), batch_size,
            replacement=True, generator=generator)                    # (B,)
        level = (timesteps.float() + 1.0) / self.T                    # (B,) in (0,1]

        # translation offset T_t: sign-symmetric, magnitude grows with t
        # ((rand<0.5)*2-1 avoids torch.where scalar issues on old torch)
        sign_t = (torch.rand(batch_size, device=device,
                             generator=generator) < 0.5).float() * 2.0 - 1.0
        trans = sign_t * self.translation_range * level               # (B,)

        # scale S_t: interpolate 1.0 -> random endpoint in scale_range
        lo, hi = self.scale_range
        end = lo + (hi - lo) * torch.rand(
            batch_size, device=device, generator=generator)           # (B,)
        scale = 1.0 + (end - 1.0) * level                             # (B,)

        # rotation angle theta_t: sign-symmetric, grows with t
        sign_r = (torch.rand(batch_size, device=device,
                             generator=generator) < 0.5).float() * 2.0 - 1.0
        theta = sign_r * self.rotation_range * level                  # (B,)

        return {'types': types, 'trans': trans,
                'scale': scale, 'theta': theta}

    # ------------------------------------------------------------------
    # the affine transform g_t (== h_t) and its inverse
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_channels(x, theta):
        """Apply Givens rotations with angle theta over channel pairs.

        x: (B, C, H, W); theta: (B,). Pairs (0,1), (2,3), ... share theta,
        the composite is a rigid rotation of the C-dim latent vector at
        every BEV location (RSDNet supplement, Eq. 19). Odd trailing
        channel (if C is odd) is left unchanged.
        """
        B, C, H, W = x.shape
        n_pairs = C // 2
        cos = torch.cos(theta).view(B, 1, 1, 1, 1)
        sin = torch.sin(theta).view(B, 1, 1, 1, 1)
        head = x[:, :n_pairs * 2].reshape(B, n_pairs, 2, H, W)
        a, b = head[:, :, 0:1], head[:, :, 1:2]
        rot = torch.cat([cos * a - sin * b,
                         sin * a + cos * b], dim=2)
        rot = rot.reshape(B, n_pairs * 2, H, W)
        if C % 2 == 1:
            rot = torch.cat([rot, x[:, -1:]], dim=1)
        return rot

    def apply(self, x, params):
        """g_t(x) applied per sample according to sampled types."""
        types = params['types']
        out = x.clone()

        m = types == self.TYPE_TRANSLATION
        if m.any():
            out[m] = x[m] - params['trans'][m].view(-1, 1, 1, 1)

        m = types == self.TYPE_SCALING
        if m.any():
            out[m] = x[m] * params['scale'][m].view(-1, 1, 1, 1)

        m = types == self.TYPE_ROTATION
        if m.any():
            out[m] = self._rotate_channels(x[m], params['theta'][m])

        return out

    def invert(self, x, params):
        """g_t^{-1}(x): recover x0-space tensors from transformed space.

        Needed before the task loss when prediction_type == 'sample',
        so the detection head always sees an un-transformed BEV.
        """
        types = params['types']
        out = x.clone()

        m = types == self.TYPE_TRANSLATION
        if m.any():
            out[m] = x[m] + params['trans'][m].view(-1, 1, 1, 1)

        m = types == self.TYPE_SCALING
        if m.any():
            out[m] = x[m] / params['scale'][m].view(-1, 1, 1, 1)

        m = types == self.TYPE_ROTATION
        if m.any():
            out[m] = self._rotate_channels(x[m], -params['theta'][m])

        return out

    # ------------------------------------------------------------------
    # main entry point used by the training loop
    # ------------------------------------------------------------------
    def construct(self, latents, noise, timesteps, noise_scheduler,
                  prediction_type):
        """Build multi-type noisy latents + matching training target.

        Returns:
            noisy_latents: f_t(x_t) = add_noise(g_t(x0), h_t(eps), t)
            target:        h_t(eps) if 'epsilon' else g_t(x0)
            params:        pass to .invert() for the task loss
        """
        if prediction_type not in ('epsilon', 'sample'):
            raise ValueError(
                f"NCM supports 'epsilon' or 'sample' prediction, got "
                f"'{prediction_type}'. Disable --use_ncm for "
                f"'{prediction_type}'.")

        params = self.sample_params(
            latents.shape[0], timesteps, latents.device)

        g_x0 = self.apply(latents, params)   # g_t(x0)
        h_eps = self.apply(noise, params)    # h_t(eps), same transform

        # sqrt(a_bar)*g(x0) + sqrt(1-a_bar)*h(eps)  == RSDNet Eq. 6
        noisy_latents = noise_scheduler.add_noise(g_x0, h_eps, timesteps)

        target = h_eps if prediction_type == 'epsilon' else g_x0
        return noisy_latents, target, params
