import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2


def _valid_group_count(dim: int, num_groups: int = 8) -> int:
    groups = min(num_groups, dim)
    while dim % groups != 0 and groups > 1:
        groups -= 1
    return groups


class MultiScaleLocalBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8):
        super().__init__()
        groups = _valid_group_count(dim, num_groups)
        self.pre_norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.branch_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.branch_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.branch_dilated = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=dim,
            bias=False,
        )
        self.branch_strip = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm(x)
        features = torch.cat(
            [
                self.branch_3x3(x_norm),
                self.branch_5x5(x_norm),
                self.branch_dilated(x_norm),
                self.branch_strip(x_norm),
            ],
            dim=1,
        )
        return x + self.fuse(features)


class MultiScaleStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[MultiScaleLocalBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ChannelMixedMultiScaleLocalBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, channel_expand: int = 2):
        super().__init__()
        groups = _valid_group_count(dim, num_groups)
        hidden_dim = dim * channel_expand
        self.local_block = MultiScaleLocalBlock(dim, num_groups=num_groups)
        self.channel_mixer = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        return x + self.channel_mixer(x)


class ChannelMixedMultiScaleStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ChannelMixedMultiScaleLocalBlock(dim) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LocalResidualConvBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8):
        super().__init__()
        groups = _valid_group_count(dim, num_groups)
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PolarCoordinateTransform(nn.Module):
    def __init__(
        self,
        radial_bins: int = 256,
        theta_bins: int = 256,
        radius_mode: str = "linear",
        align_corners: bool = True,
    ):
        super().__init__()
        if radius_mode not in {"linear", "log"}:
            raise ValueError("radius_mode must be 'linear' or 'log'.")
        self.radial_bins = int(radial_bins)
        self.theta_bins = int(theta_bins)
        self.radius_mode = str(radius_mode)
        self.align_corners = bool(align_corners)

    @staticmethod
    def _max_corner_radius(tx_coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
        device = tx_coords.device
        corners_y = torch.tensor(
            [0.0, 0.0, float(height - 1), float(height - 1)],
            device=device,
            dtype=torch.float32,
        )
        corners_x = torch.tensor(
            [0.0, float(width - 1), 0.0, float(width - 1)],
            device=device,
            dtype=torch.float32,
        )
        dy = corners_y[None, :] - tx_coords[:, 0:1].float()
        dx = corners_x[None, :] - tx_coords[:, 1:2].float()
        return torch.sqrt(dx * dx + dy * dy).amax(dim=1).clamp_min(1.0)

    def _radius_values(self, unit_r: torch.Tensor, max_r: torch.Tensor) -> torch.Tensor:
        if self.radius_mode == "log":
            return torch.expm1(unit_r * torch.log1p(max_r[:, None, None]))
        return unit_r * max_r[:, None, None]

    def cartesian_to_polar(self, x: torch.Tensor, tx_coords: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        tx = tx_coords.to(device=x.device, dtype=torch.float32)
        max_r = self._max_corner_radius(tx, height, width)
        unit_r = torch.linspace(
            0.0,
            1.0,
            self.radial_bins,
            device=x.device,
            dtype=torch.float32,
        )[None, :, None]
        theta = torch.linspace(
            -math.pi,
            math.pi,
            self.theta_bins + 1,
            device=x.device,
            dtype=torch.float32,
        )[:-1]
        radius = self._radius_values(unit_r, max_r)
        y = tx[:, 0:1, None] + radius * torch.sin(theta[None, None, :])
        x_coord = tx[:, 1:2, None] + radius * torch.cos(theta[None, None, :])
        grid_x = 2.0 * x_coord / max(float(width - 1), 1.0) - 1.0
        grid_y = 2.0 * y / max(float(height - 1), 1.0) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )

    def polar_to_cartesian(
        self,
        polar: torch.Tensor,
        tx_coords_input: torch.Tensor,
        input_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        target_h, target_w = target_hw
        input_h, input_w = input_hw
        tx = tx_coords_input.to(device=polar.device, dtype=torch.float32).clone()
        tx[:, 0] = tx[:, 0] * float(target_h - 1) / max(float(input_h - 1), 1.0)
        tx[:, 1] = tx[:, 1] * float(target_w - 1) / max(float(input_w - 1), 1.0)
        max_r = self._max_corner_radius(tx, target_h, target_w)
        yy, xx = torch.meshgrid(
            torch.arange(target_h, device=polar.device, dtype=torch.float32),
            torch.arange(target_w, device=polar.device, dtype=torch.float32),
            indexing="ij",
        )
        dy = yy[None, :, :] - tx[:, 0:1, None]
        dx = xx[None, :, :] - tx[:, 1:2, None]
        radius = torch.sqrt(dx * dx + dy * dy)
        if self.radius_mode == "log":
            unit_r = torch.log1p(radius) / torch.log1p(max_r[:, None, None])
        else:
            unit_r = radius / max_r[:, None, None]
        theta = torch.atan2(dy, dx)
        grid = torch.stack(
            [
                (theta / math.pi).clamp(-1.0, 1.0),
                2.0 * unit_r.clamp(0.0, 1.0) - 1.0,
            ],
            dim=-1,
        )
        return F.grid_sample(
            polar,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )


class PolarDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PolarThetaStem(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8):
        super().__init__()
        groups = _valid_group_count(dim, num_groups)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.theta_depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, 3),
            padding=0,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = F.pad(out, (1, 1, 0, 0), mode="circular")
        out = self.theta_depthwise(out)
        return x + self.pointwise(out)


class PolarRadialMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _valid_group_count(dim, num_groups)
        self.norm = nn.RMSNorm(dim)
        self.mamba = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.theta_norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.theta_depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, 3),
            padding=0,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _scan_radial(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, radial, theta = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(batch * theta, radial, channels)
        out = self.mamba(self.norm(seq))
        return out.view(batch, theta, radial, channels).permute(0, 3, 2, 1).contiguous()

    def _mix_theta(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(self.theta_norm(x), (1, 1, 0, 0), mode="circular")
        return self.theta_depthwise(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_radial = self.out_proj(self._scan_radial(x))
        delta_theta = self._mix_theta(x)
        return x + self.alpha * delta_radial + self.beta * delta_theta


class PolarRadialMambaStage(nn.Module):
    def __init__(self, dim: int, depth: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[PolarRadialMambaBlock(dim, d_state, d_conv, expand) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PolarRadialMambaBranch(nn.Module):
    """
    Tx-centered polar branch.

    Main PGCM change:
    This branch returns p1-p4 instead of only returning the bottleneck feature.
    The model converts these polar features back to Cartesian coordinates and
    uses them to modulate Cartesian encoder stages.
    """

    def __init__(
        self,
        in_channels: int,
        dims: List[int],
        polar_mamba_depths: List[int],
        ssm_d_state: int,
        ssm_d_conv: int,
        ssm_expand: int,
        polar_radial_bins: int,
        polar_theta_bins: int,
        add_valid_mask: bool = True,
        add_tx_polar_map: bool = True,
        add_tx_anchor: Optional[bool] = None,
    ):
        super().__init__()
        if add_tx_anchor is not None:
            add_tx_polar_map = add_tx_anchor
        self.add_valid_mask = bool(add_valid_mask)
        self.add_tx_polar_map = bool(add_tx_polar_map)
        extra_channels = int(self.add_valid_mask) + int(self.add_tx_polar_map)
        self.polar_transform = PolarCoordinateTransform(
            radial_bins=polar_radial_bins,
            theta_bins=polar_theta_bins,
            radius_mode="linear",
        )
        self.polar_patch_embed = nn.Conv2d(
            in_channels + extra_channels,
            dims[0],
            kernel_size=3,
            padding=1,
        )
        self.polar_stem = PolarThetaStem(dims[0])
        self.polar_down1 = PolarDownsample(dims[0], dims[1])
        self.polar_enc2 = PolarRadialMambaStage(
            dims[1], polar_mamba_depths[1], ssm_d_state, ssm_d_conv, ssm_expand
        )
        self.polar_down2 = PolarDownsample(dims[1], dims[2])
        self.polar_enc3 = PolarRadialMambaStage(
            dims[2], polar_mamba_depths[2], ssm_d_state, ssm_d_conv, ssm_expand
        )
        self.polar_down3 = PolarDownsample(dims[2], dims[3])
        self.polar_enc4 = PolarRadialMambaStage(
            dims[3], polar_mamba_depths[3], ssm_d_state, ssm_d_conv, ssm_expand
        )

    def _build_polar_input(self, x: torch.Tensor, tx_coords: torch.Tensor) -> torch.Tensor:
        xp = self.polar_transform.cartesian_to_polar(x, tx_coords)
        extras = []
        if self.add_valid_mask:
            valid = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            extras.append(self.polar_transform.cartesian_to_polar(valid, tx_coords))
        if self.add_tx_polar_map:
            tx_polar_map = xp.new_zeros(xp.shape[0], 1, xp.shape[2], xp.shape[3])
            tx_polar_map[:, :, 0, :] = 1.0
            extras.append(tx_polar_map)
        if extras:
            xp = torch.cat([xp, *extras], dim=1)
        return xp

    def forward(self, x: torch.Tensor, tx_coords: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        xp = self._build_polar_input(x, tx_coords)
        p1 = self.polar_stem(self.polar_patch_embed(xp))
        p2 = self.polar_enc2(self.polar_down1(p1))
        p3 = self.polar_enc3(self.polar_down2(p2))
        p4 = self.polar_enc4(self.polar_down3(p3))
        return p1, p2, p3, p4


class PolarGuidedCartesianModulation(nn.Module):
    """
    PGCM: FiLM-like polar-guided modulation for Cartesian features.

    Fc_mod = (1 + gamma) * Fc + beta.

    Unlike standard channel-wise FiLM, gamma and beta are spatial maps with
    shape B x 1 x H x W and are broadcast over channels. This matches the
    radio-map setting where the modulation should primarily decide which
    regions need Tx-conditioned correction.
    """

    def __init__(
        self,
        cart_dim: int,
        polar_dim: int,
        hidden_ratio: float = 0.5,
        num_groups: int = 8,
        gamma_limit: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()
        hidden_dim = max(int(cart_dim * hidden_ratio), 16)
        hidden_groups = _valid_group_count(hidden_dim, num_groups)
        self.gamma_limit = float(gamma_limit)
        self.condition = nn.Sequential(
            nn.Conv2d(polar_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=hidden_groups, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1, bias=True),
        )
        if zero_init:
            nn.init.zeros_(self.condition[-1].weight)
            nn.init.zeros_(self.condition[-1].bias)

    def forward(self, cart: torch.Tensor, polar_cart: torch.Tensor) -> torch.Tensor:
        if polar_cart.shape[2:] != cart.shape[2:]:
            polar_cart = F.interpolate(
                polar_cart,
                size=cart.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        gamma, beta = self.condition(polar_cart).chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.gamma_limit
        return (1.0 + gamma) * cart + beta


class PolarGuidedCartesianBranch(nn.Module):
    """
    Cartesian obstruction correction branch with optional PGCM at each scale.

    Main PGCM change:
    Polar guidance is injected before each Cartesian encoder stage, so the
    Cartesian branch receives Tx-centered propagation context early instead of
    only at the bottleneck fusion.
    """

    def __init__(
        self,
        in_channels: int,
        dims: List[int],
        depths: List[int],
        use_pgcm: bool = True,
        use_pgcm_at_stage1: bool = False,
        pgcm_gamma_limit: float = 1.0,
        pgcm_zero_init: bool = True,
    ):
        super().__init__()
        self.use_pgcm = bool(use_pgcm)
        self.use_pgcm_at_stage1 = bool(use_pgcm_at_stage1)
        self.cart_patch_embed = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)
        self.pgcm1 = PolarGuidedCartesianModulation(
            dims[0], dims[0], gamma_limit=pgcm_gamma_limit, zero_init=pgcm_zero_init
        )
        self.cart_enc1 = MultiScaleStage(dims[0], depths[0])
        self.cart_down1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.pgcm2 = PolarGuidedCartesianModulation(
            dims[1], dims[1], gamma_limit=pgcm_gamma_limit, zero_init=pgcm_zero_init
        )
        self.cart_enc2 = ChannelMixedMultiScaleStage(dims[1], depths[1])
        self.cart_down2 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.pgcm3 = PolarGuidedCartesianModulation(
            dims[2], dims[2], gamma_limit=pgcm_gamma_limit, zero_init=pgcm_zero_init
        )
        self.cart_enc3 = ChannelMixedMultiScaleStage(dims[2], depths[2])
        self.cart_down3 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        self.pgcm4 = PolarGuidedCartesianModulation(
            dims[3], dims[3], gamma_limit=pgcm_gamma_limit, zero_init=pgcm_zero_init
        )
        self.cart_bottleneck_refine = nn.Sequential(
            MultiScaleStage(dims[3], depths[3]),
            MultiScaleLocalBlock(dims[3]),
            LocalResidualConvBlock(dims[3]),
        )

    def _modulate(
        self,
        module: PolarGuidedCartesianModulation,
        cart: torch.Tensor,
        polar_cart: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_pgcm:
            return cart
        return module(cart, polar_cart)

    def forward(
        self,
        x: torch.Tensor,
        polar_cart_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_cart, p2_cart, p3_cart, p4_cart = polar_cart_features
        c0 = self.cart_patch_embed(x)
        if self.use_pgcm_at_stage1:
            c0 = self._modulate(self.pgcm1, c0, p1_cart)
        x1 = self.cart_enc1(c0)

        c1 = self.cart_down1(x1)
        c1 = self._modulate(self.pgcm2, c1, p2_cart)
        x2 = self.cart_enc2(c1)

        c2 = self.cart_down2(x2)
        c2 = self._modulate(self.pgcm3, c2, p3_cart)
        x3 = self.cart_enc3(c2)

        c3 = self.cart_down3(x3)
        c3 = self._modulate(self.pgcm4, c3, p4_cart)
        cart = self.cart_bottleneck_refine(c3)
        return x1, x2, x3, cart


class GatedDualCoordinateFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        cart_correction_dropout: float = 0.0,
        cart_gate_bias_init: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        if not 0.0 <= cart_correction_dropout < 1.0:
            raise ValueError("cart_correction_dropout must be in [0, 1).")
        groups = _valid_group_count(dim, num_groups)
        hidden_dim = max(dim // 2, 16)
        gate_groups = _valid_group_count(hidden_dim, num_groups)
        self.cart_correction_dropout = float(cart_correction_dropout)
        self.polar_adapter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )
        self.cart_adapter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gate_groups, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[3].bias, cart_gate_bias_init)

    def _drop_correction(self, correction: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cart_correction_dropout <= 0.0:
            return correction
        keep_prob = 1.0 - self.cart_correction_dropout
        keep = correction.new_empty(correction.shape[0], 1, 1, 1).bernoulli_(keep_prob)
        return correction * keep / keep_prob

    def forward(self, cart: torch.Tensor, polar_cart: torch.Tensor) -> torch.Tensor:
        polar_base = self.polar_adapter(polar_cart)
        correction = self.cart_adapter(cart)
        gate = self.gate(torch.cat([polar_base, correction], dim=1))
        correction = self._drop_correction(correction)
        return polar_base + gate * correction


class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        groups = _valid_group_count(out_channels, num_groups)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shortcut(x) + self.refine(x))


class ResidualDecoderUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        groups = _valid_group_count(out_channels)
        self.up_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
        )
        self.refine = ResidualDecoderBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        return self.refine(torch.cat([x, skip], dim=1))


class PADCNet(nn.Module):
    """
    PA-DCNet: a propagation-aware dual-coordinate network.

    The model keeps the dual-coordinate design:
    - Polar branch: Tx-centered radial Mamba for global propagation context.
    - Cartesian branch: local obstruction correction, now modulated by polar
      features at multiple encoder scales through PGCM.
    - Fusion: polar base plus gated Cartesian residual correction.

    The first convolution always follows in_channels, so different task modes
    do not change parameter shapes. If a task only needs the first two
    channels, the unused channels are zeroed in forward instead of being
    removed from the tensor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        dims: List[int] = [48, 96, 192, 384],
        depths: List[int] = [2, 3, 4, 2],
        ssm_d_state: int = 32,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        polar_radial_bins: int = 256,
        polar_theta_bins: int = 256,
        polar_mamba_depths: List[int] = [0, 1, 1, 2],
        cart_correction_dropout: float = 0.0,
        cart_gate_bias_init: float = 0.0,
        use_pgcm: bool = True,
        use_pgcm_at_stage1: bool = False,
        pgcm_gamma_limit: float = 1.0,
        pgcm_zero_init: bool = True,
        task: str = "srm",
        input_mode: str = "auto",
        tx_channel: int = 1,
        drop_static_third_channel: Optional[bool] = None,
        add_polar_valid_mask: bool = True,
        add_polar_tx_map: bool = True,
        add_polar_tx_anchor: Optional[bool] = None,
    ):
        super().__init__()
        if len(dims) != 4 or len(depths) != 4 or len(polar_mamba_depths) != 4:
            raise ValueError("dims, depths, and polar_mamba_depths must each contain four stages.")
        if tx_channel < 0 or tx_channel >= in_channels:
            raise ValueError("tx_channel must be a valid input channel index.")

        self.in_channels = int(in_channels)
        self.tx_channel = int(tx_channel)
        self.use_pgcm_at_stage1 = bool(use_pgcm_at_stage1)
        self.task = str(task).lower()
        self.input_mode = self._resolve_input_mode(input_mode, drop_static_third_channel)

        self.polar_branch = PolarRadialMambaBranch(
            in_channels=self.in_channels,
            dims=dims,
            polar_mamba_depths=polar_mamba_depths,
            ssm_d_state=ssm_d_state,
            ssm_d_conv=ssm_d_conv,
            ssm_expand=ssm_expand,
            polar_radial_bins=polar_radial_bins,
            polar_theta_bins=polar_theta_bins,
            add_valid_mask=add_polar_valid_mask,
            add_tx_polar_map=add_polar_tx_map,
            add_tx_anchor=add_polar_tx_anchor,
        )
        self.cartesian_branch = PolarGuidedCartesianBranch(
            in_channels=self.in_channels,
            dims=dims,
            depths=depths,
            use_pgcm=use_pgcm,
            use_pgcm_at_stage1=use_pgcm_at_stage1,
            pgcm_gamma_limit=pgcm_gamma_limit,
            pgcm_zero_init=pgcm_zero_init,
        )
        self.dual_coordinate_fusion = GatedDualCoordinateFusion(
            dims[3],
            cart_correction_dropout=cart_correction_dropout,
            cart_gate_bias_init=cart_gate_bias_init,
        )
        self.dec3 = ResidualDecoderUpBlock(dims[3], dims[2], dims[2])
        self.dec2 = ResidualDecoderUpBlock(dims[2], dims[1], dims[1])
        self.dec1 = ResidualDecoderUpBlock(dims[1], dims[0], dims[0])
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def _resolve_input_mode(
        self,
        input_mode: str,
        drop_static_third_channel: Optional[bool],
    ) -> str:
        if drop_static_third_channel is not None:
            return "first2" if bool(drop_static_third_channel) else "all"
        mode = str(input_mode).lower()
        if mode == "auto":
            if self.task == "srm":
                return "first2"
            if self.task == "drm":
                return "all"
            raise ValueError("task must be 'srm' or 'drm' when input_mode='auto'.")
        if mode in {"first2", "2ch", "two_channel", "two-channel"}:
            return "first2"
        if mode in {"all", "3ch", "full"}:
            return "all"
        raise ValueError("input_mode must be 'auto', 'first2', or 'all'.")

    def _model_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "first2" and x.shape[1] > 2:
            x = x.clone()
            x[:, 2:] = 0.0
        return x

    def _extract_tx_coords(self, x: torch.Tensor) -> torch.Tensor:
        tx_map = x[:, self.tx_channel:self.tx_channel + 1]
        _, _, _, width = tx_map.shape
        flat_idx = tx_map.flatten(2).argmax(dim=-1).squeeze(1)
        y = torch.div(flat_idx, width, rounding_mode="floor")
        x_coord = flat_idx % width
        return torch.stack([y, x_coord], dim=1).to(dtype=torch.float32)

    def _polar_to_cartesian_features(
        self,
        polar_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tx_coords: torch.Tensor,
        input_hw: Tuple[int, int],
        target_hws: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_cart = None
        if self.use_pgcm_at_stage1:
            p1_cart = self.polar_branch.polar_transform.polar_to_cartesian(
                polar_features[0],
                tx_coords,
                input_hw,
                target_hws[0],
            )
        p2_to_p4 = tuple(
            self.polar_branch.polar_transform.polar_to_cartesian(
                polar_feat,
                tx_coords,
                input_hw,
                target_hw,
            )
            for polar_feat, target_hw in zip(polar_features[1:], target_hws[1:])
        )
        return (p1_cart, *p2_to_p4)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_hw = x.shape[2:]
        tx_coords = self._extract_tx_coords(x)
        x_eff = self._model_input(x)

        polar_features = self.polar_branch(x_eff, tx_coords)
        target_hws = (
            input_hw,
            (input_hw[0] // 2, input_hw[1] // 2),
            (input_hw[0] // 4, input_hw[1] // 4),
            (input_hw[0] // 8, input_hw[1] // 8),
        )
        polar_cart_features = self._polar_to_cartesian_features(
            polar_features,
            tx_coords,
            input_hw,
            target_hws,
        )

        x1, x2, x3, cart = self.cartesian_branch(x_eff, polar_cart_features)
        polar_bottleneck_cart = polar_cart_features[-1]
        fused = self.dual_coordinate_fusion(cart, polar_bottleneck_cart)
        d3 = self.dec3(fused, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        out = self.final_conv(d1)

        features = {
            "model_input": x_eff,
            "tx_coords": tx_coords,
            "polar_p1": polar_features[0],
            "polar_p2": polar_features[1],
            "polar_p3": polar_features[2],
            "polar_p4": polar_features[3],
            "polar_cart_p1": polar_cart_features[0],
            "polar_cart_p2": polar_cart_features[1],
            "polar_cart_p3": polar_cart_features[2],
            "polar_cart_p4": polar_cart_features[3],
            "cart_x1": x1,
            "cart_x2": x2,
            "cart_x3": x3,
            "cart_bottleneck": cart,
            "fused_bottleneck": fused,
            "decoder_d3": d3,
            "decoder_d2": d2,
            "decoder_d1": d1,
        }
        return out, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_features(x)
        return out


PADCNetModel = PADCNet
Model = PADCNet


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PADCNet().to(device).eval()
    x = torch.randn(1, 3, 256, 256, device=device)
    x[:, 1].zero_()
    x[:, 1, 128, 128] = 1.0
    x[:, 2] = x[:, 0]
    with torch.no_grad():
        y, features = model.forward_features(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model.__class__.__name__}")
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
    print(f"Params: {params:.3f} M")
    for name, value in features.items():
        if torch.is_tensor(value):
            print(f"{name}: {tuple(value.shape)}")
