"""Optional physics-aware output wrappers shared by every PINN method."""

from __future__ import annotations

import torch
import torch.nn as nn


class CavityHardBoundaryWrapper(nn.Module):
    """Enforce cavity velocity walls with a corner-smoothed moving lid.

    The lid discontinuity cannot be represented continuously at the two top
    corners. The configured corner width makes that regularization explicit;
    all remaining wall values are enforced exactly by construction.
    """

    def __init__(
        self,
        base: nn.Module,
        bounds: tuple[float, float, float, float],
        lid_velocity: float = 1.0,
        corner_width: float = 0.02,
        lid_lifting: str = "linear",
        lid_vertical_power: int = 6,
    ) -> None:
        super().__init__()
        self.base = base
        self.bounds = tuple(float(value) for value in bounds)
        self.lid_velocity = float(lid_velocity)
        self.corner_width = max(float(corner_width), 1e-6)
        self.lid_lifting = str(lid_lifting).lower()
        self.lid_vertical_power = max(2, int(lid_vertical_power))
        if self.lid_lifting not in {"linear", "harmonic", "divergence_free"}:
            raise ValueError(
                "cavity hard-boundary lid_lifting must be 'linear', "
                "'harmonic', or 'divergence_free'."
            )
        self.physics_formulation = "cavity_hard_boundary"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.base(coords)
        x0, x1, y0, y1 = self.bounds
        x = (coords[:, 0:1] - x0) / max(x1 - x0, 1e-12)
        y = (coords[:, 1:2] - y0) / max(y1 - y0, 1e-12)
        left_arg = x / self.corner_width
        right_arg = (1.0 - x) / self.corner_width
        left = _smoothstep01(left_arg)
        right = _smoothstep01(right_arg)
        horizontal_lift = left * right
        if self.lid_lifting == "divergence_free":
            # Derive the lifting from
            #   psi = U * Ly * eta^m(eta - 1) * h(xi).
            # It gives the regularized moving lid at eta=1, stationary other
            # walls, and exactly zero divergence without prescribing a CFD
            # solution or a Couette-like interior profile. m localizes the
            # geometric lift near the moving lid so corner smoothing does not
            # create artificial side jets deep inside the cavity.
            left_derivative = (
                _smoothstep01_derivative(left_arg) / self.corner_width
            )
            right_derivative = (
                -_smoothstep01_derivative(right_arg) / self.corner_width
            )
            horizontal_derivative = (
                left_derivative * right + left * right_derivative
            )
            power = self.lid_vertical_power
            y_power = y.pow(power)
            vertical_streamfunction = y_power * (y - 1.0)
            vertical_derivative = (
                power * y.pow(power - 1) * (y - 1.0) + y_power
            )
            aspect = (y1 - y0) / max(x1 - x0, 1e-12)
            u_lift = (
                self.lid_velocity
                * vertical_derivative
                * horizontal_lift
            )
            v_lift = (
                -self.lid_velocity
                * aspect
                * vertical_streamfunction
                * horizontal_derivative
            )
        elif self.lid_lifting == "harmonic":
            # A reference-free, rapidly decaying lifting avoids imposing a
            # Couette-like horizontal flow through the whole cavity. It still
            # satisfies the regularized lid and all stationary walls exactly.
            aspect_wave = torch.as_tensor(
                torch.pi * max((y1 - y0) / max(x1 - x0, 1e-12), 1e-6),
                dtype=coords.dtype,
                device=coords.device,
            )
            vertical_lift = torch.sinh(aspect_wave * y) / torch.sinh(aspect_wave)
            u_lift = self.lid_velocity * vertical_lift * horizontal_lift
            v_lift = torch.zeros_like(u_lift)
        else:
            vertical_lift = y
            u_lift = self.lid_velocity * vertical_lift * horizontal_lift
            v_lift = torch.zeros_like(u_lift)
        envelope = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        u = u_lift + envelope * raw[:, 0:1]
        v = v_lift + envelope * raw[:, 1:2]
        p = raw[:, 2:3]
        return torch.cat([u, v, p], dim=1)


class CavityUVPVelocityLiftWrapper(nn.Module):
    """Direct ``u,v,p`` cavity ansatz with analytically lifted velocity walls."""

    def __init__(
        self,
        base: nn.Module,
        bounds: tuple[float, float, float, float],
        lid_velocity: float = 1.0,
        corner_width: float = 0.05,
        lift_scale: float = 16.0,
        lid_vertical_power: int = 3,
    ) -> None:
        super().__init__()
        self.base = base
        self.bounds = tuple(float(value) for value in bounds)
        self.lid_velocity = float(lid_velocity)
        self.corner_width = max(float(corner_width), 1e-6)
        self.lift_scale = float(lift_scale)
        self.lid_vertical_power = max(int(lid_vertical_power), 1)
        self.physics_formulation = "cavity_uvp_velocity_lift"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.base(coords)
        x0, x1, y0, y1 = self.bounds
        xi = (coords[:, 0:1] - x0) / max(x1 - x0, 1e-12)
        eta = (coords[:, 1:2] - y0) / max(y1 - y0, 1e-12)
        lid_profile = _smoothstep01(xi / self.corner_width) * _smoothstep01(
            (1.0 - xi) / self.corner_width
        )
        u_lift = (
            self.lid_velocity
            * eta.pow(self.lid_vertical_power)
            * lid_profile
        )
        distance = xi * (1.0 - xi) * eta * (1.0 - eta)
        u = u_lift + self.lift_scale * distance * raw[:, 0:1]
        v = self.lift_scale * distance * raw[:, 1:2]
        p = raw[:, 2:3]
        return torch.cat([u, v, p], dim=1)


class StreamfunctionPressureWrapper(nn.Module):
    """Convert a two-output ``(psi, p)`` network into divergence-free ``u,v,p``."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.physics_formulation = "streamfunction_pressure"

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            working = coords if coords.requires_grad else coords.clone().detach().requires_grad_(True)
            output = self.base(working)
            psi = output[:, 0:1]
            p = output[:, 1:2]
            gradient = torch.autograd.grad(
                psi,
                working,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = gradient[:, 1:2]
            v = -gradient[:, 0:1]
            return torch.cat([u, v, p], dim=1)


class HardBoundaryStreamfunctionPressureWrapper(nn.Module):
    """Divergence-free cavity ansatz with regularized hard velocity walls.

    The base network predicts ``(raw_psi, raw_p)``. The streamfunction is
    assembled as

        psi_total = psi_lift + envelope_psi * raw_psi

    where the correction envelope and its first derivatives vanish on all
    walls. Velocity is then recovered from the streamfunction, so the interior
    continuity residual is zero up to autograd/numerical precision while the
    regularized moving lid remains imposed by construction.
    """

    def __init__(
        self,
        base: nn.Module,
        bounds: tuple[float, float, float, float],
        lid_velocity: float = 1.0,
        corner_width: float = 0.02,
        lid_vertical_power: int = 6,
        correction_scale: float = 64.0,
        correction_wall_boost: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.bounds = tuple(float(value) for value in bounds)
        self.lid_velocity = float(lid_velocity)
        self.corner_width = max(float(corner_width), 1e-6)
        self.lid_vertical_power = max(2, int(lid_vertical_power))
        self.correction_scale = float(correction_scale)
        self.correction_wall_boost = max(float(correction_wall_boost), 0.0)
        self.physics_formulation = "hard_boundary_streamfunction_pressure"
        self.latest_streamfunction_diagnostics: dict[str, float] = {}

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            working = coords if coords.requires_grad else coords.clone().detach().requires_grad_(True)
            raw = self.base(working)
            components = self.streamfunction_components(working, raw[:, 0:1])
            psi_total = components["psi_total"]
            p = raw[:, 1:2]
            self._record_diagnostics(components)
            gradient = torch.autograd.grad(
                psi_total,
                working,
                grad_outputs=torch.ones_like(psi_total),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            u = gradient[:, 1:2]
            v = -gradient[:, 0:1]
            return torch.cat([u, v, p], dim=1)

    def streamfunction(self, coords: torch.Tensor, raw_psi: torch.Tensor) -> torch.Tensor:
        return self.streamfunction_components(coords, raw_psi)["psi_total"]

    def streamfunction_components(
        self,
        coords: torch.Tensor,
        raw_psi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x0, x1, y0, y1 = self.bounds
        lx = max(x1 - x0, 1e-12)
        ly = max(y1 - y0, 1e-12)
        xi = (coords[:, 0:1] - x0) / lx
        eta = (coords[:, 1:2] - y0) / ly
        horizontal = self._lid_profile(xi)
        vertical = self._vertical_lid_shape(eta)
        psi_lift = self.lid_velocity * ly * horizontal * vertical
        horizontal_correction = (
            xi.pow(2)
            * (1.0 - xi).pow(2)
            * (1.0 + self.correction_wall_boost * (xi - 0.5).pow(2))
        )
        vertical_correction = (
            eta.pow(2)
            * (1.0 - eta).pow(2)
            * (1.0 + self.correction_wall_boost * (eta - 0.5).pow(2))
        )
        correction_envelope = horizontal_correction * vertical_correction
        scaled_correction = self.correction_scale * correction_envelope * raw_psi
        return {
            "raw_psi": raw_psi,
            "psi_lift": psi_lift,
            "correction_envelope": correction_envelope,
            "scaled_correction": scaled_correction,
            "psi_total": psi_lift + scaled_correction,
        }

    def streamfunction_auxiliary(self, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.base(coords)
        components = self.streamfunction_components(coords, raw[:, 0:1])
        components["raw_p"] = raw[:, 1:2]
        return components

    def _lid_profile(self, xi: torch.Tensor) -> torch.Tensor:
        left = _smoothstep01(xi / self.corner_width)
        right = _smoothstep01((1.0 - xi) / self.corner_width)
        return left * right

    def _vertical_lid_shape(self, eta: torch.Tensor) -> torch.Tensor:
        power = self.lid_vertical_power
        return eta.pow(power) * (eta - 1.0)

    def _record_diagnostics(self, components: dict[str, torch.Tensor]) -> None:
        diagnostics: dict[str, float] = {}
        for name in ["raw_psi", "scaled_correction", "psi_total"]:
            values = components[name].detach()
            diagnostics[f"{name}_mean"] = float(values.mean().cpu())
            diagnostics[f"{name}_std"] = float(values.std(unbiased=False).cpu())
            diagnostics[f"{name}_abs_max"] = float(values.abs().max().cpu())
        self.latest_streamfunction_diagnostics = diagnostics


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _smoothstep01_derivative(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    derivative = 6.0 * clipped * (1.0 - clipped)
    return torch.where(
        (value > 0.0) & (value < 1.0),
        derivative,
        torch.zeros_like(derivative),
    )
