"""Autograd Navier-Stokes residuals."""

from __future__ import annotations

import time

import torch


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx with graph retention for higher derivatives."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def navier_stokes_residuals(
    model: torch.nn.Module,
    coords: torch.Tensor,
    nu: float,
    steady: bool = True,
    detach_coords: bool = True,
    runtime_profile: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute velocity-pressure residuals for 2D incompressible Navier-Stokes."""
    total_start = time.perf_counter()
    if detach_coords:
        xyt = coords.clone().detach().requires_grad_(True)
    else:
        xyt = coords.requires_grad_(True)

    forward_start = time.perf_counter()
    out = model(xyt)
    if runtime_profile is not None:
        runtime_profile["forward_pass_sec"] = runtime_profile.get(
            "forward_pass_sec", 0.0
        ) + (time.perf_counter() - forward_start)
    residual_start = time.perf_counter()
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    grad_u = gradients(u, xyt)
    grad_v = gradients(v, xyt)
    grad_p = gradients(p, xyt)

    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    if steady or xyt.shape[1] < 3:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
    else:
        u_t = grad_u[:, 2:3]
        v_t = grad_v[:, 2:3]

    u_xx = gradients(u_x, xyt)[:, 0:1]
    u_yy = gradients(u_y, xyt)[:, 1:2]
    v_xx = gradients(v_x, xyt)[:, 0:1]
    v_yy = gradients(v_y, xyt)[:, 1:2]

    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    omega = v_x - u_y
    p_grad_norm = torch.sqrt(p_x * p_x + p_y * p_y + 1e-18)
    pde_residual = torch.sqrt(f_u * f_u + f_v * f_v + f_c * f_c + 1e-18)
    if runtime_profile is not None:
        runtime_profile["pde_residual_autograd_sec"] = runtime_profile.get(
            "pde_residual_autograd_sec", 0.0
        ) + (time.perf_counter() - residual_start)
        runtime_profile["navier_stokes_total_sec"] = runtime_profile.get(
            "navier_stokes_total_sec", 0.0
        ) + (time.perf_counter() - total_start)

    return {
        "coords": xyt,
        "u": u,
        "v": v,
        "p": p,
        "u_x": u_x,
        "u_y": u_y,
        "u_t": u_t,
        "v_x": v_x,
        "v_y": v_y,
        "v_t": v_t,
        "p_x": p_x,
        "p_y": p_y,
        "u_xx": u_xx,
        "u_yy": u_yy,
        "v_xx": v_xx,
        "v_yy": v_yy,
        "f_c": f_c,
        "f_u": f_u,
        "f_v": f_v,
        "omega": omega,
        "p_grad_norm": p_grad_norm,
        "pde_residual": pde_residual,
    }
