"""Reference-data utilities for lid-driven cavity benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "references" / "lid_driven_cavity"


@dataclass(frozen=True)
class CavityProfileReference:
    """Centerline profile data for lid-driven cavity validation."""

    reynolds: float
    source: str
    u_profile: pd.DataFrame
    v_profile: pd.DataFrame

    @property
    def has_u(self) -> bool:
        return not self.u_profile.empty

    @property
    def has_v(self) -> bool:
        return not self.v_profile.empty


def load_lid_cavity_profile_reference(
    reference: str,
    reynolds: float,
    reference_path: str | Path | None = None,
) -> CavityProfileReference:
    """Load built-in Ghia or external centerline data for one Reynolds number."""
    reference = (reference or "none").lower()
    if reference == "none":
        return CavityProfileReference(reynolds, "none", pd.DataFrame(), pd.DataFrame())
    if reference == "ghia":
        return _load_ghia(reynolds)
    if reference == "external":
        if reference_path is None:
            raise ValueError("--reference external requires --reference_path.")
        return _load_external_profile(Path(reference_path), reynolds)
    raise ValueError(f"Unknown cavity reference '{reference}'. Use ghia, external, or none.")


def load_full_field_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Load optional structured full-field CFD reference from CSV/NPZ/NPY."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Full-field cavity reference not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return _columns_to_field_dict(df.to_dict("list"), path)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return _columns_to_field_dict({k: data[k] for k in data.files}, path)
    if getattr(data, "dtype", None) is not None and data.dtype.names:
        return _columns_to_field_dict({name: data[name] for name in data.dtype.names}, path)
    if data.shape[-1] < 4:
        raise ValueError("Full-field NPY reference must have columns x,y,u,v and optional p,omega.")
    names = ["x", "y", "u", "v", "p", "omega"]
    return _columns_to_field_dict({name: data[..., i].reshape(-1) for i, name in enumerate(names[: data.shape[-1]])}, path)


def interpolate_full_field(reference: dict[str, np.ndarray], xy: np.ndarray) -> dict[str, np.ndarray]:
    """Bilinearly interpolate a structured full-field reference onto points."""
    x = np.asarray(reference["x"], dtype=float).reshape(-1)
    y = np.asarray(reference["y"], dtype=float).reshape(-1)
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)
    if nx * ny != x.size:
        raise ValueError("Full-field reference must be a structured tensor-product grid.")
    order = np.lexsort((x, y))
    out: dict[str, np.ndarray] = {}
    for name in ["u", "v", "p", "omega"]:
        values = np.asarray(reference.get(name, np.full_like(x, np.nan)), dtype=float).reshape(-1)[order].reshape(ny, nx)
        out[name] = _interp2(xu, yu, values, xy[:, 0], xy[:, 1]).reshape(-1, 1)
    out["speed"] = np.sqrt(out["u"] ** 2 + out["v"] ** 2)
    out["p_x"] = np.zeros_like(out["u"])
    out["p_y"] = np.zeros_like(out["u"])
    out["has_p_reference"] = bool(reference.get("has_p_reference", np.isfinite(reference.get("p", [])).any()))
    out["has_omega_reference"] = bool(reference.get("has_omega_reference", np.isfinite(reference.get("omega", [])).any()))
    out["omega_reference_source"] = str(reference.get("omega_reference_source", "provided"))
    out["source_path"] = str(reference.get("source_path", ""))
    return out


def validate_full_field_against_ghia(path: str | Path, reynolds: float) -> dict[str, float | str]:
    """Compare a full-field CFD reference against built-in Ghia centerline profiles."""
    reference = load_full_field_reference(path)
    profile = load_lid_cavity_profile_reference("ghia", reynolds)
    out: dict[str, float | str] = {
        "reynolds": float(reynolds),
        "full_field_reference_path": str(path),
        "ghia_reference_source": profile.source,
    }
    if profile.has_u:
        u_xy = np.column_stack([np.full(len(profile.u_profile), 0.5), profile.u_profile["y"].to_numpy(dtype=float)])
        pred = interpolate_full_field(reference, u_xy)["u"]
        ref = profile.u_profile["u_ref"].to_numpy(dtype=float).reshape(-1, 1)
        out["cfd_vs_ghia_u_centerline_rmse"] = _rmse_np(pred, ref)
        out["cfd_vs_ghia_u_centerline_rel_l2"] = _relative_l2_np(pred, ref)
    else:
        out["cfd_vs_ghia_u_centerline_rmse"] = float("nan")
        out["cfd_vs_ghia_u_centerline_rel_l2"] = float("nan")
    if profile.has_v:
        v_xy = np.column_stack([profile.v_profile["x"].to_numpy(dtype=float), np.full(len(profile.v_profile), 0.5)])
        pred = interpolate_full_field(reference, v_xy)["v"]
        ref = profile.v_profile["v_ref"].to_numpy(dtype=float).reshape(-1, 1)
        out["cfd_vs_ghia_v_centerline_rmse"] = _rmse_np(pred, ref)
        out["cfd_vs_ghia_v_centerline_rel_l2"] = _relative_l2_np(pred, ref)
    else:
        out["cfd_vs_ghia_v_centerline_rmse"] = float("nan")
        out["cfd_vs_ghia_v_centerline_rel_l2"] = float("nan")
    return out


def _load_ghia(reynolds: float) -> CavityProfileReference:
    u_path = REFERENCE_DIR / "ghia_1982_u_centerline.csv"
    v_path = REFERENCE_DIR / "ghia_1982_v_centerline.csv"
    u = _filter_re(pd.read_csv(u_path), reynolds, "u_ref", u_path)
    v = _filter_re(pd.read_csv(v_path), reynolds, "v_ref", v_path)
    return CavityProfileReference(float(reynolds), "ghia_1982", u, v)


def _load_external_profile(path: Path, reynolds: float) -> CavityProfileReference:
    if not path.exists():
        raise FileNotFoundError(f"External cavity profile reference not found: {path}")
    df = pd.read_csv(path)
    required = {"re", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"External cavity profile CSV is missing columns: {sorted(missing)}")
    if "u_ref" not in df.columns and "v_ref" not in df.columns:
        raise ValueError("External cavity profile CSV must include u_ref and/or v_ref.")
    u = _filter_re(df.dropna(subset=["u_ref"]) if "u_ref" in df.columns else pd.DataFrame(), reynolds, "u_ref", path)
    v = _filter_re(df.dropna(subset=["v_ref"]) if "v_ref" in df.columns else pd.DataFrame(), reynolds, "v_ref", path)
    return CavityProfileReference(float(reynolds), str(path), u, v)


def _filter_re(df: pd.DataFrame, reynolds: float, value_col: str, source: Path) -> pd.DataFrame:
    if df.empty:
        return df
    required = {"re", "x", "y", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    sub = df[np.isclose(df["re"].astype(float), float(reynolds))].copy()
    if sub.empty:
        available = sorted(float(v) for v in df["re"].dropna().unique())
        raise ValueError(f"No lid-driven cavity reference rows for Re={reynolds:g} in {source}. Available Re: {available}")
    return sub.sort_values(["y", "x"]).reset_index(drop=True)


def _columns_to_field_dict(columns: dict[str, Any], source: Path) -> dict[str, np.ndarray]:
    required = {"x", "y", "u", "v"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Full-field reference {source} is missing columns/keys: {sorted(missing)}")
    out = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in columns.items() if name in {"x", "y", "u", "v", "p", "omega"}}
    n = len(out["x"])
    if any(len(values) != n for values in out.values()):
        raise ValueError(f"Full-field reference {source} has inconsistent column lengths.")
    out["has_p_reference"] = bool("p" in out and np.isfinite(out["p"]).any())
    if "p" not in out:
        out["p"] = np.full(n, np.nan, dtype=float)
    provided_omega = bool("omega" in out and np.isfinite(out["omega"]).any())
    if not provided_omega:
        computed = _compute_structured_vorticity(out)
        out["omega"] = computed if computed is not None else np.full(n, np.nan, dtype=float)
        out["omega_reference_source"] = "computed_from_velocity" if computed is not None else "missing"
    else:
        out["omega_reference_source"] = "provided"
    out["has_omega_reference"] = bool(np.isfinite(out["omega"]).any())
    out["source_path"] = str(source)
    return out


def _compute_structured_vorticity(columns: dict[str, np.ndarray]) -> np.ndarray | None:
    x = np.asarray(columns["x"], dtype=float).reshape(-1)
    y = np.asarray(columns["y"], dtype=float).reshape(-1)
    u = np.asarray(columns["u"], dtype=float).reshape(-1)
    v = np.asarray(columns["v"], dtype=float).reshape(-1)
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = len(xu), len(yu)
    if nx < 2 or ny < 2 or nx * ny != x.size:
        return None
    order = np.lexsort((x, y))
    u_grid = u[order].reshape(ny, nx)
    v_grid = v[order].reshape(ny, nx)
    du_dy = np.gradient(u_grid, yu, axis=0, edge_order=1)
    dv_dx = np.gradient(v_grid, xu, axis=1, edge_order=1)
    omega_grid = dv_dx - du_dy
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return omega_grid.reshape(-1)[inverse]


def _relative_l2_np(pred: np.ndarray, true: np.ndarray, min_reference_norm: float = 1e-8) -> float:
    ref_norm = float(np.linalg.norm(true))
    if ref_norm < min_reference_norm:
        return float("nan")
    return float(np.linalg.norm(pred - true) / ref_norm)


def _rmse_np(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))


def _interp2(xu: np.ndarray, yu: np.ndarray, values: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    xq = np.clip(xq, xu[0], xu[-1])
    yq = np.clip(yq, yu[0], yu[-1])
    ix = np.clip(np.searchsorted(xu, xq, side="right") - 1, 0, len(xu) - 2)
    iy = np.clip(np.searchsorted(yu, yq, side="right") - 1, 0, len(yu) - 2)
    x0, x1 = xu[ix], xu[ix + 1]
    y0, y1 = yu[iy], yu[iy + 1]
    tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq, dtype=float), where=(x1 != x0))
    ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq, dtype=float), where=(y1 != y0))
    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
