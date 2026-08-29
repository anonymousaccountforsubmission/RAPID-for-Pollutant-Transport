

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import json
import time
import pandas as pd


# =============================================================================
# CUDA CHECK
# =============================================================================

print("=" * 60)
print("CUDA DIAGNOSTIC")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. In Colab, go to Runtime > Change runtime type > T4 GPU."
    )

device = torch.device("cuda")
print(f"Device:          {device}")
print(f"GPU:             {torch.cuda.get_device_name(0)}")

a = torch.randn(1000, 1000, device=device)
print(f"GPU tensor OK:   {a.device}")
del a
torch.cuda.empty_cache()

# =============================================================================
# PATHS
# =============================================================================
import os
REPO_BASE = Path(__file__).resolve().parent.parent  #pare"nt repo
DATA_BASE = Path(os.environ.get("AERMOD Data"))
SAVE_PATH = Path(__file__).resolve().parent / "outputs"
SAVE_PATH.mkdir(parents=True, exist_ok=True)
SCENARIO_PATH = DATA_BASE / "no2pvsm"
COORD_FRAME_PATH = SCENARIO_PATH / "coordinate_frame.json"
NORM_META_PATH = SCENARIO_PATH / "normalization_metadata.json"


MODEL_BEST    = DATA_BASE / 'no2_pinn_reaction_lowwind_best.pth'
MODEL_FINAL   = DATA_BASE / 'no2_pinn_reaction_lowwind_final.pth'
METRICS_OUT   = DATA_BASE / 'no2_pinn_reaction_lowwind_metrics.csv'
TRAIN_LOG_OUT = DATA_BASE / 'no2_pinn_reaction_lowwind_training_log.csv'

if not DATA_BASE.exists():
    raise FileNotFoundError(f"NO2 data folder not found: {DRIVE_BASE}")


REGIME_SIGNAL = 1              # phi_AERMOD > threshold: trusted supervised AERMOD label
REGIME_PHYSICAL_ZERO = 2       # phi_AERMOD == 0 and wind is not low: weak near-zero constraint
REGIME_STRUCTURAL_ZERO = 3     # phi_AERMOD == 0 and low wind/calm: no AERMOD label loss
PHI_ZERO_THRESHOLD = 1e-12
LOW_WIND_THRESHOLD = 1.0        # m/s
PHYSICAL_ZERO_UPPER_BOUND = 0.1 # ug/m3; NO2 fields are larger than benzene, revisit if p2 preds saturate here
PHYSICAL_ZERO_LOSS_WEIGHT = 0.3

SIGNAL_HIGH_VALUE_WEIGHT_POWER = 0.75
SIGNAL_UNDERPRED_PENALTY = 3.0

AUTO_HIGH_THRESHOLD_FROM_DATA = True
HIGH_LABEL_PERCENTILE     = 95.0
HIGH_LABEL_THRESHOLD      = 5.0   # ug/m3, overwritten when auto-thresholding
HIGH_CAPTURE_TARGET       = 5.0   # ug/m3, pull high rows up toward this
HIGH_CAPTURE_METRIC_FLOOR = 2.5   # ug/m3, capture bar used by the benchmark metric
HIGH_CAPTURE_WEIGHT       = 0.5

STRUCTURAL_PHYSICS_WEIGHT = 1.0
STRUCTURAL_PHYSICS_N_POINTS = 256
STRUCTURAL_PHYSICS_GLOBAL = True
PHYSICS_LENGTH_SCALE = 2000.0
PHYSICS_CONC_FLOOR = 1e-4

PHYS_ADAPTIVE_BALANCE = True
SIGNAL_PHYS_TARGET_RATIO = 0.25
STRUCTURAL_PHYS_TARGET_RATIO = 0.5
PHYS_EFF_WEIGHT_MAX = 1.0

# =============================================================================
# CHEMISTRY CONFIG
# =============================================================================


CHEM_AS_NETWORK_INPUT = False
CHEM_KEYS = ['T_K', 'C_NOx', 'C_O3', 'J_NO2']

REACTION_ENABLED = True
REACTION_WEIGHT  = 1.0     # damp the reaction term if chemistry swamps transport
REACTION_CLAMP_NO = True   # phi > C_NOx is unphysical, so floor the NO reservoir at 0


# k_eff with units (ug/m3)^-1 s^-1:
#   'direct'    : use the column value as k_eff (literal reading of the formula
#                 k(C_NOx - phi)C_O3 with the column already being the rate)
#   'arrhenius' : treat the column as temperature in Kelvin and evaluate
#                 k(T) = A exp(-Ea/R / T) for NO + O3 -> NO2 + O2, then convert
#                 from cm3 molecule^-1 s^-1 to (ug/m3)^-1 s^-1
#   'auto'      : arrhenius if the column looks like ambient Kelvin, else direct
REACTION_RATE_MODE = 'auto'
ARRHENIUS_A        = 1.4e-12   # cm3 molecule^-1 s^-1
ARRHENIUS_EA_OVER_R = 1310.0   # K
# rate_ug = k_cm3 * N_A/1e12 * M_NO2/(M_NO * M_O3) * C_NO * C_O3, masses in g/mol
M_NO, M_O3, M_NO2 = 30.006, 47.997, 46.006
CM3_TO_UGM3_RATE = (6.02214076e23 / 1e12) * M_NO2 / (M_NO * M_O3)

if COORD_FRAME_PATH.exists():
    with open(COORD_FRAME_PATH) as f:
        COORD_FRAME = json.load(f)
    DOMAIN_X_MIN = COORD_FRAME['x_local_min']
    DOMAIN_X_MAX = COORD_FRAME['x_local_max']
    DOMAIN_Y_MIN = COORD_FRAME['y_local_min']
    DOMAIN_Y_MAX = COORD_FRAME['y_local_max']
    print(f"\nDomain: x=[{DOMAIN_X_MIN/1000:.1f}, {DOMAIN_X_MAX/1000:.1f}] km  "
          f"y=[{DOMAIN_Y_MIN/1000:.1f}, {DOMAIN_Y_MAX/1000:.1f}] km")
else:
    COORD_FRAME = None
    DOMAIN_X_MIN = DOMAIN_X_MAX = DOMAIN_Y_MIN = DOMAIN_Y_MAX = None
    print(f"\ncoordinate_frame.json not found at {COORD_FRAME_PATH}")
    print("Domain bounds will be derived from the loaded collocation data.")


def resolve_scenario_path(base):
    """
    Find the folder that holds the scenario_* collocation folders.

    Accepts scenario_* directly under base, a scenarios*/ container, or any
    ancestor of collocation_points_shard_*.npz found recursively.
    """
    base = Path(base)
    if any(p.is_dir() and p.name.startswith('scenario_') for p in base.iterdir()):
        return base

    containers = sorted(p for p in base.glob('scenarios*') if p.is_dir())
    for c in containers:
        if any(p.is_dir() and p.name.startswith('scenario_') for p in c.iterdir()):
            return c

    shards = sorted(base.rglob('collocation_points_shard_*.npz'))
    if not shards:
        shards = sorted(base.rglob('collocation_points.npz'))
    if shards:
        parents = {s.parent.parent for s in shards}
        return parents.pop() if len(parents) == 1 else base

    raise FileNotFoundError(f"No collocation shards found anywhere under {base}")


SCENARIO_PATH = resolve_scenario_path(DRIVE_BASE)
print(f"Drive base:    {DRIVE_BASE}")
print(f"Scenario path: {SCENARIO_PATH}")

# =============================================================================
# ARCHITECTURE
# =============================================================================

class ResidualBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm1   = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2   = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.skip    = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.norm1(x))
        h = self.linear1(h)
        h = torch.nn.functional.gelu(self.norm2(h))
        h = self.linear2(h)
        return self.skip(x) + h


class ParametricADRPINN(nn.Module):
    """
    Parametric 3D advection-diffusion-reaction PINN surrogate for AERMOD NO2 labels.

    Transport/source inputs (always used):
        x, y, z : receptor position in local frame (m), including height z
        t       : fixed quasi-steady hour coordinate, usually 3600 s
        cx, cy  : source horizontal position in local frame (m)
        h0      : source release height (m)
        u, v    : effective horizontal wind components (m/s)
        d       : effective plume spread diameter (m)
        kappa   : turbulent diffusivity (m2/s)
        Q       : emission rate (g/s)

    Chemistry inputs (T_K, C_NOx, C_O3, J_NO2) are frozen out of the network by
    default and only enter the PDE residual. use_chem_inputs=True enables the
    ablation where they are concatenated as four extra inputs.

    Output:
        phi     : AERMOD-labeled NO2 concentration (ug/m3)
    """

    N_TRANSPORT_INPUTS = 12
    N_CHEM_INPUTS = 4

    def __init__(self, num_fourier_features=0, hidden_dim=256,
                 num_layers=8, fourier_scale=0.0, use_chem_inputs=False):
        super().__init__()
        self.use_chem_inputs = bool(use_chem_inputs)
        self.n_inputs = self.N_TRANSPORT_INPUTS + (self.N_CHEM_INPUTS if self.use_chem_inputs else 0)
        self.use_fourier = num_fourier_features > 0

        if self.use_fourier:
            self.register_buffer('B', torch.randn(self.n_inputs, num_fourier_features) * fourier_scale)
            input_dim = self.n_inputs + 2 * num_fourier_features
        else:
            self.register_buffer('B', torch.zeros(1))
            input_dim = self.n_inputs

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        self._init_weights()

        self.register_buffer('x_min',     torch.tensor(-43000.0))
        self.register_buffer('x_max',     torch.tensor( 43000.0))
        self.register_buffer('y_min',     torch.tensor(-49000.0))
        self.register_buffer('y_max',     torch.tensor( 49000.0))
        self.register_buffer('z_min',     torch.tensor(    0.0))
        self.register_buffer('z_max',     torch.tensor(  500.0))
        self.register_buffer('t_min',     torch.tensor(  3600.0))
        self.register_buffer('t_max',     torch.tensor(  3600.0))
        self.register_buffer('cx_min',    torch.tensor(-43000.0))
        self.register_buffer('cx_max',    torch.tensor( 43000.0))
        self.register_buffer('cy_min',    torch.tensor(-49000.0))
        self.register_buffer('cy_max',    torch.tensor( 49000.0))
        self.register_buffer('h0_min',    torch.tensor(    0.0))
        self.register_buffer('h0_max',    torch.tensor(  200.0))
        self.register_buffer('u_min',     torch.tensor(  -15.0))
        self.register_buffer('u_max',     torch.tensor(   15.0))
        self.register_buffer('v_min',     torch.tensor(  -15.0))
        self.register_buffer('v_max',     torch.tensor(   15.0))
        self.register_buffer('d_min',     torch.tensor( 2000.0))
        self.register_buffer('d_max',     torch.tensor( 2000.0))
        self.register_buffer('kappa_min', torch.tensor(    0.1))
        self.register_buffer('kappa_max', torch.tensor(  500.0))
        self.register_buffer('Q_min',     torch.tensor(   1e-6))
        self.register_buffer('Q_max',     torch.tensor(    1.0))
        # Registered even when frozen so checkpoints stay comparable across the ablation.
        self.register_buffer('T_K_min',   torch.tensor(  250.0))
        self.register_buffer('T_K_max',   torch.tensor(  320.0))
        self.register_buffer('C_NOx_min', torch.tensor(    0.0))
        self.register_buffer('C_NOx_max', torch.tensor(  200.0))
        self.register_buffer('C_O3_min',  torch.tensor(    0.0))
        self.register_buffer('C_O3_max',  torch.tensor(  200.0))
        self.register_buffer('J_NO2_min', torch.tensor(    0.0))
        self.register_buffer('J_NO2_max', torch.tensor(   0.02))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization_from_data(self, data_dict):
        mapping = [
            ('x_min', 'x_max', 'x'),
            ('y_min', 'y_max', 'y'),
            ('z_min', 'z_max', 'z'),
            ('t_min', 't_max', 't'),
            ('cx_min', 'cx_max', 'cx'),
            ('cy_min', 'cy_max', 'cy'),
            ('h0_min', 'h0_max', 'h0'),
            ('u_min', 'u_max', 'u'),
            ('v_min', 'v_max', 'v'),
            ('d_min', 'd_max', 'd'),
            ('kappa_min', 'kappa_max', 'kappa'),
            ('Q_min', 'Q_max', 'Q'),
            ('T_K_min', 'T_K_max', 'T_K'),
            ('C_NOx_min', 'C_NOx_max', 'C_NOx'),
            ('C_O3_min', 'C_O3_max', 'C_O3'),
            ('J_NO2_min', 'J_NO2_max', 'J_NO2'),
        ]
        for min_name, max_name, key in mapping:
            if key in data_dict:
                getattr(self, min_name).fill_(float(data_dict[key].min()))
                getattr(self, max_name).fill_(float(data_dict[key].max()))

    def normalize_input(self, x, y, z, t, cx, cy, h0, u, v, d, kappa, Q, chem=None):
        eps = 1e-8
        def n(val, lo, hi):
            return 2.0 * (val - lo) / (hi - lo + eps) - 1.0
        parts = (
            n(x, self.x_min, self.x_max),
            n(y, self.y_min, self.y_max),
            n(z, self.z_min, self.z_max),
            n(t, self.t_min, self.t_max),
            n(cx, self.cx_min, self.cx_max),
            n(cy, self.cy_min, self.cy_max),
            n(h0, self.h0_min, self.h0_max),
            n(u, self.u_min, self.u_max),
            n(v, self.v_min, self.v_max),
            n(d, self.d_min, self.d_max),
            n(kappa, self.kappa_min, self.kappa_max),
            n(Q, self.Q_min, self.Q_max),
        )
        if self.use_chem_inputs:
            T_K, C_NOx, C_O3, J_NO2 = chem
            parts = parts + (
                n(T_K, self.T_K_min, self.T_K_max),
                n(C_NOx, self.C_NOx_min, self.C_NOx_max),
                n(C_O3, self.C_O3_min, self.C_O3_max),
                n(J_NO2, self.J_NO2_min, self.J_NO2_max),
            )
        return parts

    def forward(self, x, y, z, t, cx, cy, h0, u, v, d, kappa, Q,
                chem=None, normalize=True):
        if self.use_chem_inputs and chem is None:
            raise ValueError("Model was built with use_chem_inputs=True but chem=None was passed.")

        if normalize:
            parts = self.normalize_input(x, y, z, t, cx, cy, h0, u, v, d, kappa, Q, chem)
        else:
            parts = (x, y, z, t, cx, cy, h0, u, v, d, kappa, Q)
            if self.use_chem_inputs:
                parts = parts + tuple(chem)

        inputs = torch.cat(parts, dim=1)

        if self.use_fourier:
            proj   = 2.0 * np.pi * torch.matmul(inputs, self.B)
            inputs = torch.cat([inputs, torch.sin(proj), torch.cos(proj)], dim=-1)

        h = torch.nn.functional.gelu(self.input_layer(inputs))
        for block in self.residual_blocks:
            h = block(h)

        return self.output_head(h)

# =============================================================================
# DATA LOADING — AERMOD NO2 ONLY
# =============================================================================

COLS = ['t', 'x', 'y', 'z', 'cx', 'cy', 'h0', 'd', 'Q', 'u', 'v', 'kappa',
        'T_K', 'C_NOx', 'C_O3', 'J_NO2', 'r', 'phi']


def _load_npz_array(npz_file):
    """Load the data matrix from a collocation shard, tolerant of the npz key name."""
    with np.load(npz_file) as npz:
        if 'data' in npz.files:
            arr = npz['data']
        else:
            arr = npz[npz.files[0]]
    return arr.astype(np.float32)


def resolve_rate_coefficient(T_col):
    """
    Build k_eff in (ug/m3)^-1 s^-1 from the T_K column, per REACTION_RATE_MODE.

    Returns (k_eff array, mode string actually used).
    """
    mode = REACTION_RATE_MODE
    med = float(np.median(T_col))
    if mode == 'auto':
        looks_like_kelvin = 200.0 <= med <= 350.0
        mode = 'arrhenius' if looks_like_kelvin else 'direct'
        print(f"  Rate mode 'auto' -> '{mode}' (median T_K column = {med:.4g})")

    if mode == 'direct':
        k_eff = T_col.astype(np.float32)
    elif mode == 'arrhenius':
        T = np.clip(T_col.astype(np.float64), 180.0, 350.0)
        k_cm3 = ARRHENIUS_A * np.exp(-ARRHENIUS_EA_OVER_R / T)
        k_eff = (k_cm3 * CM3_TO_UGM3_RATE).astype(np.float32)
    else:
        raise ValueError(f"Unknown REACTION_RATE_MODE: {REACTION_RATE_MODE}")

    print(f"  k_eff [(ug/m3)^-1 s^-1]: min={k_eff.min():.4e}  "
          f"median={np.median(k_eff):.4e}  max={k_eff.max():.4e}")
    return k_eff, mode


def load_no2_data(scenario_path):
    """
    Load AERMOD-only NO2 scenario npz files (3D, sharded).

    Column order:
      [t, x, y, z, cx, cy, h0, d, Q, u, v, kappa, T_K, C_NOx, C_O3, J_NO2, r, phi]
    'r' is loaded for reference only; it is not a network input and not used in
    the PDE in this version.
    """
    coll_list = []
    n_scenarios = 0
    n_shards = 0

    for folder in sorted(Path(scenario_path).iterdir()):
        if not (folder.is_dir() and folder.name.startswith('scenario_')):
            continue
        shard_files = sorted(folder.glob('collocation_points_shard_*.npz'))
        if not shard_files:
            legacy = folder / 'collocation_points.npz'
            if legacy.exists():
                shard_files = [legacy]
        if not shard_files:
            continue
        for shard in shard_files:
            coll_list.append(_load_npz_array(shard))
            n_shards += 1
        n_scenarios += 1

    # Fallback: shards stored flat under the scenario path, or nested under
    # folders that do not follow the scenario_* naming convention.
    if n_scenarios == 0:
        flat = sorted(Path(scenario_path).glob('collocation_points_shard_*.npz'))
        if not flat:
            flat = sorted(Path(scenario_path).rglob('collocation_points_shard_*.npz'))
        if not flat:
            flat = sorted(Path(scenario_path).rglob('collocation_points.npz'))
        for shard in flat:
            coll_list.append(_load_npz_array(shard))
            n_shards += 1
        n_scenarios = len({s.parent for s in flat})

    assert n_scenarios > 0, f"No collocation shards found at {scenario_path}"
    combined = np.vstack(coll_list)

    if combined.shape[1] != len(COLS):
        raise ValueError(
            f"Expected {len(COLS)} columns {COLS}, but collocation data has "
            f"{combined.shape[1]} columns. Check the dataset export/column order."
        )

    good_mask = np.all(np.isfinite(combined), axis=1)
    n_removed = int((~good_mask).sum())
    if n_removed:
        print(f"  Removed {n_removed:,} NaN/Inf rows")
    combined = combined[good_mask]

    data = {col: combined[:, i] for i, col in enumerate(COLS)}

    # Keep the d override used by the current source-term formulation.
    data['d'] = np.full_like(data['d'], 2000.0, dtype=np.float32)
    assert np.isclose(data['d'].min(), 2000.0)
    assert np.isclose(data['d'].max(), 2000.0)
    print(f"  d override OK: {data['d'].min():.1f} {data['d'].max():.1f}")

    n_neg = int((data['phi'] < 0).sum())
    if n_neg:
        raise ValueError(f"Found {n_neg:,} negative AERMOD phi values. Check dataset units/export.")

    for key in ['C_NOx', 'C_O3', 'J_NO2']:
        n_bad = int((data[key] < 0).sum())
        if n_bad:
            print(f"  WARNING: {n_bad:,} negative {key} values clipped to 0")
            data[key] = np.maximum(data[key], 0.0)

    print("\n  Chemistry setup:")
    data['k_eff'], rate_mode = resolve_rate_coefficient(data['T_K'])
    data['rate_mode'] = rate_mode

    n_exceed = int((data['phi'] > data['C_NOx']).sum())
    if n_exceed:
        pct = 100.0 * n_exceed / len(data['phi'])
        tail = ("the NO reservoir is floored at 0 in the reaction term"
                if REACTION_CLAMP_NO else
                "these rows give negative NO2 production")
        print(f"  Note: {n_exceed:,} rows ({pct:.2f}%) have phi > C_NOx; {tail}.")

    wind_speed = np.sqrt(data['u']**2 + data['v']**2)
    print("\n  Wind speed percentiles (m/s):")
    for q in [0, 1, 5, 10, 25, 50]:
        print(f"    p{q:<2}: {np.percentile(wind_speed, q):.4f}")

    regime = np.full(len(data['phi']), REGIME_SIGNAL, dtype=np.int64)
    is_zero = data['phi'] <= PHI_ZERO_THRESHOLD
    is_low_wind = wind_speed < LOW_WIND_THRESHOLD
    regime[is_zero & (~is_low_wind)] = REGIME_PHYSICAL_ZERO
    regime[is_zero & is_low_wind] = REGIME_STRUCTURAL_ZERO
    data['regime'] = regime
    data['wind_speed'] = wind_speed.astype(np.float32)

    print(f"\n  Loaded {n_scenarios} scenarios across {n_shards} collocation shards")
    print(f"  AERMOD rows: {len(combined):,}")
    print(f"  No internal row filtering applied")
    print(f"  No phi clipping applied")
    print(f"  Regime 1 signal rows:          {(regime == REGIME_SIGNAL).sum():,}")
    print(f"  Regime 2 physical-zero rows:   {(regime == REGIME_PHYSICAL_ZERO).sum():,}")
    print(f"  Regime 3 structural-zero rows: {(regime == REGIME_STRUCTURAL_ZERO).sum():,}")

    print("\n  Data ranges:")
    for col, arr in data.items():
        if isinstance(arr, np.ndarray):
            print(f"    {col:8s}: min={arr.min():.4e}  max={arr.max():.4e}  mean={arr.mean():.4e}")

    return data


def configure_high_value_thresholds(phi):
    """
    Set the high-tail thresholds from the NO2 label distribution.

    The benzene script hardcoded 5 / 2.5 ug/m3. NO2 lives on a different scale,
    so the training hinge and the capture metric are anchored to a percentile of
    the positive labels instead.
    """
    global HIGH_LABEL_THRESHOLD, HIGH_CAPTURE_TARGET, HIGH_CAPTURE_METRIC_FLOOR
    pos = phi[phi > PHI_ZERO_THRESHOLD]
    if not AUTO_HIGH_THRESHOLD_FROM_DATA or pos.size < 100:
        print(f"  High-tail thresholds (fixed): high>{HIGH_LABEL_THRESHOLD:.3f}  "
              f"capture>{HIGH_CAPTURE_METRIC_FLOOR:.3f} ug/m3")
        return
    thr = float(np.percentile(pos, HIGH_LABEL_PERCENTILE))
    HIGH_LABEL_THRESHOLD = thr
    HIGH_CAPTURE_TARGET = thr
    HIGH_CAPTURE_METRIC_FLOOR = 0.5 * thr
    print(f"  High-tail thresholds from data p{HIGH_LABEL_PERCENTILE:.0f}: "
          f"high>{HIGH_LABEL_THRESHOLD:.3f}  capture>{HIGH_CAPTURE_METRIC_FLOOR:.3f} ug/m3")

# =============================================================================
# PHYSICS RESIDUAL — ADVECTION-DIFFUSION-REACTION
# =============================================================================

def reaction_term(phi, k_eff, C_NOx, C_O3, J_NO2):
    """
    Net NO2 chemical production (ug/m3/s):

        R = k_eff * [NO] * [O3] - J_NO2 * [NO2],   [NO] = C_NOx - phi

    First term: NO titration of ozone producing NO2.
    Second term: NO2 photolysis destroying NO2.
    R > 0 means chemistry is a net NO2 source at that point.

    This assumes phi and C_NOx share a concentration basis (both ug/m3 of NO2
    mass), i.e. phi is the NO2 partition of the NOx pool rather than an isolated
    facility increment. If the dataset's phi is an increment on top of a
    background, C_NOx must be the matching increment or the term is mis-scaled.
    """
    no_conc = C_NOx - phi
    if REACTION_CLAMP_NO:
        no_conc = no_conc.clamp(min=0.0)
    production = k_eff * no_conc * C_O3
    photolysis = J_NO2 * phi
    return REACTION_WEIGHT * (production - photolysis)


def compute_pde_residual_reactive(model, x, y, z, t, cx, cy, h0, u, v, d, kappa, Q,
                                  k_eff, C_NOx, C_O3, J_NO2, T_K, dev,
                                  return_terms=False):
    """
    Steady 3D advection-diffusion-reaction residual for the PINN regularizer.

    t is fixed at one quasi-steady hour so phi_t is zero, and the dataset carries
    no vertical wind so w*phi_z is omitted. Diffusion acts in all three
    dimensions. The reaction term is the NO2-specific addition relative to the
    benzene formulation.

    Chemistry tensors are detached: they drive the residual but the network never
    sees them as inputs unless the CHEM_AS_NETWORK_INPUT ablation is on.
    """
    x = x.detach().to(dev).requires_grad_(True)
    y = y.detach().to(dev).requires_grad_(True)
    z = z.detach().to(dev).requires_grad_(True)
    t = t.detach().to(dev).requires_grad_(True)

    cx    = cx.detach().to(dev)
    cy    = cy.detach().to(dev)
    h0    = h0.detach().to(dev)
    d     = d.detach().to(dev)
    kappa = kappa.detach().to(dev)
    Q     = Q.detach().to(dev)

    k_eff = k_eff.detach().to(dev)
    C_NOx = C_NOx.detach().to(dev)
    C_O3  = C_O3.detach().to(dev)
    J_NO2 = J_NO2.detach().to(dev)
    T_K   = T_K.detach().to(dev)

    u_raw = u.detach().to(dev)
    v_raw = v.detach().to(dev)
    spd_raw = torch.sqrt(u_raw**2 + v_raw**2).clamp(min=1e-6)
    spd_eff = spd_raw.clamp(min=0.5)
    scale   = spd_eff / spd_raw
    u_phys  = u_raw * scale
    v_phys  = v_raw * scale

    chem_in = (T_K, C_NOx, C_O3, J_NO2) if model.use_chem_inputs else None
    phi  = model(x, y, z, t, cx, cy, h0, u_phys, v_phys, d, kappa, Q,
                 chem=chem_in, normalize=True)
    ones = torch.ones_like(phi)

    phi_t  = torch.zeros_like(phi)
    phi_x  = torch.autograd.grad(phi,   x, ones, create_graph=True, retain_graph=True)[0]
    phi_y  = torch.autograd.grad(phi,   y, ones, create_graph=True, retain_graph=True)[0]
    phi_z  = torch.autograd.grad(phi,   z, ones, create_graph=True, retain_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, x, ones, create_graph=True, retain_graph=True)[0]
    phi_yy = torch.autograd.grad(phi_y, y, ones, create_graph=True, retain_graph=True)[0]
    phi_zz = torch.autograd.grad(phi_z, z, ones, create_graph=True)[0]

    r_src   = d / 2.0
    r_sq    = r_src ** 2
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - h0)**2
    # 3D isotropic Gaussian source normalized over the volume.
    source  = (Q / (np.pi**1.5 * r_src**3)) * torch.exp(-dist_sq / r_sq)

    advection = u_phys*phi_x + v_phys*phi_y
    diffusion = kappa*(phi_xx + phi_yy + phi_zz)
    reaction  = reaction_term(phi, k_eff, C_NOx, C_O3, J_NO2) if REACTION_ENABLED \
        else torch.zeros_like(phi)

    residual = phi_t + advection - diffusion - source - reaction

    # Dimensionless residual scaling: compare the residual against characteristic
    # advection, diffusion, source and reaction magnitudes so the physics loss is
    # unit-aware and cannot be made decorative by a small absolute residual.
    speed = torch.sqrt(u_phys**2 + v_phys**2).clamp(min=0.5)
    c_ref = phi.abs().clamp(min=PHYSICS_CONC_FLOOR)
    L = PHYSICS_LENGTH_SCALE
    scale_n = (speed * c_ref / L
               + kappa * c_ref / (L * L)
               + source.abs()
               + reaction.abs()).clamp(min=1e-12)
    res_scaled = (residual / scale_n).clamp(-10, 10)

    if return_terms:
        terms = {
            'phi': phi.detach(),
            'advection': advection.detach(),
            'diffusion': diffusion.detach(),
            'source': source.detach(),
            'reaction': reaction.detach(),
            'residual_raw': residual.detach(),
            'scale': scale_n.detach(),
        }
        return res_scaled, terms
    return res_scaled

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_regime_aware_data_loss(phi_pred, phi_true, regime):
    """
    Regime-aware data loss.

    Regime 1: positive AERMOD signal. Full log-space supervised AERMOD loss.
    Regime 2: zero with non-low wind. Weak near-zero upper-bound penalty.
    Regime 3: zero with low wind/calm. No AERMOD data loss; PDE handles it.

    Chemistry never appears here. That is the "frozen chemistry" half of the
    design: supervision stays a 12-input transport problem, and chemical
    information reaches the weights only through the reactive residual.
    """
    regime = regime.flatten().long()
    loss = torch.tensor(0.0, device=phi_pred.device)
    parts = {
        'signal_log_loss': 0.0,
        'high_capture_loss': 0.0,
        'physical_zero_loss': 0.0,
        'n_signal': int((regime == REGIME_SIGNAL).sum().item()),
        'n_physical_zero': int((regime == REGIME_PHYSICAL_ZERO).sum().item()),
        'n_structural_zero': int((regime == REGIME_STRUCTURAL_ZERO).sum().item()),
    }

    mask_signal = regime == REGIME_SIGNAL
    if mask_signal.any():
        pred1 = phi_pred[mask_signal]
        true1 = phi_true[mask_signal]
        log_pred = torch.log10(pred1.clamp(min=1e-15))
        log_true = torch.log10(true1.clamp(min=1e-15))
        log_diff = (log_pred - log_true).clamp(-5, 5)

        asym_w = torch.where(
            log_diff < 0,
            torch.full_like(log_diff, SIGNAL_UNDERPRED_PENALTY),
            torch.ones_like(log_diff),
        )
        sq_err = asym_w * log_diff**2

        if SIGNAL_HIGH_VALUE_WEIGHT_POWER > 0:
            peak_w = 1.0 + true1.clamp(min=1e-15) ** SIGNAL_HIGH_VALUE_WEIGHT_POWER
            signal_loss = (peak_w * sq_err).sum() / (peak_w.sum() + 1e-12)
        else:
            signal_loss = torch.mean(sq_err)
        loss = loss + signal_loss * (mask_signal.float().mean())
        parts['signal_log_loss'] = float(signal_loss.detach().cpu().item())

        high_mask = true1 > HIGH_LABEL_THRESHOLD
        if high_mask.any() and HIGH_CAPTURE_WEIGHT > 0:
            log_target = float(np.log10(max(HIGH_CAPTURE_TARGET, 1e-15)))
            deficit = torch.relu(log_target - log_pred[high_mask])
            high_capture_loss = torch.mean(deficit**2)
            loss = loss + HIGH_CAPTURE_WEIGHT * high_capture_loss
            parts['high_capture_loss'] = float(high_capture_loss.detach().cpu().item())

    mask_phys_zero = regime == REGIME_PHYSICAL_ZERO
    if mask_phys_zero.any():
        pred2 = phi_pred[mask_phys_zero]
        zero_loss = torch.mean(torch.relu(pred2 - PHYSICAL_ZERO_UPPER_BOUND)**2)
        loss = loss + PHYSICAL_ZERO_LOSS_WEIGHT * zero_loss * (mask_phys_zero.float().mean())
        parts['physical_zero_loss'] = float(zero_loss.detach().cpu().item())

    return loss, parts

# =============================================================================
# BENCHMARKING METRICS — PINN VS AERMOD NO2 LABELS
# =============================================================================

def compute_benchmark_metrics(pred_np, true_np, label):
    """Agreement metrics for PINN predictions against held-out AERMOD NO2 labels."""
    pred_s = np.maximum(pred_np, 1e-15)
    true_s = np.maximum(true_np, 1e-15)
    ratio  = pred_s / true_s

    mean_true = true_s.mean()
    mean_pred = pred_s.mean()

    FB    = 2.0 * (mean_true - mean_pred) / (mean_true + mean_pred + 1e-10)
    NMSE  = np.mean((pred_s - true_s)**2) / (mean_true * mean_pred + 1e-10)
    FAC2  = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100.0
    MG    = np.exp(np.mean(np.log(ratio + 1e-15)))
    VG    = np.exp(np.var(np.log(ratio + 1e-15)))
    RMSE  = np.sqrt(np.mean((pred_s - true_s)**2))
    NMB   = (pred_s.sum() - true_s.sum()) / (true_s.sum() + 1e-10)
    NME   = np.abs(pred_s - true_s).sum() / (true_s.sum() + 1e-10)
    r     = float(np.corrcoef(pred_s, true_s)[0, 1]) if len(pred_s) > 1 else float('nan')
    rho, _= spearmanr(pred_s, true_s) if len(pred_s) > 1 else (float('nan'), None)

    log_p    = np.log10(pred_s)
    log_t    = np.log10(true_s)
    log_rmse = np.sqrt(np.mean((log_p - log_t)**2))
    ss_res   = np.sum((log_p - log_t)**2)
    ss_tot   = np.sum((log_t - log_t.mean())**2)
    r2_log   = 1.0 - ss_res / (ss_tot + 1e-10)

    high_true = true_s > HIGH_LABEL_THRESHOLD
    high_capture = float((pred_s[high_true] > HIGH_CAPTURE_METRIC_FLOOR).mean()) \
        if high_true.sum() > 0 else 0.0

    within_2x  = float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) * 100
    within_5x  = float(np.mean((ratio >= 0.2) & (ratio <= 5.0))) * 100
    within_10x = float(np.mean((ratio >= 0.1) & (ratio <= 10.0))) * 100

    metrics = {
        'label':        label,
        'n':            len(pred_s),
        'mean_aermod':  mean_true,
        'mean_pred':    mean_pred,
        'FB':           FB,
        'NMSE':         NMSE,
        'FAC2':         FAC2,
        'MG':           MG,
        'VG':           VG,
        'RMSE':         RMSE,
        'NMB':          NMB,
        'NME':          NME,
        'r':            r,
        'rho':          float(rho),
        'log_rmse':     log_rmse,
        'r2_log':       r2_log,
        'high_capture': high_capture,
        'high_threshold': HIGH_LABEL_THRESHOLD,
        'within_2x':    within_2x,
        'within_5x':    within_5x,
        'within_10x':   within_10x,
    }

    print(f"\n  -- {label} (n={len(pred_s):,}) --")
    print(f"    mean_aermod={mean_true:.4f}  mean_pred={mean_pred:.4f}")
    print(f"    FB={FB:.3f}  NMSE={NMSE:.2f}  FAC2={FAC2:.1f}%")
    print(f"    MG={MG:.3f}  VG={VG:.3f}")
    print(f"    r={r:.3f}  rho={float(rho):.3f}  r2_log={r2_log:.3f}")
    print(f"    RMSE={RMSE:.4f}  NMB={NMB:.3f}  NME={NME:.3f}")
    print(f"    Log RMSE={log_rmse:.4f}")
    print(f"    High-label capture={high_capture:.3f}  "
          f"(AERMOD>{HIGH_LABEL_THRESHOLD:.2f} & PINN>{HIGH_CAPTURE_METRIC_FLOOR:.2f} ug/m3)")
    print(f"    Within 2x={within_2x:.1f}%  5x={within_5x:.1f}%  10x={within_10x:.1f}%")

    return metrics


def predict_batched(model, tensors, idx, chunk=200000):
    """Model prediction over an index set, supplying chem only if the model uses it."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            sl = idx[s:s + chunk]
            chem_in = (tensors['T_K'][sl], tensors['C_NOx'][sl],
                       tensors['C_O3'][sl], tensors['J_NO2'][sl]) \
                if model.use_chem_inputs else None
            outs.append(model(
                tensors['x'][sl], tensors['y'][sl], tensors['z'][sl], tensors['t'][sl],
                tensors['cx'][sl], tensors['cy'][sl], tensors['h0'][sl],
                tensors['u'][sl], tensors['v'][sl], tensors['d'][sl],
                tensors['kappa'][sl], tensors['Q'][sl],
                chem=chem_in, normalize=True,
            ))
    return torch.cat(outs)


def evaluate_full_benchmark(model, tensors, set_name, max_samples=100000):
    model.eval()
    N        = tensors['x'].shape[0]
    n_eval   = min(max_samples, N)
    eval_idx = torch.randperm(N, device=device)[:n_eval]

    pred_t = predict_batched(model, tensors, eval_idx)
    true_t = tensors['phi'][eval_idx]
    regime_t = tensors['regime'][eval_idx]

    pred_np = pred_t.cpu().numpy().flatten()
    true_np = true_t.cpu().numpy().flatten()
    regime_np = regime_t.cpu().numpy().flatten().astype(int)

    nonzero_mask = true_np > PHI_ZERO_THRESHOLD
    zero_mask    = true_np <= PHI_ZERO_THRESHOLD
    signal_mask = regime_np == REGIME_SIGNAL
    physical_zero_mask = regime_np == REGIME_PHYSICAL_ZERO
    structural_zero_mask = regime_np == REGIME_STRUCTURAL_ZERO
    high_mask    = true_np > HIGH_LABEL_THRESHOLD

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {set_name}")
    print(f"{'='*60}")

    rows = []
    rows.append(compute_benchmark_metrics(pred_np, true_np, f"{set_name} — All AERMOD rows"))

    if signal_mask.sum() > 100:
        rows.append(compute_benchmark_metrics(
            pred_np[signal_mask], true_np[signal_mask],
            f"{set_name} — Regime 1 signal rows"))

    if nonzero_mask.sum() > 100 and not np.array_equal(signal_mask, nonzero_mask):
        rows.append(compute_benchmark_metrics(
            pred_np[nonzero_mask], true_np[nonzero_mask],
            f"{set_name} — Nonzero AERMOD rows"))

    if high_mask.sum() > 20:
        rows.append(compute_benchmark_metrics(
            pred_np[high_mask], true_np[high_mask],
            f"{set_name} — High AERMOD labels (>{HIGH_LABEL_THRESHOLD:.2f})"))

    if physical_zero_mask.sum() > 0:
        print(f"\n  Regime 2 physical-zero rows: {physical_zero_mask.sum():,}")
        print(f"    PINN mean:               {pred_np[physical_zero_mask].mean():.4e}")
        print(f"    PINN > 0.1 ug/m3:        {(pred_np[physical_zero_mask] > 0.1).mean()*100:.1f}%")
        print(f"    PINN > 0.5 ug/m3:        {(pred_np[physical_zero_mask] > 0.5).mean()*100:.1f}%")

    if structural_zero_mask.sum() > 0:
        print(f"\n  Regime 3 structural-zero rows: {structural_zero_mask.sum():,}")
        print(f"    PINN mean:               {pred_np[structural_zero_mask].mean():.4e}")
        print(f"    PINN median:             {np.median(pred_np[structural_zero_mask]):.4e}")
        print(f"    PINN > 0.1 ug/m3:        {(pred_np[structural_zero_mask] > 0.1).mean()*100:.1f}%")
        print(f"    PINN > 0.5 ug/m3:        {(pred_np[structural_zero_mask] > 0.5).mean()*100:.1f}%")
        print("    Note: these rows are not scored against AERMOD zeros because the")
        print("    training objective intentionally masks their AERMOD data label.")

    if zero_mask.sum() > 0:
        print(f"\n  All zero-label AERMOD rows: {zero_mask.sum():,}")
        print(f"    PINN mean on zero-label rows: {pred_np[zero_mask].mean():.4e}")
        print(f"    PINN > 0.1 ug/m3 on zero-label rows: {(pred_np[zero_mask] > 0.1).mean()*100:.1f}%")

    return rows


def diagnose_physics_terms(model, tensors, index_pool, label, n=4096):
    """
    Print the magnitude of each PDE term, plus a photostationary-state check.

    This is the main auditing tool for the NO2 formulation: if the reaction term
    dwarfs advection/diffusion by orders of magnitude, the normalized residual is
    effectively enforcing chemical equilibrium and REACTION_WEIGHT (or
    REACTION_RATE_MODE) is the knob to revisit.
    """
    if len(index_pool) == 0:
        print(f"\n  Physics diagnostics ({label}): empty index pool, skipped")
        return
    model.eval()
    n = min(n, len(index_pool))
    idx = index_pool[torch.randperm(len(index_pool), device=device)[:n]]

    res, terms = compute_pde_residual_reactive(
        model,
        tensors['x'][idx], tensors['y'][idx], tensors['z'][idx], tensors['t'][idx],
        tensors['cx'][idx], tensors['cy'][idx], tensors['h0'][idx],
        tensors['u'][idx], tensors['v'][idx], tensors['d'][idx],
        tensors['kappa'][idx], tensors['Q'][idx],
        tensors['k_eff'][idx], tensors['C_NOx'][idx], tensors['C_O3'][idx],
        tensors['J_NO2'][idx], tensors['T_K'][idx],
        device, return_terms=True,
    )

    def m(t_):
        return float(t_.abs().mean().item())

    print(f"\n  Physics diagnostics ({label}, n={n:,}) — mean |term| in ug/m3/s:")
    print(f"    advection      = {m(terms['advection']):.4e}")
    print(f"    diffusion      = {m(terms['diffusion']):.4e}")
    print(f"    emission src   = {m(terms['source']):.4e}")
    print(f"    reaction (net) = {m(terms['reaction']):.4e}")
    print(f"    raw residual   = {m(terms['residual_raw']):.4e}")
    print(f"    scaled |res|   = {m(res):.4f}   mean res^2 = {float((res**2).mean().item()):.4f}")

    # Photostationary state: k[NO][O3] = J[NO2] gives a chemistry-only NO2 level.
    k_eff = tensors['k_eff'][idx]
    C_NOx = tensors['C_NOx'][idx]
    C_O3  = tensors['C_O3'][idx]
    J     = tensors['J_NO2'][idx]
    pss = (k_eff * C_NOx * C_O3) / (J + k_eff * C_O3 + 1e-20)
    phi_pred = terms['phi']
    print(f"    PSS NO2 (chem only): mean={float(pss.mean().item()):.4f}  "
          f"median={float(pss.median().item()):.4f} ug/m3")
    print(f"    PINN phi on same rows: mean={float(phi_pred.mean().item()):.4f}  "
          f"median={float(phi_pred.median().item()):.4f} ug/m3")


def norm_buffer_dict(model):
    keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 't_min', 't_max',
            'cx_min', 'cx_max', 'cy_min', 'cy_max', 'h0_min', 'h0_max',
            'u_min', 'u_max', 'v_min', 'v_max', 'd_min', 'd_max',
            'kappa_min', 'kappa_max', 'Q_min', 'Q_max',
            'T_K_min', 'T_K_max', 'C_NOx_min', 'C_NOx_max',
            'C_O3_min', 'C_O3_max', 'J_NO2_min', 'J_NO2_max']
    return {k: getattr(model, k).item() for k in keys}

# =============================================================================
# MAIN — DATA LOADING
# =============================================================================

print("\n" + "="*60)
print("HOUSTON SHIP CHANNEL NO2 PINN — REACTIVE AERMOD-DERIVED MODEL")
print("="*60)

print("\n1. Loading AERMOD-only NO2 data...")
no2_data = load_no2_data(SCENARIO_PATH)
configure_high_value_thresholds(no2_data['phi'])

if DOMAIN_X_MIN is None:
    DOMAIN_X_MIN = float(no2_data['x'].min())
    DOMAIN_X_MAX = float(no2_data['x'].max())
    DOMAIN_Y_MIN = float(no2_data['y'].min())
    DOMAIN_Y_MAX = float(no2_data['y'].max())
    print(f"\n  Derived domain: x=[{DOMAIN_X_MIN/1000:.1f}, {DOMAIN_X_MAX/1000:.1f}] km  "
          f"y=[{DOMAIN_Y_MIN/1000:.1f}, {DOMAIN_Y_MAX/1000:.1f}] km")
DOMAIN_Z_MIN = float(no2_data['z'].min())
DOMAIN_Z_MAX = float(no2_data['z'].max())
print(f"  z range: [{DOMAIN_Z_MIN:.1f}, {DOMAIN_Z_MAX:.1f}] m")

print(f"\n  AERMOD-derived rows kept without filtering: {len(no2_data['phi']):,}")
print(f"  Chemistry as network input: {CHEM_AS_NETWORK_INPUT} "
      f"({'ablation' if CHEM_AS_NETWORK_INPUT else 'frozen — physics-only'})")
print(f"  'r' column loaded but unused by design")

# Train/test split 80/20
np.random.seed(42)
N_total   = len(no2_data['phi'])
all_idx   = np.random.permutation(N_total)
n_test    = int(N_total * 0.20)
n_train   = N_total - n_test
test_idx  = np.sort(all_idx[:n_test])
train_idx = np.sort(all_idx[n_test:])

print(f"\n  Train: {n_train:,}  Test: {n_test:,}")

ARRAY_KEYS = [k for k, v in no2_data.items() if isinstance(v, np.ndarray)]
train_data = {k: no2_data[k][train_idx] for k in ARRAY_KEYS}
test_data  = {k: no2_data[k][test_idx] for k in ARRAY_KEYS}


def to_dev(arr):
    return torch.tensor(arr, dtype=torch.float32).view(-1, 1).to(device)


def build_tensors(split):
    """
    GPU tensors for one split.

    Chemistry (T_K, C_NOx, C_O3, J_NO2, k_eff) is uploaded because the physics
    residual needs it, not because the network consumes it. 'r' is deliberately
    left on the host.
    """
    tensors = {k: to_dev(split[k]) for k in
               ['x', 'y', 'z', 't', 'cx', 'cy', 'h0', 'u', 'v', 'd', 'kappa', 'Q',
                'T_K', 'C_NOx', 'C_O3', 'J_NO2', 'k_eff', 'wind_speed']}
    tensors['phi'] = to_dev(split['phi']).clamp(min=1e-15)
    tensors['regime'] = torch.tensor(split['regime'], dtype=torch.long, device=device).view(-1)
    return tensors


train_tensors = build_tensors(train_data)
test_tensors  = build_tensors(test_data)

N_TRAIN = train_tensors['x'].shape[0]
N_TEST  = test_tensors['x'].shape[0]
print(f"\n  GPU tensors: {N_TRAIN:,} train  {N_TEST:,} test")
print(f"  PHI train: {train_tensors['phi'].min().item():.2e} - {train_tensors['phi'].max().item():.2e}")
print(f"  GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
print(f"  Train regimes: signal={(train_tensors['regime'] == REGIME_SIGNAL).sum().item():,} | "
      f"physical-zero={(train_tensors['regime'] == REGIME_PHYSICAL_ZERO).sum().item():,} | "
      f"structural-zero={(train_tensors['regime'] == REGIME_STRUCTURAL_ZERO).sum().item():,}")
print(f"  Test regimes:  signal={(test_tensors['regime'] == REGIME_SIGNAL).sum().item():,} | "
      f"physical-zero={(test_tensors['regime'] == REGIME_PHYSICAL_ZERO).sum().item():,} | "
      f"structural-zero={(test_tensors['regime'] == REGIME_STRUCTURAL_ZERO).sum().item():,}")

STRUCTURAL_TRAIN_IDX = torch.where(train_tensors['regime'] == REGIME_STRUCTURAL_ZERO)[0]
SIGNAL_TRAIN_IDX = torch.where(train_tensors['regime'] == REGIME_SIGNAL)[0]
print(f"  Global physics pools: structural={len(STRUCTURAL_TRAIN_IDX):,} | signal={len(SIGNAL_TRAIN_IDX):,}")
if len(STRUCTURAL_TRAIN_IDX) == 0:
    print("  WARNING: no structural-zero rows. Low-wind physics correction will not activate.")

# Median chemistry, used only for the CHEM_AS_NETWORK_INPUT ablation when
# plotting synthetic scenarios that carry no chemistry of their own.
CHEM_MEDIANS = {k: float(np.median(no2_data[k])) for k in CHEM_KEYS}
print(f"  Chemistry medians for plotting: "
      + "  ".join(f"{k}={v:.4g}" for k, v in CHEM_MEDIANS.items()))


def chem_like(ref, values=None):
    """Chem tuple broadcast to ref's shape (dataset medians unless overridden)."""
    vals = dict(CHEM_MEDIANS)
    if values:
        vals.update(values)
    return tuple(torch.full_like(ref, vals[k]) for k in CHEM_KEYS)

# =============================================================================
# MODEL INIT
# =============================================================================

print("\n2. Initializing model...")
torch.cuda.empty_cache()

model = ParametricADRPINN(
    num_fourier_features=0,
    hidden_dim=256,
    num_layers=8,
    fourier_scale=0.0,
    use_chem_inputs=CHEM_AS_NETWORK_INPUT,
).to(device)

model.set_normalization_from_data(train_data)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Network inputs: {model.n_inputs}")
print(f"  Parameters:     {n_params:,}")
print(f"  cx range:       [{model.cx_min.item():.0f}, {model.cx_max.item():.0f}] m")
print(f"  cy range:       [{model.cy_min.item():.0f}, {model.cy_max.item():.0f}] m")
print(f"  z range:        [{model.z_min.item():.1f}, {model.z_max.item():.1f}] m")
print(f"  h0 range:       [{model.h0_min.item():.1f}, {model.h0_max.item():.1f}] m")
print(f"  u range:        [{model.u_min.item():.2f}, {model.u_max.item():.2f}] m/s")
print(f"  kappa range:    [{model.kappa_min.item():.1f}, {model.kappa_max.item():.1f}] m2/s")
print(f"  Q range:        [{model.Q_min.item():.2e}, {model.Q_max.item():.2e}] g/s")
print(f"  d range:        [{model.d_min.item():.0f}, {model.d_max.item():.0f}] m")
print(f"  T_K range:      [{model.T_K_min.item():.2f}, {model.T_K_max.item():.2f}]")
print(f"  C_NOx range:    [{model.C_NOx_min.item():.3f}, {model.C_NOx_max.item():.3f}] ug/m3")
print(f"  C_O3 range:     [{model.C_O3_min.item():.3f}, {model.C_O3_max.item():.3f}] ug/m3")
print(f"  J_NO2 range:    [{model.J_NO2_min.item():.4e}, {model.J_NO2_max.item():.4e}] 1/s")

model.eval()
sanity_out = predict_batched(model, train_tensors,
                            torch.arange(1, device=device))
print(f"\n  Sanity check — output: {sanity_out.item():.4e}  "
      f"target: {train_tensors['phi'][0].item():.4e}")
if sanity_out.item() == 0 or np.isnan(sanity_out.item()):
    print("  *** WARNING: zero/NaN output — check normalization ***")

diagnose_physics_terms(model, train_tensors, SIGNAL_TRAIN_IDX, "untrained, signal rows")

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================

print("\n3. Setting up optimizer...")

EPOCH_SIZE = min(500000, N_TRAIN)
BATCH_SIZE = 4096
N_BATCHES  = max(EPOCH_SIZE // BATCH_SIZE, 1)
num_epochs = 1000

# Curriculum: supervised only -> add reactive PDE on positive rows -> add
# PDE-only correction on structural low-wind zeros.
DATA_ONLY_EPOCHS         = 150
SIGNAL_PHYSICS_START     = 250
SIGNAL_PHYSICS_RAMP      = 150
STRUCTURAL_PHYSICS_START = 300
STRUCTURAL_PHYSICS_RAMP  = 200

PHYSICS_WEIGHT                = 0.05
STRUCTURAL_PHYSICS_WEIGHT_MAX = 1.0
PHYSICS_EVERY_N               = 5
PHYSICS_N_POINTS              = 256
early_stop_patience           = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1500, eta_min=1e-6
)

print(f"  Epochs:             {num_epochs}")
print(f"  Batch size:         {BATCH_SIZE}")
print(f"  Epoch size:         {EPOCH_SIZE:,}")
print(f"  Data-only epochs:   {DATA_ONLY_EPOCHS}")
print(f"  Signal physics:     start={SIGNAL_PHYSICS_START}, ramp={SIGNAL_PHYSICS_RAMP}, max={PHYSICS_WEIGHT}")
print(f"  Structural physics: start={STRUCTURAL_PHYSICS_START}, ramp={STRUCTURAL_PHYSICS_RAMP}, max={STRUCTURAL_PHYSICS_WEIGHT_MAX}")
print(f"  Low-wind threshold: {LOW_WIND_THRESHOLD} m/s")
print(f"  Physical-zero upper bound: {PHYSICAL_ZERO_UPPER_BOUND} ug/m3")
print(f"  Reaction:           enabled={REACTION_ENABLED}  weight={REACTION_WEIGHT}  "
      f"rate_mode={no2_data['rate_mode']}")
print(f"  Early stop:         {early_stop_patience}")

# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n4. Training reactive regime-aware AERMOD-derived NO2 model...")
print("-" * 60)


def physics_residual_on(idx):
    """Reactive residual on a set of training-row indices."""
    return compute_pde_residual_reactive(
        model,
        train_tensors['x'][idx], train_tensors['y'][idx], train_tensors['z'][idx],
        train_tensors['t'][idx], train_tensors['cx'][idx], train_tensors['cy'][idx],
        train_tensors['h0'][idx], train_tensors['u'][idx], train_tensors['v'][idx],
        train_tensors['d'][idx], train_tensors['kappa'][idx], train_tensors['Q'][idx],
        train_tensors['k_eff'][idx], train_tensors['C_NOx'][idx],
        train_tensors['C_O3'][idx], train_tensors['J_NO2'][idx], train_tensors['T_K'][idx],
        device,
    )


loss_history         = []
data_loss_history    = []
physics_loss_history = []
physics_term_history = []
signal_phys_weight_history = []
structural_phys_weight_history = []
lr_history           = []
best_loss            = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    t_start = time.time()

    epoch_idx = torch.randperm(N_TRAIN, device=device)[:EPOCH_SIZE]

    epoch_data_loss    = 0.0
    epoch_physics_loss = 0.0
    epoch_physics_term = 0.0
    n_batches          = 0
    n_physics_batches  = 0

    if epoch < SIGNAL_PHYSICS_START:
        sig_ramp = 0.0
    elif epoch < SIGNAL_PHYSICS_START + SIGNAL_PHYSICS_RAMP:
        sig_ramp = (epoch - SIGNAL_PHYSICS_START) / max(SIGNAL_PHYSICS_RAMP, 1)
    else:
        sig_ramp = 1.0

    if epoch < STRUCTURAL_PHYSICS_START:
        struct_ramp = 0.0
    elif epoch < STRUCTURAL_PHYSICS_START + STRUCTURAL_PHYSICS_RAMP:
        struct_ramp = (epoch - STRUCTURAL_PHYSICS_START) / max(STRUCTURAL_PHYSICS_RAMP, 1)
    else:
        struct_ramp = 1.0

    epoch_w_sig = 0.0
    epoch_w_struct = 0.0
    n_w_sig = 0
    n_w_struct = 0

    for b in range(N_BATCHES):
        bs  = b * BATCH_SIZE
        be  = bs + BATCH_SIZE
        idx = epoch_idx[bs:be]

        optimizer.zero_grad(set_to_none=True)

        chem_batch = (train_tensors['T_K'][idx], train_tensors['C_NOx'][idx],
                      train_tensors['C_O3'][idx], train_tensors['J_NO2'][idx]) \
            if model.use_chem_inputs else None

        phi_pred = model(
            train_tensors['x'][idx], train_tensors['y'][idx], train_tensors['z'][idx],
            train_tensors['t'][idx], train_tensors['cx'][idx], train_tensors['cy'][idx],
            train_tensors['h0'][idx], train_tensors['u'][idx], train_tensors['v'][idx],
            train_tensors['d'][idx], train_tensors['kappa'][idx], train_tensors['Q'][idx],
            chem=chem_batch, normalize=True,
        )

        batch_regime = train_tensors['regime'][idx]
        data_loss, _ = compute_regime_aware_data_loss(phi_pred, train_tensors['phi'][idx], batch_regime)

        structural_physics_loss = torch.tensor(0.0, device=device)
        signal_physics_loss = torch.tensor(0.0, device=device)

        # Regime 3: low-wind structural zeros carry no AERMOD label, so the
        # reactive PDE is their only constraint. Sampled from the global pool so
        # the correction stays active even when such rows are rare in a batch.
        if struct_ramp > 0 and STRUCTURAL_PHYSICS_GLOBAL and len(STRUCTURAL_TRAIN_IDX) > 0:
            n_struct = min(STRUCTURAL_PHYSICS_N_POINTS, len(STRUCTURAL_TRAIN_IDX))
            s_idx = STRUCTURAL_TRAIN_IDX[torch.randperm(len(STRUCTURAL_TRAIN_IDX), device=device)[:n_struct]]
            res_struct = physics_residual_on(s_idx)
            structural_physics_loss = torch.mean(res_struct**2)
            epoch_physics_loss += structural_physics_loss.item()
            n_physics_batches += 1

        # Regime 1: reactive PDE regularization on positive AERMOD rows, sampled
        # globally so physics covers the whole source/receptor/met/chemistry space.
        if b % PHYSICS_EVERY_N == 0 and sig_ramp > 0 and len(SIGNAL_TRAIN_IDX) > 0:
            n_phys = min(PHYSICS_N_POINTS, len(SIGNAL_TRAIN_IDX))
            r_idx = SIGNAL_TRAIN_IDX[torch.randperm(len(SIGNAL_TRAIN_IDX), device=device)[:n_phys]]
            res_signal = physics_residual_on(r_idx)
            signal_physics_loss = torch.mean(res_signal**2)
            epoch_physics_loss += signal_physics_loss.item()
            n_physics_batches += 1

        # Adaptive balancing: keep each weighted physics term at a target
        # fraction of the current data loss so physics regularizes and never
        # dominates the gradient.
        eps_w = 1e-12
        if PHYS_ADAPTIVE_BALANCE:
            data_ref = data_loss.detach().clamp(min=1e-8)
            if struct_ramp > 0 and float(structural_physics_loss) > 0:
                w_struct = (struct_ramp * STRUCTURAL_PHYS_TARGET_RATIO
                            * data_ref / (structural_physics_loss.detach() + eps_w))
                w_struct = float(w_struct.clamp(max=PHYS_EFF_WEIGHT_MAX))
            else:
                w_struct = 0.0
            if sig_ramp > 0 and float(signal_physics_loss) > 0:
                w_sig = (sig_ramp * SIGNAL_PHYS_TARGET_RATIO
                         * data_ref / (signal_physics_loss.detach() + eps_w))
                w_sig = float(w_sig.clamp(max=PHYS_EFF_WEIGHT_MAX))
            else:
                w_sig = 0.0
        else:
            w_sig = PHYSICS_WEIGHT * sig_ramp
            w_struct = STRUCTURAL_PHYSICS_WEIGHT_MAX * struct_ramp

        if w_sig > 0:
            epoch_w_sig += w_sig
            n_w_sig += 1
        if w_struct > 0:
            epoch_w_struct += w_struct
            n_w_struct += 1

        physics_loss = w_struct * structural_physics_loss + w_sig * signal_physics_loss
        total_loss = data_loss + physics_loss
        epoch_physics_term += physics_loss.item()

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_data_loss += data_loss.item()
        n_batches += 1

    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    avg_data    = epoch_data_loss / max(n_batches, 1)
    avg_physics = epoch_physics_loss / max(n_physics_batches, 1) if n_physics_batches > 0 else 0.0
    phys_term   = epoch_physics_term / max(n_batches, 1)
    avg_total   = avg_data + phys_term
    avg_w_sig   = epoch_w_sig / max(n_w_sig, 1)
    avg_w_struct = epoch_w_struct / max(n_w_struct, 1)

    scheduler.step()

    loss_history.append(avg_total)
    data_loss_history.append(avg_data)
    physics_loss_history.append(avg_physics)
    physics_term_history.append(phys_term)
    signal_phys_weight_history.append(avg_w_sig)
    structural_phys_weight_history.append(avg_w_struct)
    lr_history.append(optimizer.param_groups[0]['lr'])

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        if sig_ramp > 0 or struct_ramp > 0 or phys_term > 0:
            phys_str = f"PhysRaw={avg_physics:.3e} | PhysTerm={phys_term:.3e}"
        else:
            phys_str = "Phys=off"
        ratio = phys_term / max(avg_data, 1e-12)
        print(f"Epoch {epoch:4d}/{num_epochs}: "
              f"Data={avg_data:.4f} | {phys_str} | "
              f"Phys/Data={ratio:.3e} | "
              f"w_sig={avg_w_sig:.3e} | "
              f"w_struct={avg_w_struct:.3e} | "
              f"{elapsed:.1f}s | LR={optimizer.param_groups[0]['lr']:.1e}")

    if not np.isnan(avg_data) and avg_data < best_loss:
        best_loss = avg_data
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'data_loss': avg_data,
            'physics_loss': avg_physics,
            'physics_term': phys_term,
            'signal_physics_weight': avg_w_sig,
            'structural_physics_weight': avg_w_struct,
            'coord_frame': COORD_FRAME,
            'norm_buffers': norm_buffer_dict(model),
            'chem_as_network_input': CHEM_AS_NETWORK_INPUT,
            'reaction': {
                'enabled': REACTION_ENABLED,
                'weight': REACTION_WEIGHT,
                'rate_mode': no2_data['rate_mode'],
                'clamp_no': REACTION_CLAMP_NO,
            },
            'training_mode': 'curriculum_lowwind_reaction_no2_aermod_derived',
        }, MODEL_BEST)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stop_patience:
        print(f"\n*** Early stopping at epoch {epoch} — no improvement for {early_stop_patience} epochs ***")
        break

print("-" * 60)
print(f"Training complete. Best data loss: {best_loss:.6f}")
n_epochs_run = len(loss_history)
history_cols = {
    'total_loss': loss_history,
    'data_loss': data_loss_history,
    'physics_loss': physics_loss_history,
    'physics_term': physics_term_history,
    'signal_physics_weight': signal_phys_weight_history,
    'structural_physics_weight': structural_phys_weight_history,
    'lr': lr_history,
}
for name, values in history_cols.items():
    if len(values) != n_epochs_run:
        raise ValueError(
            f"Training log length mismatch for '{name}': "
            f"expected {n_epochs_run}, got {len(values)}"
        )

log_df = pd.DataFrame({
    'epoch': range(1, n_epochs_run + 1),
    **history_cols,
})
log_df.to_csv(TRAIN_LOG_OUT, index=False)
print(f"Training log saved: {TRAIN_LOG_OUT}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'coord_frame': COORD_FRAME,
    'norm_buffers': norm_buffer_dict(model),
    'chem_as_network_input': CHEM_AS_NETWORK_INPUT,
    'training_mode': 'curriculum_lowwind_reaction_no2_aermod_derived',
    'reaction': {
        'enabled': REACTION_ENABLED,
        'weight': REACTION_WEIGHT,
        'rate_mode': no2_data['rate_mode'],
        'clamp_no': REACTION_CLAMP_NO,
        'arrhenius_A': ARRHENIUS_A,
        'arrhenius_Ea_over_R': ARRHENIUS_EA_OVER_R,
    },
    'curriculum': {
        'DATA_ONLY_EPOCHS': DATA_ONLY_EPOCHS,
        'SIGNAL_PHYSICS_START': SIGNAL_PHYSICS_START,
        'SIGNAL_PHYSICS_RAMP': SIGNAL_PHYSICS_RAMP,
        'STRUCTURAL_PHYSICS_START': STRUCTURAL_PHYSICS_START,
        'STRUCTURAL_PHYSICS_RAMP': STRUCTURAL_PHYSICS_RAMP,
        'PHYSICS_WEIGHT': PHYSICS_WEIGHT,
        'STRUCTURAL_PHYSICS_WEIGHT_MAX': STRUCTURAL_PHYSICS_WEIGHT_MAX,
    },
    'high_value_thresholds': {
        'HIGH_LABEL_THRESHOLD': HIGH_LABEL_THRESHOLD,
        'HIGH_CAPTURE_TARGET': HIGH_CAPTURE_TARGET,
        'HIGH_CAPTURE_METRIC_FLOOR': HIGH_CAPTURE_METRIC_FLOOR,
    },
}, MODEL_FINAL)
print(f"Final model saved: {MODEL_FINAL}")

# =============================================================================
# LOAD BEST MODEL FOR BENCHMARKING
# =============================================================================

print("\n5. Loading best checkpoint for benchmarking...")
checkpoint = torch.load(MODEL_BEST, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"  Best epoch: {checkpoint['epoch']}  Best data loss: {checkpoint['data_loss']:.6f}")

diagnose_physics_terms(model, train_tensors, SIGNAL_TRAIN_IDX, "trained, signal rows")
diagnose_physics_terms(model, train_tensors, STRUCTURAL_TRAIN_IDX, "trained, structural-zero rows")

# =============================================================================
# FULL BENCHMARKING
# =============================================================================

print("\n6. Running PINN-vs-AERMOD NO2 benchmark suite...")

all_metric_rows = []
train_metrics = evaluate_full_benchmark(model, train_tensors, "TRAINING SET", max_samples=100000)
all_metric_rows.extend(train_metrics)

test_metrics = evaluate_full_benchmark(model, test_tensors, "TEST SET (held-out)", max_samples=100000)
all_metric_rows.extend(test_metrics)

metrics_df = pd.DataFrame(all_metric_rows)
metrics_df.to_csv(METRICS_OUT, index=False)
print(f"\nRegime-aware benchmark metrics saved: {METRICS_OUT}")
print(f"  Rows: {len(metrics_df)}  Cols: {list(metrics_df.columns)}")

# =============================================================================
# TRAINING PROGRESS PLOTS
# =============================================================================

print("\n7. Generating training visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(loss_history) + 1)

ax = axes[0]
ax.semilogy(epochs_range, data_loss_history, 'b-', lw=2, label='Data Loss')
phys_nonzero = [(i+1, p) for i, p in enumerate(physics_loss_history) if p > 0]
if phys_nonzero:
    pe, pv = zip(*phys_nonzero)
    ax.semilogy(pe, pv, 'r-', lw=1.5, alpha=0.7, label='Reactive Physics Loss')
ax.axvline(x=SIGNAL_PHYSICS_START, color='orange', ls='--', alpha=0.8,
           label=f'Signal physics start (ep {SIGNAL_PHYSICS_START})')
ax.axvline(x=STRUCTURAL_PHYSICS_START, color='red', ls='--', alpha=0.6,
           label=f'Structural physics start (ep {STRUCTURAL_PHYSICS_START})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log)')
ax.set_title('Loss Components')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(epochs_range, lr_history, 'purple', lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('LR Schedule')
ax.grid(True, alpha=0.3)

ax = axes[2]
min_idx = int(np.argmin(loss_history))
ax.semilogy(epochs_range, loss_history, 'g-', lw=2, label='Total Loss')
ax.scatter([min_idx + 1], [loss_history[min_idx]], color='red', s=120,
           zorder=5, label=f'Best: {loss_history[min_idx]:.4f}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss (log)')
ax.set_title('Total Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = SAVE_PATH / 'no2_training_progress_reaction.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Training plot saved: {fig_path}")

# =============================================================================
# HOUSTON DOMAIN NO2 CONCENTRATION FIELD
# =============================================================================

def visualize_no2_field(model, u_eff, v_eff, kappa_val, source_params,
                        z_eval=0.0, n_grid=80, save_path=None):
    model.eval()
    dev = next(model.parameters()).device

    x_range = np.linspace(DOMAIN_X_MIN, DOMAIN_X_MAX, n_grid)
    y_range = np.linspace(DOMAIN_Y_MIN, DOMAIN_Y_MAX, n_grid)
    X_g, Y_g = np.meshgrid(x_range, y_range)
    n_pts = n_grid * n_grid

    x_flat = torch.tensor(X_g.flatten(), dtype=torch.float32).view(-1, 1).to(dev)
    y_flat = torch.tensor(Y_g.flatten(), dtype=torch.float32).view(-1, 1).to(dev)
    z_flat = torch.full_like(x_flat, float(z_eval))
    t_flat = torch.full_like(x_flat, 3600.0)
    u_flat = torch.full_like(x_flat, float(u_eff))
    v_flat = torch.full_like(x_flat, float(v_eff))
    k_flat = torch.full_like(x_flat, float(kappa_val))
    chem_in = chem_like(x_flat) if model.use_chem_inputs else None

    total = torch.zeros(n_pts, 1, device=dev)

    with torch.no_grad():
        for src in source_params:
            cx_f = torch.full_like(x_flat, float(src['cx_local']))
            cy_f = torch.full_like(x_flat, float(src['cy_local']))
            h0_f = torch.full_like(x_flat, float(src.get('h0_m', 10.0)))
            d_f  = torch.full_like(x_flat, float(src.get('d_m', 2000.0)))
            q_f  = torch.full_like(x_flat, float(src['Q_gs']))
            pred = model(x_flat, y_flat, z_flat, t_flat, cx_f, cy_f, h0_f,
                         u_flat, v_flat, d_f, k_flat, q_f,
                         chem=chem_in, normalize=True)
            total += pred

    conc = total.cpu().numpy().reshape(n_grid, n_grid)
    log_conc = np.log10(np.maximum(conc, 1e-6))

    fig, ax = plt.subplots(figsize=(12, 10))
    cs = ax.contourf(X_g / 1000, Y_g / 1000, log_conc, levels=50, cmap='jet', extend='both')
    cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    cbar.set_label('log10(AERMOD-surrogate NO2 ug/m3)')

    for src in source_params:
        ax.scatter(src['cx_local'] / 1000, src['cy_local'] / 1000,
                   color='white', marker='x', s=60, linewidths=1.5, zorder=5)

    ax.set_xlabel('Local Easting (km)')
    ax.set_ylabel('Local Northing (km)')
    ax.set_title(f"AERMOD-Surrogate NO2 Field | z={z_eval:.0f}m u={u_eff:.1f} v={v_eff:.1f} "
                 f"kappa={kappa_val:.0f} m2/s")
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Field plot saved: {save_path}")
    plt.show()
    return fig


met_path = DRIVE_ROOT / 'aermet_2021_hourly.parquet'
src_path = DRIVE_ROOT / 'sources_utm_2021.parquet'
# Prefer NO2-folder copies when present.
if (DRIVE_BASE / 'aermet_2021_hourly.parquet').exists():
    met_path = DRIVE_BASE / 'aermet_2021_hourly.parquet'
if (DRIVE_BASE / 'sources_utm_2021.parquet').exists():
    src_path = DRIVE_BASE / 'sources_utm_2021.parquet'

if met_path.exists() and src_path.exists():
    met_df = pd.read_parquet(met_path)
    src_df = pd.read_parquet(src_path)

    nc_met = met_df[met_df['wspd'] > 2.0].iloc[500]
    ex_u   = float(nc_met.get('u_eff', nc_met.get('u', 3.0)))
    ex_v   = float(nc_met.get('v_eff', nc_met.get('v', 1.5)))
    ex_k   = float(nc_met.get('kappa', 150.0))

    needed_cols = [c for c in ['x_local', 'y_local', 'ann_value_gs', 'facility_id'] if c in src_df.columns]
    src_top = (src_df[needed_cols]
               .dropna()
               .groupby('facility_id')
               .agg(cx_local=('x_local', 'mean'), cy_local=('y_local', 'mean'), Q_gs=('ann_value_gs', 'sum'))
               .nlargest(20, 'Q_gs')
               .reset_index())

    source_params_ex = [
        {'cx_local': r['cx_local'], 'cy_local': r['cy_local'], 'Q_gs': r['Q_gs'],
         'd_m': 2000.0, 'h0_m': 10.0}
        for _, r in src_top.iterrows()
    ]

    z_field = float(DOMAIN_Z_MIN)
    print(f"\n  Generating field plot: z={z_field:.0f} u={ex_u:.2f} v={ex_v:.2f} kappa={ex_k:.0f}")
    visualize_no2_field(
        model, ex_u, ex_v, ex_k, source_params_ex,
        z_eval=z_field,
        save_path=SAVE_PATH / 'no2_concentration_field_reaction.png'
    )
else:
    print("\n  Met/source parquet files not found on Drive — skipping field plot")
    print(f"  Expected: {met_path}")
    print(f"            {src_path}")

# =============================================================================
# 3D PLUME SCENARIO
# =============================================================================

def visualize_plume_3d(model, scenario, n_xy=48, n_z=28, half_width_m=15000.0,
                       conc_percentile=90.0, save_path=None):
    """
    Render a single 3D NO2 plume scenario as a volumetric scatter.

    Points above a concentration percentile are drawn, colored by log10
    concentration, so the visible cloud traces the plume body. Chemistry is not
    required here: with frozen chemistry the network is a pure transport-input
    surrogate, so any scenario can be rendered without chemical fields.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    model.eval()
    dev = next(model.parameters()).device

    cx = float(scenario['cx_local'])
    cy = float(scenario['cy_local'])
    h0 = float(scenario['h0_m'])
    d  = float(scenario.get('d_m', 2000.0))
    Q  = float(scenario['Q_gs'])
    u  = float(scenario['u'])
    v  = float(scenario['v'])
    k  = float(scenario['kappa'])

    x_lo = max(DOMAIN_X_MIN, cx - half_width_m)
    x_hi = min(DOMAIN_X_MAX, cx + half_width_m)
    y_lo = max(DOMAIN_Y_MIN, cy - half_width_m)
    y_hi = min(DOMAIN_Y_MAX, cy + half_width_m)
    z_lo = float(DOMAIN_Z_MIN)
    z_hi = float(DOMAIN_Z_MAX)
    if z_hi - z_lo < 1.0:
        z_lo, z_hi = 0.0, max(2.0 * h0, 200.0)

    x_range = np.linspace(x_lo, x_hi, n_xy)
    y_range = np.linspace(y_lo, y_hi, n_xy)
    z_range = np.linspace(z_lo, z_hi, n_z)
    X_g, Y_g, Z_g = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    n_pts = X_g.size

    x_flat = torch.tensor(X_g.reshape(-1, 1), dtype=torch.float32).to(dev)
    y_flat = torch.tensor(Y_g.reshape(-1, 1), dtype=torch.float32).to(dev)
    z_flat = torch.tensor(Z_g.reshape(-1, 1), dtype=torch.float32).to(dev)
    t_flat  = torch.full_like(x_flat, 3600.0)
    cx_flat = torch.full_like(x_flat, cx)
    cy_flat = torch.full_like(x_flat, cy)
    h0_flat = torch.full_like(x_flat, h0)
    u_flat  = torch.full_like(x_flat, u)
    v_flat  = torch.full_like(x_flat, v)
    d_flat  = torch.full_like(x_flat, d)
    k_flat  = torch.full_like(x_flat, k)
    q_flat  = torch.full_like(x_flat, Q)

    preds = []
    with torch.no_grad():
        for s in range(0, n_pts, 200000):
            e = min(s + 200000, n_pts)
            chem_in = chem_like(x_flat[s:e], scenario.get('chem')) \
                if model.use_chem_inputs else None
            preds.append(model(
                x_flat[s:e], y_flat[s:e], z_flat[s:e], t_flat[s:e],
                cx_flat[s:e], cy_flat[s:e], h0_flat[s:e],
                u_flat[s:e], v_flat[s:e], d_flat[s:e], k_flat[s:e], q_flat[s:e],
                chem=chem_in, normalize=True,
            ).cpu())
    conc = torch.cat(preds).numpy().reshape(-1)

    xs = X_g.reshape(-1) / 1000.0
    ys = Y_g.reshape(-1) / 1000.0
    zs = Z_g.reshape(-1)

    thresh = max(np.percentile(conc, conc_percentile), 1e-6)
    body = conc >= thresh
    if body.sum() < 50:
        body = conc >= np.percentile(conc, 50.0)

    log_c = np.log10(np.maximum(conc[body], 1e-6))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        xs[body], ys[body], zs[body],
        c=log_c, cmap='turbo',
        s=18, alpha=0.35, edgecolors='none', depthshade=True,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('log10(AERMOD-surrogate NO2 ug/m3)')

    ax.scatter([cx / 1000.0], [cy / 1000.0], [h0],
               color='black', marker='*', s=200, depthshade=False,
               label='Source')

    spd = max(np.hypot(u, v), 1e-6)
    arr_len = 0.35 * (x_hi - x_lo) / 1000.0
    ax.quiver(cx / 1000.0, cy / 1000.0, h0,
              u / spd, v / spd, 0.0, length=arr_len, color='red',
              linewidth=2, label='Wind')

    ax.set_xlabel('Local Easting (km)')
    ax.set_ylabel('Local Northing (km)')
    ax.set_zlabel('Height z (m)')
    ax.set_title(
        f"3D NO2 Plume Scenario | h0={h0:.0f}m  Q={Q:.2e} g/s\n"
        f"u={u:.2f} v={v:.2f} (|U|={spd:.2f} m/s)  kappa={k:.0f} m2/s"
    )
    ax.legend(loc='upper left')
    ax.view_init(elev=22, azim=-60)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D plume plot saved: {save_path}")
    plt.show()
    return fig


print("\n7b. Generating 3D plume scenario...")

plume_src_idx = int(np.argmax(no2_data['phi']))
plume_scenario = {
    'cx_local': float(no2_data['cx'][plume_src_idx]),
    'cy_local': float(no2_data['cy'][plume_src_idx]),
    'h0_m':     float(no2_data['h0'][plume_src_idx]),
    'd_m':      float(no2_data['d'][plume_src_idx]),
    'Q_gs':     float(no2_data['Q'][plume_src_idx]),
    'u':        float(no2_data['u'][plume_src_idx]),
    'v':        float(no2_data['v'][plume_src_idx]),
    'kappa':    float(no2_data['kappa'][plume_src_idx]),
    'chem':     {k: float(no2_data[k][plume_src_idx]) for k in CHEM_KEYS},
}
print(f"  Source @ ({plume_scenario['cx_local']/1000:.1f}, "
      f"{plume_scenario['cy_local']/1000:.1f}) km  h0={plume_scenario['h0_m']:.0f} m  "
      f"Q={plume_scenario['Q_gs']:.2e} g/s  "
      f"u={plume_scenario['u']:.2f} v={plume_scenario['v']:.2f}  "
      f"kappa={plume_scenario['kappa']:.0f}")
print(f"  Row chemistry: " + "  ".join(f"{k}={v:.4g}" for k, v in plume_scenario['chem'].items()))

visualize_plume_3d(
    model, plume_scenario,
    save_path=SAVE_PATH / 'no2_plume_3d_reaction.png'
)

# =============================================================================
# PARITY PLOT — TEST SET
# =============================================================================

print("\n8. Generating parity plot...")

model.eval()
signal_test_idx_all = torch.where(test_tensors['regime'] == REGIME_SIGNAL)[0]
n_parity = min(50000, len(signal_test_idx_all))
if n_parity > 0:
    parity_idx = signal_test_idx_all[torch.randperm(len(signal_test_idx_all), device=device)[:n_parity]]
else:
    parity_idx = torch.randperm(N_TEST, device=device)[:min(50000, N_TEST)]

pred_parity = predict_batched(model, test_tensors, parity_idx)
true_parity = test_tensors['phi'][parity_idx]

pred_np = pred_parity.cpu().numpy().flatten()
true_np = true_parity.cpu().numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
valid_mask = (true_np > 1e-6) & (pred_np > 1e-6)

ax = axes[0]
if valid_mask.sum() > 0:
    hb = ax.hexbin(np.log10(true_np[valid_mask]), np.log10(pred_np[valid_mask]),
                   gridsize=50, cmap='inferno', mincnt=1, bins='log')
    lim_min = min(np.log10(true_np[valid_mask]).min(), np.log10(pred_np[valid_mask]).min())
    lim_max = max(np.log10(true_np[valid_mask]).max(), np.log10(pred_np[valid_mask]).max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'w--', lw=2, label='1:1 line')
    ax.plot([lim_min, lim_max], [lim_min+0.301, lim_max+0.301], 'w:', alpha=0.5, label='Factor of 2')
    ax.plot([lim_min, lim_max], [lim_min-0.301, lim_max-0.301], 'w:', alpha=0.5)
    plt.colorbar(hb, ax=ax, label='log10(Count)')
    ax.legend(fontsize='small')
else:
    ax.text(0.5, 0.5, 'No positive rows for log parity plot', ha='center', va='center')
ax.set_xlabel('AERMOD NO2 label log10(ug/m3)')
ax.set_ylabel('PINN prediction log10(ug/m3)')
ax.set_title('Parity Plot — Test Regime 1 Signal Rows')
ax.grid(True, alpha=0.3)

ax = axes[1]
if valid_mask.sum() > 0:
    residuals = np.log10(pred_np[valid_mask]) - np.log10(true_np[valid_mask])
    ax.hist(residuals, bins=60, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    mu, sigma = residuals.mean(), residuals.std()
    ax.axvline(mu, color='red', lw=2, label=f'Mean: {mu:.3f}')
    ax.axvline(mu - sigma, color='orange', lw=1.5, ls='--', label=f'+/-1 sigma: {sigma:.3f}')
    ax.axvline(mu + sigma, color='orange', lw=1.5, ls='--')
    ax.axvline(0, color='green', lw=1, ls=':')
    ax.legend(fontsize='small')
else:
    ax.text(0.5, 0.5, 'No positive rows for residual histogram', ha='center', va='center')
ax.set_xlabel('Log10 Residual (PINN - AERMOD)')
ax.set_ylabel('Density')
ax.set_title('Residual Distribution — Test Regime 1 Signal Rows')
ax.grid(True, alpha=0.3)

plt.tight_layout()
parity_path = SAVE_PATH / 'no2_parity_plot_reaction.png'
plt.savefig(parity_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Parity plot saved: {parity_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NO2 REACTIVE PINN TRAINING & BENCHMARKING COMPLETE")
print("="*60)
print(f"  Best checkpoint:     {MODEL_BEST}")
print(f"  Final model:         {MODEL_FINAL}")
print(f"  Benchmark metrics:   {METRICS_OUT}")
print(f"  Training log:        {TRAIN_LOG_OUT}")
print(f"  Training plot:       {fig_path}")
print(f"  3D plume plot:       {SAVE_PATH / 'no2_plume_3d_reaction.png'}")
print(f"  Parity plot:         {parity_path}")

print(f"\n  Training summary:")
print(f"    Mode:              Curriculum low-wind advection-diffusion-reaction NO2 PINN")
print(f"    Chemistry inputs:  {'network + physics' if CHEM_AS_NETWORK_INPUT else 'physics only (frozen)'}")
print(f"    Rate mode:         {no2_data['rate_mode']}")
print(f"    Epochs run:        {len(loss_history)}")
print(f"    Best data loss:    {best_loss:.6f}")
print(f"    Final LR:          {lr_history[-1]:.2e}")

signal_test = next((m for m in test_metrics if 'Regime 1 signal rows' in m.get('label', '')), None)
all_test = next((m for m in test_metrics if 'All AERMOD rows' in m.get('label', '')), None)
summary_m = signal_test or all_test
if summary_m:
    print(f"\n  Held-out Regime 1 signal-row agreement summary:")
    print(f"    FAC2:        {summary_m.get('FAC2', float('nan')):.1f}%")
    print(f"    Pearson r:   {summary_m.get('r', float('nan')):.3f}")
    print(f"    Spearman rho:{summary_m.get('rho', float('nan')):.3f}")
    print(f"    RMSE:        {summary_m.get('RMSE', float('nan')):.4f}")
    print(f"    Log RMSE:    {summary_m.get('log_rmse', float('nan')):.4f}")

print("\nAll NO2 outputs saved to Google Drive.")
print("="*60)
