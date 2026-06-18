
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.sst_preprocessing import build_ocean_mask_bitmask
from src.utils.config import Config


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Backfill normalization stats for a trained model")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory containing the trained model")
    parser.add_argument("--data", type=str, default=cfg.data.nc_path,
                        help="Path to source NetCDF (must match training)")
    parser.add_argument("--train-start", type=str, default=cfg.data.train_slice[0])
    parser.add_argument("--train-end", type=str, default=cfg.data.train_slice[1])
    parser.add_argument("--mask-mode", type=str, default=cfg.data.mask_mode,
                        choices=["first", "majority"])
    parser.add_argument("--ice-threshold", type=float, default=cfg.data.ice_threshold)
    parser.add_argument("--out-name", type=str, default="norm_stats.npz")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing norm_stats.npz")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

    out_path = run_dir / args.out_name
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Use --force to overwrite.")

    print(f"Opening {args.data}")
    ds = xr.open_dataset(args.data, decode_cf=True)

    print(f"Building ocean mask (mode={args.mask_mode}, ice_threshold={args.ice_threshold})")
    ocean_mask = build_ocean_mask_bitmask(
        ds, mode=args.mask_mode, ice_threshold=args.ice_threshold,
    )

    print(f"Computing μ, σ over training period {args.train_start} → {args.train_end}")
    sst_train = ds["sst"].sel(time=slice(args.train_start, args.train_end))
    mu = sst_train.where(ocean_mask).mean("time").values.astype(np.float32)
    sigma = sst_train.where(ocean_mask).std("time").values.astype(np.float32)
    sigma = np.where((sigma < 1e-6) | ~np.isfinite(sigma), 1.0, sigma).astype(np.float32)

    np.savez(
        out_path,
        mu=mu,
        sigma=sigma,
        ocean_mask=ocean_mask,
        train_start=np.array(args.train_start),
        train_end=np.array(args.train_end),
        mask_mode=np.array(args.mask_mode),
        ice_threshold=np.array(args.ice_threshold, dtype=np.float32),
    )

    n_ocean = int(ocean_mask.sum())
    print(f"\nSaved: {out_path}")
    print(f"  mu     shape={mu.shape} dtype={mu.dtype}")
    print(f"  sigma  shape={sigma.shape} dtype={sigma.dtype}")
    print(f"  ocean  shape={ocean_mask.shape} pixels={n_ocean}")
    print(f"  σ range over ocean: [{sigma[ocean_mask].min():.3f}, {sigma[ocean_mask].max():.3f}]")
    print(f"  μ range over ocean: [{mu[ocean_mask].min():.3f}, {mu[ocean_mask].max():.3f}] °C")


if __name__ == "__main__":
    main()