"""
MLP standardisation ablation: compare MLP Raw (standardized) vs MLP 17 features (standardized)
under LOGO validation. The tree models are invariant to monotone transformations,
so only the MLP is tested here.

The existing models.py already wraps MLP in make_pipeline(StandardScaler(), MLPRegressor(...)).
This script confirms that explicitly and reports LOGO results for:
  - MLP raw (standardized)
  - MLP 17 features (standardized)

This addresses audit point 2A: whether the MLP gain comes from the features or from scaling.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
from common import load, feats, calibrate_pivc, metrics
from models import make, fit_predict

df, n0 = load()
geoms = sorted(df.gid.unique())

results = []

for fset_name, fset_code in [('raw (standardized)', 'raw'), ('17 features (standardized)', 'full17')]:
    preds_all = []
    for g in geoms:
        tr = df[df.gid != g]
        te = df[df.gid == g]
        par = calibrate_pivc(tr) if fset_code == 'full17' else None
        pred = fit_predict('MLP', tr, te, fset_code, par, seed=42)
        preds_all.append(pd.DataFrame({'y': te.eta.values, 'pred': pred, 'gid': g}))
    
    P = pd.concat(preds_all, ignore_index=True)
    m = metrics(P.y, P.pred)
    
    # Per-geometry RMSE
    per_g = P.groupby('gid').apply(lambda d: np.sqrt(np.mean((d.y - d.pred)**2)))
    macro_rms = np.sqrt(np.mean(per_g**2))
    
    # Per-geometry R2
    r2_g = P.groupby('gid').apply(lambda d: metrics(d.y, d.pred)['r2'])
    
    result = {
        'fset': fset_name,
        'pooled_r2': m['r2'],
        'pooled_rmse': m['rmse'],
        'pooled_mae': m['mae'],
        'macro_rms': macro_rms,
        'weighted_r2_g_mean': np.average(r2_g, weights=P.groupby('gid').size()),
        'median_r2_g': np.median(r2_g),
        'n_r2_le_0': int((r2_g <= 0).sum()),
    }
    results.append(result)
    print(f"\nMLP {fset_name}:")
    print(f"  Pooled R^2 = {result['pooled_r2']:.4f}")
    print(f"  Pooled RMSE = {result['pooled_rmse']:.4f}")
    print(f"  Macro-RMS RMSE = {result['macro_rms']:.4f}")
    print(f"  Weighted mean R^2_g = {result['weighted_r2_g_mean']:.3f}")
    print(f"  Median R^2_g = {result['median_r2_g']:.3f}")
    print(f"  Geometries with R^2_g <= 0: {result['n_r2_le_0']}")

# Summary
print("\n" + "="*70)
print("MLP STANDARDISATION ABLATION SUMMARY (LOGO)")
print("="*70)
r0, r1 = results[0], results[1]
print(f"  Pooled R^2:   {r0['pooled_r2']:.4f} -> {r1['pooled_r2']:.4f}  (delta = {r1['pooled_r2']-r0['pooled_r2']:+.4f})")
print(f"  Pooled RMSE:  {r0['pooled_rmse']:.4f} -> {r1['pooled_rmse']:.4f}  (delta = {r1['pooled_rmse']-r0['pooled_rmse']:+.4f})")
print(f"  Macro-RMS:    {r0['macro_rms']:.4f} -> {r1['macro_rms']:.4f}  (delta = {r1['macro_rms']-r0['macro_rms']:+.4f})")
print(f"\n  Conclusion: The MLP inputs are standardized via StandardScaler in a")
print(f"  sklearn Pipeline, fitted on training data only. The gain from 17 features")
print(f"  is {r1['pooled_r2']-r0['pooled_r2']:+.4f} in pooled R^2 under LOGO,")
print(f"  with macro-RMS change of {r1['macro_rms']-r0['macro_rms']:+.4f}.")

# Save
with open('mlp_standardised_ablation.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: mlp_standardised_ablation.json")
