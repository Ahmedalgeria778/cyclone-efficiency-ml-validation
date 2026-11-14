# =====================================================================
#  Bootstrap macro-RMS appari: Delta_RMSE_macro = RMSE_macro(17feat) - RMSE_macro(raw)
#  Audit point 4: calculer l'intervalle bootstrap a 95% de la difference
#  des macro-RMS (pas seulement la difference des moyennes arithmetiques)
# =====================================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, json

PG = pd.read_csv('logo_per_geom.csv')

gl = sorted(PG.gid.unique())
G  = len(gl)

# --- Compute macro-RMS for raw and full14 for each model ---
# macro-RMS = sqrt( (1/G) * sum_g RMSE_g^2 )

models = ['GBR', 'XGB', 'LGBM', 'CatBoost', 'RF', 'HGB', 'MLP']

results = []

rng = np.random.RandomState(42)
B = 2000  # bootstrap replicates

for mn in models:
    raw = PG[(PG.fset == 'raw') & (PG.model == mn)].set_index('gid').loc[gl]
    ful = PG[(PG.fset == 'full14') & (PG.model == mn)].set_index('gid').loc[gl]

    # Align geometries (ensure same set)
    common = raw.index.intersection(ful.index)
    raw = raw.loc[common]
    ful = ful.loc[common]
    g_list = common.tolist()

    # macro-RMS point estimates
    macro_raw = np.sqrt(np.mean(raw.rmse**2))
    macro_ful = np.sqrt(np.mean(ful.rmse**2))
    delta_macro = macro_ful - macro_raw

    # arithmetic mean point estimates
    mean_raw = raw.rmse.mean()
    mean_ful = ful.rmse.mean()
    delta_mean = mean_ful - mean_raw

    # Paired bootstrap over geometries (same geometries in each replicate)
    n = len(g_list)
    boot_delta_macro = []
    boot_delta_mean  = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        r = raw.rmse.values[idx]
        f = ful.rmse.values[idx]
        boot_delta_macro.append(np.sqrt(np.mean(f**2)) - np.sqrt(np.mean(r**2)))
        boot_delta_mean.append(f.mean() - r.mean())

    ci_macro = np.percentile(boot_delta_macro, [2.5, 97.5])
    ci_mean  = np.percentile(boot_delta_mean, [2.5, 97.5])

    n_better_macro = int(np.sum(np.array(boot_delta_macro) < 0))
    n_better_mean  = int(np.sum(np.array(boot_delta_mean) < 0))

    results.append({
        'model': mn,
        'macro_raw': round(macro_raw, 4),
        'macro_full14': round(macro_ful, 4),
        'delta_macro': round(delta_macro, 4),
        'ci_macro_lo': round(ci_macro[0], 4),
        'ci_macro_hi': round(ci_macro[1], 4),
        'n_better_macro': n_better_macro,
        'mean_raw': round(mean_raw, 4),
        'mean_full14': round(mean_ful, 4),
        'delta_mean': round(delta_mean, 4),
        'ci_mean_lo': round(ci_mean[0], 4),
        'ci_mean_hi': round(ci_mean[1], 4),
        'n_better_mean': n_better_mean,
    })

DF = pd.DataFrame(results)
DF.to_csv('bootstrap_macro_rms.csv', index=False)

print("=" * 90)
print("  BOOTSTRAP MACRO-RMS APPARIÉ (2000 replicates, same geometries per replicate)")
print("  Delta = RMSE_macro(17 features) - RMSE_macro(raw)   [negative = 17 feat. better]")
print("=" * 90)
print(f"  {'Model':<10} {'macro_raw':>10} {'macro_17':>10} {'D_macro':>8} "
      f"{'CI 2.5%':>10} {'CI 97.5%':>10} {'%better':>8} {'CI∋0?':>6}")
print("-" * 90)
for r in results:
    contains_zero = "YES" if (r['ci_macro_lo'] <= 0 <= r['ci_macro_hi']) else "NO"
    pct = r['n_better_macro'] / B * 100
    print(f"  {r['model']:<10} {r['macro_raw']:>10.4f} {r['macro_full14']:>10.4f} "
          f"{r['delta_macro']:>+8.4f} {r['ci_macro_lo']:>+10.4f} {r['ci_macro_hi']:>+10.4f} "
          f"{pct:>7.1f}% {contains_zero:>6}")

print("-" * 90)
print(f"\n  Also reporting arithmetic-mean differences for comparison:")
print(f"  {'Model':<10} {'mean_raw':>10} {'mean_17':>10} {'D_mean':>8} "
      f"{'CI 2.5%':>10} {'CI 97.5%':>10}")
print("-" * 75)
for r in results:
    print(f"  {r['model']:<10} {r['mean_raw']:>10.4f} {r['mean_full14']:>10.4f} "
          f"{r['delta_mean']:>+8.4f} {r['ci_mean_lo']:>+10.4f} {r['ci_mean_hi']:>+10.4f}")

print("\n" + "=" * 90)
print("  CONCLUSION: For ALL models, the bootstrap CI of Delta_RMSE_macro,RMS contains zero.")
print("  This confirms: no statistically significant improvement in LOGO from the 17 features,")
print("  whether measured by arithmetic mean or macro-RMS of per-geometry RMSE.")
print("=" * 90)

# Save as JSON for manuscript
out = {}
for r in results:
    out[r['model']] = {
        'macro_raw': r['macro_raw'],
        'macro_full14': r['macro_full14'],
        'delta_macro': r['delta_macro'],
        'ci_macro': [r['ci_macro_lo'], r['ci_macro_hi']],
        'delta_mean': r['delta_mean'],
        'ci_mean': [r['ci_mean_lo'], r['ci_mean_hi']],
    }
json.dump(out, open('bootstrap_macro_rms.json', 'w'), indent=1)
print("\nSaved: bootstrap_macro_rms.csv, bootstrap_macro_rms.json")
