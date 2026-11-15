import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PG = pd.read_csv('logo_per_geom.csv')
gl = sorted(PG.gid.unique())

# Also load pooled predictions for pooled RMSE
try:
    P = pd.read_pickle('logo_preds.pkl')
except:
    P = None

print("Verification Table 6 (LOGO macro-RMS vs pooled vs arithmetic mean)")
print(f"  {'Model':<10} {'feat':>5} {'pooled_R2':>10} {'pooled_RMSE':>12} {'macro_RMS':>10} {'arith_mean':>10}")
print("-" * 65)

for mn in ['GBR','XGB','LGBM','CatBoost','RF','HGB','MLP']:
    for fs in ['raw','full17']:
        sub = PG[(PG.fset==fs) & (PG.model==mn)].set_index('gid').loc[gl]
        macro = np.sqrt(np.mean(sub.rmse**2))
        arith = sub.rmse.mean()
        
        # Pooled from pickle if available
        if P is not None:
            pred = P[(P.fset==fs) & (P.model==mn)]
            ss = ((pred.y - pred.pred)**2).sum()
            st = ((pred.y - pred.y.mean())**2).sum()
            r2_pool = 1 - ss/st if st > 0 else np.nan
            rmse_pool = np.sqrt(np.mean((pred.y - pred.pred)**2))
        else:
            r2_pool = np.nan
            rmse_pool = np.nan
        
        print(f"  {mn:<10} {fs:>5} {r2_pool:>10.4f} {rmse_pool:>12.4f} {macro:>10.4f} {arith:>10.4f}")
    print()
