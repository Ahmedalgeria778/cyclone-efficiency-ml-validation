import time,json
from models import *
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
df,_=load(); gids=np.array(sorted(df.gid.unique()))
def lapple(d,N=5.0):
    s50=d.b.values/(4*np.pi*N); return 1/(1+(s50/d.Stk.values)**2)
def fit_lapple_N(tr):
    f=lambda N: np.mean((lapple(tr,N)-tr.eta.values)**2)
    Ns=np.linspace(0.2,40,400); return Ns[np.argmin([f(n) for n in Ns])]
def stk_only(tr,te):
    f=lambda s,A,al: 1-np.exp(-A*s**al)
    p,_=curve_fit(f,tr.Stk.values,tr.eta.values,p0=[10,0.5],maxfev=20000); return np.clip(f(te.Stk.values,*p),0,1)
rows=[]
t0=time.time()
# ---- A: repeated grouped 23/5 splits, fixed HP (identical budget), raw vs full17
for rep in range(20):
    rng=np.random.RandomState(100+rep); te_g=rng.choice(gids,5,replace=False)
    te=df[df.gid.isin(te_g)]; tr=df[~df.gid.isin(te_g)]
    par=calibrate_pivc(tr)
    for nm,p in [('Lapple(N=5)',lapple(te)),('Lapple(N fitted)',lapple(te,fit_lapple_N(tr))),('Stk-only fit',stk_only(tr,te))]:
        rows.append(dict(protocol='fixed',rep=rep,model=nm,fset='-',**metrics(te.eta,p),n=len(te)))
    for fs in ['raw','full17']:
        for mn in ['GBR','XGB','LGBM','CatBoost','RF','HGB','MLP']:
            p=fit_predict(mn,tr,te,fs,par)
            rows.append(dict(protocol='fixed',rep=rep,model=mn,fset=fs,**metrics(te.eta,p),n=len(te)))
    print('A',rep,round(time.time()-t0),flush=True)
pd.DataFrame(rows).to_csv('splits_fixed.csv',index=False)
# ---- B: nested tuning (inner 3-fold GroupKFold on training geometries), 10 splits
rows=[]
for rep in range(10):
    rng=np.random.RandomState(100+rep); te_g=rng.choice(gids,5,replace=False)
    te=df[df.gid.isin(te_g)]; tr=df[~df.gid.isin(te_g)].reset_index(drop=True)
    par=calibrate_pivc(tr)
    for fs in ['raw','full17']:
        for mn in ['GBR','XGB','LGBM','CatBoost','RF','HGB']:
            best=None
            for hp in GRID[mn]:
                sc=[]
                for itr,iva in GroupKFold(3).split(tr,groups=tr.gid):
                    a=tr.iloc[itr];b=tr.iloc[iva]; pa=calibrate_pivc(a) if fs=='full17' else None
                    p=fit_predict(mn,a,b,fs,pa,hp); sc.append(np.mean((p-b.eta.values)**2))
                m=np.mean(sc)
                if best is None or m<best[0]: best=(m,hp)
            p=fit_predict(mn,tr,te,fs,par,best[1])
            rows.append(dict(protocol='tuned',rep=rep,model=mn,fset=fs,**metrics(te.eta,p),n=len(te)))
    print('B',rep,round(time.time()-t0),flush=True)
    pd.DataFrame(rows).to_csv('splits_tuned.csv',index=False)
# ---- C: random point-wise 80/20 (the protocol used by Bregolin et al.)
rows=[]
for rep in range(20):
    rng=np.random.RandomState(200+rep); idx=rng.permutation(len(df)); k=int(.8*len(df))
    tr=df.iloc[idx[:k]]; te=df.iloc[idx[k:]]; par=calibrate_pivc(tr)
    for fs,mn in [('raw','GBR'),('full17','GBR'),('raw','MLP'),('full17','MLP'),('raw','CatBoost')]:
        p=fit_predict(mn,tr,te,fs,par); rows.append(dict(protocol='random_pointwise',rep=rep,model=mn,fset=fs,**metrics(te.eta,p),n=len(te)))
pd.DataFrame(rows).to_csv('splits_random.csv',index=False)
# ---- D: L2GO (60 random pairs), GBR raw vs full17
rows=[]
rng=np.random.RandomState(7); allp=[(i,j) for i in gids for j in gids if i<j]
sel=[allp[k] for k in rng.choice(len(allp),60,replace=False)]
for (i,j) in sel:
    te=df[df.gid.isin([i,j])]; tr=df[~df.gid.isin([i,j])]; par=calibrate_pivc(tr)
    for fs in ['raw','full17']:
        p=fit_predict('GBR',tr,te,fs,par)
        for g in (i,j):
            m=te.gid.values==g; rows.append(dict(pair=f'{i}-{j}',gid=g,fset=fs,y=te.eta.values[m],pred=p[m]))
pd.to_pickle(pd.DataFrame(rows),'l2go.pkl')
print('done',round(time.time()-t0))
