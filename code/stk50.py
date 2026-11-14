from common import *
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit, brentq
df,_=load()
SIG=0.025
def extract(s,e):
    """return (stk50, mode) mode: 'interp' if data span 0.5, 'fit' if fitted exponential model inverted, None if failed"""
    o=np.argsort(s); s=s[o]; e=np.maximum.accumulate(e[o])
    # collapse duplicate Stk
    us,ix=np.unique(s,return_index=True)
    if len(us)>=3 and e.min()<0.5<e.max():
        ls=np.log(us); ee=np.array([e[s==u].mean() for u in us]); ee=np.maximum.accumulate(ee)
        # strictly increasing for pchip inverse: invert eta->logStk
        ek,ik=np.unique(ee,return_index=True)
        if len(ek)>=3 and ek.min()<0.5<ek.max():
            try: return float(np.exp(PchipInterpolator(ek,ls[ik])(0.5))),'interp'
            except Exception: pass
    if len(us)>=4:
        f=lambda x,A,k: 1-np.exp(-A*x**k)
        try:
            p,_=curve_fit(f,s,e,p0=[50,0.7],bounds=([1e-6,0.05],[1e9,6]),maxfev=20000)
            r=np.sum((f(s,*p)-e)**2)
            if r/len(s)<0.05**2: return float((np.log(2)/p[0])**(1/p[1])),'fit'
        except Exception: pass
    return np.nan,None
rows=[]; rng=np.random.RandomState(0)
for c,d in df.groupby('cond'):
    s=d.Stk.values; e=d.eta.values
    v,mode=extract(s,e)
    if mode is None: continue
    bs=[]
    for _ in range(1000):
        eb=np.clip(e+rng.normal(0,SIG,len(e)),0,1); vb,_=extract(s,eb)
        if np.isfinite(vb): bs.append(np.log(vb))
    if len(bs)<500: continue
    r=d.iloc[0]
    rows.append(dict(cond=c,gid=r.gid,De=r.De,a=r.a,b=r.b,Re=r.Re,rr=r.rr,n=len(d),stk50=v,mode=mode,sd_ln=np.std(bs),cv=np.std(np.exp(bs))/np.mean(np.exp(bs))))
S=pd.DataFrame(rows); S.to_csv('stk50_table.csv',index=False)
print(len(S),'conditions with Stk50 of',df.cond.nunique(), S['mode'].value_counts().to_dict(), 'median CV',S.cv.median())
def wls(X,y,w):
    W=np.sqrt(w); beta=np.linalg.lstsq(X*W[:,None],y*W,rcond=None)[0]; return beta
S['y']=np.log(S.stk50); S['lnRe']=np.log(S.Re); S['w']=1/np.maximum(S.sd_ln,0.02)**2
def design(S,kind):
    one=np.ones(len(S))
    if kind=='pooled': return np.c_[S.lnRe,one]
    if kind=='geom_covariates': return np.c_[S.lnRe,np.log(S.De),np.log(S.a*S.b),np.log(S.rr),one]
    if kind=='fixed_effects':
        G=pd.get_dummies(S.gid).values.astype(float); return np.c_[S.lnRe,G]
res={}
gl=S.gid.unique()
for kind in ['pooled','geom_covariates','fixed_effects']:
    sub=S
    if kind=='fixed_effects':
        k=S.groupby('gid').Re.nunique(); sub=S[S.gid.isin(k[k>=2].index)]
    X=design(sub,kind); b=wls(X,sub.y.values,sub.w.values)[0]
    bs=[]
    gs=sub.gid.unique()
    for _ in range(2000):
        pick=rng.choice(gs,len(gs),replace=True)
        bb=pd.concat([sub[sub.gid==g].assign(gid=i) for i,g in enumerate(pick)])
        try:
            Xb=design(bb,kind); bs.append(wls(Xb,bb.y.values,bb.w.values)[0])
        except Exception: pass
    bs=np.array(bs)
    # unweighted version too
    bu=wls(X,sub.y.values,np.ones(len(sub)))[0]
    res[kind]=dict(alpha=float(b),se_cluster=float(np.std(bs)),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],alpha_unweighted=float(bu),n=int(len(sub)),n_geom=int(sub.gid.nunique()))
    print(kind,res[kind])
import json; json.dump(res,open('stk50_reg.json','w'),indent=1)
# how much of ln Stk50 variance is between-geometry vs within
print('ln Re range within geometry (median ratio max/min):',S.groupby('gid').Re.agg(lambda x:x.max()/x.min()).median())
print('corr(lnRe, geometry mean lnRe)',np.corrcoef(S.lnRe,S.groupby('gid').lnRe.transform('mean'))[0,1])
