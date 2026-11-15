import json
from models import *
from scipy.stats import spearmanr, rankdata
df,_=load(); P=pd.read_pickle('logo_preds.pkl'); out={}
gsz=df.groupby('gid').size()
def per_geom(P):
    r=[]
    for (fs,mn,g),d in P.groupby(['fset','model','gid']):
        r.append(dict(fset=fs,model=mn,gid=g,n=len(d),**metrics(d.y,d.pred)))
    return pd.DataFrame(r)
PG=per_geom(P); PG.to_csv('logo_per_geom.csv',index=False)
summ=[]
for (fs,mn),d in PG.groupby(['fset','model']):
    w=d.n/d.n.sum(); wm=(w*d.r2).sum(); wsd=np.sqrt((w*(d.r2-wm)**2).sum())
    sub=P[(P.fset==fs)&(P.model==mn)]; pm=metrics(sub.y,sub.pred)
    summ.append(dict(fset=fs,model=mn,r2_w=wm,r2_sd=wsd,r2_med=d.r2.median(),n_good=(d.r2>0.5).sum(),n_marg=((d.r2>0)&(d.r2<=0.5)).sum(),n_fail=(d.r2<=0).sum(),
                     pooled_r2=pm['r2'],pooled_rmse=pm['rmse'],pooled_mae=pm['mae'],mean_geom_rmse=d.rmse.mean(),median_geom_rmse=d.rmse.median()))
SM=pd.DataFrame(summ); SM.to_csv('logo_summary.csv',index=False); print(SM.round(3).to_string())
# --- paired ablation significance (per-geometry RMSE, raw vs full17) via Wilcoxon + bootstrap over geometries
from scipy.stats import wilcoxon
abl=[]
for mn in ['GBR','XGB','LGBM','CatBoost','RF','HGB','MLP']:
    a=PG[(PG.fset=='raw')&(PG.model==mn)].set_index('gid').rmse; b=PG[(PG.fset=='full17')&(PG.model==mn)].set_index('gid').rmse
    dlt=(b-a); rng=np.random.RandomState(0); bs=[dlt.sample(len(dlt),replace=True,random_state=rng).mean() for _ in range(2000)]
    try: pw=wilcoxon(a,b).pvalue
    except Exception: pw=np.nan
    abl.append(dict(model=mn,drmse_mean=dlt.mean(),ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),n_better=int((dlt<0).sum()),p_wilcoxon=pw))
AB=pd.DataFrame(abl); AB.to_csv('logo_ablation_paired.csv',index=False); print(AB.round(4).to_string())
# --- Mahalanobis (geometry space), distances relative to the 27 training geometries
G=df.groupby('gid')[['De','a','b']].first(); gl=G.index.values
rows=[]
for g in gl:
    tr=G.drop(g).values; x=G.loc[g].values; mu=tr.mean(0); S=np.cov(tr.T)
    dm=lambda v: np.sqrt((v-mu)@np.linalg.inv(S)@(v-mu))
    dtr=np.array([dm(v) for v in tr]); thr=np.percentile(dtr,95)
    sd=tr.std(0); knn=np.sort(np.sqrt((((tr-x)/sd)**2).sum(1)))[:3].mean()
    rows.append(dict(gid=g,d_mahal=dm(x),thr95=thr,knn3=knn))
MD=pd.DataFrame(rows).set_index('gid')
ref=PG[(PG.fset=='full17')&(PG.model=='GBR')].set_index('gid')
MD['r2']=ref.r2; MD['rmse']=ref.rmse; MD['n']=ref.n; MD['flag']=MD.d_mahal>MD.thr95
MD.to_csv('mahal.csv'); print(MD.round(3).to_string())
def perm_p(x,y,B=5000):
    r0=spearmanr(x,y)[0]; rng=np.random.RandomState(1); c=0
    for _ in range(B):
        if abs(spearmanr(x,rng.permutation(y))[0])>=abs(r0): c+=1
    return r0,(c+1)/(B+1)
for nm,col in [('d_mahal','rmse'),('knn3','rmse'),('d_mahal','r2'),('knn3','r2'),('n','rmse')]:
    r,p=perm_p(MD[nm].values,MD[col].values); print('spearman',nm,col,round(r,3),'p=',round(p,4)); out[f'sp_{nm}_{col}']=(r,p)
from sklearn.metrics import roc_auc_score
fail=(MD.rmse>MD.rmse.quantile(0.75)).astype(int)  # worst quartile by RMSE
print('AUC d_mahal vs worst-quartile RMSE',roc_auc_score(fail,MD.d_mahal),' knn3',roc_auc_score(fail,MD.knn3))
rng=np.random.RandomState(3); aucs=[]
for _ in range(2000):
    i=rng.choice(len(MD),len(MD)); 
    if fail.values[i].min()==fail.values[i].max(): continue
    aucs.append(roc_auc_score(fail.values[i],MD.d_mahal.values[i]))
print('AUC bootstrap CI',np.percentile(aucs,[2.5,97.5])); out['auc']=(roc_auc_score(fail,MD.d_mahal),*np.percentile(aucs,[2.5,97.5]))
print('flagged',MD.flag.sum(),' of which poor(R2<=0.5):',(MD.flag&(MD.r2<=0.5)).sum(),' poor total',(MD.r2<=0.5).sum())
# --- Uncertainty: quadratic sigma(eta_hat) fitted on OOF residuals of the other 27 geometries
R=P[(P.fset=='full17')&(P.model=='GBR')].copy(); R['res']=R.y-R.pred
def sig_model(Rtr,nb=10):
    bins=np.linspace(0,1,nb+1); ix=np.clip(np.digitize(Rtr.pred,bins)-1,0,nb-1); xs=[];ss=[]
    for k in range(nb):
        m=ix==k
        if m.sum()>=8: xs.append(Rtr.pred.values[m].mean()); ss.append(Rtr.res.values[m].std())
    c=np.polyfit(xs,ss,2); return lambda e: np.maximum(np.polyval(c,e),0.01)
cov=[]; z=[]
for g in gl:
    f=sig_model(R[R.gid!=g]); t=R[R.gid==g]; s=f(t.pred.values); zz=np.abs(t.res.values)/s
    for _,(zi,ri,pi) in enumerate(zip(zz,t.res.values,t.pred.values)): z.append(dict(gid=g,z=zi))
Z=pd.DataFrame(z); Z=Z.merge(MD[['d_mahal','thr95','flag']],left_on='gid',right_index=True)
for k,nom in [(1,.683),(2,.954),(3,.997)]:
    print(f'coverage k={k}: pooled {np.mean(Z.z<=k):.3f} (nominal {nom}); per-geom mean {Z.groupby("gid").z.apply(lambda x:np.mean(x<=k)).mean():.3f}; flagged {np.mean(Z[Z.flag].z<=k):.3f} unflagged {np.mean(Z[~Z.flag].z<=k):.3f}')
    out[f'cov{k}']=(float(np.mean(Z.z<=k)),float(np.mean(Z[Z.flag].z<=k)) if Z.flag.any() else None,float(np.mean(Z[~Z.flag].z<=k)))
q_all=np.quantile(Z.z,0.95); q_fl=np.quantile(Z[Z.flag].z,0.95) if Z.flag.any() else np.nan; q_un=np.quantile(Z[~Z.flag].z,0.95)
print('95% quantile of |z| all',q_all,'unflagged',q_un,'flagged',q_fl,' => inflation needed vs 1.96:',q_all/1.96,q_fl/1.96)
# group-wise (geometry-clustered) conformal alternative: scale-normalised, quantile from other geometries
cv=[]
for g in gl:
    f=sig_model(R[R.gid!=g]); tr=R[R.gid!=g]; ztr=np.abs(tr.res.values)/f(tr.pred.values)
    q=np.quantile(ztr,0.95); t=R[R.gid==g]; zz=np.abs(t.res.values)/f(t.pred.values); cv.append(dict(gid=g,cov=np.mean(zz<=q),q=q,n=len(t)))
CV=pd.DataFrame(cv); print('conformal-normalised 95%: pooled cov',(CV['cov']*CV.n).sum()/CV.n.sum(),'per-geom mean',CV['cov'].mean(),'min',CV['cov'].min(),'q mean',CV.q.mean())
out['conf']=((CV['cov']*CV.n).sum()/CV.n.sum(),CV['cov'].mean(),CV.q.mean()); CV.to_csv('conformal_cov.csv')
# Breusch-Pagan on OOF residuals vs eta_hat
import numpy.linalg as la
e2=R.res.values**2; X=np.c_[np.ones(len(R)),R.pred.values]; b=la.lstsq(X,e2,rcond=None)[0]; fit=X@b
r2=1-((e2-fit)**2).sum()/((e2-e2.mean())**2).sum(); LM=len(R)*r2
from scipy.stats import chi2; print('BP LM',LM,'p',chi2.sf(LM,1))
X2=np.c_[np.ones(len(R)),R.pred.values,R.pred.values**2]; b2=la.lstsq(X2,e2,rcond=None)[0]; f2=X2@b2
r22=1-((e2-f2)**2).sum()/((e2-e2.mean())**2).sum(); print('BP quad LM',len(R)*r22,'p',chi2.sf(len(R)*r22,2)); out['bp']=(LM,chi2.sf(LM,1),len(R)*r22,chi2.sf(len(R)*r22,2))
fq=sig_model(R); c=np.polyfit(np.linspace(0,1,11),fq(np.linspace(0,1,11)),2); print('quad coeffs (c2,c1,c0)',c, ' sigma(0.5)=',fq(0.5))
out['sig_coeff']=list(c)
# --- Re-quartile stratified OOF metrics + composition
df['Req']=pd.qcut(df.Re,4,labels=['Q1','Q2','Q3','Q4']); R=R.join(df[['Req','Re','gid']].rename(columns={'gid':'g2'}))
tabq=[]
for q,d in R.groupby('Req'):
    m=metrics(d.y,d.pred); sub=df[df.Req==q]; tabq.append(dict(q=q,Re_lo=sub.Re.min(),Re_hi=sub.Re.max(),n=len(d),n_geom=sub.gid.nunique(),n_cond=sub.cond.nunique(),eta_std=d.y.std(),frac_transition=((d.y>0.2)&(d.y<0.8)).mean(),**m))
TQ=pd.DataFrame(tabq); TQ.to_csv('re_quartile.csv',index=False); print(TQ.round(3).to_string())
# leave-Re-quartile-out (extrapolation in Re), GBR raw vs full17
rows=[]
for q in ['Q1','Q2','Q3','Q4']:
    te=df[df.Req==q]; tr=df[df.Req!=q]; par=calibrate_pivc(tr)
    for fs in ['raw','full17']:
        p=fit_predict('GBR',tr,te,fs,par); rows.append(dict(q=q,fset=fs,**metrics(te.eta,p)))
LQ=pd.DataFrame(rows); LQ.to_csv('leave_Re_out.csv',index=False); print(LQ.round(3).to_string())
# PiVC constants across folds
pars=pd.read_pickle('logo_pars.pkl'); pa=np.array([pars[g][0] for g in gl]); pm=np.array([pars[g][1] for g in gl])
print('PiVC (m=-0.5) a,b,c across folds: median',np.median(pa,0),'IQR',np.percentile(pa,[25,75],axis=0))
print('PiVC fitted m: median',np.median(pm[:,3]),'range',pm[:,3].min(),pm[:,3].max(),' a med',np.median(pm[:,0]))
out['pivc_m']=(float(np.median(pm[:,3])),float(pm[:,3].min()),float(pm[:,3].max()))
json.dump(out,open('logo_out.json','w'),indent=1,default=float)
