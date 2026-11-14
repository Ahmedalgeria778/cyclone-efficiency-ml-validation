import pickle,time,sys
from models import *
df,_=load(); gids=sorted(df.gid.unique())
FS=['raw','raw+log','raw+pivc','raw+sep+geo','full14']
MODELS=['GBR','XGB','LGBM','CatBoost','RF','HGB']
rows=[]; pars={}
t0=time.time()
for g in gids:
    te=df[df.gid==g]; tr=df[df.gid!=g]
    par=calibrate_pivc(tr); parm=calibrate_pivc(tr,fit_m=True); pars[g]=(par,parm)
    for fs in FS:
        for mn in MODELS:
            p=fit_predict(mn,tr,te,fs,par)
            rows.append(pd.DataFrame(dict(gid=g,idx=te.index,y=te.eta.values,pred=p,fset=fs,model=mn)))
    for fs in ['raw','full14']:
        p=fit_predict('MLP',tr,te,fs,par)
        rows.append(pd.DataFrame(dict(gid=g,idx=te.index,y=te.eta.values,pred=p,fset=fs,model='MLP')))
    p=fit_predict('GBR',tr,te,'full14',parm)   # PiVC with fitted Re exponent
    rows.append(pd.DataFrame(dict(gid=g,idx=te.index,y=te.eta.values,pred=p,fset='full14_m',model='GBR')))
    print(g,round(time.time()-t0),flush=True)
pd.concat(rows).to_pickle('logo_preds.pkl'); pickle.dump(pars,open('logo_pars.pkl','wb'))
