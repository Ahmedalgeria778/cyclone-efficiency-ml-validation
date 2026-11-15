from models import *
df,_=load(); P=pd.read_pickle('logo_preds.pkl')
R=P[(P.fset=='full17')&(P.model=='GBR')].set_index('idx'); Rr=P[(P.fset=='raw')&(P.model=='GBR')].set_index('idx')
edges=[0,2.3e4,5.15e4,2.5e5,1e7]; labs=['Q1','Q2','Q3','Q4']
df['band']=pd.cut(df.Re,edges,labels=labs)
rows=[];rows2=[]
for q in labs:
    sub=df[df.band==q]; d=R.loc[sub.index]; m=metrics(d.y,d.pred)
    rows.append(dict(band=q,Re_lo=sub.Re.min(),Re_hi=sub.Re.max(),n=len(sub),n_geom=sub.gid.nunique(),n_cond=sub.cond.nunique(),eta_std=sub.eta.std(),frac_trans=((sub.eta>0.2)&(sub.eta<0.8)).mean(),rr_med=sub.rr.median(),**m))
    tr=df[df.band!=q]; par=calibrate_pivc(tr)
    for fs in ['raw','full17']:
        p=fit_predict('GBR',tr,sub,fs,par); rows2.append(dict(band=q,fset=fs,**metrics(sub.eta,p)))
A=pd.DataFrame(rows);B=pd.DataFrame(rows2); A.to_csv('re_bands.csv',index=False); B.to_csv('leave_Re_band_out.csv',index=False)
print(A.round(3).to_string()); print(B.round(3).to_string())
# geometries by band
print(df.groupby('band').gid.apply(lambda x: sorted(x.unique())))
