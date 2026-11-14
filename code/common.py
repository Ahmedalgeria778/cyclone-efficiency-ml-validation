import numpy as np, pandas as pd, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
RAW=['De','a','b','Re','Stk','Dp','rr']

def load():
    import os
    _here=os.path.dirname(os.path.abspath(__file__))
    df=pd.read_csv(os.path.join(_here,'Dataset training_validation_testing BPNN cyclone.csv'))
    df.columns=['De','a','b','Re','Stk','Dp','rr','eta']
    n0=len(df); df=df.drop_duplicates().reset_index(drop=True)
    df['gid']=df.groupby(['De','a','b']).ngroup()
    df['cond']=df.groupby(['De','a','b','Re','rr']).ngroup()
    return df,n0

def pivc(Stk,Re,rr,a,b,c,m=-0.5):
    return 1-np.exp(-a*Stk*Re**m/(1+b*(rr/1000.)**c))

def calibrate_pivc(d, fit_m=False):
    """fit (a,b,c[,m]) on TRAINING data only: minimise MSE between PiVC and eta"""
    def res(p):
        a,b,c=p[:3]; m=p[3] if fit_m else -0.5
        return pivc(d.Stk.values,d.Re.values,d.rr.values,a,b,c,m)-d.eta.values
    lo=[1e-3,0,0.1]+([-1.5] if fit_m else []); hi=[1e9,50,6]+([0.5] if fit_m else [])
    best=None
    for la in [1e2,1e3,1e4,1e5]:
        x0=[la,0.3,2.0]+([-0.5] if fit_m else [])
        r=least_squares(res,x0,bounds=(lo,hi),max_nfev=300,x_scale=[la,1,1]+([1] if fit_m else []))
        if best is None or r.cost<best.cost: best=r
    return best.x

def feats(d,fset,par=None):
    X=pd.DataFrame(index=d.index)
    for k in ['De','a','b','Re','Stk','Dp','rr']: X[k]=d[k]
    if fset=='raw': return X
    L=pd.DataFrame(index=d.index)
    for k in ['Re','Stk','Dp','rr']: L['ln'+k]=np.log(d[k])
    sep=d.Stk/np.sqrt(d.Re); geo=d.a*d.b/d.De**2
    if fset=='raw+log': return pd.concat([X,L],axis=1)
    if fset=='raw+pivc':
        X['PiVC']=pivc(d.Stk.values,d.Re.values,d.rr.values,*par[:3],(par[3] if len(par)>3 else -0.5)); return X
    if fset=='raw+sep+geo':
        X['sep']=sep; X['lnsep']=np.log(sep); X['geo']=geo; X['Stk_geo']=d.Stk*geo; X['Re_De']=d.Re*d.De; return X
    if fset=='full14':
        X=pd.concat([X,L],axis=1)
        X['PiVC']=pivc(d.Stk.values,d.Re.values,d.rr.values,*par[:3],(par[3] if len(par)>3 else -0.5))
        X['sep']=sep; X['lnsep']=np.log(sep); X['geo']=geo; X['Stk_geo']=d.Stk*geo; X['Re_De']=d.Re*d.De
        return X
    raise ValueError(fset)

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    ss=((y-p)**2).sum(); st=((y-y.mean())**2).sum()
    return dict(r2=1-ss/st if st>0 else np.nan, rmse=np.sqrt(np.mean((y-p)**2)), mae=np.mean(np.abs(y-p)))
