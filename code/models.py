from common import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb
HP0={ # identical-protocol fixed hyper-parameters (same budget of 400 trees, depth 4, lr 0.05 for all boosters)
 'GBR':dict(n_estimators=400,learning_rate=0.05,max_depth=4,subsample=0.85),
 'XGB':dict(n_estimators=400,learning_rate=0.05,max_depth=4,subsample=0.85),
 'LGBM':dict(n_estimators=400,learning_rate=0.05,max_depth=4,subsample=0.85),
 'CatBoost':dict(iterations=400,learning_rate=0.05,depth=4),
 'RF':dict(n_estimators=300,min_samples_leaf=2),
 'HGB':dict(max_iter=300,learning_rate=0.05,max_depth=4),
 'MLP':dict(alpha=1e-3),
}
GRID={ # small random-search grids, identical budget (4 configs) for every model
 'GBR':[dict(n_estimators=n,learning_rate=l,max_depth=d,subsample=0.85) for n,l,d in [(300,0.05,3),(300,0.05,5),(500,0.03,4),(200,0.1,3)]],
 'XGB':[dict(n_estimators=n,learning_rate=l,max_depth=d,subsample=0.85) for n,l,d in [(300,0.05,3),(300,0.05,5),(500,0.03,4),(200,0.1,3)]],
 'LGBM':[dict(n_estimators=n,learning_rate=l,max_depth=d,subsample=0.85) for n,l,d in [(300,0.05,3),(300,0.05,5),(500,0.03,4),(200,0.1,3)]],
 'CatBoost':[dict(iterations=n,learning_rate=l,depth=d) for n,l,d in [(300,0.05,3),(300,0.05,5),(500,0.03,4),(200,0.1,3)]],
 'RF':[dict(n_estimators=250,min_samples_leaf=m,max_features=f) for m,f in [(1,1.0),(2,0.6),(3,1.0),(5,0.6)]],
 'HGB':[dict(max_iter=n,learning_rate=l,max_depth=d) for n,l,d in [(300,0.05,3),(300,0.05,5),(500,0.03,4),(200,0.1,3)]],
}
def make(name,hp=None,seed=0):
    hp=dict(HP0[name] if hp is None else hp)
    if name=='GBR': return GradientBoostingRegressor(random_state=seed,**hp)
    if name=='XGB': return xgb.XGBRegressor(n_jobs=1,verbosity=0,random_state=seed,**hp)
    if name=='LGBM': return lgb.LGBMRegressor(subsample_freq=1,n_jobs=1,verbose=-1,random_state=seed,**hp)
    if name=='CatBoost': return cb.CatBoostRegressor(verbose=0,thread_count=1,random_seed=seed,**hp)
    if name=='RF': return RandomForestRegressor(n_jobs=1,random_state=seed,**hp)
    if name=='HGB': return HistGradientBoostingRegressor(random_state=seed,**hp)
    if name=='MLP': # Bregolin-type BPNN: 3x15 hidden, ReLU, Adam lr .008, batch 5, 100 epochs
        return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(15,15,15),activation='relu',solver='adam',learning_rate_init=0.008,batch_size=5,max_iter=100,tol=0,n_iter_no_change=1000,random_state=seed,**hp))
def fit_predict(name,tr,te,fset,par,hp=None,seed=0):
    m=make(name,hp,seed); m.fit(feats(tr,fset,par),tr.eta); return np.clip(m.predict(feats(te,fset,par)),0,1)
