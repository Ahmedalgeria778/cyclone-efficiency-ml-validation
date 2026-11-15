import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.size':9,'figure.dpi':200,'axes.grid':True,'grid.alpha':.3})
df,_=load(); P=pd.read_pickle('logo_preds.pkl'); MD=pd.read_csv('mahal.csv',index_col=0)
S=pd.read_csv('stk50_table.csv'); reg=json.load(open('stk50_reg.json'))
# Fig A: ablation (repeated grouped splits)
F=pd.read_csv('splits_fixed.csv'); M=F[F.fset!='-'].groupby(['model','fset']).r2.agg(['mean','std']).unstack()
ms=['GBR','XGB','LGBM','CatBoost','RF','HGB','MLP']; x=np.arange(len(ms)); w=.38
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for k,(fs,lab) in enumerate([('raw','7 raw inputs'),('full17','17 features')]):
    ax[0].bar(x+(k-.5)*w,[M.loc[m,('mean',fs)] for m in ms],w,yerr=[M.loc[m,('std',fs)] for m in ms],capsize=2,label=lab)
ax[0].set_xticks(x);ax[0].set_xticklabels(ms,rotation=30);ax[0].set_ylim(.4,1);ax[0].set_ylabel('$R^2$ (grouped 23/5 splits)');ax[0].legend(fontsize=7);ax[0].set_title('(a) Repeated grouped hold-out (20 splits)')
L=pd.read_csv('logo_summary.csv')
for k,(fs,lab) in enumerate([('raw','7 raw inputs'),('full17','17 features')]):
    ax[1].bar(x+(k-.5)*w,[L[(L.fset==fs)&(L.model==m)].pooled_rmse.values[0] for m in ms],w,label=lab)
ax[1].set_xticks(x);ax[1].set_xticklabels(ms,rotation=30);ax[1].set_ylabel('pooled LOGO RMSE');ax[1].set_title('(b) Leave-one-geometry-out');ax[1].legend(fontsize=7)
plt.tight_layout();plt.savefig('figs/fig_ablation.png');plt.close()
# Fig B: LOGO parity + per geometry
R=P[(P.fset=='full17')&(P.model=='GBR')]
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6))
ax[0].scatter(R.y,R.pred,s=4,alpha=.5);ax[0].plot([0,1],[0,1],'k--',lw=.8);ax[0].set_xlabel('measured $\\eta$');ax[0].set_ylabel('LOGO prediction');ax[0].set_title('(a) Out-of-geometry parity')
MDs=MD.sort_values('rmse'); ax[1].barh(range(28),MDs.rmse,color=['tab:red' if v else 'tab:blue' for v in MDs.flag]);ax[1].set_yticks(range(28));ax[1].set_yticklabels([f'G{g}' for g in MDs.index],fontsize=5);ax[1].set_xlabel('LOGO RMSE');ax[1].set_title('(b) Per geometry (red: flagged)')
sc=ax[2].scatter(MD.d_mahal,MD.rmse,c=np.log10(MD.n),cmap='viridis',s=18);ax[2].set_xlabel('$d_{Mahal}$ (leave-one-out)');ax[2].set_ylabel('LOGO RMSE');ax[2].set_title('(c) Distance vs error');plt.colorbar(sc,ax=ax[2],label='log$_{10}$ n')
plt.tight_layout();plt.savefig('figs/fig_logo.png');plt.close()
# Fig C: Stk50 vs Re
fig,ax=plt.subplots(figsize=(3.6,3))
ax.errorbar(S.Re,S.stk50,yerr=S.stk50*S.sd_ln*2,fmt='none',ecolor='gray',lw=.6,zorder=1); sc=ax.scatter(S.Re,S.stk50,c=S.gid,cmap='tab20',s=14,zorder=2)
rr=np.array([S.Re.min(),S.Re.max()]); C=np.exp(np.mean(np.log(S.stk50))-reg['pooled']['alpha']*np.mean(np.log(S.Re)))
ax.plot(rr,C*rr**reg['pooled']['alpha'],'r',label=f"pooled $\\alpha$={reg['pooled']['alpha']:.2f}")
ax.plot(rr,C*rr**-0.5*(np.exp(np.mean(np.log(S.stk50))+0.5*np.mean(np.log(S.Re)))/C/ (1)) if False else np.exp(np.mean(np.log(S.stk50))+0.5*np.mean(np.log(S.Re)))*rr**-0.5,'k--',label='slope $-0.5$')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Re');ax.set_ylabel('Stk$_{50}$');ax.legend(fontsize=7);plt.tight_layout();plt.savefig('figs/fig_stk50.png');plt.close()
print('ok')
