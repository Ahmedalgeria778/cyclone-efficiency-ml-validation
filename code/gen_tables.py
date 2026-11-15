import json
from common import *
df,_=load(); f=lambda x:f'{x:.3f}'
F=pd.read_csv('splits_fixed.csv'); T=pd.read_csv('splits_tuned.csv'); L=pd.read_csv('logo_summary.csv'); Rn=pd.read_csv('splits_random.csv')
MD=pd.read_csv('mahal.csv',index_col=0); PG=pd.read_csv('logo_per_geom.csv')
# Table data ranges
rows=[('$D_e/D$','De'),('$a/D$','a'),('$b/D$','b'),('Re','Re'),('Stk','Stk'),('$D_p/D$','Dp'),('$\\rho_p/\\rho_g$','rr'),('$\\eta$','eta')]
s='\\begin{tabular}{lrr}\\toprule Variable & Min & Max\\\\\\midrule\n'
for n,c in rows:
    a,b=df[c].min(),df[c].max(); fm=(lambda v:f'{v:.3g}'); s+=f'{n} & {fm(a)} & {fm(b)}\\\\\n'
open('out/t_ranges.tex','w').write(s+'\\bottomrule\\end{tabular}')
# Table benchmark fixed
ms=['GBR','XGB','LGBM','CatBoost','RF','HGB','MLP']; s='\\begin{tabular}{lcccc}\\toprule Model & \\multicolumn{2}{c}{Raw 7 inputs} & \\multicolumn{2}{c}{17 features}\\\\ & $R^2$ & RMSE & $R^2$ & RMSE\\\\\\midrule\n'
for m in ms:
    r=[]
    for fs in ['raw','full17']:
        d=F[(F.model==m)&(F.fset==fs)]; r+= [f'{d.r2.mean():.3f}$\\pm${d.r2.std():.3f}',f'{d.rmse.mean():.3f}']
    s+=f'{m} & '+' & '.join(r)+'\\\\\n'
s+='\\midrule\\multicolumn{5}{l}{Baselines (no machine learning)}\\\\\n'
for m in ['Lapple(N=5)','Lapple(N fitted)','Stk-only fit']:
    d=F[F.model==m]; s+=f'{m.replace("Lapple","Lapple")} & \\multicolumn{{2}}{{c}}{{{d.r2.mean():.3f}$\\pm${d.r2.std():.3f}}} & \\multicolumn{{2}}{{c}}{{RMSE {d.rmse.mean():.3f}}}\\\\\n'
open('out/t_fixed.tex','w').write(s+'\\bottomrule\\end{tabular}')
# tuned
s='\\begin{tabular}{lcccc}\\toprule Model & $R^2$ raw & $R^2$ 17 feat. & $\\Delta$RMSE & splits improved\\\\\\midrule\n'
for m in ['GBR','XGB','LGBM','CatBoost','RF','HGB']:
    a=T[(T.model==m)&(T.fset=='raw')].sort_values('rep');b=T[(T.model==m)&(T.fset=='full17')].sort_values('rep')
    s+=f'{m} & {a.r2.mean():.3f}$\\pm${a.r2.std():.3f} & {b.r2.mean():.3f}$\\pm${b.r2.std():.3f} & {(b.rmse.values-a.rmse.values).mean():+.4f} & {(b.rmse.values<a.rmse.values).sum()}/10\\\\\n'
open('out/t_tuned.tex','w').write(s+'\\bottomrule\\end{tabular}')
# random vs grouped
s='\\begin{tabular}{lccc}\\toprule Model / features & Random point-wise 80/20 & Grouped 23/5 (fixed HP) & LOGO (pooled)\\\\\\midrule\n'
for m,fs in [('GBR','raw'),('GBR','full17'),('CatBoost','raw'),('MLP','raw'),('MLP','full17')]:
    a=Rn[(Rn.model==m)&(Rn.fset==fs)].r2; b=F[(F.model==m)&(F.fset==fs)].r2; l=L[(L.model==m)&(L.fset==fs)].pooled_r2.values[0]
    s+=f'{m} / {"raw" if fs=="raw" else "17 feat."} & {a.mean():.3f}$\\pm${a.std():.3f} & {b.mean():.3f}$\\pm${b.std():.3f} & {l:.3f}\\\\\n'
open('out/t_gap.tex','w').write(s+'\\bottomrule\\end{tabular}')
# LOGO table
s='\\begin{tabular}{llcccccc}\\toprule Model & Features & Pooled $R^2$ & Pooled RMSE & Macro RMS RMSE & Weighted mean $R^2_g$ & Median $R^2_g$ & $R^2_g\\le0$\\\\\\midrule\n'
for m in ms:
    for fs in ['raw','full17']:
        d=L[(L.model==m)&(L.fset==fs)].iloc[0]; s+=f'{m} & {"raw" if fs=="raw" else "17"} & {d.pooled_r2:.3f} & {d.pooled_rmse:.3f} & {np.sqrt((PG[(PG.model==m)&(PG.fset==fs)].rmse**2).mean()):.3f} & {d.r2_w:.2f}$\\pm${d.r2_sd:.2f} & {d.r2_med:.2f} & {int(d.n_fail)}\\\\\n'
open('out/t_logo.tex','w').write(s+'\\bottomrule\\end{tabular}')
# Re bands
A=pd.read_csv('re_bands.csv'); B=pd.read_csv('leave_Re_band_out.csv')
s='\\begin{tabular}{lcccccc}\\toprule Band & Re range & $n$ & Geom. & Cond. & LOGO RMSE & Leave-band-out $R^2$ (17 feat.)\\\\\\midrule\n'
for _,r in A.iterrows():
    s+=f'{r.band} & {r.Re_lo:.2g}--{r.Re_hi:.2g} & {int(r.n)} & {int(r.n_geom)} & {int(r.n_cond)} & {r.rmse:.3f} & {B[(B.band==r.band)&(B.fset=="full17")].r2.values[0]:.2f}\\\\\n'
open('out/t_re.tex','w').write(s+'\\bottomrule\\end{tabular}')
# Stk50
reg=json.load(open('stk50_reg.json')); nm={'pooled':'Pooled, Re only','geom_covariates':'Pooled + $D_e/D$, $ab$, $\\rho_p/\\rho_g$','fixed_effects':'Geometry fixed effects'}
s='\\begin{tabular}{lcccc}\\toprule Specification & $\\alpha$ (weighted) & 95\\% CI (geometry bootstrap) & $\\alpha$ (unweighted) & $n$ cond./geom.\\\\\\midrule\n'
for k,v in reg.items(): s+=f'{nm[k]} & {v["alpha"]:.3f} & [{v["ci"][0]:.2f}, {v["ci"][1]:.2f}] & {v["alpha_unweighted"]:.3f} & {v["n"]}/{v["n_geom"]}\\\\\n'
open('out/t_stk.tex','w').write(s+'\\bottomrule\\end{tabular}')
# geometry appendix
g=df.groupby('gid').agg(De=('De','first'),a=('a','first'),b=('b','first'),n=('eta','size'),Remin=('Re','min'),Remax=('Re','max'),rrmin=('rr','min'),rrmax=('rr','max'))
s='{\\scriptsize\\begin{longtable}{rrrrrrrrr}\\toprule ID & $D_e/D$ & $a/D$ & $b/D$ & $n$ & Re range & $\\rho_p/\\rho_g$ & LOGO RMSE & $d_{Mahal}$\\\\\\midrule\n'
ref=PG[(PG.fset=='full17')&(PG.model=='GBR')].set_index('gid')
for i,r in g.iterrows():
    s+=f'G{i} & {r.De:.3f} & {r.a:.3f} & {r.b:.3f} & {int(r.n)} & {r.Remin:.2g}--{r.Remax:.2g} & {r.rrmin:.0f}--{r.rrmax:.0f} & {ref.loc[i,"rmse"]:.3f} & {MD.loc[i,"d_mahal"]:.2f}\\\\\n'
open('out/t_geom.tex','w').write(s+'\\bottomrule\\end{longtable}}')
print(open('out/t_gap.tex').read())

# ablation table (LOGO pooled RMSE / R2 per configuration)
cfg=[('raw','Raw (7)'),('raw+log','Raw + logs (11)'),('raw+pivc','Raw + $\\Pi_{VC}$ (8)'),('raw+sep+geo','Raw + separation/geometry terms (12)'),('full17','All features (17)')]
mm=['GBR','XGB','LGBM','CatBoost','RF','HGB']
t='\\begin{tabular}{l'+'c'*len(mm)+'}\\toprule Configuration (no. of inputs) & '+' & '.join(mm)+'\\\\\\midrule\n'
for k,lab in cfg:
    t+=lab+' & '+' & '.join(f"{L[(L.fset==k)&(L.model==m)].pooled_r2.values[0]:.3f} / {L[(L.fset==k)&(L.model==m)].pooled_rmse.values[0]:.3f}" for m in mm)+'\\\\\n'
open('out/t_abl.tex','w').write(t+'\\bottomrule\\end{tabular}')
sens={}
import json
for sg,fn in [('0.01','stk50_reg_0.01.json'),('0.025','stk50_reg.json'),('0.05','stk50_reg_0.05.json')]:
    sens[sg]=json.load(open(fn))
tc=pd.read_csv('stk50_table.csv').cv.median(); 
t='\\begin{tabular}{lcccc}\\toprule $\\sigma_\\epsilon$ & Median CV of Stk$_{50}$ & $\\alpha$ pooled & $\\alpha$ + covariates & $\\alpha$ fixed effects\\\\\\midrule\n'
cvs={'0.01':0.0154,'0.025':tc,'0.05':0.0816}
for sg,r in sens.items(): t+=f"{sg} & {100*cvs[sg]:.1f}\\% & {r['pooled']['alpha']:.3f} & {r['geom_covariates']['alpha']:.3f} & {r['fixed_effects']['alpha']:.3f}\\\\\n"
open('out/t_sens.tex','w').write(t+'\\bottomrule\\end{tabular}')
