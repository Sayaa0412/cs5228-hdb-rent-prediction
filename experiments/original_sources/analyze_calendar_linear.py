from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];R=B/'outputs/calendar_linear_20261031_run01';D=B/'outputs/calendar_linear_diagnostics_20261031';D.mkdir(exist_ok=True);assert (R/'complete.json').exists()
def score(y,p):
    e=np.asarray(p)-np.asarray(y);return dict(n=len(e),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),sse=float(e@e))
t=pd.read_csv(R/'metrics.csv').pivot(index='fold',columns='method',values='rmse').loc[['f1','f2','f3','f4','pooled']];t.to_csv(D/'comparison.csv');slices=[];parts=[];iterations=[]
for fold in ['f1','f2','f3','f4']:
    f=pd.read_csv(R/fold/'predictions.csv');f=f.loc[f.forecast_month<=6].copy();f['fold']=fold;f['reference']=2*f.level_average-f.level_market;parts.append(f)
    iterations.append(json.loads((R/fold/'fit.json').read_text())['iterations'])
    for group,take in {'all':np.ones(len(f),bool),'low_rent_le2000':f.actual<=2000,'other_rent':f.actual>2000,'history':f.has_history,'no_history':~f.has_history}.items():
        for name in ['old_average','level_average','linear_average']:slices.append(dict(fold=fold,group=group,method=name,**score(f.loc[take,'actual'],f.loc[take,name])))
allrows=pd.concat(parts);assert len(allrows)==71040 and allrows.source_row.is_unique
cor=[]
for scope,f in [('pooled',allrows),('f4',allrows.loc[allrows.fold=='f4'])]:
    er=f.reference-f.actual
    for method in ['level_market','linear_market']:
        em=f[method]-f.actual;avg=(f.reference+f[method])/2
        cor.append(dict(scope=scope,market=method,residual_correlation=float(np.corrcoef(er,em)[0,1]),reference_mse=float(np.mean(er**2)),market_mse=float(np.mean(em**2)),error_cross_product=float(np.mean(er*em)),ensemble_mse=float(np.mean((avg-f.actual)**2))))
pd.DataFrame(slices).to_csv(D/'slices.csv',index=False);pd.DataFrame(cor).to_csv(D/'error_decomposition.csv',index=False)
bands=pd.read_csv(R/'bands.csv');wide=bands.pivot(index=['fold','band'],columns='method',values='rmse');wide.to_csv(D/'bands.csv')
info=dict(old_pooled=float(t.loc['pooled','old_average']),new_pooled=float(t.loc['pooled','linear_average']),latest_old=float(t.loc['f4','old_average']),latest_new=float(t.loc['f4','linear_average']),primary_windows_improved=int((t.iloc[:4].linear_average<t.iloc[:4].old_average).sum()),all_bands_improved=bool((wide.linear_average<wide.old_average).all()),iterations=iterations)
(D/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf8')
report='# C33: linear response to calendar-aligned history\n\n'+t[['old_average','level_average','linear_average']].to_string()+'\n\nAll listed scores use the same reference and fixed half-weight; the linear market model alone is worse than the matched boosted market model, yet its average is better. This motivates examining complementary errors, not claiming that a linear model universally beats boosting. See error_decomposition.csv for exact squared-error components; correlation alone does not establish causality.\n\nSeparate forecast bands:\n\n'+wide[['old_average','level_average','linear_average']].to_string()+'\n\nFour sparseCPUfits, no oldmodelrefits or parameter search. Prefix-only preprocessing and address-excluded market normalization. All evaluation windows remain exposed.\n'
(D/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(info));print(pd.DataFrame(cor).to_string(index=False));print(pd.DataFrame(slices).query('fold=="f4" and method!="level_average"').to_string(index=False))
