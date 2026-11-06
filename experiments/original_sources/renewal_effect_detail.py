from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];R=B/'outputs/renewal_gpu_20261031_run01';D=B/'outputs/renewal_diagnostics_20261031'
rows=[]
for fold in ['f1','f2','f3','f4']:
    f=pd.read_csv(R/fold/'predictions.csv')
    for lo,hi in [(1,6),(7,12),(13,18),(19,24)]:
        if f.forecast_month.max()<hi:continue
        take=f.forecast_month.between(lo,hi);y=f.loc[take,'actual']
        rows.append(dict(fold=fold,band=f'{lo}-{hi}',n=int(take.sum()),old_rmse=float(np.sqrt(np.mean((f.loc[take,'old_average']-y)**2))),exact_rmse=float(np.sqrt(np.mean((f.loc[take,'exact_average']-y)**2)))))
pd.DataFrame(rows).to_csv(D/'forecast_bands.csv',index=False)
s=pd.read_csv(D/'slices.csv');f=s[(s.fold=='f4')&s.method.isin(['old','exact'])].copy();p=f.pivot(index='group',columns='method',values=['n','rmse','sse']);out=[]
for group in p.index:
    out.append(dict(group=group,n=int(p.loc[group,('n','old')]),old_rmse=float(p.loc[group,('rmse','old')]),exact_rmse=float(p.loc[group,('rmse','exact')]),sse_reduction=float(p.loc[group,('sse','old')]-p.loc[group,('sse','exact')])))
(D/'latest_effects.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(out,indent=2))
