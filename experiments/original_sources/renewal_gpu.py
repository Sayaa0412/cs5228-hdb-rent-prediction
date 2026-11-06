"""Fixed exact-lag versus neighboring-lag comparisons; no baseline refits."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,shutil,sys
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'work'));import renewal_features as features
s=importlib.util.spec_from_file_location('old_gpu',B/'work/census_gpu_experiment.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);g=m.g
OUT=B/'outputs/renewal_gpu_20261031_run01';g.OUT=OUT;g.rail.OUT=OUT
ARMS={'exact':(12,24),'lag_minus1':(11,23),'lag_plus1':(13,25)}
def main():
    g.rail.guard();OUT.mkdir(exist_ok=True)
    if (OUT/'complete.json').exists():raise RuntimeError('Already completed; do not reproduce')
    if (OUT/'runner.py').exists():assert g.rail.sha(__file__)==g.rail.sha(OUT/'runner.py')
    else:
        shutil.copy2(__file__,OUT/'runner.py');shutil.copy2(B/'work/renewal_protocol.md',OUT/'protocol.md');shutil.copy2(B/'work/renewal_features.py',OUT/'features.py')
    assert g.rail.sha(B/'work/renewal_features.py')==g.rail.sha(OUT/'features.py')
    d=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=d['raw'],d['x'],d['y'];scores=[];pooled={};manifest={}
    g.rail.dump(OUT/'settings.json',dict(params=g.PARAMS,arms=ARMS,new_fits=12,baseline_refits=0))
    for fold,cutoff,start in g.rail.FOLDS:
        g.rail.guard();dest=OUT/fold;dest.mkdir(exist_ok=True);tr=raw.RENT_APPROVAL_DATE<=cutoff;va=raw.RENT_APPROVAL_DATE>=start;ids=np.flatnonzero(va)
        rp=m.REF/fold/'gpu_reference/predictions.csv';mp=B/'outputs/census_gpu_20261031_run01'/fold/'gpu_market_ratio/predictions.csv'
        for p in [rp,mp]:manifest[str(p)]=g.rail.sha(p)
        ref=pd.read_csv(rp);market=pd.read_csv(mp);np.testing.assert_array_equal(ref.source_row,ids);np.testing.assert_array_equal(ref.actual,market.actual)
        out=ref.drop(columns='predicted').copy();out['old_reference']=ref.predicted;out['old_average']=(ref.predicted+market.predicted)/2
        for name,lags in ARMS.items():
            g.rail.guard();folder=dest/name;extra=features.build(raw,x,y,cutoff,lags);z=pd.concat([x,extra],axis=1)
            if (folder/'predictions.npy').exists():pred=np.load(folder/'predictions.npy')
            else:
                g.rail.event('fit_start',fold=fold,method=name,lags=lags);model=g.fit(folder,z.loc[tr],y.loc[tr],g.rail.CAT);pred=model.predict(z.loc[va]);np.save(folder/'predictions.npy',pred)
            assert np.isfinite(pred).all() and len(pred)==len(ids);out[name+'_reference']=pred;out[name+'_average']=(pred+market.predicted)/2
            if name=='exact':
                for k in range(2):out[f'lag{k}_has_history']=extra.loc[va,f'PAST{k}_LOG_COUNT'].gt(0).to_numpy()
        out.to_csv(dest/'predictions.csv',index=False)
        for name in ['old_reference','old_average']+[a+'_'+b for a in ARMS for b in ['reference','average']]:
            f=out.loc[out.forecast_month<=6];scores.append(dict(fold=fold,method=name,**g.rail.score(f.actual,f[name])));pooled.setdefault(name,[]).append(f[['source_row','actual',name]].rename(columns={name:'predicted'}))
        pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False);g.rail.dump(dest/'complete.json',dict(fold=fold));g.rail.event('fold_complete',fold=fold,averages={r['method']:r['rmse'] for r in scores[-8:] if 'average' in r['method']})
    for name,parts in pooled.items():
        f=pd.concat(parts);assert len(f)==71040 and f.source_row.is_unique;scores.append(dict(fold='pooled',method=name,**g.rail.score(f.actual,f.predicted)))
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False);assert all(g.rail.sha(p)==h for p,h in manifest.items());g.rail.dump(OUT/'input_manifest.json',manifest);g.rail.dump(OUT/'complete.json',dict(time=g.rail.now(),new_fits=12));g.rail.event('complete',scores=scores[-8:])
if __name__=='__main__':
    try:main()
    except g.rail.Pause as e:g.rail.dump(OUT/'paused.json',dict(reason=str(e)));sys.exit(75)
