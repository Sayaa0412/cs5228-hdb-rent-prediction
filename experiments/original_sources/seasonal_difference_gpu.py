"""C32: matched level/difference targets on a common historical representation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,shutil,sys
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('prior_gpu',B/'work/census_gpu_experiment.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);g=m.g
OUT=B/'outputs/seasonal_difference_gpu_20261031_run01';g.OUT=OUT;g.rail.OUT=OUT

def history_features(x,tr,z,fold):
    source=x.loc[tr,['ADDRESS','FLAT_TYPE','MONTH_INDEX']].copy();source['normalized_rent']=z;source['excluded_fold']=fold
    assert source.groupby('ADDRESS').excluded_fold.nunique().max()==1
    source['key']=source.ADDRESS+'|'+source.FLAT_TYPE
    table=source.groupby(['key','MONTH_INDEX']).normalized_rent.agg(['mean','size'])
    month=x.MONTH_INDEX-24;available=month.between(source.MONTH_INDEX.min(),source.MONTH_INDEX.max());keys=pd.MultiIndex.from_arrays([x.ADDRESS+'|'+x.FLAT_TYPE,month]);v=table.reindex(keys).set_axis(x.index)
    extra=pd.DataFrame(index=x.index);extra['HISTORY24_NORMALIZED_MEAN']=v['mean'].where(available);extra['HISTORY24_LOG_COUNT']=np.log1p(v['size'].fillna(0)).where(available);extra['HISTORY24_UNAVAILABLE']=(~available).astype(float)
    assert (month<x.MONTH_INDEX).all()
    return extra,extra.HISTORY24_NORMALIZED_MEAN.fillna(0).to_numpy()

def main():
    g.rail.guard();assert not OUT.exists();OUT.mkdir();shutil.copy2(__file__,OUT/'runner.py');shutil.copy2(B/'work/seasonal_difference_protocol.md',OUT/'protocol.md')
    d=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=d['raw'],d['x'],d['y'];base=m.static(raw,x);rows=[];parts={};inputs={}
    g.rail.dump(OUT/'settings.json',dict(params=g.PARAMS,new_fits=8,targets=['level','difference'],lag=24,weights='original SGD loss',baseline_refits=0))
    for name,cutoff,start in g.rail.FOLDS:
        g.rail.guard();dest=OUT/name;dest.mkdir();tr=raw.RENT_APPROVAL_DATE<=cutoff;va=raw.RENT_APPROVAL_DATE>=start;ids=np.flatnonzero(va)
        rp=B/'outputs/renewal_gpu_20261031_run01'/name/'predictions.csv';r24p=B/'outputs/renewal_ablation_gpu_20261031_run01'/name/'predictions.csv';op=B/'outputs/renewal_market_gpu_20261031_run01'/name/'predictions.csv';ip=m.MARKET/name/'ratio_index/market_crossfit.csv';fp=m.MARKET/name/'ratio_forecast.json'
        for p in [rp,r24p,op,ip,fp]:inputs[str(p)]=g.rail.sha(p)
        ref=pd.read_csv(rp);r24=pd.read_csv(r24p);old=pd.read_csv(op);idx=pd.read_csv(ip);fc=json.loads(fp.read_text())
        for f in [ref,r24,old]:np.testing.assert_array_equal(f.source_row,ids)
        np.testing.assert_array_equal(idx.source_row,np.flatnonzero(tr));scale=np.exp(idx.market_offset.to_numpy());z=y.loc[tr].to_numpy()/scale;w=scale**2;w/=w.mean()
        extra,anchor=history_features(x,tr,z,idx.crossfit_fold.to_numpy());xx=pd.concat([base,extra],axis=1)
        h=ref.forecast_month.to_numpy();phi=fc['phi'];qscale=np.exp(fc['level']+fc['slope']*phi*(1-phi**h)/(1-phi));refpred=np.where(h<=12,ref.exact_reference,r24.lag24_only_reference)
        out=ref[['source_row','RENT_APPROVAL_DATE','TOWN','FLAT_TYPE','actual','forecast_month']].copy();out['old_average']=old.old_routed_average;out['has_history']=extra.loc[va,'HISTORY24_LOG_COUNT'].gt(0).to_numpy()
        for arm in ['level','difference']:
            g.rail.guard();g.rail.event('fit_start',fold=name,arm=arm);subtract=anchor[tr] if arm=='difference' else 0
            model=g.fit(dest/arm,xx.loc[tr],z-subtract,g.rail.CAT,w);p=(model.predict(xx.loc[va])+(anchor[va] if arm=='difference' else 0))*qscale
            assert np.isfinite(p).all();out[arm+'_market']=p;out[arm+'_average']=(refpred+p)/2
        out.to_csv(dest/'predictions.csv',index=False)
        for method in ['old_average','level_average','difference_average','level_market','difference_market']:
            f=out.loc[out.forecast_month<=6];rows.append(dict(fold=name,method=method,**g.rail.score(f.actual,f[method])));parts.setdefault(method,[]).append(f[['source_row','actual',method]].rename(columns={method:'predicted'}))
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);g.rail.dump(dest/'complete.json',dict(fold=name));g.rail.event('fold_complete',fold=name,results=rows[-5:-2])
    for method,ls in parts.items():
        f=pd.concat(ls);assert len(f)==71040 and f.source_row.is_unique;rows.append(dict(fold='pooled',method=method,**g.rail.score(f.actual,f.predicted)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);assert all(g.rail.sha(p)==v for p,v in inputs.items());g.rail.dump(OUT/'input_manifest.json',inputs);g.rail.dump(OUT/'complete.json',dict(time=g.rail.now(),new_fits=8));g.rail.event('complete',results=rows[-5:])
if __name__=='__main__':
    try:main()
    except g.rail.Pause as e:g.rail.dump(OUT/'paused.json',dict(reason=str(e)));sys.exit(75)
