"""Eight new market-component fits for two availability profiles."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,shutil,sys
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'work'));import renewal_features as feat
s=importlib.util.spec_from_file_location('prior_gpu',B/'work/census_gpu_experiment.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);g=m.g
OUT=B/'outputs/renewal_market_gpu_20261031_run01';g.OUT=OUT;g.rail.OUT=OUT
def main():
    g.rail.guard();OUT.mkdir(exist_ok=True)
    if (OUT/'complete.json').exists():raise RuntimeError('Already complete')
    if (OUT/'runner.py').exists():assert g.rail.sha(__file__)==g.rail.sha(OUT/'runner.py')
    else:shutil.copy2(__file__,OUT/'runner.py');shutil.copy2(B/'work/renewal_market_protocol.md',OUT/'protocol.md')
    data=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=data['raw'],data['x'],data['y'];base=m.static(raw,x);rows=[];parts={};inputs={}
    g.rail.dump(OUT/'settings.json',dict(params=g.PARAMS,profiles={'both':[12,24],'only24':[24]},new_fits=8,reference_refits=0,routing='both through horizon12,only24 thereafter'))
    for fold,cutoff,start in g.rail.FOLDS:
        g.rail.guard();dest=OUT/fold;dest.mkdir(exist_ok=True);tr=raw.RENT_APPROVAL_DATE<=cutoff;va=raw.RENT_APPROVAL_DATE>=start;ids=np.flatnonzero(va)
        rp=B/'outputs/renewal_gpu_20261031_run01'/fold/'predictions.csv';r24p=B/'outputs/renewal_ablation_gpu_20261031_run01'/fold/'predictions.csv';ip=m.MARKET/fold/'ratio_index/market_crossfit.csv';fp=m.MARKET/fold/'ratio_forecast.json'
        for p in [rp,r24p,ip,fp]:inputs[str(p)]=g.rail.sha(p)
        ref=pd.read_csv(rp);ref24=pd.read_csv(r24p);index=pd.read_csv(ip);conf=json.loads(fp.read_text());np.testing.assert_array_equal(index.source_row,np.flatnonzero(tr));np.testing.assert_array_equal(ref.source_row,ids);np.testing.assert_array_equal(ref24.source_row,ids)
        scale=np.exp(index.market_offset.to_numpy());weights=scale**2;weights/=weights.mean()
        out=ref[['source_row','RENT_APPROVAL_DATE','TOWN','FLAT_TYPE','actual','forecast_month','lag0_has_history','lag1_has_history']].copy();h=ref.forecast_month.to_numpy();phi=conf['phi'];future_scale=np.exp(conf['level']+conf['slope']*phi*(1-phi**h)/(1-phi))
        out['old_average']=ref.exact_average;out['old24_average']=ref24.lag24_only_average
        for profile,lags,reference in [('both',[12,24],ref.exact_reference),('only24',[24],ref24.lag24_only_reference)]:
            g.rail.guard();folder=dest/profile;folder.mkdir(exist_ok=True);z=pd.concat([base,feat.build(raw,x,y,cutoff,lags)],axis=1)
            if (folder/'market_prediction.npy').exists():pred=np.load(folder/'market_prediction.npy')
            else:
                g.rail.event('fit_start',fold=fold,method='history_market',profile=profile);model=g.fit(folder/'market',z.loc[tr],y.loc[tr].to_numpy()/scale,g.rail.CAT,weights)
                pred=model.predict(z.loc[va])*future_scale;assert np.isfinite(pred).all();np.save(folder/'market_prediction.npy',pred)
            out[profile+'_market']=pred;out[profile+'_average']=(reference+pred)/2
        out['old_routed_average']=np.where(h<=12,out.old_average,out.old24_average);out['new_average']=np.where(h<=12,out.both_average,out.only24_average);out.to_csv(dest/'predictions.csv',index=False)
        methods=['old_average','old24_average','both_market','only24_market','both_average','only24_average','old_routed_average','new_average']
        for name in methods:
            f=out.loc[out.forecast_month<=6];rows.append(dict(fold=fold,method=name,**g.rail.score(f.actual,f[name])));parts.setdefault(name,[]).append(f[['source_row','actual',name]].rename(columns={name:'predicted'}))
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);g.rail.dump(dest/'complete.json',dict(fold=fold));g.rail.event('fold_complete',fold=fold,old=rows[-8]['rmse'],new=rows[-1]['rmse'])
    for name,ls in parts.items():
        f=pd.concat(ls);assert len(f)==71040 and f.source_row.is_unique;rows.append(dict(fold='pooled',method=name,**g.rail.score(f.actual,f.predicted)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);assert all(g.rail.sha(p)==h for p,h in inputs.items());g.rail.dump(OUT/'input_manifest.json',inputs);g.rail.dump(OUT/'complete.json',dict(time=g.rail.now(),new_fits=8));g.rail.event('complete',pooled=rows[-8:])
if __name__=='__main__':
    try:main()
    except g.rail.Pause as e:g.rail.dump(OUT/'paused.json',dict(reason=str(e)));sys.exit(75)
