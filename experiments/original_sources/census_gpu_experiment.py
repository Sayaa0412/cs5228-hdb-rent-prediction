"""Substantive external-data addition with geography control, and GPU ensemble validation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
B=Path(__file__).resolve().parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=module('course_gpu',B/'work/course_gpu_common.py');prep=module('census_prep',B/'work/prepare_census_features.py')
OUT=None;REF=B/'outputs/rail_centrality_gpu_20261031_run01';MARKET=B/'outputs/structural_refinement_20261030_run01'
DEPS=['course_gpu_common.py','prepare_census_features.py','rail_centrality_gpu.py','rail_history_experiment.py']
def framehash(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()).hexdigest()
def static(raw,x):
    z=x.drop(columns=['MONTH_INDEX','LEASE_AGE','BUILDING_AGE']).copy();year=raw.RENT_APPROVAL_DATE.str[:4].astype(int)
    z['LEASE_START_YEAR']=year-x.LEASE_AGE;z['COMPLETION_YEAR']=year-x.BUILDING_AGE
    return z[['FLOOR_AREA_SQM','LATITUDE','LONGITUDE','MAX_FLOOR','LEASE_START_YEAR','COMPLETION_YEAR']+g.rail.CAT]
def save_scores(dest,raw,y,query,cutoff,pred):
    f=raw.loc[query,['source_row','RENT_APPROVAL_DATE','TOWN','FLAT_TYPE','FLOOR_AREA_SQM']].copy();f['actual']=y.loc[query].to_numpy();f['predicted']=pred
    f['forecast_month']=[pd.Period(d,freq='M').ordinal-pd.Period(cutoff,freq='M').ordinal for d in f.RENT_APPROVAL_DATE]
    f.to_csv(dest/'predictions.csv',index=False)
    scores={f'horizon_{h}m':g.rail.score(f.loc[f.forecast_month<=h,'actual'],f.loc[f.forecast_month<=h,'predicted']) for h in [6,12,18] if f.forecast_month.max()>=h}
    g.rail.dump(dest/'evaluation.json',scores);return f,scores
def run():
    g.rail.guard()
    if not OUT.exists():
        OUT.mkdir(parents=True);shutil.copy2(__file__,OUT/'runner.py');(OUT/'dependencies').mkdir()
        for name in DEPS:shutil.copy2(B/'work'/name,OUT/'dependencies'/name)
        g.rail.dump(OUT/'source_manifest_before.json',g.rail.manifest())
        files=[B/'outputs/census_preparation_20261031/features.pkl',B/'outputs/census_preparation_20261031/audit.json',prep.SOURCE]
        for fold,_,_ in g.rail.FOLDS:files += [REF/fold/'gpu_reference/predictions.csv',MARKET/fold/'ratio_index/market_crossfit.csv',MARKET/fold/'ratio_forecast.json']
        g.rail.dump(OUT/'input_manifest.json',{str(p):g.rail.sha(p) for p in files})
        g.rail.dump(OUT/'protocol.json',dict(created=g.rail.now(),plan='refine-logs/EXPERIMENT_PLAN.md B1; B4 GPU validation',lecture_basis='Lecture1 integration; Lectures4-6 controlled regression/ensembles.',hypothesis='Neighborhood housing composition may add transferable information beyond property characteristics or subzone identity. A geography-only arm separates new numeric information from finer place labels.',methods=['preserved_gpu_reference','geography_only','geography_census','gpu_market_ratio','fixed_gpu_average'],params=g.PARAMS,total_new_fits=12,order=['f4','f1','f2','f3'],selection='No outer hyperparameter or tree-count selection; fixed650. Census composition definitions fixed before scores. GPUmarket is implementation validation, not a new method.',data='Official Census2020 released18June2021; mirror portal created2024. Exact supplied planning area/subzone join; counts of residents, not units. No sensitive personal features.',limits='All four windows exposed; one seed; same GPU defaults across fitted arms, CPU history preserved. Static geography and2020 snapshot can age.'))
    else:
        assert g.rail.sha(__file__)==g.rail.sha(OUT/'runner.py')
        for name in DEPS:assert g.rail.sha(B/'work'/name)==g.rail.sha(OUT/'dependencies'/name)
    manifest=json.loads((OUT/'input_manifest.json').read_text());assert all(g.rail.sha(p)==h for p,h in manifest.items())
    data=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=data['raw'],data['x'],data['y'];geo,census=data['geography'],data['census']
    xx={'geography_only':pd.concat([x,geo],axis=1),'geography_census':pd.concat([x,geo,census],axis=1),'gpu_market_ratio':static(raw,x)}
    g.rail.dump(OUT/'feature_contract.json',{name:dict(columns=list(z),sha256=framehash(z)) for name,z in xx.items()})
    for fold in ['f4','f1','f2','f3']:
        _,cutoff,start=next(t for t in g.rail.FOLDS if t[0]==fold);train=raw.RENT_APPROVAL_DATE<=cutoff;query=(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<='2025-03')
        for name,z in xx.items():
            g.rail.guard();dest=OUT/fold/name
            if (dest/'complete.json').exists():continue
            target=y.loc[train];weights=None;cats=g.rail.CAT+(['PLANNING_AREA','SUBZONE'] if name!='gpu_market_ratio' else [])
            if name=='gpu_market_ratio':
                log=pd.read_csv(MARKET/fold/'ratio_index/market_crossfit.csv');np.testing.assert_array_equal(log.source_row,x.index[train]);np.testing.assert_array_equal(log.month,x.loc[train,'MONTH_INDEX'])
                scale=np.exp(log.market_offset.to_numpy());target=target.to_numpy()/scale;weights=scale**2;weights/=weights.mean()
            g.rail.event('fit_start',fold=fold,method=name,rows=int(train.sum()))
            model=g.fit(dest,z.loc[train],target,cats,weights);rawpred=model.predict(z.loc[query]);loaded=CatBoostRegressor();loaded.load_model(dest/'model.cbm');np.testing.assert_array_equal(rawpred[:128],loaded.predict(z.loc[query].iloc[:128]))
            pred=rawpred
            if name=='gpu_market_ratio':
                info=json.loads((MARKET/fold/'ratio_forecast.json').read_text());h=x.loc[query,'MONTH_INDEX'].to_numpy()-x.loc[train,'MONTH_INDEX'].max();ph=info['phi'];offset=info['level']+info['slope']*ph*(1-ph**h)/(1-ph);pred=pred*np.exp(offset)
                g.rail.dump(dest/'normalization.json',dict(crossfit_path=str(MARKET/fold/'ratio_index/market_crossfit.csv'),crossfit_sha256=g.rail.sha(MARKET/fold/'ratio_index/market_crossfit.csv'),forecast=info,weights_mean=float(weights.mean())))
            frame,scores=save_scores(dest,raw,y,query,cutoff,pred)
            reference=pd.read_csv(REF/fold/'gpu_reference/predictions.csv')
            for col in ['source_row','actual','forecast_month']:np.testing.assert_array_equal(frame[col],reference[col])
            if name=='gpu_market_ratio':
                avgdest=OUT/fold/'fixed_gpu_average';avgdest.mkdir(exist_ok=True);save_scores(avgdest,raw,y,query,cutoff,.5*(pred+reference.predicted.to_numpy()))
            g.rail.dump(dest/'complete.json',dict(fold=fold,method=name,training_rows=int(train.sum()),training_row_sha256=hashlib.sha256(x.index[train].to_numpy(dtype=np.int64).tobytes()).hexdigest(),feature_sha256=framehash(z),reload_equal=True));g.rail.event('evaluation_complete',fold=fold,method=name,metrics=scores)
    assert all(g.rail.sha(p)==h for p,h in manifest.items());assert g.rail.manifest()==json.loads((OUT/'source_manifest_before.json').read_text())
    g.rail.dump(OUT/'complete.json',dict(time=g.rail.now(),new_fits=12,original_files_verified=len(g.rail.manifest()),preserved_inputs=len(manifest)));g.rail.event('batch_complete')
def smoke():
    OUT.mkdir(parents=True,exist_ok=False);t=prep.table();assert len(t)==332 and t[(t.PLANNING_AREA=='changi')&(t.SUBZONE=='changi airport')].shape[0]==1
    data=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');geo,census,joined,_=prep.build(data['x']);pd.testing.assert_frame_equal(geo,data['geography']);pd.testing.assert_frame_equal(census,data['census']);assert joined['_merge'].eq('both').all()
    rng=np.random.default_rng(42);n=256;x=pd.DataFrame({'area':rng.normal(size=n),'town':[f't{i%20}' for i in range(n)],'share':rng.uniform(0,1,n)});y=3000+200*x.area;scale=rng.uniform(.8,1.4,n);weights=scale**2;weights/=weights.mean()
    np.testing.assert_allclose((y-scale*3000)**2,scale**2*(y/scale-3000)**2,rtol=1e-10,atol=1e-7)
    model=g.fit(OUT/'weighted_gpu',x.iloc[:200],y.iloc[:200]/scale[:200],['town'],weights[:200],iterations=20);pred=model.predict(x.iloc[200:]);r=CatBoostRegressor();r.load_model(OUT/'weighted_gpu/model.cbm');np.testing.assert_array_equal(pred,r.predict(x.iloc[200:]))
    result=dict(status='pass',census332_subzones=True,changi_parent_format=True,exact150000_row_join=True,no_target_in_feature_function=True,weighted_loss_identity=True,gpu_weighted_fit_reload_equal=True);g.rail.dump(OUT/'witness.json',result);print('WITNESS '+json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();OUT=a.output.resolve();g.OUT=OUT;g.rail.OUT=OUT
    if a.smoke:smoke()
    else:
        try:run()
        except g.rail.Pause as exc:g.rail.dump(OUT/'paused.json',dict(time=g.rail.now(),reason=str(exc)));sys.exit(75)
