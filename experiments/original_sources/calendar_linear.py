"""Bounded C33: shared linear response to the newly successful history inputs."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,shutil,threading,sys
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge
B=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('c32',B/'work/seasonal_difference_gpu.py');c32=importlib.util.module_from_spec(s);s.loader.exec_module(c32);m=c32.m;g=m.g
OUT=B/'outputs/calendar_linear_20261031_run01';g.rail.OUT=OUT
def guarded_fit(model,X,y,weight):
    done=threading.Event()
    def watch():
        while not done.wait(2):
            try:g.rail.guard()
            except g.rail.Pause as exc:g.rail.dump(OUT/'paused.json',dict(reason=str(exc)));os._exit(75)
    t=threading.Thread(target=watch,daemon=True);t.start()
    try:model.fit(X,y,sample_weight=weight)
    finally:done.set();t.join(timeout=3)
def main():
    g.rail.guard();assert not OUT.exists();OUT.mkdir();shutil.copy2(__file__,OUT/'runner.py');shutil.copy2(B/'work/calendar_linear_protocol.md',OUT/'protocol.md')
    d=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=d['raw'],d['x'],d['y'];base=m.static(raw,x);rows=[];parts={};bands=[];manifest={}
    for fold,cutoff,start in g.rail.FOLDS:
        g.rail.guard();dest=OUT/fold;dest.mkdir();tr=raw.RENT_APPROVAL_DATE<=cutoff;va=raw.RENT_APPROVAL_DATE>=start
        cp=B/'outputs/seasonal_difference_gpu_20261031_run01'/fold/'predictions.csv';ip=m.MARKET/fold/'ratio_index/market_crossfit.csv';fp=m.MARKET/fold/'ratio_forecast.json'
        for p in [cp,ip,fp]:manifest[str(p)]=g.rail.sha(p)
        old=pd.read_csv(cp);idx=pd.read_csv(ip);fc=json.loads(fp.read_text());np.testing.assert_array_equal(old.source_row,np.flatnonzero(va));np.testing.assert_array_equal(idx.source_row,np.flatnonzero(tr))
        scale=np.exp(idx.market_offset.to_numpy());z=y.loc[tr].to_numpy()/scale;w=scale**2;w/=w.mean();extra,_=c32.history_features(x,tr,z,idx.crossfit_fold.to_numpy());xx=pd.concat([base,extra],axis=1)
        numeric=[c for c in xx if c not in g.rail.CAT];prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),numeric),('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float64),g.rail.CAT)],sparse_threshold=1)
        X=prep.fit_transform(xx.loc[tr]);Q=prep.transform(xx.loc[va]);model=Ridge(alpha=10,solver='lsqr',tol=1e-6,max_iter=5000)
        g.rail.event('fit_start',fold=fold,rows=X.shape[0],columns=X.shape[1],device='bounded CPU sparse LSQR');guarded_fit(model,X,z,w);iterations=int(np.max(model.n_iter_));assert iterations<5000
        h=old.forecast_month.to_numpy();phi=fc['phi'];qscale=np.exp(fc['level']+fc['slope']*phi*(1-phi**h)/(1-phi));pred=model.predict(Q)*qscale;assert np.isfinite(pred).all()
        reference=2*old.level_average-old.level_market
        out=old[['source_row','RENT_APPROVAL_DATE','TOWN','FLAT_TYPE','actual','forecast_month','old_average','level_average','level_market','has_history']].copy();out['linear_market']=pred;out['linear_average']=(reference+pred)/2;out.to_csv(dest/'predictions.csv',index=False)
        joblib.dump(dict(preprocessor=prep,model=model,columns=list(xx)),dest/'linear.joblib');g.rail.dump(dest/'fit.json',dict(iterations=iterations,alpha=10,columns=list(xx),market_target='y/s',weights='s^2'))
        for method in ['old_average','level_average','linear_average','level_market','linear_market']:
            p=out.loc[out.forecast_month<=6];rows.append(dict(fold=fold,method=method,**g.rail.score(p.actual,p[method])));parts.setdefault(method,[]).append(p[['source_row','actual',method]].rename(columns={method:'predicted'}))
            for lo,hi in [(1,6),(7,12),(13,18),(19,24)]:
                if h.max()<hi:continue
                take=out.forecast_month.between(lo,hi);bands.append(dict(fold=fold,band=f'{lo}-{hi}',method=method,**g.rail.score(out.loc[take,'actual'],out.loc[take,method])))
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);g.rail.event('fold_complete',fold=fold,iterations=iterations,results=rows[-5:-2])
    for method,ls in parts.items():
        p=pd.concat(ls);assert len(p)==71040 and p.source_row.is_unique;rows.append(dict(fold='pooled',method=method,**g.rail.score(p.actual,p.predicted)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(bands).to_csv(OUT/'bands.csv',index=False);assert all(g.rail.sha(p)==v for p,v in manifest.items());g.rail.dump(OUT/'input_manifest.json',manifest);g.rail.dump(OUT/'complete.json',dict(time=g.rail.now(),new_fits=4,device='CPU',baseline_refits=0));g.rail.event('complete',scores=rows[-5:])
if __name__=='__main__':
    try:main()
    except g.rail.Pause as exc:g.rail.dump(OUT/'paused.json',dict(reason=str(exc)));sys.exit(75)
