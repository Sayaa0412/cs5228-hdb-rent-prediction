"""Matched input fields and temporal windows for conventional course regressors."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,sys,threading
import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('census_parent',B/'work/census_gpu_experiment.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
OUT=None;DEPS=['census_gpu_experiment.py']+m.DEPS
def prep(x,train):
    cats=m.g.rail.CAT+['PLANNING_AREA','SUBZONE'];nums=[c for c in x if c not in cats]
    p=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),nums),('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float64),cats)],sparse_threshold=1)
    a=p.fit_transform(x.loc[train]);return p,a.tocsr(),p.transform(x).tocsr()
def run():
    m.g.rail.guard()
    if not OUT.exists():
        OUT.mkdir(parents=True);shutil.copy2(__file__,OUT/'runner.py');(OUT/'dependencies').mkdir()
        for name in DEPS:shutil.copy2(B/'work'/name,OUT/'dependencies'/name)
        files=[B/'outputs/census_preparation_20261031/features.pkl'];m.g.rail.dump(OUT/'input_manifest.json',{str(p):m.g.rail.sha(p) for p in files});m.g.rail.dump(OUT/'source_manifest_before.json',m.g.rail.manifest())
        m.g.rail.dump(OUT/'protocol.json',dict(lecture='Lecture4 CART; Lecture5 regularized regression; Lecture6 pp19-25 bagging/random forests.',purpose='Matched current field/window comparison requested by teacher; a different bias/variance family, not another boosting parameter adjustment.',inputs='Same25 raw-enriched fields as GPUcensus; numerical median/standardization and categorical one-hot fitted on each training prefix. NativeCatBoost vs one-hot encoding remains an explicit implementation difference.',methods=dict(ridge='alpha10,lsqr,tol1e-6,max_iter5000',cart='depth12,min_samples_leaf20',random_forest='300trees,depth12,min_samples_leaf20,max_features0.5,bootstrap,seed42'),selection='One fixed specification per family; not an exhaustive optimized comparison. Outer labels never choose parameters or tree count.',hardware='CPUsklearn implementation,4threads. Installed environment has no GPUsklearn forest; bounded CPU fallback preserves standard course algorithms. No package install.',source='Original files, baseline and priorRidge variants preserved. New comparison uses identical current enriched inputs and all4timewindows.'))
    else:
        assert m.g.rail.sha(__file__)==m.g.rail.sha(OUT/'runner.py')
        for name in DEPS:assert m.g.rail.sha(B/'work'/name)==m.g.rail.sha(OUT/'dependencies'/name)
    manifest=json.loads((OUT/'input_manifest.json').read_text());assert all(m.g.rail.sha(p)==h for p,h in manifest.items())
    data=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');raw,x,y=data['raw'],pd.concat([data['x'],data['geography'],data['census']],axis=1),data['y']
    done=threading.Event()
    def watchdog():
        while not done.wait(2):
            try:m.g.rail.guard()
            except m.g.rail.Pause as exc:m.g.rail.dump(OUT/'paused.json',dict(time=m.g.rail.now(),reason=str(exc)));os._exit(75)
    thread=threading.Thread(target=watchdog,daemon=True);thread.start()
    try:
        with threadpool_limits(limits=4):
            for fold,cutoff,start in m.g.rail.FOLDS:
                train=raw.RENT_APPROVAL_DATE<=cutoff;query=(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<='2025-03');d=OUT/fold;d.mkdir(exist_ok=True)
                processor,a,allx=prep(x,train);joblib.dump(processor,d/'preprocessor.joblib');target=y.loc[train];q=allx[query.to_numpy()]
                for name in ['ridge','cart','random_forest']:
                    dest=d/name;dest.mkdir(exist_ok=True)
                    if (dest/'complete.json').exists():continue
                    m.g.rail.guard();m.g.rail.event('fit_start',fold=fold,method=name,rows=int(train.sum()),encoded_columns=a.shape[1])
                    if name=='ridge':model=Ridge(alpha=10,solver='lsqr',tol=1e-6,max_iter=5000).fit(a,target)
                    elif name=='cart':model=DecisionTreeRegressor(max_depth=12,min_samples_leaf=20,random_state=42).fit(a,target)
                    else:
                        path=dest/'checkpoint.joblib';model=joblib.load(path) if path.exists() else RandomForestRegressor(n_estimators=0,max_depth=12,min_samples_leaf=20,max_features=.5,bootstrap=True,random_state=42,n_jobs=4,warm_start=True)
                        while model.n_estimators<300:
                            m.g.rail.guard();model.set_params(n_estimators=model.n_estimators+25);model.fit(a,target);joblib.dump(model,path);m.g.rail.event('progress',fold=fold,method=name,trees=model.n_estimators)
                    joblib.dump(model,dest/'model.joblib');pred=model.predict(q);loaded=joblib.load(dest/'model.joblib');np.testing.assert_allclose(pred,loaded.predict(q),rtol=0,atol=1e-8)
                    _,scores=m.save_scores(dest,raw,y,query,cutoff,pred);m.g.rail.dump(dest/'complete.json',dict(train_rows=int(train.sum()),train_row_sha256=hashlib.sha256(x.index[train].to_numpy(dtype=np.int64).tobytes()).hexdigest(),feature_sha256=m.framehash(x),encoded_columns=a.shape[1],params=model.get_params(),solver_iterations=int(np.max(model.n_iter_)) if name=='ridge' else None,reload_equal=True));m.g.rail.event('evaluation_complete',fold=fold,method=name,metrics=scores)
    finally:done.set();thread.join(timeout=3)
    assert all(m.g.rail.sha(p)==h for p,h in manifest.items());assert m.g.rail.manifest()==json.loads((OUT/'source_manifest_before.json').read_text());m.g.rail.dump(OUT/'complete.json',dict(time=m.g.rail.now(),fits=12,source_preserved=True))
def smoke():
    OUT.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(42);x=pd.DataFrame({'a':rng.normal(size=200),'c':['x','y']*100});p=ColumnTransformer([('num',StandardScaler(),['a']),('cat',OneHotEncoder(handle_unknown='ignore'),['c'])],sparse_threshold=1);a=p.fit_transform(x.iloc[:160]);q=p.transform(pd.DataFrame({'a':[.5,1.],'c':['x','new']}));y=2000+100*x.a.iloc[:160]
    for model in [Ridge(alpha=10,solver='lsqr'),DecisionTreeRegressor(max_depth=4,min_samples_leaf=5,random_state=42),RandomForestRegressor(n_estimators=10,max_depth=4,min_samples_leaf=5,random_state=42,n_jobs=4)]:
        model.fit(a,y);pred=model.predict(q);assert np.isfinite(pred).all();joblib.dump(model,OUT/(type(model).__name__+'.joblib'));np.testing.assert_allclose(pred,joblib.load(OUT/(type(model).__name__+'.joblib')).predict(q),atol=1e-8)
    result=dict(status='pass',sparse_preprocessing=True,unknown_category=True,three_standard_regressors=True,save_reload=True);m.g.rail.dump(OUT/'witness.json',result);print('WITNESS '+json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();OUT=a.output.resolve();m.g.rail.OUT=OUT
    if a.smoke:smoke()
    else:
        try:run()
        except m.g.rail.Pause as exc:m.g.rail.dump(OUT/'paused.json',dict(reason=str(exc),time=m.g.rail.now()));sys.exit(75)
