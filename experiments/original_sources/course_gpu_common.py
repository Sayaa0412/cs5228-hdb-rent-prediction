"""Shared guarded GPU fitter for the course-based comparison; no search defaults."""
from pathlib import Path
import importlib.util,json,os,threading,time
from catboost import CatBoostRegressor
B=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prior_gpu',B/'work/rail_centrality_gpu.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
PARAMS=prior.PARAMS.copy();rail=prior.rail;OUT=None
def fit(dest,x,y,cat,weights=None,iterations=650):
    rail.guard();dest.mkdir(parents=True,exist_ok=True);done=threading.Event();start=time.monotonic()
    def watchdog():
        last=-1
        while not done.wait(2):
            try:rail.guard()
            except rail.Pause as exc:
                rail.dump(OUT/'paused.json',dict(time=rail.now(),reason=str(exc),job=str(dest)));os._exit(75)
            try:
                native=json.loads((dest/'native/catboost_training.json').read_text());tree=native['iterations'][-1]['iteration']+1
                if tree>=last+150:rail.event('progress',job=str(dest.relative_to(OUT)),tree=tree,seconds=round(time.monotonic()-start,1));last=tree
            except (OSError,ValueError,KeyError,IndexError):pass
    params=PARAMS.copy();params['iterations']=iterations
    model=CatBoostRegressor(**params,train_dir=str(dest/'native'));thread=threading.Thread(target=watchdog,daemon=True);thread.start()
    try:model.fit(x,y,cat_features=cat,sample_weight=weights,save_snapshot=True,snapshot_file=str(dest/'snapshot.cbsnapshot'),snapshot_interval=30)
    finally:done.set();thread.join(timeout=3)
    assert model.tree_count_==iterations;model.save_model(dest/'model.cbm');rail.dump(dest/'actual_params.json',model.get_all_params());return model
