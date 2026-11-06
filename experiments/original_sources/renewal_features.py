"""Strictly past, exact-month group rent histories with explicit availability."""
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1]
def build(raw,x,y,cutoff,lags):
    prefix=raw.RENT_APPROVAL_DATE<=cutoff
    history=pd.DataFrame({'key':x.ADDRESS+'|'+x.FLAT_TYPE,'type':x.FLAT_TYPE,'month':x.MONTH_INDEX,'log_rent':np.log(np.asarray(y))}).loc[prefix].copy()
    assert history.log_rent.notna().all()
    history['anomaly']=history.log_rent-history.groupby(['type','month']).log_rent.transform('median')
    table=history.groupby(['key','month']).anomaly.agg(['mean','size']);out=pd.DataFrame(index=x.index)
    for k,lag in enumerate(lags):
        source_month=x.MONTH_INDEX-lag;available=source_month.between(history.month.min(),history.month.max())
        key=pd.MultiIndex.from_arrays([x.ADDRESS+'|'+x.FLAT_TYPE,source_month]);look=table.reindex(key).set_axis(x.index)
        assert (source_month<x.MONTH_INDEX).all()
        out[f'PAST{k}_LOG_COUNT']=np.log1p(look['size'].fillna(0)).where(available)
        out[f'PAST{k}_LOG_RENT_ANOMALY']=look['mean'].where(available)
        out[f'PAST{k}_UNAVAILABLE']=(~available).astype(float)
    return out
if __name__=='__main__':
    import json
    d=pd.read_pickle(B/'outputs/census_preparation_20261031/features.pkl');r,x,y=d['raw'],d['x'],d['y'];a=build(r,x,y,'2023-03',[12,24]);changed=y.copy();changed.loc[r.RENT_APPROVAL_DATE>'2023-03']=99999;b=build(r,x,changed,'2023-03',[12,24]);pd.testing.assert_frame_equal(a,b)
    # Alter a query's own target: its strictly earlier descriptors must not change.
    i=np.flatnonzero(r.RENT_APPROVAL_DATE.eq('2022-01'))[0];changed=y.copy();changed.iloc[i]=99999;c=build(r,x,changed,'2023-03',[12,24]);pd.testing.assert_series_equal(a.iloc[i],c.iloc[i])
    assert a.loc[r.RENT_APPROVAL_DATE<'2022-01','PAST0_UNAVAILABLE'].eq(1).all()
    assert a.loc[r.RENT_APPROVAL_DATE>'2024-03','PAST0_UNAVAILABLE'].eq(1).all()
    D=B/'outputs/renewal_feature_checks_20261031';D.mkdir(exist_ok=True);summary=dict(future_target_mutation_pass=True,own_target_mutation_pass=True,availability_boundaries_pass=True,columns=list(a));(D/'checks.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary))
