"""Deterministic join of official 2020 housing composition, without rent targets."""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];D=B/'outputs/census_preparation_20261031';D.mkdir(exist_ok=True)
SOURCE=B/'data_external/census2020/census2020_dwelling_raw.json'
BLOCKS=Path('D:/LSF/NUS/courses/CS5228/project/cs5228-hdb-rent-prediction/data/auxiliary/sg-hdb-block.csv')
def norm(s):return re.sub(r'\s+',' ',str(s).strip().lower())
def table():
    records=json.loads(SOURCE.read_text())['result']['records'];area=None;rows=[]
    assert [r['_id'] for r in records]==list(range(1,len(records)+1))
    for record in records:
        name=record['Number']
        if name=='Total':continue
        parent=re.fullmatch(r'(.+?)\s*-\s*Total',name)
        if parent:
            area=norm(parent.group(1));continue
        assert area is not None
        row={'PLANNING_AREA':area,'SUBZONE':norm(name)}
        for k,v in record.items():
            if k in ['_id','Number']:continue
            row[k]=0. if v=='-' else np.nan if v is None or str(v).lower() in ['na','n.a.',''] else float(str(v).replace(',',''))
        rows.append(row)
    t=pd.DataFrame(rows);assert len(t)==332 and not t.duplicated(['PLANNING_AREA','SUBZONE']).any()
    return t
def build(x):
    blocks=pd.read_csv(BLOCKS,dtype={'BLOCK':str})
    for c in ['TOWN','BLOCK','STREET','PLANNING_AREA','SUBZONE']:blocks[c]=blocks[c].map(norm)
    blocks['ADDRESS']=blocks.TOWN+'|'+blocks.BLOCK+'|'+blocks.STREET
    assert blocks.ADDRESS.is_unique
    b=blocks.set_index('ADDRESS').reindex(x.ADDRESS)
    geo=b[['PLANNING_AREA','SUBZONE']].reset_index(drop=True);geo.index=x.index
    assert geo.notna().all().all()
    t=table();joined=geo.reset_index(names='source_row').merge(t,on=['PLANNING_AREA','SUBZONE'],how='left',validate='many_to_one',indicator=True)
    np.testing.assert_array_equal(joined.source_row,x.index);joined.index=x.index
    total=joined.Total.where(joined.Total>0);hdb=joined.HDBDwellings_Total.where(joined.HDBDwellings_Total>0)
    extra=pd.DataFrame(index=x.index)
    extra['CENSUS_LOG_RESIDENTS']=np.log1p(joined.Total)
    extra['CENSUS_HDB_SHARE']=joined.HDBDwellings_Total/total
    for label,key in [('SMALL','HDBDwellings_1_and2_RoomFlats1'),('3ROOM','HDBDwellings_3_RoomFlats'),('4ROOM','HDBDwellings_4_RoomFlats'),('5ROOM','HDBDwellings_5_RoomandExecutiveFlats')]:extra[f'CENSUS_{label}_WITHIN_HDB']=joined[key]/hdb
    for label,key in [('CONDO','CondominiumsandOtherApartments'),('LANDED','LandedProperties'),('OTHER','Others')]:extra[f'CENSUS_{label}_SHARE']=joined[key]/total
    extra['CENSUS_MISSING']=(joined['_merge'].ne('both')|total.isna()).astype(float)
    # Rounded census counts can slightly violate compositional sums; do not silently renormalize them.
    assert not np.isinf(extra.to_numpy()).any()
    return geo,extra,joined,t
if __name__=='__main__':
    raw,x,y=pd.read_pickle(B/'outputs/ple_timecap_20261031_run01/features.pkl');geo,extra,joined,t=build(x)
    t.to_csv(D/'census_subzone_counts.csv',index=False)
    unmatched=joined.loc[joined['_merge'].ne('both'),['PLANNING_AREA','SUBZONE']].value_counts().reset_index(name='rental_rows');unmatched.to_csv(D/'unmatched_subzones.csv',index=False)
    pd.to_pickle(dict(raw=raw,x=x,y=y,geography=geo,census=extra),D/'features.pkl')
    audit=dict(status='prepared',census_subzones=len(t),training_rows=len(x),matched_rows=int(joined['_merge'].eq('both').sum()),unmatched_rows=int(joined['_merge'].ne('both').sum()),nonpositive_or_unknown_total=int(extra.CENSUS_MISSING.sum()),extra_features=list(extra),geography_columns=list(geo),census_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),blocks_sha256=hashlib.sha256(BLOCKS.read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(pd.util.hash_pandas_object(extra,index=True).values.tobytes()).hexdigest(),join='Exact normalized planning area plus subzone; no fuzzy fallback or row removal.',semantics='Resident population composition, not unit counts, individual income, tenancy or population density. Hyphen means nil/negligible; na remains missing.',availability='Original Census2020 statistical release18June2021 precedes all origins. Training rows before release use a snapshot available at fitting origin, not a claimed as-of-approval feature.',limits='Uses supplied static geography; no re-geocoding or proof of every historical boundary. Portal mirror created2024; original statistics published2021. Rounding retained.')
    (D/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit,indent=2));print(unmatched.to_string(index=False))
