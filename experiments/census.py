"""Exact public census join; residents by dwelling type, not dwelling counts."""
from pathlib import Path
import json,re
import numpy as np,pandas as pd

def norm(s):return re.sub(r'\s+',' ',str(s).strip().lower())
def table(source):
    records=json.loads(Path(source).read_text())['result']['records'];area=None;rows=[]
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
def build(x, blocks, source):
    blocks=blocks.copy()
    for c in ['TOWN','BLOCK','STREET','PLANNING_AREA','SUBZONE']:blocks[c]=blocks[c].map(norm)
    blocks['ADDRESS']=blocks.TOWN+'|'+blocks.BLOCK+'|'+blocks.STREET
    assert blocks.ADDRESS.is_unique
    b=blocks.set_index('ADDRESS').reindex(x.ADDRESS)
    geo=b[['PLANNING_AREA','SUBZONE']].reset_index(drop=True);geo.index=x.index
    assert geo.notna().all().all()
    t=table(source);joined=geo.reset_index(names='source_row').merge(t,on=['PLANNING_AREA','SUBZONE'],how='left',validate='many_to_one',indicator=True)
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
