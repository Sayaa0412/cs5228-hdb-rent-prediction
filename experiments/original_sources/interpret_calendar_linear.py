"""Interpret saved C33 coefficients and bound value of another scalar-weight search."""
from pathlib import Path
import json
import pandas as pd,numpy as np,joblib
B=Path(__file__).resolve().parents[1];D=B/'outputs/calendar_linear_interpretation_20261031';D.mkdir(exist_ok=True)
rows=[]
for fold in ['f1','f2','f3','f4','full_training']:
    path=B/'outputs/calendar_linear_20261031_run01'/fold/'linear.joblib' if fold!='full_training' else B/'outputs/calendar_linear_final_20261031_run01/linear_market.joblib'
    saved=joblib.load(path);prep=saved['preprocessor'];model=saved['model'];names=prep.transformers_[0][2];scaler=prep.named_transformers_['numeric'].named_steps['standardscaler']
    for i,name in enumerate(names):
        if name.startswith('HISTORY24'):
            rows.append(dict(fold=fold,feature=name,standardized_coefficient=float(model.coef_[i]),input_scale=float(scaler.scale_[i]),natural_unit_coefficient=float(model.coef_[i]/scaler.scale_[i])))
pd.DataFrame(rows).to_csv(D/'coefficients.csv',index=False)
decomp=pd.read_csv(B/'outputs/calendar_linear_diagnostics_20261031/error_decomposition.csv');r=decomp[(decomp.scope=='pooled')&(decomp.market=='linear_market')].iloc[0]
a,b,c=r.reference_mse,r.market_mse,r.error_cross_product;w=float(np.clip((a-c)/(a+b-2*c),0,1));oracle=(1-w)**2*a+w*w*b+2*w*(1-w)*c;gain=float(np.sqrt(r.ensemble_mse)-np.sqrt(oracle))
info=dict(retrospective_oracle_market_weight=w,retrospective_oracle_rmse=float(np.sqrt(oracle)),fixed_average_rmse=float(np.sqrt(r.ensemble_mse)),optimistic_scalar_weight_gain=gain,warning='Uses exposed evaluation labels; this is only an optimistic diagnostic, not a selected/deployable weight or validation gain.',decision='Do not initiate another scalar-weight search on a sub-dollar pooled oracle signal. Preserve fixed0.5and existing predictions.')
(D/'weight_diagnostic.json').write_text(json.dumps(info,indent=2),encoding='utf8')
text='# C33: conditional history response and limited value of scalar reweighting\n\nThe natural-unit coefficient on normalized historical rent estimates the change in the normalized prediction per1SGD of that historical field, holding the encoded fields fixed. It is a penalized conditional association, not a causal effect, a tenancy-retention probability, or evidence that the same unit recurs. Missing histories are imputed and have explicit count/availability fields, so the coefficient describes observed-history changes. Correlated categorical fields also limit standalone interpretation.\n\n'+pd.DataFrame(rows).query('feature=="HISTORY24_NORMALIZED_MEAN"').to_string(index=False)+'\n\nA pooled scalar-weight oracle based on exposed errors gives only '+f'{gain:.3f}'+'SGD beyond fixed averaging. This does not justify another weight grid or chronological-stack rerun; the oracle is not a validated performance result. No training or prediction reproduction was performed, and neither weights nor predictions changed.\n'
(D/'REPORT.md').write_text(text,encoding='utf8');print(pd.DataFrame(rows).query('feature=="HISTORY24_NORMALIZED_MEAN"').to_string(index=False));print(json.dumps(info))
