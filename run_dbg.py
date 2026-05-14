import pandas as pd
import numpy as np
import joblib
import xgboost_filter_model.train_filter_v13_wf_image as tv13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
df = tv13.prepare_data_v13(start_date='2020-01-01', end_date='2026-05-14', use_cache=True)
df = add_directional_features(df)
df = add_ma_features(df)
df = add_momentum_features(df)
df.dropna(inplace=True)
df_test = df[df.index >= pd.to_datetime('2026-05-12').tz_localize('UTC')].copy()
prod_s1 = joblib.load("xgboost_filter_model/filter_model_v13_wf_image.joblib")
prod_s2 = joblib.load("xgboost_filter_model/directional_model_v13_wf.joblib")
s1_cols = list(prod_s1.feature_names_in_)
s2_cols = list(prod_s2.feature_names_in_)
df_test['s1_prob'] = prod_s1.predict_proba(df_test[s1_cols])[:, 1]
df_test['s2_prob'] = prod_s2.predict_proba(df_import pandas as pd
impod1import numpy as npb'import joblib
impngimport xgboos2from xgboost_filter_model.train_directional_model_v2 imposignafrom xgboost_filter_model.train_directional_model_v3 import add_ma_features
from EOF
from xgboost_filterecho "
import pandas as pd
import joblib
import xgboost_filter_model.train_filter_v13_wf_image as tv13
from xgboost_filter_model.train_directional_model_v2 import add_directional_features
from xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgboost_filter_model.train_directional_model_v9 import add_momentum_features
df = tv13.prepare_data_v13(start_date='2020-01-01', end_date='2026-05-14', use_cache=True)
df = add_directional_features(df)
df = add_ma_features(df)
df = add_momentum_features(df)
df.dropna(inplace=True)
df_test = df[df.index >= pd.to_datetime('2026-05-12').tz_localize('UTC')].copy()
prod_s1 = joblib.load('xgboost_filter_model/filter_model_v13_wf_image.joblib')
prod_s2 = joblib.load('xgboost_filter_model/directional_model_v13_wf.joblib')
s1_cols = list(prod_s1.feature_names_in_)
s2_cols = list(prod_s2.feature_names_in_)
df_test['s1_prob'] = prod_s1.predict_proba(df_test[s1_cols])[:, 1]
df_test['s2_prob'] = prod_s2.predict_proba(df_test[s2_cols])[:, 1]
cond1 = df_test['imporobimport joblib
impongimport xgboos2from xgboost_filter_model.train_directional_model_v2 import ntfrom xgboost_filter_model.train_directional_model_v3 import add_ma_features
from xgthfrom xgboost_fiEOF
