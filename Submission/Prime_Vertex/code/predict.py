"""Regenerate predictions from the SAVED models in model/ (no training).
Usage: python code/predict.py --parquet full.parquet --sample sample_submission.csv --out submission_from_models.csv --model_dir model
"""
import argparse, os
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from common import prepare, to_cb

ap = argparse.ArgumentParser()
ap.add_argument('--parquet', default='full.parquet')
ap.add_argument('--sample', required=True)
ap.add_argument('--out', default='submission_from_models.csv')
ap.add_argument('--model_dir', default='model')
args = ap.parse_args()

train2, test2, FEATS7, cats = prepare(args.parquet)
p_lgb = np.mean([lgb.Booster(model_file=os.path.join(args.model_dir, f'lgb_seed{s}.txt')).predict(test2[FEATS7])
                 for s in [42, 7, 11]], axis=0)
cb = CatBoostClassifier(); cb.load_model(os.path.join(args.model_dir, 'catboost.cbm'))
p_cb = cb.predict_proba(Pool(to_cb(test2, FEATS7, cats), cat_features=cats))[:, 1]
r = lambda x: rankdata(x) / len(x)
blend = 0.5 * r(p_lgb) + 0.5 * r(p_cb)
sub = pd.read_csv(args.sample)
sub['returned'] = sub['order_line_id'].map(pd.Series(blend, index=test2['order_line_id'].values)).values
assert len(sub) == len(test2) and sub['returned'].notna().all() and sub['returned'].between(0, 1).all()
sub.to_csv(args.out, index=False)
print('saved', args.out, sub.shape)
