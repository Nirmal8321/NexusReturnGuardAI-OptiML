"""Train final models on ALL 24 train months and write the submission.
LightGBM (seeds 42, 7, 11) + CatBoost (seed 42), rank-average 50/50.
Usage: python code/train.py --parquet full.parquet --sample sample_submission.csv --out submission_rerun.csv --model_dir model [--compare submission.csv]
"""
import argparse, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from common import prepare, to_cb

ap = argparse.ArgumentParser()
ap.add_argument('--parquet', default='full.parquet')
ap.add_argument('--sample', required=True)
ap.add_argument('--out', default='submission_rerun.csv')
ap.add_argument('--model_dir', default='model')
ap.add_argument('--compare', default=None, help='the submitted Kaggle csv, to print agreement')
ap.add_argument('--smoke', action='store_true', help='tiny plumbing test, NOT for submission')
args = ap.parse_args()
os.makedirs(args.model_dir, exist_ok=True)

train2, test2, FEATS7, cats = prepare(args.parquet)
print(len(FEATS7), 'features |', len(cats), 'categorical')   # expected: 65 | 16

dtrain = train2.sample(min(100_000, len(train2)), random_state=1) if args.smoke else train2
n_lgb, n_cb = (30, 30) if args.smoke else (940, 700)

fast = dict(objective='binary', metric='auc', learning_rate=0.1, num_leaves=31, min_child_samples=200,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, max_bin=63,
            cat_smooth=20, min_data_per_group=200, n_jobs=-1, verbose=-1, seed=42)

t0, pl = time.time(), []
dall = lgb.Dataset(dtrain[FEATS7], dtrain['returned'], categorical_feature=cats, free_raw_data=False)
for s in [42, 7, 11]:
    m = lgb.train(dict(fast, seed=s), dall, n_lgb)
    m.save_model(os.path.join(args.model_dir, f'lgb_seed{s}.txt'))
    pl.append(m.predict(test2[FEATS7]))
    print('lgb seed', s, f'{time.time()-t0:.0f}s')
p_lgb = np.mean(pl, axis=0)

cb = CatBoostClassifier(iterations=n_cb, learning_rate=0.1, depth=7, border_count=64, l2_leaf_reg=5,
                        thread_count=-1, random_seed=42, verbose=100)
cb.fit(Pool(to_cb(dtrain, FEATS7, cats), dtrain['returned'], cat_features=cats))
cb.save_model(os.path.join(args.model_dir, 'catboost.cbm'))
p_cb = cb.predict_proba(Pool(to_cb(test2, FEATS7, cats), cat_features=cats))[:, 1]
print(f'cat done {time.time()-t0:.0f}s')

r = lambda x: rankdata(x) / len(x)
blend = 0.5 * r(p_lgb) + 0.5 * r(p_cb)
print('lgb mean', p_lgb.mean().round(4), '| cat mean', p_cb.mean().round(4), '| corr', np.corrcoef(p_lgb, p_cb)[0, 1].round(4))

sub = pd.read_csv(args.sample)
sub['returned'] = sub['order_line_id'].map(pd.Series(blend, index=test2['order_line_id'].values)).values
assert len(sub) == len(test2) and sub['returned'].notna().all() and sub['returned'].between(0, 1).all()
sub.to_csv(args.out, index=False)
print('saved', args.out, sub.shape)

if args.compare:
    ref = pd.read_csv(args.compare)
    m_ = sub.merge(ref, on='order_line_id', suffixes=('_new', '_ref'))
    print('rows matched:', len(m_), '| Pearson corr:', np.corrcoef(m_.returned_new, m_.returned_ref)[0, 1].round(6),
          '| max abs diff:', (m_.returned_new - m_.returned_ref).abs().max().round(6))
