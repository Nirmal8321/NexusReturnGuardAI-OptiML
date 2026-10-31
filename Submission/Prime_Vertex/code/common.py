"""Shared data prep. Builds train2/test2 (past-only target encoding) and the 65-feature list.
Used by train.py and predict.py so both see identical features."""
import numpy as np
import pandas as pd

TE_KEYS = {'store_id': 50, 'product_id': 20, 'customer_id': 10, 'supplier_code': 50, 'banner_id': 50}  # id: smoothing alpha
ID_COLS = ['order_line_id', 'order_id', 'customer_id', 'product_id', 'store_id', 'supplier_code', 'banner_id']


def add_te(df_fit, df_oth, keys=TE_KEYS):
    """Rows of df_fit get return rate from EARLIER months only (expanding, smoothed).
    Rows of df_oth get the rate from ALL rows of df_fit."""
    a, b = df_fit.copy(), df_oth.copy()
    prior = a['returned'].mean()
    for k, alpha in keys.items():
        c = k + '_cd'
        g = (a.groupby([c, 'month_idx'])['returned'].agg(s='sum', n='count')
               .reset_index().sort_values([c, 'month_idx']))
        g['ps'] = g.groupby(c)['s'].cumsum() - g['s']   # sum of earlier months
        g['pn'] = g.groupby(c)['n'].cumsum() - g['n']   # count of earlier months
        a = a.merge(g[[c, 'month_idx', 'ps', 'pn']], on=[c, 'month_idx'], how='left')
        a[k + '_te'] = ((a['ps'] + prior * alpha) / (a['pn'] + alpha)).astype('float32')
        a = a.drop(columns=['ps', 'pn'])
        tot = df_fit.groupby(c)['returned'].agg(s='sum', n='count')
        s = b[c].map(tot['s']).fillna(0)
        n = b[c].map(tot['n']).fillna(0)
        b[k + '_te'] = ((s + prior * alpha) / (n + alpha)).astype('float32')
    return a, b


def prepare(parquet):
    full = pd.read_parquet(parquet)
    NOT_FEATS = ID_COLS + ['order_ts', 'returned', 'is_train', 'month_idx', 'year']
    FEATS = [c for c in full.columns if c not in NOT_FEATS]
    CATS = [c for c in FEATS if str(full[c].dtype) == 'category']
    for k in TE_KEYS:
        full[k + '_cd'] = pd.factorize(full[k])[0].astype('int32')
    train = full[full.is_train == 1]
    test = full[full.is_train == 0]
    train2, test2 = add_te(train, test)
    assert len(train2) == len(train) and len(test2) == len(test)

    BASE = [c for c in FEATS if c != 'ord_dup_prod']
    P = ['prod_age', 'prod_prelaunch']
    T = ['cpi', 'unemployment', 'fuel_price']
    A = ['store_age', 'cust_tenure']
    TE_COLS = [k + '_te' for k in TE_KEYS]
    FEATS3 = [c for c in BASE if c not in (P + T + A)] + TE_COLS
    FEATS4 = [c for c in FEATS3 if c != 'banner_id_te']
    FEATS5 = [c for c in FEATS4 if c not in ('promo_num', 'promo_channel_code')] + ['banner_id_te']
    CNT = [c for c in ['cust_order_seq', 'cust_days_since_first', 'cust_days_since_prev'] if c in FEATS5]
    FRQ = [c for c in FEATS5 if c.endswith('_freq') or c == 'prod_freq_month']
    ORD = [c for c in ['items_in_order', 'ord_total_qty', 'ord_max_price', 'ord_sum_line_value', 'ord_mean_price',
                       'ord_cat_repeat', 'ord_nunique_prod', 'ord_nunique_cat', 'ord_nunique_color',
                       'ord_n_same_cat', 'hour'] if c in FEATS5]
    FEATS7 = [c for c in FEATS5 if c not in CNT + FRQ + ORD + ['minute1', 'is_placeholder_time']]
    cats = [c for c in FEATS7 if c in CATS]
    return train2, test2, FEATS7, cats


def to_cb(df, feats, cats):
    """CatBoost wants strings for categoricals."""
    x = df[feats].copy()
    for c in cats:
        x[c] = x[c].astype(str)
    return x
