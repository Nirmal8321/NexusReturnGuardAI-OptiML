import os
# Feature engineering (NB02). Set DATA_DIR to the folder with the 7 competition csv files.


import pandas as pd, numpy as np, re, difflib, gc
from pathlib import Path

DATA = Path(os.environ.get('DATA_DIR', '/kaggle/input/competitions/nexus-opti-ml-2026-operation-insight'))

tr = pd.read_csv(DATA/'train.csv')
te = pd.read_csv(DATA/'test.csv')
cu = pd.read_csv(DATA/'customers.csv')
pr = pd.read_csv(DATA/'products.csv')
st = pd.read_csv(DATA/'stores.csv')
ca = pd.read_csv(DATA/'calendar.csv')

N_TR, N_TE = len(tr), len(te)
assert (N_TR, N_TE) == (1_400_005, 479_995)

tr['is_train'] = 1
te['is_train'] = 0
te['returned'] = np.nan
full = pd.concat([tr, te], ignore_index=True)
del tr, te; gc.collect()

full['order_ts']          = pd.to_datetime(full['order_ts'])
cu['signup_date']         = pd.to_datetime(cu['signup_date'])
cu['record_updated_at']   = pd.to_datetime(cu['record_updated_at'])
pr['launch_date']         = pd.to_datetime(pr['launch_date'])
st['opened_date']         = pd.to_datetime(st['opened_date'])
ca['week_start']          = pd.to_datetime(ca['week_start'])

N = len(full)
print(full.shape, full.is_train.sum(), (full.is_train == 0).sum())

def norm(s):
    return re.sub('[^a-z]', '', str(s).lower())

def fuzzy_map(raw, ref, cutoff=0.8):
    ref_norm = {norm(r): r for r in ref}
    keys = list(ref_norm)
    out = {}
    for v in raw.dropna().unique():
        n = norm(v)
        if n in ref_norm:
            out[v] = ref_norm[n]
        else:
            m = difflib.get_close_matches(n, keys, n=1, cutoff=cutoff)
            out[v] = ref_norm[m[0]] if m else np.nan
    return out

# customer region -> 22 clean store regions
ref_region = sorted(st['region'].dropna().unique())
cu['region'] = cu['region'].map(fuzzy_map(cu['region'], ref_region))

# product category -> 12 most frequent labels
ref_cat = pr['category'].value_counts().head(12).index.tolist()
pr['category'] = pr['category'].map(fuzzy_map(pr['category'], ref_cat))

print('cust regions :', cu['region'].nunique(), '| NaN:', cu['region'].isna().sum())
print('prod category:', pr['category'].nunique(), '| NaN:', pr['category'].isna().sum())

# rename clashing columns BEFORE merge
cu = cu.rename(columns={'region': 'cust_region'})
st = st.rename(columns={'channel': 'store_channel',
                        'region':  'store_region',
                        'format':  'store_format'})

# 1. latest record per customer (static fields + fallback tier)
cu_latest = (cu.sort_values('record_updated_at')
               .drop_duplicates('customer_id', keep='last'))
assert cu_latest['customer_id'].is_unique and len(cu_latest) == 201_901

# 2. merge, one row per customer guaranteed by validate
full = full.merge(cu_latest.drop(columns='record_updated_at'),
                  on='customer_id', how='left', validate='m:1')
assert len(full) == N

# 3. as-of loyalty tier for the duplicated customers only
cu_hist = (cu[cu['customer_id'].duplicated(keep=False)]
             [['customer_id', 'record_updated_at', 'loyalty_tier']]
             .sort_values('record_updated_at'))

sub = full.loc[full['customer_id'].isin(cu_hist['customer_id']),
               ['customer_id', 'order_ts']].copy()
sub['row'] = sub.index
sub = sub.sort_values('order_ts')

m = pd.merge_asof(sub, cu_hist,
                  left_on='order_ts', right_on='record_updated_at',
                  by='customer_id', direction='backward')
m = m.dropna(subset=['loyalty_tier'])

changed = (full.loc[m['row'], 'loyalty_tier'].values != m['loyalty_tier'].values).sum()
full.loc[m['row'].values, 'loyalty_tier'] = m['loyalty_tier'].values
print('rows with as-of tier different from latest:', changed)
assert len(full) == N

full['week_start'] = (full['order_ts'].dt.normalize()
                      - pd.to_timedelta(full['order_ts'].dt.weekday, unit='D'))

full = full.merge(pr, on='product_id', how='left', validate='m:1')
assert len(full) == N

full = full.merge(st, on='store_id', how='left', validate='m:1')
assert len(full) == N

full = full.merge(ca, on='week_start', how='left', validate='m:1')
assert len(full) == N

print('orphans -> category:', full['category'].isna().mean().round(4),
      '| store_format:', full['store_format'].isna().mean().round(4),
      '| cpi:', full['cpi'].isna().mean().round(4))

ts = full['order_ts']
full['year']        = ts.dt.year.astype('int16')
full['month']       = ts.dt.month.astype('int8')
full['weekday']     = ts.dt.weekday.astype('int8')
full['hour']        = ts.dt.hour.astype('int8')
full['minute']      = ts.dt.minute.astype('int8')
full['week_of_year']= ts.dt.isocalendar().week.astype('int8')

full['minute1']  = (full['minute'] == 1).astype('int8')
full['is_placeholder_time'] = (full['hour'].isin([0, 3, 23]) & (full['minute'] == 1)).astype('int8')

# months since Jan 2023: Jan 2023 = 1, Dec 2024 = 24, Jun 2025 = 30
full['month_idx'] = ((ts.dt.year - 2023) * 12 + ts.dt.month).astype('int8')

print(full.groupby('is_train')['month_idx'].agg(['min', 'max']))
print(full['minute1'].mean().round(3), full['is_placeholder_time'].mean().round(3))

d = full['order_ts'].dt.normalize()

full['cust_tenure'] = (d - full['signup_date']).dt.days.clip(lower=0).astype('float32')
full['prod_age']    = (d - full['launch_date']).dt.days.astype('float32')      # negative allowed
full['store_age']   = ((d - full['opened_date']).dt.days / 365.25).astype('float32')

full['prod_prelaunch'] = (full['prod_age'] < 0).astype('int8')

print('prod_age < 0 :', full['prod_prelaunch'].mean().round(3))
print('signup after order (pre-clip approx):', ((d - full['signup_date']).dt.days < 0).mean().round(4))

full['line_value'] = (full['unit_price'] * full['quantity']).astype('float32')

full['disc_price_ratio'] = full['unit_price'] / full['list_price']
full['value_gap']        = full['order_value'] / full['line_value']
full['freight_per_unit'] = full['freight_cost'] / full['quantity']
full['freight_ratio']    = full['freight_cost'] / full['order_value']

for c in ['unit_price', 'quantity', 'order_value', 'line_value', 'freight_cost']:
    full['log_' + c] = np.log1p(full[c].clip(lower=0)).astype('float32')

# missing flags (Phase 1: small but real signal)
for c in ['session_clicks', 'freight_cost', 'discount']:
    full[c + '_missing'] = full[c].isna().astype('int8')
full['has_coupon'] = full['coupon_code'].notna().astype('int8')

full = full.replace([np.inf, -np.inf], np.nan)
num = ['disc_price_ratio', 'value_gap', 'freight_per_unit', 'freight_ratio']
full[num] = full[num].astype('float32')
print(full[num].describe().T[['mean', '50%', 'max']])

g = full.groupby('order_id')

agg = g.agg(
    ord_sum_line_value = ('line_value',  'sum'),
    ord_max_price      = ('unit_price',  'max'),
    ord_mean_price     = ('unit_price',  'mean'),
    ord_nunique_prod   = ('product_id',  'nunique'),
    ord_nunique_cat    = ('category',    'nunique'),
    ord_nunique_color  = ('color',       'nunique'),
    ord_total_qty      = ('quantity',    'sum'),
)
full = full.join(agg, on='order_id')
assert len(full) == N

full['line_share']      = full['line_value'] / full['ord_sum_line_value']
full['price_vs_max']    = full['unit_price'] / full['ord_max_price']
full['price_rank']      = full.groupby('order_id')['unit_price'].rank(ascending=False, method='average')
full['price_rank_pct']  = full['price_rank'] / full['items_in_order']
full['line_pos']        = full['line_no'] / full['items_in_order']

# bracketing: same product / same category repeated inside the order
full['ord_n_same_prod'] = full.groupby(['order_id', 'product_id'])['product_id'].transform('size')
full['ord_n_same_cat']  = full.groupby(['order_id', 'category'])['category'].transform('size')
full['ord_dup_prod']    = (full['ord_n_same_prod'] > 1).astype('int8')
full['ord_cat_repeat']  = (full['items_in_order'] - full['ord_nunique_cat']).astype('float32')

new = [c for c in full.columns if c.startswith('ord_') or c in
       ['line_share','price_vs_max','price_rank','price_rank_pct','line_pos']]
full[new] = full[new].astype('float32')
print(full[['ord_nunique_prod', 'ord_n_same_prod', 'line_share']].describe().T[['mean', 'max']])

# region match: customer region == store region
ok = full['cust_region'].notna() & full['store_region'].notna()
full['region_match'] = np.where(ok, (full['cust_region'] == full['store_region']).astype('float32'), np.nan)

# promo code as ordinal + keep original as category later
full['promo_num'] = full['promo_channel_code'].astype(str).str.extract(r'(\d+)')[0].astype('float32')

# channel x format
full['chan_format'] = full['channel'].astype(str) + '_' + full['store_format'].astype(str)

# customer order history (label-free): one row per order
od = (full.drop_duplicates('order_id')[['order_id', 'customer_id', 'order_ts']]
          .sort_values(['customer_id', 'order_ts', 'order_id']))
gc_ = od.groupby('customer_id')['order_ts']
od['cust_order_seq']        = od.groupby('customer_id').cumcount().astype('float32')
od['cust_days_since_first'] = (od['order_ts'] - gc_.transform('min')).dt.days.astype('float32')
od['cust_days_since_prev']  = gc_.diff().dt.days.astype('float32')

full = full.merge(od[['order_id', 'cust_order_seq', 'cust_days_since_first', 'cust_days_since_prev']],
                  on='order_id', how='left', validate='m:1')
assert len(full) == N

print(full['region_match'].value_counts(dropna=False, normalize=True).round(3))
print(full['promo_num'].value_counts().sort_index())

for c in ['store_id', 'product_id', 'customer_id', 'supplier_code', 'banner_id']:
    full[c + '_freq'] = full[c].map(full[c].value_counts()).astype('float32')

# product/store popularity relative to its own month (volume drift guard)
full['prod_freq_month'] = (full.groupby(['product_id', 'month_idx'])['product_id']
                               .transform('size').astype('float32'))

print(full[[c for c in full.columns if c.endswith('_freq')]].isna().sum())

drop_cols = ['coupon_code', 'size_sqm', 'warehouse_zone', 'picker_shift', 'gift_wrap', 'tz_hint',
             'signup_date', 'launch_date', 'opened_date', 'week_start', 'minute']
full = full.drop(columns=[c for c in drop_cols if c in full.columns])

cat_cols = ['promo_channel_code', 'channel', 'payment_method', 'ship_mode', 'device_type',
            'store_channel', 'store_region', 'cust_region', 'store_format', 'customer_type',
            'loyalty_tier', 'category', 'packaging_type', 'color', 'age_band',
            'preferred_contact', 'chan_format']
for c in cat_cols:
    full[c] = full[c].astype('category')

for c in full.select_dtypes('float64').columns:
    full[c] = full[c].astype('float32')
for c in full.select_dtypes('int64').columns:
    full[c] = pd.to_numeric(full[c], downcast='integer')
full = full.replace([np.inf, -np.inf], np.nan)

# final checks
assert len(full) == N == 1_880_000
assert full['order_line_id'].is_unique
assert full.loc[full.is_train == 1, 'returned'].notna().all()
assert full.loc[full.is_train == 0, 'returned'].isna().all()
print('train return rate:', full.loc[full.is_train == 1, 'returned'].mean().round(4))
print('memory MB:', round(full.memory_usage(deep=True).sum() / 1e6))
print('cols:', full.shape[1])
print(full.isna().mean().sort_values(ascending=False).head(10).round(3))

full.to_parquet('full.parquet', index=False)
