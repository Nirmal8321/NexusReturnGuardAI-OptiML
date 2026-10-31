Team: Prime Vertex | NEXUS OptiML 2026 - Operation Insight | Project name: NexusReturnGuard
Submitted file (submission.csv): rank blend of LightGBM (3 seeds) + CatBoost, public LB 0.56367.

CONTENTS
  submission.csv        exact file submitted on Kaggle (not regenerated, not edited)
  writeup.txt           one-page write-up
  declaration.txt       signed by both members
  requirements.txt      library versions
  code/01_features.py   cleaning, merges, feature engineering (from Kaggle notebook nexusreturnguard-02-features)
  code/common.py        target encoding + final 65-feature list (shared)
  code/train.py         trains final models on all 24 train months, writes submission
  code/predict.py       regenerates the submission from the saved models in model/
  code/notebooks/       all Kaggle notebooks (EDA, features, modelling experiments, final), incl. failed experiments
  model/                lgb_seed42.txt, lgb_seed7.txt, lgb_seed11.txt, catboost.cbm

HOW TO REPRODUCE (working directory = the unzipped folder, Python version: 3.13.15)
  1. pip install -r requirements.txt
  2. Put the 7 competition files (train, test, sample_submission, customers, products, stores, calendar) in ./data
  3. DATA_DIR=./data python code/01_features.py
        -> writes full.parquet (1,880,000 rows x 104 columns), about 10 min
  4. python code/train.py --parquet full.parquet --sample data/sample_submission.csv --out submission_rerun.csv --model_dir model --compare submission.csv
        -> retrains and writes submission_rerun.csv, about 50 min on CPU; prints agreement with submission.csv
  5. (optional, no training) python code/predict.py --parquet full.parquet --sample data/sample_submission.csv --out submission_from_models.csv --model_dir model
        -> uses the saved models in model/ and reproduces the submitted probabilities

RANDOM SEEDS
  LightGBM seeds 42, 7, 11 (averaged). CatBoost seed 42. Fuzzy label matching is deterministic.

NOTES
  - Final prediction = 0.5 * rank(LightGBM mean of 3 seeds) + 0.5 * rank(CatBoost), ranks scaled to 0-1.
  - Multi-threaded training can change the last decimals between runs. The ranking matches, and submission.csv
    is the exact file submitted. model/ was saved from rerun with same code and seeds.
  - Forum notebooks used as starting point: none.
