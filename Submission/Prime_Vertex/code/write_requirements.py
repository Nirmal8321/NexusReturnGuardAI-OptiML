"""Writes requirements.txt with the exact installed versions. Run once in the environment that produced the submission."""
from importlib.metadata import version
pk = ['pandas', 'numpy', 'scipy', 'scikit-learn', 'lightgbm', 'catboost', 'pyarrow']
open('requirements.txt', 'w').write('\n'.join(f'{p}=={version(p)}' for p in pk) + '\n')
import sys; print('python', sys.version.split()[0]); print(open('requirements.txt').read())
