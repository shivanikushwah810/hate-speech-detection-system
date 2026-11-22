"""
retrain_production.py
=====================
Fixes the "Model saved with scikit-learn X but installed version is Y" problem
the honest way: it re-creates the production model files with the scikit-learn
version that is installed right now.

What it does
------------
1. Loads data/dataset.csv and makes the SAME stratified split as every other
   script (seed 42, 80/20, duplicates removed first).
2. Fits TF-IDF + Logistic Regression on the TRAIN split only:
       TF-IDF : max_features=25000, ngram_range=(1,3), sublinear_tf, max_df=0.95
       LR     : class_weight='balanced', C=1.0, solver='lbfgs', max_iter=1000
   The original model used solver='saga' (max_iter=2000). On the full 19.5k-text
   training split saga does NOT converge within 2000 iterations (macro-F1 ~0.66
   instead of ~0.73), so the default is now lbfgs, which converges. Use
   `--solver saga` to reproduce the original settings anyway.
   (Because only the train split is used, Model Evaluation on the test split is
    a genuine held-out result.)
3. BEFORE overwriting anything, copies the current models/adam_*.pkl to
   models/legacy_sklearn_<version>/ (never overwritten if it already exists).
4. Writes models/adam_classifier.pkl, models/adam_vectorizer.pkl and
   models/production_model_meta.json (versions, dataset fingerprint, split).

It asks for confirmation (type YES) unless you pass --yes.
Restart the Flask app afterwards so it loads the new files.

Run:
    python retrain_production.py          # asks first
    python retrain_production.py --yes    # no question
Undo:
    python promote_model.py --restore models/legacy_sklearn_<version>
"""

import argparse
import hashlib
import os
import pickle
import shutil
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from utils import production_model as pm  # noqa: E402
from utils.ml_common import (DEFAULT_DATASET, MODELS_DIR, RANDOM_STATE, DataError,  # noqa: E402
                             Timer, compute_metrics, environment_info, now_iso,
                             prepare_data, write_json)

VECTORIZER_PARAMS = dict(max_features=25000, ngram_range=(1, 3), sublinear_tf=True,
                         max_df=0.95, token_pattern=r'\b\w+\b')
LR_PARAMS = dict(solver='lbfgs', class_weight='balanced', C=1.0, max_iter=1000,
                 random_state=RANDOM_STATE)
SAGA_PARAMS = dict(LR_PARAMS, solver='saga', max_iter=2000)   # the original production settings


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description='Retrain the production model with the installed scikit-learn.')
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--yes', action='store_true', help='do not ask for confirmation')
    ap.add_argument('--solver', choices=['lbfgs', 'saga'], default='lbfgs',
                    help='lbfgs (default, converges) or saga (original settings, may not converge)')
    args = ap.parse_args()
    lr_params = LR_PARAMS if args.solver == 'lbfgs' else SAGA_PARAMS

    try:
        data = prepare_data(args.dataset)
    except DataError as e:
        print(f'[ERROR] {e}')
        return 1
    info = data['info']

    import sklearn
    clf0, vec0, err0 = pm.load_artifacts()
    st = pm.status(clf0, vec0, err0)
    print(f"Installed scikit-learn : {sklearn.__version__}")
    print(f"Current model files    : saved with scikit-learn {st.get('pickle_sklearn')}"
          if st['loaded'] else f"Current model files    : {st['message']}")
    print(f"Training data          : {info['n_train']} train texts of {info['rows_used_after_dedup']} "
          f"(test split of {info['n_test']} stays untouched)")
    if not args.yes:
        print('\nThis REPLACES models/adam_classifier.pkl and models/adam_vectorizer.pkl '
              '(a backup is made first).')
        if input('Type YES to continue: ').strip() != 'YES':
            print('Cancelled. Nothing was changed.')
            return 0

    # ── backup ────────────────────────────────────────────────────────────
    # If the files on disk were already produced by this script (meta matches their
    # hashes) they are not an original: keep the existing backup, do not back them up.
    prev = pm.read_meta() if st['loaded'] and st.get('meta_matches_files') else None
    if prev:
        old_ver = prev.get('replaces_model_saved_with_sklearn') or 'unknown'
        backup = os.path.join(BASE_DIR, prev['backup_folder']) if prev.get('backup_folder') else None
        print('Current files were created by this script earlier; the original backup is kept: '
              f"{prev.get('backup_folder')}")
    else:
        old_ver = st.get('pickle_sklearn') or 'unknown'
        backup = os.path.join(MODELS_DIR, f'legacy_sklearn_{old_ver}')
        if st['loaded'] and not os.path.exists(backup):
            os.makedirs(backup)
            for fn in ('adam_classifier.pkl', 'adam_vectorizer.pkl'):
                shutil.copy2(os.path.join(MODELS_DIR, fn), os.path.join(backup, fn))
            print(f'Backup of the old model: {os.path.relpath(backup, BASE_DIR)}')
        elif os.path.exists(backup):
            print(f'Backup folder already exists, kept as it is: {os.path.relpath(backup, BASE_DIR)}')

    print(f'Fitting TF-IDF + Logistic Regression ({lr_params['solver']}) on the train split ...')
    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train = vec.fit_transform(data['train_clean'])
    clf = LogisticRegression(**lr_params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with Timer() as t:
            clf.fit(X_train, data['y_train'])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f'  trained in {t.seconds:.1f}s, converged={converged}, iterations={int(clf.n_iter_[0])}')
    if not converged:
        print('[WARNING] The solver did not converge; the model is still saved (see the meta file).')

    X_test = vec.transform(data['test_clean'])
    m = compute_metrics(data['y_test'], clf.predict(X_test).astype(int))
    print(f"  held-out test: accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    for fn, obj in (('adam_classifier.pkl', clf), ('adam_vectorizer.pkl', vec)):
        tmp = os.path.join(MODELS_DIR, fn + '.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(MODELS_DIR, fn))

    write_json(os.path.join(MODELS_DIR, 'production_model_meta.json'), {
        'trained_at': now_iso(),
        'environment': environment_info(),
        'replaces_model_saved_with_sklearn': old_ver,
        'backup_folder': os.path.relpath(backup, BASE_DIR) if backup and os.path.exists(backup) else None,
        'trained_on': 'training split only (test split never seen)',
        'dataset': {k: info[k] for k in ('file', 'sha256_prefix', 'test_size', 'random_state',
                                        'n_train', 'n_test', 'rows_used_after_dedup')},
        'vectorizer_params': {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in VECTORIZER_PARAMS.items()},
        'classifier_params': dict(lr_params),
        'converged': bool(converged),
        'n_iter': int(clf.n_iter_[0]),
        'train_time_sec': round(t.seconds, 3),
        'file_sha256': {'adam_classifier.pkl': _sha(os.path.join(MODELS_DIR, 'adam_classifier.pkl')),
                        'adam_vectorizer.pkl': _sha(os.path.join(MODELS_DIR, 'adam_vectorizer.pkl'))},
        'heldout_metrics': {k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall',
                                              'macro_f1', 'weighted_f1')},
    })

    c2, v2, e2 = pm.load_artifacts()
    st2 = pm.status(c2, v2, e2)
    print(f"\nNew status: [{st2['level']}] {st2['message']}")
    print('Restart the Flask app so it loads the new model, then run: python evaluate_model.py')
    return 0 if st2['level'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
