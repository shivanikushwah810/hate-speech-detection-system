"""
train_and_compare.py
====================
Model Comparison pipeline: trains FOUR algorithms on the same data and tests them
on the same held-out split.

    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Linear SVM (LinearSVC)
    4. Random Forest

Fairness rules (no data leakage):
  * Duplicate texts are removed BEFORE splitting.
  * ONE stratified 80/20 split (random_state=42) is used for every model.
  * TF-IDF is fitted on the TRAIN split only; the test split is only transformed.
  * Every model receives exactly the same TF-IDF matrix.
  * No hyper-parameter is tuned on the test set (fixed, documented settings).
  * Training time is measured per model (model.fit only; vectorizer time is reported separately).

Output:
  reports/comparison_results.json         (the dashboard reads this)
  reports/history/comparison_<time>.json  (archived copy of every run)
  models/candidates/<slug>.pkl            (vectorizer + model pipeline)
  models/candidates/<slug>.meta.json      (metadata: versions, params, metrics, dataset fingerprint)
The production model files in models/ are NEVER touched by this script.

Run:
    python train_and_compare.py
"""

import argparse
import os
import shutil
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from utils.ml_common import (CANDIDATES_DIR, COMPARISON_REPORT, DEFAULT_DATASET,  # noqa: E402
                             HISTORY_DIR, RANDOM_STATE, DataError, Timer,
                             bootstrap_macro_f1, compute_metrics, environment_info,
                             now_iso, prepare_data, write_json)

# Same TF-IDF settings as the production vectorizer (models/adam_vectorizer.pkl)
VECTORIZER_PARAMS = dict(max_features=25000, ngram_range=(1, 3), sublinear_tf=True,
                         max_df=0.95, token_pattern=r'\b\w+\b')

CANDIDATES = [
    {
        'name': 'Logistic Regression',
        'slug': 'logistic_regression',
        'build': lambda: LogisticRegression(solver='lbfgs', class_weight='balanced',
                                            max_iter=1000, random_state=RANDOM_STATE),
        'supports_proba': True,
        'compatibility': 'Linear model on TF-IDF features (lbfgs solver, balanced class weights). '
                         'Gives class probabilities, so it can drop into the current app.',
    },
    {
        'name': 'Multinomial Naive Bayes',
        'slug': 'multinomial_nb',
        'build': lambda: MultinomialNB(alpha=1.0),
        'supports_proba': True,
        'compatibility': 'Fast probabilistic model that works on non-negative TF-IDF features. '
                         'Has no class_weight option, so class imbalance is not compensated.',
    },
    {
        'name': 'Linear SVM',
        'slug': 'linear_svm',
        'build': lambda: LinearSVC(C=1.0, class_weight='balanced', dual='auto',
                                   random_state=RANDOM_STATE, max_iter=5000),
        'supports_proba': False,
        'compatibility': 'Strong on high-dimensional sparse text. Gives no probabilities, so it '
                         'cannot replace the production model (which blends probabilities) '
                         'without extra calibration.',
    },
    {
        'name': 'Random Forest',
        'slug': 'random_forest',
        'build': lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample',
                                                n_jobs=-1, random_state=RANDOM_STATE),
        'supports_proba': True,
        'compatibility': '200 decision trees (balanced class weights). Handles non-linear patterns but '
                         'is usually the slowest and largest model on sparse 25,000-feature TF-IDF text.',
    },
]


def main():
    ap = argparse.ArgumentParser(description='Fair comparison of LR / MNB / Linear SVM / Random Forest.')
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--text-col', default=None)
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--no-save-models', action='store_true',
                    help='do not write models/candidates/*.pkl')
    args = ap.parse_args()

    print('[1/5] Loading dataset, removing duplicates, creating stratified split ...')
    try:
        data = prepare_data(args.dataset, args.text_col, args.label_col)
    except DataError as e:
        print(f'\n[ERROR] {e}')
        return 1
    info = data['info']
    y_train, y_test = data['y_train'], data['y_test']
    print(f"      train={info['n_train']}  test={info['n_test']}  "
          f"(duplicates removed: {info['duplicates_or_empty_removed']})")

    print('[2/5] Fitting TF-IDF on the TRAIN split only (no leakage) ...')
    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    with Timer() as t_vec:
        X_train = vec.fit_transform(data['train_clean'])
    X_test = vec.transform(data['test_clean'])
    print(f'      features={X_train.shape[1]}  vectorizer fit time={t_vec.seconds:.2f}s')

    results, preds = {}, {}
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    for n, spec in enumerate(CANDIDATES, 1):
        print(f"[3/5] ({n}/{len(CANDIDATES)}) Training {spec['name']} ...")
        clf = spec['build']()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with Timer() as t_fit:
                clf.fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        with Timer() as t_pred:
            y_pred = clf.predict(X_test).astype(int)
        preds[spec['name']] = y_pred
        metrics = compute_metrics(y_test, y_pred)

        saved, meta_file, size_mb = None, None, None
        if not args.no_save_models:
            saved = f"models/candidates/{spec['slug']}.pkl"
            meta_file = f"models/candidates/{spec['slug']}.meta.json"
            joblib.dump(Pipeline([('tfidf', vec), ('clf', clf)]), os.path.join(BASE_DIR, saved),
                        compress=3)
            size_mb = round(os.path.getsize(os.path.join(BASE_DIR, saved)) / 1e6, 2)

        results[spec['name']] = {
            'order': n,
            'slug': spec['slug'],
            'metrics': metrics,
            'train_time_sec': round(t_fit.seconds, 3),
            'predict_time_sec': round(t_pred.seconds, 4),
            'converged': bool(converged),
            'supports_predict_proba': spec['supports_proba'],
            'compatibility': spec['compatibility'],
            'saved_model_file': saved,
            'saved_meta_file': meta_file,
            'model_size_mb': size_mb,
            'params': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in clf.get_params().items()},
        }
        print(f"      acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
              f"weightedF1={metrics['weighted_f1']:.4f}  train={t_fit.seconds:.2f}s"
              + ('' if converged else '  [WARNING: did not fully converge]'))

    print('[4/5] Bootstrap confidence intervals (1000 resamples of the test set) ...')
    boot = bootstrap_macro_f1(y_test, preds)
    for name, b in boot.items():
        results[name]['macro_f1_ci95'] = b['ci95']
        results[name]['diff_vs_best_ci95'] = b.get('diff_vs_best_ci95')
        results[name]['clearly_worse_than_best'] = b.get('clearly_worse_than_best')

    # Best = highest macro-F1 (primary metric because classes are imbalanced);
    # weighted-F1 only breaks an exact tie.
    best = max(results, key=lambda k: (round(results[k]['metrics']['macro_f1'], 12),
                                       results[k]['metrics']['weighted_f1']))
    others_clear = all(results[k].get('clearly_worse_than_best') for k in results if k != best)
    tied = [k for k in results if k != best and not results[k].get('clearly_worse_than_best')]

    print('[5/5] Saving report and model metadata ...')
    env = environment_info()
    if not args.no_save_models:
        for name, r in results.items():
            write_json(os.path.join(BASE_DIR, r['saved_meta_file']), {
                'model': name, 'slug': r['slug'], 'trained_at': now_iso(),
                'environment': env, 'dataset': {k: info[k] for k in (
                    'file', 'sha256_prefix', 'random_state', 'test_size', 'n_train', 'n_test')},
                'vectorizer_params': {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in VECTORIZER_PARAMS.items()},
                'params': r['params'], 'train_time_sec': r['train_time_sec'],
                'converged': r['converged'], 'metrics': {k: r['metrics'][k] for k in (
                    'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1')},
                'is_production_model': False,
            })
    report = {
        'generated_at': now_iso(),
        'environment': env,
        'dataset': info,
        'vectorizer': {'type': 'TfidfVectorizer', **{k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in VECTORIZER_PARAMS.items()},
                       'n_features': int(X_train.shape[1]),
                       'fit_time_sec': round(t_vec.seconds, 3),
                       'fitted_on': 'training split only'},
        'models': results,
        'best_model': best,
        'selection_criterion': 'highest macro-F1 on the held-out test split '
                               '(weighted-F1 only as tie-breaker)',
        'best_is_statistically_clear': bool(others_clear),
        'tied_with_best': tied,
        'notes': [
            'Single stratified split, single run: small differences can be noise '
            '(see the bootstrap 95% intervals).',
            'Timings depend on this computer and one run each.',
            'No hyper-parameter tuning was done; defaults / production settings were used.',
            'Random Forest uses a fixed 200 trees (not tuned); it is the slowest and largest model.',
            'The production model was NOT replaced by this script. See promote_model.py.',
        ],
    }
    write_json(COMPARISON_REPORT, report)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    shutil.copy2(COMPARISON_REPORT, os.path.join(
        HISTORY_DIR, 'comparison_' + report['generated_at'][:19].replace(':', '-') + '.json'))
    print(f'\nDone. Best by macro-F1: {best}')
    print('Saved: reports/comparison_results.json')
    print('Your production model files in models/ were NOT modified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
