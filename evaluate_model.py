"""
evaluate_model.py
=================
FEATURE 1 — Model Evaluation pipeline.

Ye script tumhare EXISTING model ko labelled test data par chalati hai aur
real predictions se metrics nikalti hai (kuch bhi hardcoded nahi hai).
Result reports/evaluation_results.json mein save hota hai, jise web page
"Model Evaluation" dikhata hai.

Evaluated (all on the same held-out test split):
  1. production_ml      -> TF-IDF + Logistic Regression in models/*.pkl (current production model)
  2. production_hybrid  -> the app's local pipeline: ML + keyword rules + overrides
                           (Claude API / BiLSTM are NOT included: paid / non-deterministic)
  3. legacy_ml / legacy_hybrid -> the ORIGINAL model saved with scikit-learn 1.8.0
                           (models/legacy_sklearn_1.8.0/), preserved for comparison

Run:
    python evaluate_model.py
    python evaluate_model.py --dataset data/dataset.csv --text-col tweet --label-col class
"""

import argparse
import collections
import contextlib
import os
import shutil
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)          # predictor.py uses relative paths like 'models/...'

import numpy as np  # noqa: E402

from utils import production_model as pm  # noqa: E402
from utils.ml_common import (DEFAULT_DATASET, EVALUATION_REPORT, HISTORY_DIR,  # noqa: E402
                             LABELS, MODELS_DIR, PREDICTION_TO_ID, DataError,
                             compute_metrics, environment_info, now_iso, prepare_data,
                             write_json)

LEGACY_DIR = os.path.join(MODELS_DIR, 'legacy_sklearn_1.8.0')

HELD_OUT_OK = ("Verified: models/production_model_meta.json shows this model was fitted on the TRAINING "
               "split only (same dataset fingerprint, seed and split sizes as this evaluation), and the "
               "model files match the recorded checksums. The test texts were never seen in training.")
HELD_OUT_NOTE = (
    "The training data of the existing production model is NOT included in this project, so it "
    "cannot be proven that none of these test rows were seen during its training. The numbers "
    "show how the model performs on this dataset's test split. For a guaranteed held-out "
    "result use the retrained candidates on the Model Comparison page (trained on the train "
    "split only)."
)


def main():
    ap = argparse.ArgumentParser(description='Evaluate the existing HateShield model.')
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--text-col', default=None)
    ap.add_argument('--label-col', default=None)
    args = ap.parse_args()

    print('[1/4] Loading dataset and creating the stratified split ...')
    try:
        data = prepare_data(args.dataset, args.text_col, args.label_col)
    except DataError as e:
        print(f'\n[ERROR] {e}')
        return 1
    info = data['info']
    y_test = data['y_test']
    print(f"      train={info['n_train']}  test={info['n_test']}  "
          f"(duplicates removed: {info['duplicates_or_empty_removed']})")

    print('[2/4] Loading production model ...')
    clf, vec, err = pm.load_artifacts()
    model_status = pm.status(clf, vec, err)
    print(f"      status: {model_status['level']} — {model_status['message']}")

    models = {}

    # Is the production model PROVABLY trained on the train split only?
    meta = pm.read_meta()
    ds = info
    held_out = bool(
        meta and model_status.get('meta_matches_files')
        and meta.get('dataset', {}).get('sha256_prefix') == ds['sha256_prefix']
        and meta.get('dataset', {}).get('random_state') == ds['random_state']
        and meta.get('dataset', {}).get('test_size') == ds['test_size']
        and meta.get('dataset', {}).get('n_test') == ds['n_test'])

    # ── 1) Pure ML component (current production) ────────────────────────
    print('[3/4] Evaluating production ML model (TF-IDF + Logistic Regression) ...')
    if clf is not None:
        X = vec.transform(data['test_clean'])
        scores = pm.decision_scores(clf, X)
        y_pred = np.asarray(clf.classes_)[np.argmax(scores, axis=1)].astype(int)
        models['production_ml'] = {
            'label': 'Production ML model (TF-IDF + Logistic Regression)',
            'description': 'models/adam_classifier.pkl + models/adam_vectorizer.pkl, '
                           'predictions taken as argmax of the linear scores.',
            'kind': 'production',
            'order': 1,
            'held_out_guarantee': held_out,
            'held_out_note': HELD_OUT_OK if held_out else HELD_OUT_NOTE,
            'active_in_app': bool(model_status['predict_proba_ok']),
            'metrics': compute_metrics(y_test, y_pred),
        }
    else:
        print(f'      skipped: {err}')

    # ── 2) Local hybrid pipeline (what the app really runs without API keys) ──
    def run_hybrid(predictor, label):
        paths = collections.Counter()
        preds = []
        # predictor.py prints one [WARN] line per text if the ML model fails; we count that instead.
        with warnings.catch_warnings(), open(os.devnull, 'w') as _null:
            warnings.simplefilter('ignore')
            for i, text in enumerate(data['test_raw'], 1):
                with contextlib.redirect_stdout(_null):
                    r = predictor._hybrid_predict(text)
                preds.append(PREDICTION_TO_ID[r['prediction']])
                paths[{'hybrid': 'rule_override', 'adam': 'ml_blend',
                       'keyword': 'keyword_only'}.get(r.get('method'), 'other')] += 1
                if i % 2000 == 0:
                    print(f'      ... {label}: {i}/{len(data["test_raw"])}')
        return np.array(preds), paths

    print('      Evaluating local hybrid pipeline (LR + keyword rules + overrides) ...')
    from utils.predictor import HateSpeechPredictor
    predictor = HateSpeechPredictor()
    preds, paths = run_hybrid(predictor, 'current')
    models['production_hybrid'] = {
        'label': 'Production local pipeline (ML + keyword rules + overrides)',
        'description': 'HateSpeechPredictor._hybrid_predict with the current production model - exactly '
                       'what Quick Analyze runs when no Claude API key is set.',
        'kind': 'production',
        'order': 0,
        'held_out_guarantee': False,
        'held_out_note': ('The ML part is held-out' + (' (verified, see Production ML model)' if held_out else ' is NOT verified')
                          + ', but the hand-written keyword rules and overrides in predictor.py were '
                          'designed outside this pipeline; it cannot be proven that no test text '
                          'influenced them. Treat these numbers as "how the app behaves on this '
                          'dataset", not as a pure generalisation estimate.'),
        'active_in_app': True,
        'prediction_paths': {
            'rule_override': int(paths['rule_override']),
            'ml_blend': int(paths['ml_blend']),
            'keyword_only': int(paths['keyword_only']),
        },
        'metrics': compute_metrics(y_test, preds),
    }

    # ── 3) The ORIGINAL model (scikit-learn 1.8.0), preserved for comparison ──
    lclf, lvec, lerr = pm.load_artifacts(LEGACY_DIR) if os.path.isdir(LEGACY_DIR) else (None, None, 'no legacy folder')
    if lclf is not None and meta:
        print('      Evaluating the preserved original model (saved with scikit-learn 1.8.0) ...')
        lscores = pm.decision_scores(lclf, lvec.transform(data['test_clean']))
        lpred = np.asarray(lclf.classes_)[np.argmax(lscores, axis=1)].astype(int)
        legacy_note = ('The original model\'s training data is not part of this project (its vocabulary '
                       'suggests only ~600 training texts), so held-out status cannot be proven. '
                       'It is kept only for comparison; it is no longer the production model.')
        models['legacy_ml'] = {
            'label': 'Original model, preserved (TF-IDF + LR, saved with scikit-learn 1.8.0)',
            'description': 'models/legacy_sklearn_1.8.0/ - the model that shipped with the project before '
                           'the scikit-learn compatibility fix. Loaded with a version warning.',
            'kind': 'legacy', 'order': 2, 'held_out_guarantee': False, 'held_out_note': legacy_note,
            'active_in_app': False,
            'metrics': compute_metrics(y_test, lpred),
        }
        lp = HateSpeechPredictor()
        lp.adam_clf, lp.adam_vectorizer = lclf, lvec
        lh_preds, lh_paths = run_hybrid(lp, 'original')
        models['legacy_hybrid'] = {
            'label': 'Original local pipeline (original ML model + keyword rules)',
            'description': 'The app pipeline as it behaved BEFORE the retraining, for before/after comparison.',
            'kind': 'legacy', 'order': 3, 'held_out_guarantee': False, 'held_out_note': legacy_note,
            'active_in_app': False,
            'prediction_paths': {'rule_override': int(lh_paths['rule_override']),
                                 'ml_blend': int(lh_paths['ml_blend']),
                                 'keyword_only': int(lh_paths['keyword_only'])},
            'metrics': compute_metrics(y_test, lh_preds),
        }

    notes = [
        'Claude API and BiLSTM routes are not evaluated (not local/deterministic).',
        'Pipeline results mix the ML model with hand-written rules; use the Model Comparison page for a clean algorithm-vs-algorithm test.',
        'Labels follow the app convention: 0=Hate Speech, 1=Offensive, 2=Normal.',
    ]
    if paths['keyword_only'] == len(preds) and clf is not None:
        notes.append('Every prediction of the hybrid pipeline fell back to keyword-only: the ML '
                     'model is NOT actually being used by the app in this environment '
                     '(see production model status).')

    print('[4/4] Saving report ...')
    report = {
        'generated_at': now_iso(),
        'environment': environment_info(),
        'dataset': info,
        'production_model_status': model_status,
        'models': models,
        'notes': notes,
    }
    write_json(EVALUATION_REPORT, report)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    shutil.copy2(EVALUATION_REPORT, os.path.join(
        HISTORY_DIR, 'evaluation_' + report['generated_at'][:19].replace(':', '-') + '.json'))
    print(f'\nDone. Saved: reports/evaluation_results.json (archived in reports/history/)')
    for key, m in models.items():
        mt = m['metrics']
        print(f"  {key:18s} acc={mt['accuracy']:.4f}  macroF1={mt['macro_f1']:.4f}  "
              f"weightedF1={mt['weighted_f1']:.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
