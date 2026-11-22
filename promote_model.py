"""
promote_model.py
================
Comparison ke baad agar tum kisi candidate ko app ka production model banana
chaho, to ye script use karo. Ye KABHI automatically nahi chalti — pehle poori
jaankari dikhati hai aur tumse "YES" type karwa ke confirm karti hai.
Replace karne se pehle purane model ka BACKUP banta hai.

Usage:
    python promote_model.py                          # candidates ki list dikhata hai
    python promote_model.py logistic_regression_lbfgs  # promote (confirmation ke baad)
    python promote_model.py --restore models/backup_20260101_120000   # purana model wapas
"""

import argparse
import os
import pickle
import shutil
import sys
import warnings
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from utils.ml_common import (COMPARISON_REPORT, EVALUATION_REPORT, MODELS_DIR,  # noqa: E402
                             read_json)

PROD_FILES = ('adam_classifier.pkl', 'adam_vectorizer.pkl')


def _confirm(prompt):
    return input(prompt).strip() == 'YES'


def list_candidates(report):
    print('\nAvailable candidates (from reports/comparison_results.json):\n')
    print(f"  {'slug':32s} {'macro-F1':>9s} {'accuracy':>9s}  can be promoted?")
    for name, m in report['models'].items():
        mt = m['metrics']
        ok = 'yes' if m['supports_predict_proba'] else 'NO (no predict_proba)'
        print(f"  {m['slug']:32s} {mt['macro_f1']:9.4f} {mt['accuracy']:9.4f}  {ok}")
    print(f"\nBest by macro-F1: {report['best_model']}")
    print('To promote one:  python promote_model.py <slug>\n')


def promote(slug, report):
    import joblib
    spec = next(((n, m) for n, m in report['models'].items() if m['slug'] == slug), None)
    if spec is None:
        print(f'[ERROR] Unknown candidate "{slug}". Run without arguments to see the list.')
        return 1
    name, m = spec
    if not m['supports_predict_proba']:
        print(f'[REFUSED] {name} has no predict_proba(). The app blends probabilities, so this '
              'model cannot replace the production classifier without extra calibration work.')
        return 1
    path = os.path.join(BASE_DIR, m['saved_model_file'] or '')
    if not os.path.exists(path):
        print(f'[ERROR] Candidate file missing: {m["saved_model_file"]}. Re-run train_and_compare.py.')
        return 1

    pipe = joblib.load(path)
    vec, clf = pipe.named_steps['tfidf'], pipe.named_steps['clf']
    if [int(c) for c in clf.classes_] != [0, 1, 2]:
        print('[ERROR] Class order is not [0, 1, 2]; refusing (the app expects 0=Hate,1=Offensive,2=Normal).')
        return 1

    ev, _ = read_json(EVALUATION_REPORT)
    mt = m['metrics']
    print('\n================ PROMOTION SUMMARY ================')
    print(f'Candidate : {name}')
    print(f"  accuracy={mt['accuracy']:.4f}  macro-F1={mt['macro_f1']:.4f}  weighted-F1={mt['weighted_f1']:.4f}")
    if ev and 'production_ml' in ev.get('models', {}):
        pm_ = ev['models']['production_ml']['metrics']
        print('Current production ML model (same test split):')
        print(f"  accuracy={pm_['accuracy']:.4f}  macro-F1={pm_['macro_f1']:.4f}  weighted-F1={pm_['weighted_f1']:.4f}")
    print('\nThings you must know before saying YES:')
    print(' 1. Predictions of the whole app will CHANGE: predictor.py blends this model with keyword')
    print('    rules using weights that were tuned for the OLD model, not for this one.')
    print(' 2. The candidate was trained on 80% of the dataset only (the test split was kept hidden).')
    print(' 3. Your old model files are backed up first and can be restored with --restore.')
    print(' 4. After promotion, re-run:  python evaluate_model.py  (to see the new numbers).')
    print('====================================================\n')
    if not _confirm('Type YES (capital letters) to replace the production model, anything else cancels: '):
        print('Cancelled. Nothing was changed.')
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = os.path.join(MODELS_DIR, f'backup_{stamp}')
    os.makedirs(backup)
    for fn in PROD_FILES:
        src = os.path.join(MODELS_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup, fn))
    with open(os.path.join(MODELS_DIR, PROD_FILES[0]), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, PROD_FILES[1]), 'wb') as f:
        pickle.dump(vec, f)
    print(f'[DONE] Promoted "{name}". Old files backed up in {os.path.relpath(backup, BASE_DIR)}')
    print('Restart the Flask app to load the new model.')
    return 0


def restore(folder):
    folder = os.path.abspath(folder)
    missing = [fn for fn in PROD_FILES if not os.path.exists(os.path.join(folder, fn))]
    if missing:
        print(f'[ERROR] Backup folder is missing: {missing}')
        return 1
    if not _confirm(f'Type YES to restore the model files from {folder}: '):
        print('Cancelled.')
        return 0
    for fn in PROD_FILES:
        shutil.copy2(os.path.join(folder, fn), os.path.join(MODELS_DIR, fn))
    print('[DONE] Restored. Restart the Flask app.')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Promote a candidate model (with confirmation).')
    ap.add_argument('slug', nargs='?')
    ap.add_argument('--restore', metavar='BACKUP_FOLDER')
    args = ap.parse_args()

    warnings.simplefilter('ignore')
    if args.restore:
        return restore(args.restore)
    report, err = read_json(COMPARISON_REPORT)
    if report is None:
        print('[ERROR] No comparison report found. Run: python train_and_compare.py')
        return 1
    if not args.slug:
        list_candidates(report)
        return 0
    return promote(args.slug, report)


if __name__ == '__main__':
    sys.exit(main())
