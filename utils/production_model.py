"""
utils/production_model.py
=========================
Tumhare EXISTING production model (models/adam_classifier.pkl +
models/adam_vectorizer.pkl) ko safely load karta hai aur check karta hai ki
wo is computer ke scikit-learn version par sach mein kaam kar raha hai ya nahi.

Kyun zaroori hai?
  Pickle files ek scikit-learn version se bani hoti hain. Agar tumhare venv ka
  scikit-learn version alag hai to `predict_proba` crash ho sakta hai, aur
  predictor.py us error ko chup-chaap pakad kar keyword classifier par fallback
  kar deta hai. Ye module us problem ko dikhata hai (hide nahi karta).
"""

import math
import os
import pickle
import warnings

import numpy as np

from utils.ml_common import MODELS_DIR

CLF_PATH = os.path.join(MODELS_DIR, 'adam_classifier.pkl')
VEC_PATH = os.path.join(MODELS_DIR, 'adam_vectorizer.pkl')


_pickle_version_seen = None   # sklearn version the pickles were created with (if known)


def load_artifacts(folder=None):
    """Returns (clf, vectorizer, error_message). error_message is None on success.
    `folder` lets evaluate_model.py load the preserved legacy copy; default = models/."""
    global _pickle_version_seen
    _pickle_version_seen = None     # forget the previous load: files may have been replaced
    clf_path = os.path.join(folder, 'adam_classifier.pkl') if folder else CLF_PATH
    vec_path = os.path.join(folder, 'adam_vectorizer.pkl') if folder else VEC_PATH
    if not (os.path.exists(clf_path) and os.path.exists(vec_path)):
        return None, None, 'Model files not found in models/ (adam_classifier.pkl, adam_vectorizer.pkl).'
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')   # we read the version warning; status() reports it
            with open(clf_path, 'rb') as f:
                clf = pickle.load(f)
            with open(vec_path, 'rb') as f:
                vec = pickle.load(f)
        for w in caught:
            orig = getattr(w.message, 'original_sklearn_version', None)
            if orig:
                _pickle_version_seen = str(orig)
                break
    except Exception as e:  # corrupted file, missing module, etc.
        return None, None, f'Could not load model files: {type(e).__name__}: {e}'
    for need in ('coef_', 'intercept_', 'classes_'):
        if not hasattr(clf, need):
            return None, None, 'Loaded classifier is not a fitted linear model (missing coef_/intercept_).'
    if not hasattr(vec, 'vocabulary_'):
        return None, None, 'Loaded vectorizer is not fitted.'
    return clf, vec, None


def _pickle_version(est):
    """Version the pickle was made with. (est.__getstate__ re-stamps the CURRENT version,
    so we use the warning captured while loading instead.)"""
    if _pickle_version_seen:
        return _pickle_version_seen
    try:
        import sklearn
        return est.__dict__.get('_sklearn_version') or sklearn.__version__
    except Exception:
        return None


META_PATH = os.path.join(MODELS_DIR, 'production_model_meta.json')


def read_meta():
    """models/production_model_meta.json written by retrain_production.py (or None)."""
    from utils.ml_common import read_json
    meta, _ = read_json(META_PATH)
    return meta


def status(clf, vec, load_error=None):
    """Plain dict describing whether the production model works here."""
    import sklearn
    installed = sklearn.__version__
    out = {
        'loaded': clf is not None and vec is not None,
        'installed_sklearn': installed,
        'pickle_sklearn': None,
        'version_mismatch': False,
        'predict_proba_ok': False,
        'predict_proba_error': None,
        'vocab_size': None,
        'estimated_training_docs': None,
        'message': None,
        'level': 'error',          # ok | warning | error
        'meta_present': False,
        'meta_matches_files': None,
    }
    if not out['loaded']:
        out['message'] = load_error or 'Production model not loaded.'
        return out

    out['pickle_sklearn'] = _pickle_version(clf)
    out['version_mismatch'] = bool(out['pickle_sklearn'] and out['pickle_sklearn'] != installed)
    out['vocab_size'] = int(len(vec.vocabulary_))

    # Rough, honest estimate of how many documents the vectorizer saw:
    # idf = ln((1+n)/(1+df)) + 1  -> for a term with df=1: n = 2*exp(idf-1) - 1
    try:
        if getattr(vec, 'min_df', 1) == 1 and hasattr(vec, 'idf_'):
            est = 2 * math.exp(float(np.max(vec.idf_)) - 1) - 1
            out['estimated_training_docs'] = int(round(est))
    except Exception:
        pass

    meta = read_meta()
    if meta:
        out['meta_present'] = True
        try:
            import hashlib
            sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
            want = meta.get('file_sha256', {})
            out['meta_matches_files'] = (want.get('adam_classifier.pkl') == sha(CLF_PATH)
                                         and want.get('adam_vectorizer.pkl') == sha(VEC_PATH))
        except OSError:
            out['meta_matches_files'] = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.predict_proba(vec.transform(['status check']))
        out['predict_proba_ok'] = True
    except Exception as e:
        out['predict_proba_error'] = f'{type(e).__name__}: {e}'

    if out['predict_proba_ok'] and not out['version_mismatch']:
        out['level'] = 'ok'
        out['message'] = 'Production model loaded and working.'
    elif out['predict_proba_ok']:
        out['level'] = 'warning'
        out['message'] = (f"Model was saved with scikit-learn {out['pickle_sklearn']} but "
                          f"{installed} is installed. It works here, but versions should match.")
    else:
        out['level'] = 'error'
        out['message'] = (
            f"Model was saved with scikit-learn {out['pickle_sklearn']} but {installed} is installed, "
            "and predict_proba fails. The app's predictor.py silently falls back to the keyword "
            "classifier in this situation. Fix: use Python 3.11+ with scikit-learn "
            f"{out['pickle_sklearn']} or newer, or retrain locally (see UPGRADE_GUIDE.md).")
    return out


def decision_scores(clf, X):
    """
    Linear scores  z = X @ coef.T + intercept  (this is exactly what a linear
    model computes). Works on any scikit-learn version, because it only uses
    the stored weights, not the estimator's methods.
    """
    return np.asarray(X @ clf.coef_.T) + clf.intercept_


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
