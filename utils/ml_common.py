"""
utils/ml_common.py
==================
Shared helpers used by:
  - train_and_compare.py  (Model Comparison)
  - evaluate_model.py     (Model Evaluation)
  - app.py                (dashboard APIs)

Is file mein 4 kaam hote hain:
  1. Dataset CSV load karna aur validate karna (koi data invent nahi hota).
  2. Duplicate texts hatana + STRATIFIED train/test split (same split har jagah).
  3. Metrics nikalna (accuracy, precision, recall, macro/weighted F1, confusion matrix).
  4. JSON reports safely save / load karna.
"""

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── Paths (absolute, so scripts work no matter where you run them from) ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CANDIDATES_DIR = os.path.join(MODELS_DIR, 'candidates')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
LOGS_DIR = os.path.join(REPORTS_DIR, 'logs')
HISTORY_DIR = os.path.join(REPORTS_DIR, 'history')

DEFAULT_DATASET = os.path.join(DATA_DIR, 'dataset.csv')
EVALUATION_REPORT = os.path.join(REPORTS_DIR, 'evaluation_results.json')
COMPARISON_REPORT = os.path.join(REPORTS_DIR, 'comparison_results.json')

# ── Label convention: SAME as the existing app (predictor.py) ───────────
# class id 0 = Hate Speech, 1 = Offensive, 2 = Normal
LABELS = ['Hate Speech', 'Offensive', 'Normal']
LABEL_IDS = [0, 1, 2]
PREDICTION_TO_ID = {name: i for i, name in enumerate(LABELS)}

RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_COLUMN_CANDIDATES = ['text', 'tweet', 'comment', 'content', 'sentence', 'clean_text']
LABEL_COLUMN_CANDIDATES = ['label', 'class', 'labels', 'category', 'target']

# Words people use for the 3 classes in public datasets.
LABEL_ALIASES = {
    '0': 0, 'hate': 0, 'hate speech': 0, 'hate_speech': 0, 'hatespeech': 0,
    '1': 1, 'offensive': 1, 'offensive language': 1, 'offensive_language': 1,
    '2': 2, 'normal': 2, 'neither': 2, 'none': 2, 'clean': 2, 'non-hate': 2,
    'neutral': 2,
}


class DataError(Exception):
    """Raised for any dataset problem. Message is safe to show to the user."""


# ═════════════════════════════════════════════════════════════════════════
# Text cleaning  — REUSES the existing project's cleaning function so that
# training / evaluation preprocessing is identical to production.
# ═════════════════════════════════════════════════════════════════════════
_clean_fn = None


def clean_text(text):
    """Same cleaning as HateSpeechPredictor._clean_text (utils/predictor.py)."""
    global _clean_fn
    if _clean_fn is None:
        from utils.predictor import HateSpeechPredictor
        _clean_fn = lambda t: HateSpeechPredictor._clean_text(None, t)  # self is unused there
    return _clean_fn(text)


# ═════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═════════════════════════════════════════════════════════════════════════
def file_fingerprint(path):
    """Short SHA-256 of the file (to prove which dataset produced which report)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def _normalise_label(value):
    s = str(value).strip().lower()
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    try:  # "1.0" -> "1"
        f = float(s)
        if f == int(f) and str(int(f)) in LABEL_ALIASES:
            return LABEL_ALIASES[str(int(f))]
    except ValueError:
        pass
    return None


def load_dataset(path=DEFAULT_DATASET, text_col=None, label_col=None):
    """
    Read a labelled CSV. Returns (texts, labels, info).
    Raises DataError with a clear message if anything is wrong.
    Nothing is invented: rows with empty text are skipped (and counted);
    rows with an unrecognised label cause an error (we never guess labels).
    """
    if not os.path.exists(path):
        raise DataError(
            f"Dataset not found: {os.path.relpath(path, BASE_DIR)}. "
            "Run `python download_dataset.py` or place your own labelled CSV there "
            "(see data/README.md)."
        )
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    texts, labels = [], []
    skipped_empty = 0
    skipped_missing_label = 0
    bad_labels = {}
    try:
        with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            header = [h.strip() for h in (reader.fieldnames or []) if h is not None]
            if not header:
                raise DataError("The dataset file is empty or has no header row.")
            lower = {h.lower(): h for h in header}

            t_col = text_col or next((lower[c] for c in TEXT_COLUMN_CANDIDATES if c in lower), None)
            l_col = label_col or next((lower[c] for c in LABEL_COLUMN_CANDIDATES if c in lower), None)
            if t_col is None or l_col is None or t_col not in header or l_col not in header:
                raise DataError(
                    f"Could not find text/label columns. Found columns: {header}. "
                    "Use columns named text/label (or tweet/class), "
                    "or pass --text-col and --label-col."
                )

            for row in reader:
                raw_text = (row.get(t_col) or '').strip()
                if not raw_text:
                    skipped_empty += 1
                    continue
                if not str(row.get(l_col) or '').strip():
                    skipped_missing_label += 1     # blank label: cannot be used, never guessed
                    continue
                lab = _normalise_label(row.get(l_col))
                if lab is None:
                    k = str(row.get(l_col))
                    bad_labels[k] = bad_labels.get(k, 0) + 1
                    continue
                texts.append(raw_text)
                labels.append(lab)
    except UnicodeError as e:
        raise DataError(f"Could not read the file as UTF-8 text: {e}")
    except csv.Error as e:
        raise DataError(f"CSV parsing problem: {e}")

    if bad_labels:
        shown = dict(list(bad_labels.items())[:8])
        raise DataError(
            f"Unrecognised label values {shown}. Allowed: 0/1/2 or "
            "hate speech / offensive / normal(neither). We never guess labels."
        )
    if not texts:
        raise DataError("No usable rows found in the dataset.")

    labels = np.array(labels, dtype=int)
    present = sorted(set(labels.tolist()))
    if present != LABEL_IDS:
        raise DataError(
            f"Dataset must contain all 3 classes (0=Hate, 1=Offensive, 2=Normal); found {present}."
        )

    info = {
        'file': os.path.basename(path),
        'sha256_prefix': file_fingerprint(path),
        'text_column': t_col,
        'label_column': l_col,
        'rows_read': int(len(texts) + skipped_empty + skipped_missing_label),
        'rows_skipped_empty_text': int(skipped_empty),
        'rows_skipped_missing_label': int(skipped_missing_label),
        'class_distribution_raw': {LABELS[i]: int(np.sum(labels == i)) for i in LABEL_IDS},
        'label_mapping': {'0': LABELS[0], '1': LABELS[1], '2': LABELS[2]},
    }
    return texts, labels, info


def prepare_data(path=DEFAULT_DATASET, text_col=None, label_col=None,
                 test_size=TEST_SIZE, seed=RANDOM_STATE):
    """
    Load -> clean -> REMOVE DUPLICATES -> stratified split.

    Leakage prevention:
      * Duplicates (same cleaned text) are removed BEFORE splitting, so the
        same sentence can never be in both train and test.
      * The split happens BEFORE anything is fitted. TF-IDF is fitted later
        on the training part only (see train_and_compare.py).
      * Split is deterministic (same seed) -> comparison and evaluation use
        exactly the same test set.
    """
    from sklearn.model_selection import train_test_split

    texts, labels, info = load_dataset(path, text_col, label_col)
    cleaned = [clean_text(t) for t in texts]

    seen, keep = {}, []
    conflicts = 0
    for i, c in enumerate(cleaned):
        if not c:
            continue
        if c in seen:
            if labels[seen[c]] != labels[i]:
                conflicts += 1            # same text, different label: first occurrence is kept
            continue
        seen[c] = i
        keep.append(i)
    removed = len(texts) - len(keep)
    if len(keep) < 50:
        raise DataError(f"Only {len(keep)} usable unique rows - too few for a train/test split.")

    raw = [texts[i] for i in keep]
    cln = [cleaned[i] for i in keep]
    y = labels[keep]

    counts = np.bincount(y, minlength=3)
    if counts.min() < 10:
        raise DataError(
            f"A class has only {int(counts.min())} samples after de-duplication. "
            "Need at least 10 per class for a stratified split."
        )

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)

    info.update({
        'rows_used_after_dedup': int(len(y)),
        'duplicates_or_empty_removed': int(removed),
        'duplicate_label_conflicts': int(conflicts),
        'test_size': float(test_size),
        'random_state': int(seed),
        'split': 'stratified',
        'n_train': int(len(tr)),
        'n_test': int(len(te)),
        'class_distribution_total': {LABELS[i]: int(np.sum(y == i)) for i in LABEL_IDS},
        'class_distribution_train': {LABELS[i]: int(np.sum(y[tr] == i)) for i in LABEL_IDS},
        'class_distribution_test': {LABELS[i]: int(np.sum(y[te] == i)) for i in LABEL_IDS},
    })
    return {
        'train_raw': [raw[i] for i in tr], 'test_raw': [raw[i] for i in te],
        'train_clean': [cln[i] for i in tr], 'test_clean': [cln[i] for i in te],
        'y_train': y[tr], 'y_test': y[te],
        'info': info,
    }


# ═════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    """All metrics computed from real predictions with scikit-learn."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kw = dict(labels=LABEL_IDS, zero_division=0)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, **kw)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', **kw)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', **kw)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)

    return {
        'n_test': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(mp), 'macro_recall': float(mr), 'macro_f1': float(mf),
        'weighted_precision': float(wp), 'weighted_recall': float(wr), 'weighted_f1': float(wf),
        'per_class': {
            LABELS[i]: {'precision': float(p[i]), 'recall': float(r[i]),
                        'f1': float(f[i]), 'support': int(s[i])}
            for i in LABEL_IDS
        },
        'labels': LABELS,
        'confusion_matrix': cm.astype(int).tolist(),   # rows = true, cols = predicted
        'classification_report': classification_report(
            y_true, y_pred, labels=LABEL_IDS, target_names=LABELS, digits=4, zero_division=0),
    }


def _macro_f1_from_cm(cm, k=3):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def bootstrap_macro_f1(y_true, preds_by_model, n_boot=1000, seed=RANDOM_STATE):
    """
    Bootstrap (resample the test set with replacement, SAME resamples for every
    model) to see how stable macro-F1 is.  Returns
      {model: {'ci95': [lo, hi]}}  and paired differences vs the best model.
    """
    y_true = np.asarray(y_true)
    n, k = len(y_true), 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    scores = {}
    for name, yp in preds_by_model.items():
        yp = np.asarray(yp)
        arr = np.empty(n_boot)
        for b in range(n_boot):
            rows = idx[b]
            cm = np.bincount(y_true[rows] * k + yp[rows], minlength=k * k).reshape(k, k)
            arr[b] = _macro_f1_from_cm(cm)
        scores[name] = arr
    point = {name: _macro_f1_from_cm(
        np.bincount(y_true * k + np.asarray(yp), minlength=k * k).reshape(k, k))
        for name, yp in preds_by_model.items()}
    best = max(point, key=point.get)
    out = {}
    for name, arr in scores.items():
        lo, hi = np.percentile(arr, [2.5, 97.5])
        entry = {'ci95': [float(lo), float(hi)]}
        if name != best:
            d = scores[best] - arr
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            entry['diff_vs_best_ci95'] = [float(dlo), float(dhi)]
            entry['clearly_worse_than_best'] = bool(dlo > 0)
        out[name] = entry
    return out


# ═════════════════════════════════════════════════════════════════════════
# JSON report helpers
# ═════════════════════════════════════════════════════════════════════════
def environment_info():
    import platform
    import sklearn
    return {
        'python': platform.python_version(),
        'scikit_learn': sklearn.__version__,
        'numpy': np.__version__,
        'platform': platform.system(),
    }


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, obj):
    """Atomic write: never leaves a half-written report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    """Returns (data, error). Never raises."""
    if not os.path.exists(path):
        return None, 'missing'
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f'unreadable: {e}'


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
