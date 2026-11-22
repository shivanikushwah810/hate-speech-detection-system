"""
utils/explainer.py
==================
FEATURE 3 — Explainable AI (XAI).

Tumhara ML component ek LINEAR model hai:  TF-IDF  ->  Logistic Regression.
Linear model ka decision exactly tod kar dikhaya ja sakta hai:

      score(class c) = intercept_c  +  sum over words/n-grams j of  ( weight[c, j] * tfidf_j )

Har term  weight * tfidf  us word ka "contribution" hai:
    contribution > 0  -> word ne is class ki taraf push kiya
    contribution < 0  -> word ne is class se door push kiya

Kyun SHAP / LIME ki jagah ye?
  * SHAP ka LinearExplainer linear models ke liye (zero-baseline ke saath) yehi
    numbers deta hai (weight * feature value). Hum wahi exact formula seedha
    use karte hain -> koi approximation nahi, koi random sampling nahi.
  * LIME random perturbations se approximate karta hai; linear model par uski
    zaroorat nahi, aur wo slower + har run mein thoda alag aata hai.
  * Extra heavy libraries (shap -> numba/llvmlite) Python 3.9 Mac par install
    problems de sakti hain. Isliye requirements.txt mein kuch add nahi hua.
  (tests/test_new_features.py mein SHAP se match bhi verify hota hai, agar shap
   installed ho.)

Important honesty rules (UI mein bhi dikhte hain):
  * Ye explain karta hai ki MODEL ne kyun aisa score diya — writer ka intention
    nahi, aur ye prove nahi karta ki text sach mein hateful hai ya nahi.
  * App ka final answer rules (slurs/threat patterns) se bhi aa sakta hai. Un
    rules ko alag se "Rule evidence" mein dikhaya jata hai.
  * Claude AI / BiLSTM routes ko ye explain nahi karta (wo local model nahi hain).
"""

import re

import numpy as np

from utils import production_model as pm
from utils.predictor import HATE_WORDS, OFFENSIVE_WORDS, PATTERNS

CLASS_KEYS = ['hate_speech', 'offensive', 'normal']
CLASS_NAMES = ['Hate Speech', 'Offensive', 'Normal']

MAX_TEXT_LEN = 5000   # same limit as /api/analyze/text


class ExplainerUnavailable(Exception):
    """The production linear model is not loaded, so nothing can be explained."""


class ExplainerInputError(ValueError):
    """Bad input text."""


DISCLAIMERS = [
    'This explains how the trained model combined word weights for THIS text. It does not show '
    'the writer\'s intent and does not prove the text is (or is not) hateful.',
    'Weights are learned from the training data, so they can contain dataset bias (topics, dialect, '
    'frequent words). A high weight means "seen with this label in training", not "true meaning".',
    'Only words/phrases that exist in the model vocabulary can contribute. Unknown words are ignored '
    'by the model (listed below).',
    'Rule evidence is separate: those rules are hand-written in predictor.py, not learned.',
    'Claude AI and BiLSTM routes (if enabled elsewhere) are not explained here.',
]


def _decision_source(final):
    if final is None:
        return 'unavailable', 'The local pipeline could not produce a final label.'
    method = final.get('method')
    detail = final.get('detail', '')
    if method == 'hybrid':
        return 'rule_override', (f'A hard rule decided the label ({detail}). '
                                 'The ML scores did not decide this one.')
    if method == 'adam':
        return 'ml_and_rules_blend', ('The label comes from blending the ML probabilities with '
                                      'keyword-rule scores (and a positive-sentiment guard).')
    return 'keyword_rules_only', ('Only keyword rules produced this label: the ML classifier was '
                                  'not used for this prediction.')


class ProductionExplainer:
    def __init__(self, predictor):
        self.predictor = predictor
        self._fn_cache = (None, None)   # (vectorizer id, feature names)

    # ── helpers ──────────────────────────────────────────────────────────
    def _feature_names(self, vec):
        if self._fn_cache[0] != id(vec):
            self._fn_cache = (id(vec), vec.get_feature_names_out())
        return self._fn_cache[1]

    @staticmethod
    def _rank(contribs, names, idx, vals, weights, top_k):
        """Build positive / negative feature lists (largest effect first)."""
        order = np.argsort(-np.abs(contribs))
        pos, neg = [], []
        for o in order:
            c = float(contribs[o])
            if c == 0.0:
                continue
            name = str(names[idx[o]])
            row = {'feature': name, 'ngram': name.count(' ') + 1,
                   'tfidf': round(float(vals[o]), 4),
                   'weight': round(float(weights[o]), 4),
                   'contribution': round(c, 4)}
            (pos if c > 0 else neg).append(row)
        return pos[:top_k], neg[:top_k], len(pos), len(neg)

    def _rule_evidence(self, text):
        lower = text.lower()
        slurs = [w for w in HATE_WORDS if re.search(r'\b' + re.escape(w) + r's?\b', lower)]
        offensive = [w for w in OFFENSIVE_WORDS if re.search(r'\b' + re.escape(w) + r'\b', lower)]
        cats = [c for c, pats in PATTERNS.items()
                if any(re.search(p, lower, re.IGNORECASE) for p in pats)]
        return {'hard_slur_words': slurs, 'offensive_words': offensive,
                'targeted_pattern_categories': cats}

    # ── main API ─────────────────────────────────────────────────────────
    def explain(self, text, top_k=8):
        if not isinstance(text, str) or not text.strip():
            raise ExplainerInputError('Please enter some text.')
        text = text.strip()
        if len(text) > MAX_TEXT_LEN:
            raise ExplainerInputError(f'Text too long (max {MAX_TEXT_LEN} characters).')
        try:
            top_k = max(3, min(int(top_k), 20))
        except (TypeError, ValueError):
            top_k = 8

        clf, vec = self.predictor.adam_clf, self.predictor.adam_vectorizer
        if clf is None or vec is None:
            raise ExplainerUnavailable(
                'The production ML model (models/adam_classifier.pkl + adam_vectorizer.pkl) is not '
                'loaded, so there is nothing to explain. Check the server log for the load error.')

        cleaned = self.predictor._clean_text(text)
        X = vec.transform([cleaned]).tocsr()
        idx, vals = X.indices, X.data
        names = self._feature_names(vec)
        coef = np.asarray(clf.coef_)
        intercept = np.asarray(clf.intercept_)
        classes = [int(c) for c in clf.classes_]
        if classes != [0, 1, 2]:
            raise ExplainerUnavailable('Unexpected class order in the model; expected [0, 1, 2].')

        # exact scores from the weights, then probabilities
        logits = pm.decision_scores(clf, X)[0]
        probs = pm.softmax(logits)

        # cross-check against scikit-learn's own predict_proba (if it works here)
        verification = {'checked': False, 'matches_predict_proba': None, 'note': None}
        try:
            ref = clf.predict_proba(X)[0]
            verification = {'checked': True,
                            'matches_predict_proba': bool(np.allclose(ref, probs, atol=1e-6)),
                            'note': None}
        except Exception as e:
            verification['note'] = (f'scikit-learn predict_proba failed in this environment '
                                    f'({type(e).__name__}); probabilities were computed directly '
                                    'from the stored weights.')

        ml_top = int(np.argmax(probs))
        order = np.argsort(-probs)
        runner_up = int(order[1])

        views = []
        decomposition_ok = True
        for c in range(3):
            w = coef[c, idx]
            contribs = w * vals
            pos, neg, n_pos, n_neg = self._rank(contribs, names, idx, vals, w, top_k)
            total = float(intercept[c] + contribs.sum())
            decomposition_ok &= bool(np.isclose(total, logits[c], atol=1e-8))
            views.append({
                'key': f'class:{CLASS_KEYS[c]}',
                'type': 'class',
                'class_key': CLASS_KEYS[c],
                'title': f'Evidence for {CLASS_NAMES[c]}',
                'intercept': round(float(intercept[c]), 4),
                'sum_contributions': round(float(contribs.sum()), 4),
                'score': round(total, 4),
                'probability': round(float(probs[c]), 4),
                'positive': pos, 'negative': neg,
                'n_positive_total': n_pos, 'n_negative_total': n_neg,
            })

        # Why predicted class A rather than runner-up B:  (w_A - w_B) * tfidf
        wd = coef[ml_top, idx] - coef[runner_up, idx]
        md = wd * vals
        pos, neg, n_pos, n_neg = self._rank(md, names, idx, vals, wd, top_k)
        views.append({
            'key': 'margin', 'type': 'margin',
            'title': f'Why {CLASS_NAMES[ml_top]} rather than {CLASS_NAMES[runner_up]}',
            'intercept': round(float(intercept[ml_top] - intercept[runner_up]), 4),
            'sum_contributions': round(float(md.sum()), 4),
            'score': round(float(logits[ml_top] - logits[runner_up]), 4),
            'positive': pos, 'negative': neg,
            'n_positive_total': n_pos, 'n_negative_total': n_neg,
        })

        # per-token view for highlighting (unigram features only)
        token_re = vec.token_pattern if isinstance(vec.token_pattern, str) else r'\b\w+\b'
        vocab = vec.vocabulary_
        row = {int(j): float(v) for j, v in zip(idx, vals)}
        tokens, unknown = [], []
        for tok in re.findall(token_re, cleaned):
            j = vocab.get(tok)
            if j is None:
                tokens.append({'token': tok, 'known': False, 'values': {}})
                if tok not in unknown:
                    unknown.append(tok)
                continue
            x = row.get(int(j), 0.0)
            values = {f'class:{CLASS_KEYS[c]}': round(float(coef[c, j] * x), 4) for c in range(3)}
            values['margin'] = round(float((coef[ml_top, j] - coef[runner_up, j]) * x), 4)
            tokens.append({'token': tok, 'known': True, 'values': values})
        n_known = sum(1 for t in tokens if t['known'])

        # what the real app pipeline says for the same text (local pipeline, no API call)
        try:
            final = self.predictor._hybrid_predict(text)
        except Exception:
            final = None
        ml_active = self.predictor._adam_classify(text) is not None
        source, source_text = _decision_source(final)

        rule_ev = self._rule_evidence(text)
        summary = self._summary(final, source_text, ml_top, probs, views, n_known, len(tokens))

        return {
            'text': text,
            'cleaned_text': cleaned,
            'method': {
                'name': 'Exact linear contribution (weight × TF-IDF value)',
                'why_not_shap_or_lime': (
                    'The ML model is linear, so its score can be split exactly into per-word '
                    'contributions. This equals SHAP LinearExplainer with a zero baseline, without '
                    'sampling or extra dependencies; LIME would only approximate it.'),
                'exact': True,
            },
            'ml_component': {
                'predicted': CLASS_NAMES[ml_top],
                'runner_up': CLASS_NAMES[runner_up],
                'probabilities': {CLASS_KEYS[c]: round(float(probs[c]), 4) for c in range(3)},
                'decomposition_exact': bool(decomposition_ok),
                'verification': verification,
                'active_in_app_pipeline': bool(ml_active),
            },
            'final_pipeline': None if final is None else {
                'prediction': final['prediction'],
                'confidence': final['confidence'],
                'scores': final['scores'],
                'method': final.get('method'),
                'detail': final.get('detail'),
                'decision_source': source,
                'decision_source_text': source_text,
                'agrees_with_ml_component': final['prediction'] == CLASS_NAMES[ml_top],
            },
            'views': views,
            'default_view': f'class:{CLASS_KEYS[ml_top]}',
            'tokens': tokens,
            'unknown_words': unknown,
            'coverage': {'known_tokens': n_known, 'total_tokens': len(tokens),
                         'vocab_size': int(len(vocab))},
            'rule_evidence': rule_ev,
            'summary': summary,
            'disclaimers': DISCLAIMERS,
        }

    @staticmethod
    def _summary(final, source_text, ml_top, probs, views, n_known, n_tokens):
        lines = []
        if final is not None:
            lines.append(f'The app\'s local pipeline labelled this text "{final["prediction"]}". '
                         + source_text)
        lines.append('The ML component on its own gives: ' + ', '.join(
            f'{CLASS_NAMES[c]} {probs[c] * 100:.0f}%' for c in range(3)) + '.')
        if float(np.max(probs)) < 0.5:
            lines.append('The ML component is not confident here (its top class is below 50%), '
                         'so treat this explanation as weak evidence.')
        v = views[ml_top]
        if n_known == 0:
            lines.append('None of the words in this text exist in the model vocabulary, so the ML '
                         'scores come only from the model\'s baseline (intercept) — there is no '
                         'word-level evidence to show.')
        else:
            if v['positive']:
                lines.append(f'Strongest evidence FOR {CLASS_NAMES[ml_top]}: ' + ', '.join(
                    f'"{f["feature"]}" (+{f["contribution"]:.2f})' for f in v['positive'][:3]) + '.')
            if v['negative']:
                lines.append(f'Strongest evidence AGAINST {CLASS_NAMES[ml_top]}: ' + ', '.join(
                    f'"{f["feature"]}" ({f["contribution"]:.2f})' for f in v['negative'][:3]) + '.')
            lines.append(f'{n_known} of {n_tokens} words were known to the model.')
        return lines
