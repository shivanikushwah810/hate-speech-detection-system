"""
tests/test_new_features.py
==========================
Automatic checks for the 3 new features. Run from the project folder:

    python -m unittest discover -s tests -v

Notes:
  * The tiny CSV files created here exist ONLY inside a temporary folder while the
    tests run, to test error handling and leak-prevention. They are never used by
    the dashboard and no metric shown in the app comes from them.
  * Metric correctness is checked against numbers computed BY HAND.
"""
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
warnings.simplefilter('ignore')

import numpy as np  # noqa: E402

from utils import ml_common as mc  # noqa: E402


def write_csv(path, rows, header='text,label'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for t, l in rows:
            f.write(f'"{t}",{l}\n')


class MetricsTest(unittest.TestCase):
    def test_metrics_match_hand_calculation(self):
        # true: 0 0 1 1 2 2   pred: 0 1 1 1 2 0
        m = mc.compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(m['accuracy'], 4 / 6)
        # class0: P=1/2 R=1/2 ; class1: P=2/3 R=1 ; class2: P=1 R=1/2
        self.assertAlmostEqual(m['per_class']['Hate Speech']['precision'], 0.5)
        self.assertAlmostEqual(m['per_class']['Offensive']['recall'], 1.0)
        self.assertAlmostEqual(m['macro_precision'], (0.5 + 2 / 3 + 1) / 3)
        self.assertAlmostEqual(m['macro_recall'], (0.5 + 1 + 0.5) / 3)
        self.assertEqual(m['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        f1 = [0.5, 0.8, 2 / 3]
        self.assertAlmostEqual(m['macro_f1'], sum(f1) / 3)
        self.assertAlmostEqual(m['weighted_f1'], sum(f1) / 3)   # equal supports

    def test_bootstrap_is_paired_and_bounded(self):
        y = np.array([0, 1, 2] * 40)
        good, bad = y.copy(), (y + 1) % 3
        out = mc.bootstrap_macro_f1(y, {'good': good, 'bad': bad}, n_boot=200)
        self.assertEqual(out['good']['ci95'][1], 1.0)
        self.assertTrue(out['bad']['clearly_worse_than_best'])


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name='d.csv'):
        return os.path.join(self.tmp.name, name)

    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(mc.DataError) as cm:
            mc.load_dataset(self.path('nope.csv'))
        self.assertIn('download_dataset.py', str(cm.exception))

    def test_unknown_labels_are_rejected_not_guessed(self):
        write_csv(self.path(), [('a b', 'spam'), ('c d', 0)])
        with self.assertRaises(mc.DataError):
            mc.load_dataset(self.path())

    def test_missing_columns(self):
        write_csv(self.path(), [('a', 0)], header='foo,bar')
        with self.assertRaises(mc.DataError):
            mc.load_dataset(self.path())

    def test_all_three_classes_required(self):
        write_csv(self.path(), [('a b', 0), ('c d', 1)])
        with self.assertRaises(mc.DataError):
            mc.load_dataset(self.path())

    def test_davidson_style_and_text_label_styles(self):
        write_csv(self.path('a.csv'), [('x', 0), ('y', 1), ('z', 2)], header='tweet,class')
        write_csv(self.path('b.csv'), [('x', 'hate speech'), ('y', 'offensive'), ('z', 'neither')])
        for f in ('a.csv', 'b.csv'):
            _, y, _ = mc.load_dataset(self.path(f))
            self.assertEqual(sorted(y.tolist()), [0, 1, 2])

    def test_split_is_stratified_deduplicated_and_leak_free(self):
        rows = []
        for i in range(300):
            rows.append((f'unique sample {i} tokenx{i % 17}', i % 3))
        rows += [('unique sample 5 tokenx5', 2)] * 5          # exact duplicates
        write_csv(self.path(), rows)
        d = mc.prepare_data(self.path())
        self.assertEqual(len(set(d['train_clean']) & set(d['test_clean'])), 0)   # no overlap
        self.assertEqual(d['info']['duplicates_or_empty_removed'], 5)
        tr = np.bincount(d['y_train']) / len(d['y_train'])
        te = np.bincount(d['y_test']) / len(d['y_test'])
        self.assertTrue(np.allclose(tr, te, atol=0.03))                           # stratified
        d2 = mc.prepare_data(self.path())                                        # deterministic
        self.assertEqual(d['test_raw'], d2['test_raw'])


class ExplainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.explainer import ProductionExplainer
        from utils.predictor import HateSpeechPredictor
        cls.predictor = HateSpeechPredictor()
        cls.explainer = ProductionExplainer(cls.predictor)

    def test_contributions_add_up_exactly(self):
        r = self.explainer.explain('you are such a stupid idiot')
        self.assertTrue(r['ml_component']['decomposition_exact'])
        for v in r['views']:
            if v['type'] == 'class':
                self.assertAlmostEqual(v['intercept'] + v['sum_contributions'], v['score'], places=3)
        self.assertAlmostEqual(sum(r['ml_component']['probabilities'].values()), 1.0, places=3)

    def test_input_validation(self):
        from utils.explainer import ExplainerInputError
        for bad in ['', '   ', None, 5, 'x' * 5001]:
            with self.assertRaises(ExplainerInputError):
                self.explainer.explain(bad)

    def test_unknown_words_reported(self):
        r = self.explainer.explain('qwertyuiop zxcvbnm')
        self.assertEqual(r['coverage']['known_tokens'], 0)
        self.assertIn('qwertyuiop', r['unknown_words'])

    def test_matches_shap_linear_explainer_if_installed(self):
        try:
            import shap
        except ImportError:
            self.skipTest('shap not installed (optional, only used for this cross-check)')
        clf, vec = self.predictor.adam_clf, self.predictor.adam_vectorizer
        text = 'you are such a stupid idiot and i hate you'
        X = vec.transform([self.predictor._clean_text(text)]).toarray()
        sv = np.asarray(shap.LinearExplainer(clf, np.zeros((1, X.shape[1]))).shap_values(X))
        ours = np.stack([clf.coef_[c] * X[0] for c in range(3)])
        theirs = sv[0].T if sv.shape[-1] == 3 else sv[:, 0, :]
        self.assertLess(float(np.abs(theirs - ours).max()), 1e-9)


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_pages_and_old_pages_still_work(self):
        for url in ['/', '/text-analysis', '/media-analysis', '/model-evaluation',
                    '/model-comparison', '/explainable-ai', '/api/health']:
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_old_text_api_unchanged(self):
        r = self.client.post('/api/analyze/text', json={'text': 'I love everyone'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()['prediction'], 'Normal')
        self.assertEqual(self.client.post('/api/analyze/text', json={}).status_code, 400)

    def test_explain_endpoint_errors(self):
        self.assertEqual(self.client.post('/api/explain', data='x', content_type='text/plain').status_code, 400)
        self.assertEqual(self.client.post('/api/explain', json={'text': ''}).status_code, 400)
        self.assertEqual(self.client.post('/api/explain', json={'text': 'hello there'}).status_code, 200)

    def test_missing_reports_do_not_crash(self):
        with mock.patch('app.EVALUATION_REPORT', '/nonexistent/e.json'), \
             mock.patch('app.COMPARISON_REPORT', '/nonexistent/c.json'):
            e = self.client.get('/api/evaluation/results').get_json()
            c = self.client.get('/api/comparison/results').get_json()
        self.assertFalse(e['available'])
        self.assertIn('evaluate_model.py', e['reason'])
        self.assertFalse(c['available'])

    def test_unknown_job_rejected(self):
        self.assertEqual(self.client.post('/api/jobs/rm-rf/start').status_code, 404)
        self.assertEqual(self.client.get('/api/jobs/rm-rf/status').status_code, 404)


class ValidationTest(unittest.TestCase):
    """Dataset validation details added for the real Davidson dataset."""

    def _rows(self, n=40):
        base = []
        for i in range(n):
            base += [(f'hateful text number {i} x', 0), (f'offensive words number {i} y', 1),
                     (f'nice normal sentence {i} z', 2)]
        return base

    def test_blank_label_rows_are_skipped_and_counted_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'd.csv')
            write_csv(path, self._rows() + [('text with no label', '')])
            texts, labels, info = mc.load_dataset(path)
        self.assertEqual(info['rows_skipped_missing_label'], 1)
        self.assertEqual(len(texts), 120)

    def test_duplicate_with_conflicting_label_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'd.csv')
            write_csv(path, self._rows() + [('hateful text number 0 x', 1)])   # same text, other label
            info = mc.prepare_data(path)['info']
        self.assertEqual(info['duplicate_label_conflicts'], 1)
        self.assertEqual(info['duplicates_or_empty_removed'], 1)

    def test_label_mapping_is_documented_in_info(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'd.csv')
            write_csv(path, self._rows())
            _, _, info = mc.load_dataset(path)
        self.assertEqual(info['label_mapping'], {'0': 'Hate Speech', '1': 'Offensive', '2': 'Normal'})


class FourModelComparisonTest(unittest.TestCase):
    def test_the_four_required_algorithms_are_configured(self):
        import train_and_compare as tc
        names = [c['name'] for c in tc.CANDIDATES]
        self.assertEqual(names, ['Logistic Regression', 'Multinomial Naive Bayes',
                                 'Linear SVM', 'Random Forest'])
        self.assertEqual(len({c['slug'] for c in tc.CANDIDATES}), 4)

    def test_every_candidate_trains_and_predicts_valid_classes(self):
        # Tiny temporary matrix, ONLY to prove the four builders work; no number from it is ever shown.
        import train_and_compare as tc
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [f'{w} item {i}' for i in range(30) for w in ('hate', 'rude', 'kind')]
        y = np.array([0, 1, 2] * 30)
        X = TfidfVectorizer().fit_transform(docs)
        for spec in tc.CANDIDATES:
            pred = spec['build']().fit(X, y).predict(X)
            self.assertTrue(set(pred.tolist()) <= {0, 1, 2}, spec['name'])


class ProductionModelStatusTest(unittest.TestCase):
    def test_status_reports_matching_versions_after_retrain(self):
        from utils import production_model as pm
        clf, vec, err = pm.load_artifacts()
        st = pm.status(clf, vec, err)
        self.assertTrue(st['loaded'], st['message'])
        self.assertEqual(st['pickle_sklearn'], st['installed_sklearn'])
        self.assertFalse(st['version_mismatch'])
        self.assertEqual(st['level'], 'ok')

    def test_stale_version_is_not_remembered_between_loads(self):
        from utils import production_model as pm
        pm._pickle_version_seen = '0.0.0'
        pm.load_artifacts()
        self.assertNotEqual(pm._pickle_version_seen, '0.0.0')


class ReportRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_history_rejects_path_traversal_and_unknown_kinds(self):
        for url in ['/api/reports/history/evaluation/..%2F..%2Fapp.py',
                    '/api/reports/history/evaluation/evaluation_x.json/../../a',
                    '/api/reports/history/comparison/evaluation_2026-01-01T00-00-00.json',
                    '/api/reports/history/secrets', '/api/reports/secrets/download']:
            self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_download_missing_report_is_a_clean_404(self):
        with mock.patch('app.EVALUATION_REPORT', '/nonexistent/e.json'):
            r = self.client.get('/api/reports/evaluation/download')
        self.assertEqual(r.status_code, 404)
        self.assertIn('error', r.get_json())

    def test_dataset_job_does_not_need_an_existing_dataset(self):
        with mock.patch('app.DEFAULT_DATASET', '/nonexistent/d.csv'), \
             mock.patch('app.job_runner.start', return_value=(True, 'Started.')) as start:
            self.assertEqual(self.client.post('/api/jobs/dataset/start').status_code, 200)
            start.assert_called_once_with('dataset')
            self.assertEqual(self.client.post('/api/jobs/evaluation/start').status_code, 400)

    def test_job_runner_only_allows_fixed_scripts_and_args(self):
        from utils import job_runner
        self.assertEqual(set(job_runner.JOBS), {'evaluation', 'comparison', 'dataset'})
        self.assertEqual(job_runner.JOBS['dataset'][1], ['--force'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
