from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import base64
from utils.predictor import HateSpeechPredictor

# ── NEW (upgrade): Model Evaluation / Model Comparison / Explainable AI ──
from utils import job_runner, production_model
from utils.explainer import ExplainerInputError, ExplainerUnavailable, ProductionExplainer
from utils.ml_common import (COMPARISON_REPORT, DEFAULT_DATASET, EVALUATION_REPORT,
                             HISTORY_DIR, DataError, prepare_data, read_json)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max (for video)

predictor = HateSpeechPredictor()
explainer = ProductionExplainer(predictor)   # NEW: reuses the already-loaded production model

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-analysis')
def text_analysis():
    return render_template('text_analysis.html')

@app.route('/media-analysis')
def media_analysis():
    return render_template('media_analysis.html')

@app.route('/api/health')
def health():
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    classifier = predictor.get_classifier_type()
    # Priority: bilstm > claude_ai > adam > keyword
    if classifier == 'keyword' and api_key:
        classifier = 'claude_ai'
    adam_ready = predictor.adam_clf is not None
    return jsonify({
        'status': 'ok',
        'classifier': classifier,
        'adam_ready': adam_ready,
        'ocr_ready': True,
        'claude_ai': bool(api_key)
    })

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 chars)'}), 413

    result = predictor.predict(text)
    return jsonify(result)

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400
    
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts must be a non-empty list'}), 400
    if len(texts) > 50:
        return jsonify({'error': 'Max 50 texts per batch'}), 413

    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            result = predictor.predict(text.strip())
            results.append({'text': text, **result})

    summary = {
        'total': len(results),
        'hate_speech': sum(1 for r in results if r.get('prediction') == 'Hate Speech'),
        'offensive': sum(1 for r in results if r.get('prediction') == 'Offensive'),
        'clean': sum(1 for r in results if r.get('prediction') == 'Normal')
    }
    return jsonify({'results': results, 'summary': summary})

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type'}), 400

    img_bytes = file.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    result = predictor.predict_image(img_b64, ext)
    return jsonify(result)

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type. Use MP4, AVI, MOV, MKV, WEBM'}), 400

    # Save temp file
    temp_path = os.path.join(UPLOAD_FOLDER, 'temp_video.' + ext)
    file.save(temp_path)

    try:
        result = predictor.predict_video(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════
#  NEW (upgrade) — page routes
# ═══════════════════════════════════════════════════════════════════════
@app.route('/model-evaluation')
def model_evaluation():
    return render_template('model_evaluation.html')

@app.route('/model-comparison')
def model_comparison():
    return render_template('model_comparison.html')

@app.route('/explainable-ai')
def explainable_ai():
    return render_template('explainable_ai.html')


# ═══════════════════════════════════════════════════════════════════════
#  NEW (upgrade) — status / dataset
# ═══════════════════════════════════════════════════════════════════════
@app.route('/api/model-status')
def model_status():
    """Is the production ML model really usable on this computer?"""
    clf, vec, err = production_model.load_artifacts()   # fresh load: also reads the pickle's sklearn version
    return jsonify(production_model.status(clf, vec, err))

_dataset_cache = {}

@app.route('/api/dataset-info')
def dataset_info():
    rel = os.path.relpath(DEFAULT_DATASET, os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(DEFAULT_DATASET):
        return jsonify({'present': False, 'valid': False, 'path': rel,
                        'error': 'No dataset yet. Press "Download dataset" or run: python download_dataset.py '
                                 '(or put your own labelled CSV at data/dataset.csv).'})
    st = os.stat(DEFAULT_DATASET)
    key = (st.st_mtime_ns, st.st_size)
    if _dataset_cache.get('key') != key:          # validate again only when the file changed
        try:
            _dataset_cache.update(key=key, ok=True, info=prepare_data(DEFAULT_DATASET)['info'], error=None)
        except DataError as e:
            _dataset_cache.update(key=key, ok=False, info=None, error=str(e))
    if not _dataset_cache['ok']:
        return jsonify({'present': True, 'valid': False, 'path': rel, 'error': _dataset_cache['error']})
    info = _dataset_cache['info']
    return jsonify({'present': True, 'valid': True, 'path': rel, 'rows': info['rows_read'],
                    'file': info['file'], 'text_column': info['text_column'],
                    'label_column': info['label_column'], 'info': info})


# ═══════════════════════════════════════════════════════════════════════
#  NEW (upgrade) — Model Evaluation + Model Comparison data
# ═══════════════════════════════════════════════════════════════════════
def _split_id(report):
    d = (report or {}).get('dataset', {})
    return (d.get('sha256_prefix'), d.get('random_state'), d.get('test_size'), d.get('n_test'))

@app.route('/api/evaluation/results')
def evaluation_results():
    ev, ev_err = read_json(EVALUATION_REPORT)
    cmp_, _ = read_json(COMPARISON_REPORT)
    payload = {'available': False, 'models': {}, 'warnings': [], 'dataset': None,
               'generated_at': None, 'environment': None, 'production_model_status': None,
               'notes': [], 'reason': None}

    if ev is not None:
        payload.update({'dataset': ev.get('dataset'), 'generated_at': ev.get('generated_at'),
                        'environment': ev.get('environment'),
                        'production_model_status': ev.get('production_model_status'),
                        'notes': ev.get('notes', [])})
        payload['models'].update(ev.get('models', {}))
    elif ev_err and ev_err != 'missing':
        payload['warnings'].append(f'evaluation_results.json could not be read ({ev_err}). Re-run evaluate_model.py.')

    # Candidates trained on the TRAIN split only -> these ARE guaranteed held-out results
    if cmp_ is not None:
        if ev is None or _split_id(ev) == _split_id(cmp_):
            if ev is None:
                payload.update({'dataset': cmp_.get('dataset'), 'generated_at': cmp_.get('generated_at'),
                                'environment': cmp_.get('environment')})
            for name, m in cmp_.get('models', {}).items():
                payload['models'][f"candidate:{m['slug']}"] = {
                    'label': f'{name} (retrained, train split only)',
                    'description': 'Trained on the training split only and tested on the held-out '
                                   'test split (see Model Comparison).',
                    'kind': 'candidate', 'held_out_guarantee': True,
                    'held_out_note': 'Vectorizer and classifier were fitted on the training split only.',
                    'metrics': m['metrics'], 'converged': m.get('converged'),
                    'order': m.get('order', 0),
                }
        else:
            payload['warnings'].append(
                'The comparison report was made with a different dataset/split than the evaluation '
                'report, so retrained candidates are hidden. Re-run both scripts.')

    if ev is None and cmp_ is not None:
        payload['warnings'].append('The existing production model has not been evaluated yet. '
                                   'Press "Run evaluation" (or run: python evaluate_model.py).')
    if payload['models']:
        payload['available'] = True
    else:
        payload['reason'] = ('No evaluation report yet. Make sure data/dataset.csv exists, then press '
                             '"Run evaluation" (or run: python evaluate_model.py).')
    payload['dataset_present'] = os.path.exists(DEFAULT_DATASET)
    return jsonify(payload)

@app.route('/api/comparison/results')
def comparison_results():
    cmp_, err = read_json(COMPARISON_REPORT)
    if cmp_ is None:
        msg = ('No comparison report yet. Make sure data/dataset.csv exists, then press '
               '"Run comparison" (or run: python train_and_compare.py).')
        if err and err != 'missing':
            msg = f'comparison_results.json could not be read ({err}). Re-run train_and_compare.py.'
        return jsonify({'available': False, 'reason': msg,
                        'dataset_present': os.path.exists(DEFAULT_DATASET)})
    return jsonify({'available': True, 'report': cmp_,
                    'dataset_present': os.path.exists(DEFAULT_DATASET)})

@app.route('/api/jobs/<name>/start', methods=['POST'])
def job_start(name):
    if name not in job_runner.JOBS:
        return jsonify({'error': 'Unknown job'}), 404
    if name != 'dataset' and not os.path.exists(DEFAULT_DATASET):
        return jsonify({'error': 'data/dataset.csv not found. Press "Download dataset" or run: '
                                 'python download_dataset.py'}), 400
    ok, msg = job_runner.start(name)
    if not ok:
        return jsonify({'error': msg}), 409
    return jsonify({'started': True})

@app.route('/api/jobs/<name>/status')
def job_status(name):
    if name not in job_runner.JOBS:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify(job_runner.status(name))


# ═══════════════════════════════════════════════════════════════════════
#  Saved reports: history + download
# ═══════════════════════════════════════════════════════════════════════
_HISTORY_NAME = re.compile(r'^(evaluation|comparison)_[0-9T\-]+\.json$')

@app.route('/api/reports/history/<kind>')
def reports_history(kind):
    if kind not in ('evaluation', 'comparison'):
        return jsonify({'error': 'Unknown report type'}), 404
    items = []
    if os.path.isdir(HISTORY_DIR):
        for fn in sorted(os.listdir(HISTORY_DIR), reverse=True):
            if fn.startswith(kind + '_') and _HISTORY_NAME.match(fn):
                rep, _ = read_json(os.path.join(HISTORY_DIR, fn))
                if rep:
                    items.append({'name': fn, 'generated_at': rep.get('generated_at'),
                                  'best_model': rep.get('best_model'),
                                  'dataset_sha': (rep.get('dataset') or {}).get('sha256_prefix')})
    return jsonify({'kind': kind, 'reports': items})

@app.route('/api/reports/history/<kind>/<name>')
def reports_history_item(kind, name):
    if kind not in ('evaluation', 'comparison') or not _HISTORY_NAME.match(name) \
            or not name.startswith(kind + '_'):
        return jsonify({'error': 'Unknown report'}), 404
    rep, err = read_json(os.path.join(HISTORY_DIR, name))
    if rep is None:
        return jsonify({'error': 'Report not found or unreadable'}), 404
    if kind == 'evaluation':
        return jsonify({'available': True, 'models': rep.get('models', {}), 'warnings': [],
                        'dataset': rep.get('dataset'), 'generated_at': rep.get('generated_at'),
                        'environment': rep.get('environment'), 'notes': rep.get('notes', []),
                        'production_model_status': rep.get('production_model_status'),
                        'archived': True})
    return jsonify({'available': True, 'report': rep, 'archived': True})

@app.route('/api/reports/<kind>/download')
def reports_download(kind):
    path = {'evaluation': EVALUATION_REPORT, 'comparison': COMPARISON_REPORT}.get(kind)
    if path is None:
        return jsonify({'error': 'Unknown report type'}), 404
    if not os.path.exists(path):
        return jsonify({'error': 'No report yet. Run it first.'}), 404
    return send_file(path, as_attachment=True, download_name=f'hateshield_{kind}_report.json',
                     mimetype='application/json')


# ═══════════════════════════════════════════════════════════════════════
#  NEW (upgrade) — Explainable AI
# ═══════════════════════════════════════════════════════════════════════
@app.route('/api/explain', methods=['POST'])
def api_explain():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'text' not in data:
        return jsonify({'error': 'Send JSON like {"text": "..."}'}), 400
    try:
        return jsonify(explainer.explain(data['text'], data.get('top_k', 8)))
    except ExplainerInputError as e:
        return jsonify({'error': str(e)}), 400
    except ExplainerUnavailable as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:                      # never leak a stack trace to the browser
        print(f'[ERROR] explain failed: {type(e).__name__}: {e}')
        return jsonify({'error': 'Could not generate the explanation.'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 200MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # PORT can be changed from the terminal:  PORT=5001 python app.py  (macOS AirPlay may use 5000)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
