# Hate Speech Detector — Improved

A Flask web app that classifies text, images, and video for hate speech using a **3-tier hybrid classifier**.

## Classifiers (Priority Order)

| Priority | Classifier | Description |
|----------|-----------|-------------|
| 1 | 🧠 BiLSTM | Deep learning model (if `models/bilstm_model.h5` present) |
| 2 | 🤖 Claude AI | Anthropic API semantic understanding (if `ANTHROPIC_API_KEY` set) |
| 3 | ⚡ Adam Hybrid | TF-IDF + Logistic Regression blended with keyword rules |
| 4 | 🔑 Keyword | Pure rule-based fallback |

### Adam Hybrid Classifier
- **Algorithm:** TF-IDF (25K features, 1-3 ngrams) + Logistic Regression (`lbfgs` solver, balanced class weights; see "scikit-learn compatibility" below)
- **Blend:** 60% Adam + 40% Keyword scores → final prediction
- **Overrides:** Direct threats, hard slurs, targeted hate phrases → always Hate Speech
- **Positive guard:** Sentiment-aware — prevents false positives on clearly normal text
- **Accuracy:** 100% on a hand-written test suite (15/15 edge cases) — *this is not a held-out evaluation; see the Model Evaluation page for real metrics*

## Labels

- **Hate Speech** — Slurs, targeted dehumanization, death threats, calls for violence against groups
- **Offensive** — Profanity, insults, crude language without targeted hate
- **Normal** — Clean, neutral, or positive content

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`

### Optional: Claude AI (best quality)
```bash
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

### Optional: OCR for images
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr
```

## Project Structure

```
hate_speech_project_improved/
├── app.py                  # Flask routes
├── requirements.txt
├── models/
│   ├── adam_classifier.pkl   # Production LR model (scikit-learn 1.9.1)
│   └── adam_vectorizer.pkl   # TF-IDF vectorizer
├── utils/
│   └── predictor.py          # HateSpeechPredictor class
└── templates/
    ├── base.html
    ├── index.html
    ├── text_analysis.html
    └── media_analysis.html
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Classifier status |
| POST | `/api/analyze/text` | `{"text": "..."}` |
| POST | `/api/analyze/batch` | `{"texts": [...]}` (max 50) |
| POST | `/api/analyze/image` | multipart file upload |
| POST | `/api/analyze/video` | multipart file upload |


---

# Model Evaluation, Model Comparison, Explainable AI

| Page | What it does |
|------|--------------|
| `/model-evaluation` | Runs the real production model on the held-out test split: accuracy, precision, recall, macro/weighted F1, confusion matrix, per-class results. **Run evaluation** executes `evaluate_model.py`; every run is archived and can be re-opened from the *Report* menu or downloaded as JSON. |
| `/model-comparison` | Trains **Logistic Regression, Multinomial Naive Bayes, Linear SVM and Random Forest** on the same data/split/TF-IDF features and compares accuracy, macro precision/recall/F1, training time, with confusion matrices and bootstrap confidence intervals. **Run comparison** executes `train_and_compare.py`. |
| `/explainable-ai` | Per-word contributions behind the ML model's prediction. |

## Setup and execution (exact commands)

Requires **Python 3.11+** (tested on 3.12).

```bash
# 1. environment
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. dataset (already included as data/dataset.csv; this re-creates + validates it)
python download_dataset.py

# 3. (re)create the production model with the installed scikit-learn - only needed if the app
#    shows "Model saved with scikit-learn X but Y is installed"
python retrain_production.py          # asks for YES; add --yes to skip the question

# 4. reports (the zip already contains reports from the run described below; re-run to reproduce)
python train_and_compare.py           # ~1 minute (Random Forest is the slow model)
python evaluate_model.py              # ~15 seconds

# 5. start the app
python app.py                         # http://localhost:5000   (PORT=5001 python app.py to change the port)

# 6. tests
python -m unittest discover -s tests -v
```

You can also press **Run evaluation** / **Run comparison** / **Download dataset** in the browser instead of steps 2, 4.
Restart `python app.py` after step 3 so the app loads the new model files.

## scikit-learn compatibility fix

The original model files were saved with scikit-learn **1.8.0**; the environment has **1.9.1**. Pickled scikit-learn models
are only guaranteed to load under the version that created them, so this was fixed properly instead of hiding the warning:

* `requirements.txt` pins `scikit-learn==1.9.1`.
* `retrain_production.py` re-created `models/adam_classifier.pkl` and `models/adam_vectorizer.pkl` with 1.9.1 (TF-IDF 25,000
  features, 1-3 grams; Logistic Regression, balanced class weights), fitted on the **training split only**, and wrote
  `models/production_model_meta.json` (versions, dataset fingerprint, split, file checksums).
* The **original files are preserved** in `models/legacy_sklearn_1.8.0/` and are evaluated next to the new model.
  Roll back with `python promote_model.py --restore models/legacy_sklearn_1.8.0`.
* One deliberate change: the original solver (`saga`, 2000 iterations) **does not converge** on the full 19.5k-text training set
  (macro-F1 0.656) while `lbfgs` converges in 50 iterations (macro-F1 0.727), so `lbfgs` is the new default.
  `python retrain_production.py --solver saga` reproduces the original settings.
* `/api/model-status` (and the banner on the Model Evaluation page) now compares the version in the files with the installed one.

## Results actually obtained (this run)

Dataset: Davidson et al. 2017, 24,783 tweets -> 24,423 unique texts after removing 360 duplicates (37 of them with conflicting
labels). Stratified split, `random_state=42`: **19,538 train / 4,885 test**. Test classes: Hate Speech 279, Offensive 3,785, Normal 821.
Environment: Python 3.12.3, scikit-learn 1.9.1. All numbers are on the held-out test split.

| Model | Accuracy | Macro precision | Macro recall | **Macro F1** (95% CI) | Training time |
|-------|---------:|----------------:|-------------:|-----------------------:|--------------:|
| Logistic Regression | 86.49% | 69.03% | 78.65% | **72.68%** (70.8-74.5) | 0.42 s |
| Linear SVM | 88.99% | 72.82% | 72.57% | 72.55% (70.7-74.5) | 0.25 s |
| Random Forest (200 trees) | 88.33% | 78.04% | 59.88% | 62.00% (60.1-63.8) | 56.9 s |
| Multinomial Naive Bayes | 81.37% | 91.56% | 41.44% | 42.84% (41.4-44.2) | 0.01 s |

* **Best by macro-F1: Logistic Regression**, but the bootstrap difference to Linear SVM includes 0, so **LR and Linear SVM are
  statistically tied**; SVM has the higher accuracy and weighted-F1. Random Forest and Naive Bayes are clearly worse on macro-F1.
* The weak spot of every model is the **Hate Speech** class (F1 0.13-0.43), which is the well-known difficulty of this dataset
  (hate speech is hard to separate from offensive language) and the smallest class.
* Naive Bayes almost never predicts Hate Speech (recall on that class is close to 0).
* Timings are one run on one machine.

Model Evaluation (same test split):

| Evaluated thing | Accuracy | Macro F1 | Held-out? |
|-----------------|---------:|---------:|-----------|
| Production ML model (retrained, TF-IDF + LR) | 86.49% | 72.68% | **Yes - verified** from `production_model_meta.json` |
| Production pipeline (ML + keyword rules; what Quick Analyze runs) | 54.02% | 48.08% | not provable (hand-written rules) |
| Original ML model (scikit-learn 1.8.0, ~600 training texts) | 51.75% | 36.38% | not provable |
| Original pipeline (original ML + rules) | 52.81% | 46.96% | not provable |

## Remaining limitations (please read)

1. **The keyword/positivity rules in `utils/predictor.py` hurt accuracy on this dataset.** The ML model alone gets 86.5%, but the
   full Quick Analyze pipeline gets 54.0%, mostly because it labels 1,612 of 3,785 truly-offensive tweets as Normal
   (offensive recall 44.6%). This was true before the retraining too (52.8%). The rule logic was **not changed** (you asked to keep
   Quick Analyze as it is), so this is a decision for you: re-tune the blend or use the ML model more directly.
2. **Retraining changed some Quick Analyze outputs.** On a 15-sentence probe set, 12 predictions are identical and 3 changed
   (e.g. "You are such a stupid idiot" Offensive -> Hate Speech; "Women are inferior and belong in the kitchen" Hate Speech ->
   Offensive) because the rule weights were tuned for the old model.
3. The Davidson labels are crowd-sourced and noisy (37 duplicate texts carry conflicting labels); the dataset is English tweets from
   ~2009-2017, so results do not transfer automatically to other platforms, languages, images or video.
4. Random Forest uses fixed untuned settings (200 trees), and no model was hyper-parameter tuned. A single split/run is used.
5. Linear SVM has no probabilities, so it cannot replace the production model without calibration; `promote_model.py` refuses it.
6. Claude API and BiLSTM routes are not evaluated (not local/deterministic). OCR/image/video analysis was not changed or re-tested
   beyond the page loading.
7. `python app.py` starts Flask with `debug=True` on `0.0.0.0` (original setting). Do not expose it to untrusted networks.
8. Frontend checks were run in a headless DOM (jsdom) with real server responses, not on physical phones; the layout rules for
   narrow screens are in `static/css/analytics.css`.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Classifier status |
| POST | `/api/analyze/text` | `{"text": "..."}` |
| POST | `/api/analyze/batch` | `{"texts": [...]}` (max 50) |
| POST | `/api/analyze/image` / `/api/analyze/video` | multipart file upload |
| GET | `/api/model-status` | Is the production model loadable; saved vs installed scikit-learn version |
| GET | `/api/dataset-info` | Dataset presence, validation summary |
| GET | `/api/evaluation/results`, `/api/comparison/results` | Latest saved reports |
| GET | `/api/reports/history/<evaluation\|comparison>[/<file>]` | Archived runs |
| GET | `/api/reports/<evaluation\|comparison>/download` | Latest report as JSON |
| POST | `/api/jobs/<evaluation\|comparison\|dataset>/start`, GET `.../status` | Run a pipeline in the background |
| POST | `/api/explain` | `{"text": "..."}` |

## Project structure (new/changed files)

```
app.py                      Flask routes (+ report history/download, dataset job)
download_dataset.py         download + validate the Davidson dataset (--force, --check)
train_and_compare.py        four-model comparison -> reports/comparison_results.json, models/candidates/
evaluate_model.py           evaluation pipeline   -> reports/evaluation_results.json
retrain_production.py       rebuild production model with the installed scikit-learn (backs up the old one)
promote_model.py            promote a candidate to production (asks YES, makes a backup)
data/dataset.csv            the labelled dataset
models/adam_*.pkl           production model (scikit-learn 1.9.1) + production_model_meta.json
models/legacy_sklearn_1.8.0 the ORIGINAL model files, preserved
models/candidates/          the four trained comparison models (*.pkl) and their metadata (*.meta.json)
reports/                    latest JSON reports; reports/history/ keeps every run
utils/ml_common.py          dataset loading/validation, split, metrics, bootstrap
utils/production_model.py   production-model loading and version check
tests/                      python -m unittest discover -s tests -v
```

`UPGRADE_GUIDE.md` is the older Hinglish walkthrough (see the note at its top).
