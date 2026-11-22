> **UPDATE (v2) — read this first.** Since this guide was written: the comparison now has **four** models
> (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest); the real Davidson dataset
> is downloaded by `download_dataset.py`; the scikit-learn 1.8.0 → 1.9.1 mismatch was fixed by
> `retrain_production.py` (the original model is kept in `models/legacy_sklearn_1.8.0/` and is evaluated
> next to the new one); every evaluation/comparison run is archived in `reports/history/`.
> The README.md has the exact commands. Where this older guide says "three algorithms" or mentions
> the saga solver, the README is correct.

# HateShield AI — Upgrade Guide (Hinglish)

Is upgrade mein 3 naye features add hue hain, aapke **existing project ke andar** (alag demo nahi):

1. **Model Evaluation** (`/model-evaluation`)
2. **Model Comparison** (`/model-comparison`)
3. **Explainable AI** (`/explainable-ai`)

Purane features (Dashboard, Quick Analyze, Text/Batch Analysis, Image, Video) ka code **touch nahi hua**.
Maine original aur upgraded app ke 15 test texts par outputs compare kiye: dono bilkul same aaye.

---

## 0. Pehle ye 4 important baatein padh lo (inspection ke findings)

1. **Project mein labelled dataset nahi tha.** Sirf do `.pkl` files thi. Isliye "genuine evaluation" ke liye
   dataset chahiye. Maine `download_dataset.py` banayi jo public **Davidson et al. (2017)** dataset (24,783 tweets,
   0=hate, 1=offensive, 2=neither — aapke model ke labels jaisi hi numbering) download karti hai.
   Aap chaho to apna CSV bhi use kar sakte ho (`data/README.md` dekho).
2. **Aapka existing model bahut chhote data par trained hai.** Vectorizer ke IDF se estimate nikalta hai ki
   training sirf **~600 documents** par hui (vocabulary sirf 2,976 words, jabki config mein 25,000 tha).
   Original training data project mein nahi hai, isliye ye **prove nahi ho sakta** ki test rows model ne
   pehle nahi dekhe. Dashboard par is model ke aage "held-out NOT guaranteed" badge dikhta hai — ye honest hai.
3. **Version problem (real bug):** `.pkl` files scikit-learn **1.8.0** se bani hain, lekin aapke `venv` mein
   Python 3.9 + scikit-learn **1.6.1** hai. Wahan `predict_proba` crash hota hai
   (`'LogisticRegression' object has no attribute 'multi_class'`) aur `predictor.py` us error ko chup-chaap
   pakad kar **keyword classifier** par fallback kar deta hai. Matlab aapke Mac par ML model shayad actually
   use hi nahi ho raha (status bar "Adam Classifier" dikhata hai, phir bhi). Naye pages ye problem
   **dikhate** hain (red banner). Fix niche section 2 mein hai.
4. **README ka "100% accuracy (15/15 edge cases)"** ek hand-written 15-sentence check hai, proper held-out
   evaluation nahi. Viva mein use "accuracy" ki tarah mat batana. Ab aapke paas real evaluation hai.

---

## 1. Kya-kya change hua

### Naye files
| File | Kaam (simple Hinglish) |
|---|---|
| `utils/ml_common.py` | Dataset CSV padhna + validate karna, duplicates hatana, **stratified 80/20 split**, metrics (accuracy/precision/recall/F1/confusion matrix) nikalna, bootstrap confidence interval, JSON report save/load. |
| `utils/production_model.py` | Aapke existing `.pkl` model ko safely load karta hai aur check karta hai ki ye is computer par chal raha hai ya nahi (version mismatch detect karta hai). |
| `utils/explainer.py` | **Explainable AI ka dimaag.** Text ke har word ka contribution = `weight × TF-IDF` nikalta hai, rules (slur/threat patterns) alag se dikhata hai. |
| `utils/job_runner.py` | Web page ka "Run" button dabane par scripts ko background mein chalata hai (browser hang nahi hota). |
| `evaluate_model.py` | **Feature 1.** Existing model ko test split par chalake real metrics nikalti hai → `reports/evaluation_results.json`. |
| `train_and_compare.py` | **Feature 2.** LR, Naive Bayes, Linear SVM aur Random Forest ko same split/same features par train karke compare karti hai → `reports/comparison_results.json`. Production model ko touch nahi karti. |
| `promote_model.py` | Agar kisi candidate ko production banana ho: summary dikhati hai, backup banati hai, aur aapse **YES** type karwa ke hi replace karti hai. |
| `download_dataset.py` | Public dataset `data/dataset.csv` mein download karti hai. |
| `templates/model_evaluation.html`, `model_comparison.html`, `explainable_ai.html` | Teen naye pages. |
| `static/css/analytics.css`, `static/js/analytics.js` | Naye pages ka design + charts (pure HTML/CSS — koi external library nahi, offline bhi chalta hai). |
| `tests/test_new_features.py` | 17 automatic tests. |
| `data/README.md`, `reports/.gitkeep`, `.gitignore`, `uploads/.gitkeep`, `UPGRADE_GUIDE.md` | Support files. |

### Modified (purani) files
| File | Kya badla |
|---|---|
| `app.py` | Sirf **add** kiya: 3 page routes + APIs (`/api/model-status`, `/api/dataset-info`, `/api/evaluation/results`, `/api/comparison/results`, `/api/jobs/<name>/start|status`, `/api/explain`). Port ab `PORT` env variable se badal sakte ho (default 5000 same). |
| `templates/base.html` | Navbar mein 3 naye links + navbar ko mobile par wrap hone layak banaya. |
| `templates/index.html` | Dashboard par 3 naye cards. |
| `requirements.txt` | Koi **nayi dependency nahi**. Sirf scikit-learn line Python version ke hisaab se (marker). |
| `README.md` | Naya section. |

**Backup:** `BACKUP_original_hate_speech_project_improved.zip` mein aapka original code (venv aur .git ke bina) hai.
Original wapas chahiye to us zip ko extract kar lo.

---

## 2. Setup — Mac Terminal commands

### Step A: Naya Python (recommended — isse version bug fix ho jaata hai)
Aapka purana venv Python 3.9 hai; naye scikit-learn ke liye Python **3.11+** chahiye.
```bash
cd ~/path/to/HATE_SPEECH_PROJECT/hate_speech_project_improved

# Homebrew se Python 3.12 (agar brew nahi hai to python.org se Python 3.12 installer download karo)
brew install python@3.12

# naya virtual environment (purane `venv` ko mat chhedo)
# Apple Silicon (M1/M2/M3):
/opt/homebrew/bin/python3.12 -m venv venv312
# Intel Mac ho to upar wali line ki jagah:  /usr/local/bin/python3.12 -m venv venv312

source venv312/bin/activate
python --version            # 3.12.x aana chahiye
pip install --upgrade pip
pip install -r requirements.txt
```
(Image/video OCR ke liye optional: `brew install tesseract`.)

> **Agar aap Python 3.9 hi rakhna chaho:** app chalega, par shipped model load nahi hoga (banner dikhega).
> Is case mein section 3 ke baad `python promote_model.py logistic_regression_lbfgs` se apne Mac par
> naya model train/promote kar sakte ho (confirmation ke baad). Ye aapka decision hai.

### Step B: Dataset + results banao (genuine numbers, aapke Mac par)
```bash
python download_dataset.py        # data/dataset.csv banata hai
python train_and_compare.py       # ~1 minute (Random Forest sabse slow hai)
python evaluate_model.py          # ~10-30 seconds
```
Ye dono scripts web page ke "Run" button se bhi chal jaati hain.

### Step C: App chalao
```bash
python app.py
```
Browser mein kholo: http://localhost:5000
Agar port 5000 busy hai (macOS AirPlay Receiver use karta hai): `PORT=5001 python app.py` → http://localhost:5001

### Step D: Automatic tests
```bash
python -m unittest discover -s tests -v
```

---

## 3. Testing checklist

**Purane features (regression)**
- [ ] Dashboard (`/`): Quick Analyze mein "I love everyone" → Normal; "you are an idiot" → Offensive.
- [ ] `/text-analysis`: Single + Batch (ek line = ek text) dono chalte hain.
- [ ] `/media-analysis`: image/video upload (OCR ke liye tesseract chahiye) pehle jaisa.
- [ ] Navbar 6 links dikhata hai; browser window chhoti karne par links neeche wrap hote hain.

**Feature 1 — Model Evaluation**
- [ ] Dataset/report ke bina page khaali-state + instructions dikhata hai (crash nahi).
- [ ] "Run evaluation" dabao → log dikhta hai → metrics tiles (Accuracy, Macro Precision/Recall, Macro F1, Weighted F1), confusion matrix, per-class bars, classification report aate hain.
- [ ] Dropdown se model badlo (Production hybrid / Production ML / retrained candidates).
- [ ] Production models par "held-out NOT guaranteed", retrained candidates par "held-out guaranteed" badge.

**Feature 2 — Model Comparison**
- [ ] "Run comparison" → table (Accuracy, Precision, Recall, Macro F1, Weighted F1, Training time), grouped bar charts, time chart, 4 confusion matrices.
- [ ] Best model macro-F1 se chuna gaya; "statistically tied" note dikhta hai agar fark noise ke andar hai.
- [ ] `models/adam_classifier.pkl` ki modified-time **nahi badli** (production touch nahi hua).
- [ ] `python promote_model.py` (bina argument) candidates ki list dikhata hai; `promote_model.py linear_svm` **refuse** karta hai (probabilities nahi hoti).

**Feature 3 — Explainable AI**
- [ ] Example 1/2/3 try karo. Chart mein positive (purple) / negative (pink) bars, highlighted words, rule evidence dikhte hain.
- [ ] Tabs se Hate/Offensive/Normal aur "Why X rather than Y" dekho.
- [ ] Gibberish (`qwertyuiop zxcvbnm`) → "unknown words" dikhte hain, "not confident" note aata hai.
- [ ] Khaali text → error message; 5000+ characters → error message.
- [ ] Footer card mein "Exactness check: yes ✔" aur "Matches scikit-learn predict_proba: yes ✔" dikhta hai.

---

## 4. Major functions — simple Hinglish

- `prepare_data()` (`ml_common.py`): CSV padhta hai → text clean karta hai (aapka hi `_clean_text`) → **duplicate** texts hatata hai (taaki same sentence train aur test dono mein na jaye) → stratified split karta hai (har class ka ratio train/test mein same).
- `compute_metrics()`: Sach mein predictions se accuracy/precision/recall/F1/confusion matrix nikalta hai (scikit-learn). Test mein haath se calculate kiye numbers se match verify hota hai.
- `bootstrap_macro_f1()`: Test set ko 1000 baar resample karke dekhta hai ki macro-F1 kitna stable hai. Isse pata chalta hai ki "best" model sach mein best hai ya fark sirf noise hai.
- `train_and_compare.py`: TF-IDF **sirf train data par** fit hota hai (leakage nahi). Sab models ko same matrix milta hai. Har model ka training time alag naapa jaata hai.
- `load_artifacts()` / `status()` (`production_model.py`): `.pkl` load karte hain aur check karte hain ki `predict_proba` chal raha hai ya nahi.
- `ProductionExplainer.explain()` (`explainer.py`): text → TF-IDF → har word ka `weight × tfidf` → positive/negative lists. Saath mein final app-pipeline ka answer, rules (slur/threat/patterns) aur limitations bhejta hai.
- `job_runner.start()/status()`: scripts ko alag process mein chalata hai aur page progress poll karta hai.

---

## 5. XAI mein SHAP/LIME kyun nahi?

Aapka ML model **linear** hai (TF-IDF + Logistic Regression). Linear model ka score exactly tod sakte hain:
`score = intercept + Σ weight × tfidf`. Har term ek word ka contribution hai. Ye **SHAP LinearExplainer (zero baseline)
ke bilkul same numbers** deta hai — maine verify kiya, max difference `0.0` (test: `test_matches_shap_linear_explainer_if_installed`).
LIME sirf approximate karta hai (random perturbations), aur `shap` library ke heavy dependencies Python 3.9 Mac par
install problems de sakte hain. Isliye exact formula use kiya — koi fake/approximate explanation nahi.

---

## 6. Actual limitations (honest)

- **Dataset aapka nahi tha.** Results Davidson tweets par hain. Tweets ka domain aapke model ke original data se alag hai, isliye production model ka score kam aata hai (mere test run mein: hybrid accuracy ≈ 52.8%, macro-F1 ≈ 0.47; sirf ML model macro-F1 ≈ 0.36). Ye real finding hai, bug nahi — aur viva mein ek achha "baseline vs improved" story banti hai.
- **Production model ke liye "held-out" prove nahi ho sakta** (training data unknown). Guaranteed held-out numbers sirf retrained candidates ke hain.
- **Single split, single run.** Chhote differences noise ho sakte hain (isliye bootstrap CI dikhaya). Training time machine-dependent hai.
- **Hyper-parameter tuning nahi hui** (jaanbujh kar, taaki test set se tuning leak na ho).
- **Linear SVM** probabilities nahi deta, isliye current app (jo probabilities blend karta hai) mein seedha promote nahi ho sakta.
- **Multinomial NB** class imbalance compensate nahi karta (class_weight support nahi) — uska macro-F1 isliye kam hai.
- **Davidson dataset ke labels** khud noisy hain (crowd-sourced); "Offensive" class 77% hai → accuracy se zyada macro-F1 dekho.
- **XAI sirf local TF-IDF+LR model** ko explain karta hai. Claude API / BiLSTM routes ko nahi. Final app label rules se bhi aa sakta hai — wo alag "Rule evidence" mein dikhta hai.
- **Explanation ≠ intent.** Word-weights dataset se seekhe hue patterns hain, writer ka irada ya "sach mein hateful hai ya nahi" prove nahi karte; dataset bias bhi ho sakta hai.
- Evaluation sirf local pipeline ka hai; Claude API route (paid, non-deterministic) evaluate nahi hota.
- Maine ye sab Linux sandbox mein (Python 3.12, scikit-learn 1.9.1, aur 1.6.1 dono) test kiya, Chromium mein pages render karke dekhe. Aapke Mac par exact timing alag hogi. Mac-specific cheezein (brew, tesseract, AirPlay port) maine run nahi ki.

---

## 7. Viva mein kaise demonstrate karein (5-7 minute)

1. **Problem statement** (30 sec): "Maine pehle model banaya tha; ab maine usse scientifically evaluate, compare aur explain kiya."
2. **Model Evaluation page**: "Ye numbers hardcoded nahi — `evaluate_model.py` ne 4,885 held-out texts par real predictions chalayi." Macro-F1 aur confusion matrix dikhao: "Accuracy ke bajaye macro-F1 kyun? kyunki 77% data Offensive class ka hai." Honestly bolo ki original model chhote data par trained tha, isliye is dataset par score kam hai — "isi wajah se maine retrain/compare kiya."
3. **Model Comparison page**: table + chart dikhao. "Chaaron algorithms same stratified split aur same TF-IDF features par; TF-IDF sirf train par fit — data leakage nahi." LR (lbfgs) aur Linear SVM ka macro-F1 statistically tied hai (bootstrap CI) — "SVM ka accuracy/weighted-F1 zyada hai, LR ka macro-F1 thoda zyada; fark noise ke andar hai." Production model **maine auto-replace nahi kiya** — `promote_model.py` confirmation maangta hai.
4. **Explainable AI page**: "you are such a stupid idiot" explain karo. Purple bars = words jo class ki taraf dhakel rahe hain. Rule evidence dikhao ("kuch decisions hand-written rules se aate hain"). Unknown words aur "intent prove nahi karta" limitation khud bolo.
5. **Terminal par** `python -m unittest discover -s tests -v` chalao — "17 automatic tests".

**Possible viva questions**
- *Stratified split kyun?* Har class ka ratio train aur test mein same rahe (hate speech rare class hai).
- *Data leakage kya hai, kaise roka?* Test ki jaankari training mein jaana. Duplicates split se pehle hataye; TF-IDF sirf train par fit; test par tuning nahi.
- *Macro vs weighted F1?* Macro: har class barabar; Weighted: class size ke hisaab se.
- *SHAP/LIME kyun nahi?* Section 5.
- *Production model replace kyun nahi kiya?* Blending rules purane model ke liye tuned hain; user confirmation zaroori.
- *Production model ka solver kyun badla?* Original `saga` (max_iter=2000) 19.5k training texts par converge nahi hota (macro-F1 ~0.66); `lbfgs` converge hota hai (~0.73). Isliye `retrain_production.py` ab lbfgs use karta hai (`--solver saga` se purani setting mil jaati hai).
