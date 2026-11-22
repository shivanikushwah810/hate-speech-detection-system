# data/

This folder holds the labelled dataset: `data/dataset.csv`.

The project ships with it already (the real Davidson et al. 2017 file, 24,783 tweets).
If it is missing or damaged, re-create it:

```bash
python download_dataset.py          # download + validate (asks before overwriting)
python download_dataset.py --force  # overwrite without asking
python download_dataset.py --check  # only validate the existing file
```

The same thing is available as the **Download dataset** button on the Model Evaluation /
Model Comparison pages when no dataset is found.

## Manual download (if the script cannot reach GitHub)

1. Open <https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv>
2. Click **Download raw file** and save it as `data/dataset.csv`.
3. Run `python download_dataset.py --check`.

## Format

Original file columns: `,count,hate_speech,offensive_language,neither,class,tweet`.
The app uses **`tweet`** as the text and **`class`** as the label:

| class | meaning |
|-------|---------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Neither (shown as "Normal") |

Your own CSV also works if it has a text column (`text`, `tweet`, `comment`, `content`, ...) and a label column
(`label`, `class`, ...) with values 0/1/2 or `hate speech` / `offensive` / `normal`. For other column names use
`--text-col` / `--label-col` on `train_and_compare.py`, `evaluate_model.py`.
Unknown label values stop the run with an error; labels are never guessed.

## What validation does

Blank texts and blank labels are skipped and counted; unknown labels are an error; all three classes must be present;
texts that become identical after cleaning are de-duplicated **before** the split (duplicates whose labels disagree are
counted); a stratified 80/20 split with `random_state=42` is made. The test part is never used for fitting.

The tweets contain real offensive and hateful language. Cite: Davidson, Warmsley, Macy & Weber (2017),
"Automated Hate Speech Detection and the Problem of Offensive Language", ICWSM.
