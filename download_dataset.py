"""
download_dataset.py
===================
Downloads the REAL public dataset used for Model Evaluation and Model Comparison:

    Davidson, Warmsley, Macy, Weber (2017). "Automated Hate Speech Detection and the
    Problem of Offensive Language". ICWSM.   24,783 tweets, 3 classes:
        0 = Hate Speech, 1 = Offensive Language, 2 = Neither (Normal)
    Source: https://github.com/t-davidson/hate-speech-and-offensive-language

Text column = "tweet", label column = "class". The numbering matches the app
(0 = Hate Speech, 1 = Offensive, 2 = Normal).

Usage (inside the project folder):
    python download_dataset.py            # download (asks before overwriting)
    python download_dataset.py --force    # download, overwrite without asking
    python download_dataset.py --check    # only validate the existing data/dataset.csv

The file is downloaded to a temporary name and VALIDATED first; an existing, good
data/dataset.csv is never replaced by a broken download.

If the download fails (no internet / firewall), do it by hand:
    1. Open  https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv
    2. Click "Download raw file" (or open the raw link and Save As)
    3. Save it as  data/dataset.csv  inside this project, then run:  python download_dataset.py --check

NOTE: the tweets contain real offensive and hateful language. Cite the paper in reports.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

URLS = [
    "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
    "https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv",
]
EXPECTED_ROWS = 24783          # size of the original Davidson file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE_DIR, 'data', 'dataset.csv')
sys.path.insert(0, BASE_DIR)

MANUAL = (
    "\nManual download:\n"
    "  1. Open https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv\n"
    "  2. Click 'Download raw file' and save it as  data/dataset.csv  in this project.\n"
    "  3. Run  python download_dataset.py --check  to validate it."
)


def validate(path):
    """Validate with the same loader the app uses. Returns (ok, summary_or_error)."""
    from utils.ml_common import DataError, prepare_data
    try:
        data = prepare_data(path)
    except DataError as e:
        return False, str(e)
    return True, data['info']


def print_summary(info):
    print(f"  file                : {info['file']} (SHA-256 {info['sha256_prefix']})")
    print(f"  text / label column : '{info['text_column']}' / '{info['label_column']}'")
    print(f"  rows read           : {info['rows_read']}  (empty text: {info['rows_skipped_empty_text']}, "
          f"missing label: {info['rows_skipped_missing_label']})")
    print(f"  raw class counts    : {info['class_distribution_raw']}")
    print(f"  duplicates removed  : {info['duplicates_or_empty_removed']} "
          f"(of which with conflicting labels: {info['duplicate_label_conflicts']})")
    print(f"  usable unique rows  : {info['rows_used_after_dedup']}")
    print(f"  stratified split    : train {info['n_train']} / test {info['n_test']} (seed {info['random_state']})")
    if info['rows_read'] != EXPECTED_ROWS:
        print(f"  [NOTE] The original Davidson file has {EXPECTED_ROWS} rows; this one has "
              f"{info['rows_read']}. Fine if you use your own data; suspicious otherwise.")


def download():
    last = None
    for url in URLS:
        for attempt in (1, 2, 3):
            try:
                print(f"Downloading (try {attempt}): {url}")
                req = urllib.request.Request(url, headers={'User-Agent': 'HateShield-AI-dataset-downloader'})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                print(f"  failed: {e}")
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"All download attempts failed (last error: {last})")


def main():
    ap = argparse.ArgumentParser(description='Download / validate the Davidson hate-speech dataset.')
    ap.add_argument('--force', action='store_true', help='overwrite data/dataset.csv without asking')
    ap.add_argument('--check', action='store_true', help='only validate the existing data/dataset.csv')
    args = ap.parse_args()

    if args.check:
        if not os.path.exists(DEST):
            print('[ERROR] data/dataset.csv not found.' + MANUAL)
            return 1
        ok, res = validate(DEST)
        if not ok:
            print(f'[ERROR] data/dataset.csv is not usable: {res}')
            return 1
        print('[OK] data/dataset.csv is valid:')
        print_summary(res)
        return 0

    if os.path.exists(DEST) and not args.force:
        if not sys.stdin.isatty():
            print('data/dataset.csv already exists; use --force to overwrite. Nothing changed.')
            return 0
        if input('data/dataset.csv already exists. Overwrite? [y/N]: ').strip().lower() != 'y':
            print('Cancelled. Existing dataset kept.')
            return 0

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    try:
        payload = download()
    except RuntimeError as e:
        print(f'\n[ERROR] {e}\nCheck your internet connection / proxy.' + MANUAL)
        return 1

    tmp = DEST + '.download'
    with open(tmp, 'wb') as f:
        f.write(payload)
    ok, res = validate(tmp)
    if not ok:
        os.remove(tmp)
        print(f'\n[ERROR] The downloaded file is not a usable dataset: {res}\n'
              'The existing data/dataset.csv (if any) was NOT changed.' + MANUAL)
        return 1
    os.replace(tmp, DEST)

    print(f'\n[OK] Saved to data/dataset.csv ({len(payload) / 1e6:.2f} MB)')
    # re-run the summary on the final path so the fingerprint shown is the saved file's
    print_summary(validate(DEST)[1])
    print('\nNext steps:  python retrain_production.py   |   python train_and_compare.py   |   python evaluate_model.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
