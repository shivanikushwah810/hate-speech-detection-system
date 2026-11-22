Original production model, preserved untouched.
Saved with scikit-learn 1.8.0 (the project now pins scikit-learn 1.9.1).
Trained on ~600 documents (vocabulary 2,976 terms); its training data is not part of this project.
It still loads under 1.9.1 (with an InconsistentVersionWarning). evaluate_model.py evaluates it
alongside the retrained model so both can be compared on the same test split.
To roll back:  python promote_model.py --restore models/legacy_sklearn_1.8.0
