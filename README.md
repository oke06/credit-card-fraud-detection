# Credit Card Fraud Detection

Detecting fraudulent credit card transactions on a heavily imbalanced dataset (0.17% fraud), with a deployed FastAPI endpoint that serves live predictions.

The focus of this project is **methodology, not leaderboard accuracy**: handling extreme class imbalance correctly, choosing metrics that actually reflect catching fraud, and shipping the trained model as a callable service rather than leaving it in a notebook.

---

## The problem

The dataset contains 284,807 transactions, of which only **492 are fraud (0.17%)**. That imbalance is the entire challenge:

- A model that predicts "not fraud" for every transaction scores **99.83% accuracy** while catching zero fraud. So accuracy is meaningless here — it rewards ignoring the rare class that actually matters.
- The model barely sees fraud during training (~1 fraud per 580 legitimate transactions), so the lazy path is to always predict "legitimate".
- The two mistakes are not equally costly: a missed fraud is realised financial loss, while a false alarm is a recoverable inconvenience. A metric that treats all errors the same is blind to this.

So the real question is: **can I catch a meaningful share of fraud without drowning in false alarms?** — measured by precision and recall, not accuracy.

The features are `Time`, `Amount`, and `V1`–`V28` (28 anonymised PCA components — the original features were transformed to protect customer privacy, so the work has to succeed on method rather than domain intuition).

---

## Key results

Final model: **Random Forest, decision threshold 0.3**

| Model | Recall (fraud caught) | Precision (flags that are correct) | Notes |
|---|---|---|---|
| v1 — Logistic Regression (naive) | 0.72 | 0.87 | 99.9% "accuracy" — misleading |
| v2 — Logistic Regression (tuned) | 0.89 | 0.39 | class weights + scaling + threshold |
| **Final — Random Forest @ 0.3** | **0.85** | **0.83** | best precision/recall trade-off |

Model selection was done on **PR-AUC** (area under the precision-recall curve), a threshold-independent metric that stays honest under heavy imbalance: Random Forest scored **0.86** vs Logistic Regression's **0.72**.

---

## Approach

**v1 — a deliberate baseline.** A plain logistic regression with no imbalance handling. It scored 99.9% accuracy but caught only 72% of fraud, missing 27 of 98 cases. The high accuracy is an artifact of the imbalance, not a sign the model works.

**Diagnosis — the metric pivot.** Because accuracy rewards ignoring fraud, I dropped it in favour of:
- **Recall** — of all real frauds, how many did I catch?
- **Precision** — of all transactions I flagged, how many were actually fraud?
- **PR-AUC** — a single threshold-independent score for comparing models.

**v2 — handling the imbalance.** Three changes, introduced one at a time so each effect was attributable:
- `class_weight='balanced'` — makes missed fraud expensive during training. Recall jumped to 92%, but precision collapsed (a wider net catches more fraud *and* more false alarms — the trade-off made concrete).
- **Feature scaling** with `StandardScaler`, fit on the training set only (the scaler's statistics must not be learned from test data, which would leak information into evaluation). This also fixed a convergence problem caused by `Amount` and the `V` features living on wildly different scales.
- **Threshold tuning** — logistic regression outputs a probability; the default 0.5 cutoff is arbitrary. Sweeping the threshold walks the precision/recall trade-off without retraining, letting me pick an operating point deliberately rather than accepting the default.

**Model comparison.** I compared logistic regression against a random forest on PR-AUC (so the comparison doesn't depend on a cherry-picked threshold). The forest won clearly (0.86 vs 0.72), which fits the problem: fraud is non-linear — it lives in pockets and feature combinations that a single linear boundary can't capture, but a forest of branching rules can.

**Final operating point.** The random forest at threshold 0.3 gives **85% recall at 83% precision**. The choice leans toward recall — in fraud, a missed case is more costly than a false alarm — while keeping false positives low enough that the flags stay actionable. At 0.3 the model catches nearly as much fraud as the tuned logistic regression but with roughly 8× fewer false alarms.

---

## Deployment

The trained model is served through a **FastAPI** endpoint (`app.py`), which is the part that takes the model out of the notebook and makes it a real service.

On startup it loads the saved model and scaler once. The `/predict` endpoint accepts a transaction as JSON, scales it with the *same* scaler used in training, scores it with the model, applies the 0.3 threshold, and returns a decision:

```json
Request  → {"Time": 406.0, "V1": -2.31, ..., "Amount": 0.0}
Response → {"fraud": true, "probability": 0.97}
```

FastAPI auto-generates an interactive test page at `/docs`, so predictions can be tried in a browser without writing any client code.

---

## Running it yourself

```bash
# 1. clone and install
git clone <YOUR_REPO_URL>
cd fraud-detection
pip install -r requirements.txt

# 2. get the data (not committed — see "Note on data" below)
#    download creditcard.csv from:
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    and place it in: data/creditcard.csv

# 3. (optional) retrain — the trained model is already saved in models/,
#    so you can skip straight to step 4. To retrain, run the notebook:
#    notebooks/fraud_detection.ipynb

# 4. launch the API
uvicorn app:app --reload
#    then open http://127.0.0.1:8000/docs
```

---

## Project structure

```
fraud-detection/
├── app.py                        # FastAPI prediction endpoint
├── models/
│   ├── fraud_model.joblib        # trained random forest
│   └── scaler.joblib             # fitted StandardScaler
├── notebooks/
│   └── fraud_detection.ipynb     # EDA, modelling, comparison, evaluation
├── data/                         # creditcard.csv (not committed)
├── requirements.txt
└── README.md
```

---

## Limitations and next steps

- **Precision is modest.** 83% precision means ~1 in 6 flags is a false alarm — much better than v2's logistic regression, but not "solved". The honest framing is that the model went from unusable to defensible.
- **Limited interpretability.** The PCA-anonymised features mean the model can't offer human-readable reasons for a flag, which matters for a real fraud system where decisions often need to be explained.
- **Single train/test split.** Results would be more robust with proper cross-validation.
- **Things I'd try next:** SMOTE or other resampling to compare against class weighting, gradient boosting (XGBoost/LightGBM) as a stronger model, and calibrating the probability outputs so the threshold has a clearer meaning.

---

## Note on data

The dataset is not included in this repository — it's ~150MB, it's not mine to redistribute, and raw input data doesn't belong in a code repo. Download it from the [Kaggle source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

*Dataset: Credit Card Fraud Detection, Machine Learning Group — Université Libre de Bruxelles (ULB).*
