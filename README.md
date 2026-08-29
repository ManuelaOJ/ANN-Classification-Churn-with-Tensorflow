# Customer Churn Prediction — Artificial Neural Network

A binary classifier that estimates the probability of a retail-bank customer leaving,
served through an interactive Streamlit app. Built with TensorFlow/Keras.

## Problem

Retaining an existing customer is far cheaper than acquiring a new one, but churn signals
are spread across demographics, product holdings and account activity. This model scores a
single customer from 10 attributes and returns a churn probability a retention team can act on.

## Data

`Churn_Modelling.csv` — 10,000 anonymised bank customers with a binary `Exited` label.

Features used: credit score, geography, gender, age, tenure, balance, number of products,
credit-card flag, active-member flag, estimated salary.

## Approach

**Preprocessing** — three artefacts are fitted on the training set and persisted so the app
applies exactly the same transformation at inference time:

| Artefact | Purpose |
|---|---|
| `label_encoder_gender.pkl` | Label-encodes `Gender` |
| `onehot_encoder_geo.pkl` | One-hot encodes `Geography` (France / Spain / Germany) |
| `scaler.pkl` | `StandardScaler` over the assembled feature matrix |

**Model** (`model.h5`) — a feed-forward ANN: dense hidden layers with ReLU activations and a
single sigmoid output. Trained with binary cross-entropy and the Adam optimiser, with
TensorBoard logging and early stopping on validation loss.

Training and evaluation live in `experiments.ipynb`; `prediction.ipynb` walks through
scoring a single record end to end.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app exposes each feature as a widget, rebuilds the feature vector in the exact order the
scaler expects, and prints the churn probability plus the resulting decision.

## Stack

TensorFlow / Keras · scikit-learn · pandas · NumPy · Streamlit · TensorBoard

## Notes

Built as hands-on practice while following Krish Naik's deep-learning course. The dataset and
the problem framing come from the course; the code here is my own implementation.
