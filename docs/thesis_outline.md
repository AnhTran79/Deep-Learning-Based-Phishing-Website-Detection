# Thesis Outline

Suggested title: **Deep Learning-Based Phishing Website Detection using URL and HTML Source Code**.

## 1. Introduction

- Phishing websites imitate legitimate services and steal credentials.
- URL-only detection can miss pages with normal-looking URLs.
- HTML source code exposes forms, password inputs, scripts, iframes, and suspicious actions.

## 2. Dataset

- Dataset: Mendeley phishing website dataset.
- Current project input starts from `output/mendeley_metadata.csv`.
- Required columns: `rec_id`, `url`, `html_file`, `label`, `created_date`.
- HTML files are loaded from the configured `dataset/` folder by `html_file`.
- Label convention: `0 = legitimate`, `1 = phishing`.

## 3. Data Processing

- Load metadata CSV.
- Normalize column names and labels.
- Read HTML source files.
- Remove invalid rows: empty URL, missing HTML file, empty/short HTML.
- Remove duplicate URL and duplicate HTML content.
- Save clean dataset to `data/processed/mendeley_url_html_label.csv.gz`.
- Split train/validation/test with stratified labels.

## 4. Baselines

- Logistic Regression + TF-IDF over URL + HTML text.
- Random Forest over handcrafted URL + HTML features.

## 5. Deep Learning Models

- URL-CNN for local URL patterns.
- URL-LSTM/BiLSTM for sequential URL context.
- HTML-CNN for source-code patterns.
- Dual-Branch CNN for URL + HTML fusion.

## 6. Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- False Positive Rate
- False Negative Rate

Recall is emphasized because missed phishing pages are high risk.

## 7. Web Demo

- User enters a URL.
- Backend fetches HTML.
- Preprocessing matches training.
- Trained Mendeley model predicts phishing probability.
- Result is logged to `reports/results/prediction_history.csv`.

## 8. Limitations

- Runtime HTML fetching may fail or be blocked.
- The dataset can age over time.
- Character-level CNN/LSTM models are simpler than transformer-based HTML models.

## 9. Future Work

- Add independent Phish360 evaluation using URL + HTML only.
- Calibrate thresholds for phishing recall.
- Add DOM-aware features.
- Compare against transformer encoders.
