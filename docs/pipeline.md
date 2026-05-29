# Mendeley URL + HTML Pipeline

The project now uses only the Mendeley metadata CSV and matching HTML files.

## 1. Input

```text
output/mendeley_metadata.csv
dataset/<html_file>
```

Expected metadata columns:

```text
rec_id,url,html_file,label,created_date
```

Labels:

```text
0 = legitimate
1 = phishing
```

## 2. Prepare Clean Dataset

```bash
python -m src.data.prepare_from_metadata ^
  --metadata output/mendeley_metadata.csv ^
  --html-root dataset ^
  --out data/processed/mendeley_url_html_label.csv.gz ^
  --min-html-length 100
```

Output:

```text
rec_id,url,html_file,html,label,created_date,html_length,source
```

Cleaning removes empty URLs, missing HTML, short HTML, duplicate URLs, and duplicate HTML content.

## 3. Split

```bash
python -m src.data.split_dataset ^
  --input data/processed/mendeley_url_html_label.csv.gz ^
  --out-dir data/splits ^
  --train-size 0.70 ^
  --val-size 0.15 ^
  --test-size 0.15
```

Splits are stratified by label.

## 4. Models

Baselines:

- Logistic Regression + TF-IDF over URL + HTML text.
- Random Forest over handcrafted URL + HTML features.

Deep learning:

- URL-CNN.
- URL-LSTM/BiLSTM.
- HTML-CNN.
- Dual-Branch CNN.

## 5. Evaluation

Results:

```text
reports/results/model_comparison.csv
reports/figures/confusion_matrix_<model>.png
reports/figures/roc_curve_<model>.png
```

Metrics include Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, False Positive Rate, and False Negative Rate.

## 6. Runtime

The web demo loads trained models from:

```text
models/saved/
```

Priority:

```text
dual_branch_cnn -> html_cnn -> url_cnn -> url_lstm -> baselines -> heuristic
```
