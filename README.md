# Deep Learning-Based Phishing Website Detection

Project phat hien website phishing bang hoc sau tu **Mendeley URL + HTML dataset**.

Scope:

- Chi dung URL va HTML source code.
- Khong dung screenshot, image branch, ResNet/EfficientNet.
- Khong phu thuoc blacklist.
- Label: `0 = legitimate`, `1 = phishing`.

## Dataset Dang Dung

Dataset chinh la metadata Mendeley da tao san:

```text
output/mendeley_metadata.csv
```

File nay co cac cot:

```text
rec_id,url,html_file,label,created_date
```

HTML source duoc doc theo cot `html_file` tu thu muc HTML, mac dinh:

```text
dataset/
```

Neu thu muc HTML nam o cho khac, truyen lai bang `--html-root`.

## Project Layout

```text
app/                 FastAPI web demo
output/              mendeley_metadata.csv
data/processed/      dataset da clean
data/splits/         train.csv, val.csv, test.csv
src/data/            load metadata, read HTML, clean, split
src/preprocessing/   URL/HTML preprocessing va char tokenizer
src/features/        handcrafted URL + HTML features
src/models/          baseline va deep model definitions
src/training/        train baselines va deep models
src/evaluation/      metrics, figures, model comparison
src/inference/       predict mot URL
models/saved/        model da train
reports/results/     model_comparison.csv, prediction_history.csv
reports/figures/     confusion matrix va ROC curve
```

## Setup

```bash
pip install -r requirements.txt
```

## Prepare Dataset Tu Metadata CSV

Neu HTML files nam trong `dataset/`:

```bash
python -m src.data.prepare_from_metadata ^
  --metadata output/mendeley_metadata.csv ^
  --html-root dataset ^
  --out data/processed/mendeley_url_html_label.csv.gz ^
  --min-html-length 100
```

Neu muon chi tao ban metadata sach truoc:

```bash
python -m src.data.prepare_from_metadata ^
  --metadata output/mendeley_metadata.csv ^
  --metadata-only
```

Lenh nay tao:

```text
data/processed/mendeley_clean_metadata.csv
```

Ban metadata sach van co the dung cho training bang cach truyen them `--html-root dataset`; HTML se duoc doc lazy theo `html_file`.

Output full:

```text
data/processed/mendeley_url_html_label.csv.gz
```

Schema:

```text
rec_id,url,html_file,html,label,created_date,html_length,source
```

## Split Train/Validation/Test

```bash
python -m src.data.split_dataset ^
  --input data/processed/mendeley_url_html_label.csv.gz ^
  --out-dir data/splits ^
  --train-size 0.70 ^
  --val-size 0.15 ^
  --test-size 0.15
```

Neu dang dung ban metadata sach:

```bash
python -m src.data.split_dataset ^
  --input data/processed/mendeley_clean_metadata.csv ^
  --html-root dataset ^
  --out-dir data/splits
```

Output:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

## Train Baselines

```bash
python -m src.training.train_baselines ^
  --data data/processed/mendeley_url_html_label.csv.gz
```

Models:

- `baseline_tfidf_logreg`: URL + HTML text -> TF-IDF -> Logistic Regression.
- `baseline_random_forest`: URL + HTML -> handcrafted features -> Random Forest.

## Train Deep Models

Train mot model:

```bash
python -m src.training.train_deep_models ^
  --data data/processed/mendeley_url_html_label.csv.gz ^
  --model dual_branch_cnn
```

Train tat ca:

```bash
python -m src.training.train_deep_models ^
  --data data/processed/mendeley_url_html_label.csv.gz ^
  --model all
```

Deep models:

- `url_cnn`: URL -> Embedding -> CNN1D -> Dense -> Sigmoid.
- `url_lstm`: URL -> Embedding -> BiLSTM -> Dense -> Sigmoid.
- `html_cnn`: HTML -> Embedding -> CNN1D -> Dense -> Sigmoid.
- `dual_branch_cnn`: URL branch + HTML branch -> Concatenate fusion -> Dense -> Sigmoid.

## Evaluation

Tat ca model ghi metrics vao:

```text
reports/results/model_comparison.csv
reports/figures/confusion_matrix_<model>.png
reports/figures/roc_curve_<model>.png
```

Xem bang so sanh:

```bash
python -m src.evaluation.compare_models --results-dir reports/results
```

Metrics gom Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, False Positive Rate, False Negative Rate.

## Predict One URL

```bash
python -m src.inference.predict ^
  --url "https://example.com/login" ^
  --model models/saved/dual_branch_cnn.pt
```

## Run Web Demo

```bash
python app.py
```

Server:

```text
http://127.0.0.1:8000
```

Runtime uu tien model train tu Mendeley trong `models/saved/` theo thu tu:

```text
dual_branch_cnn -> html_cnn -> url_cnn -> url_lstm -> baselines -> heuristic
```

Moi lan predict duoc ghi vao:

```text
reports/results/prediction_history.csv
```
