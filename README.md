# Deep Learning-Based Phishing Website Detection

Project phat hien website phishing bang hoc sau.

Project hien co 2 luong huan luyen rieng:

```text
1. Mendeley pipeline: URL + HTML
2. Phish360 pipeline: URL + HTML + Screenshot
```

Scope:

- Luong Mendeley chi dung URL va HTML source code.
- Luong Phish360 dung URL, HTML source code va screenshot.
- Khong phu thuoc blacklist.
- Label: `0 = legitimate`, `1 = phishing`.

## Dataset Dang Dung

### Luong 1: Mendeley URL + HTML

Dataset Mendeley da tao san:

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

### Luong 2: Phish360 URL + HTML + Screenshot

Dataset Phish360 dat tai:

```text
https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
```

Do dataset co dung luong lon, folder dataset khong duoc commit len GitHub. Sau khi tai dataset, dat vao:

```text
data/external/Phish360/
```

Moi sample co cau truc:

```text
<sample_id>/
  URL/url.txt
  RAW-HTML/index.html
  SCREEN-SHOT/screen_shoot.png
  Label/label.txt
```

Binary label duoc suy ra tu ten folder:

```text
L... -> 0 = legitimate
P... -> 1 = phishing
```

`Label/label.txt` duoc giu nhu metadata phu `target_brand`, vi nhieu phishing sample ghi brand bi gia mao thay vi chu `phish`.

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
reports/results/     ket qua tach theo dataset: mendeley/, phish360/
reports/figures/     chart tach theo dataset: mendeley/, phish360/
```

## Hai Luong Huan Luyen Rieng

### 1. Mendeley: URL + HTML

Lenh nay giu nguyen pipeline cu:

```bash
python train_model.py
```

Ket qua:

```text
reports/results/mendeley/model_comparison.csv
models/saved/
reports/figures/mendeley/
```

### 2. Phish360: URL + HTML + Screenshot

Lenh rieng cho pipeline multi-modal moi:

```bash
python train_phish360.py
```

Lenh nay tu dong:

```text
1. Tao index CSV tu data/external/Phish360
2. Train cac model Phish360
3. Ghi ket qua rieng cho Phish360
```

Chay nhanh de smoke test:

```bash
python train_phish360.py --quick
```

Chi train model fusion 3 nhanh:

```bash
python train_phish360.py --model tri_branch_cnn
```

Chi tao CSV index, chua train:

```bash
python train_phish360.py --force-prepare --skip-training
```

Output rieng:

```text
data/processed/phish360_url_html_screenshot.csv
reports/results/phish360/phish360_model_comparison.csv
models/saved/phish360/
reports/figures/phish360/
```

Phish360 models:

- `url_cnn`: URL only.
- `url_lstm`: URL only.
- `html_cnn`: HTML only.
- `screenshot_cnn`: Screenshot only.
- `dual_branch_cnn`: URL + HTML.
- `tri_branch_cnn`: URL + HTML + Screenshot.

## Setup

```bash
pip install -r requirements.txt
```

## One-Command Training Pipeline

Lenh khuyen dung:

```bash
python train_model.py
```

Lenh nay tu dong:

```text
1. Prepare dataset tu output/mendeley_metadata.csv + dataset/*.html
2. Train baseline models
3. Train deep models
4. Export model comparison, classification reports va charts
```

Chay nhanh de smoke test pipeline:

```bash
python train_model.py --quick
```

`--quick` tuong duong voi:

```text
--max-rows 5000 --epochs 3 --cpu
```

Mot so tuy chon hay dung:

```bash
python train_model.py --skip-deep
python train_model.py --deep-model dual_branch_cnn
python train_model.py --force-prepare
python train_model.py --skip-evaluation
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
reports/results/mendeley/model_comparison.csv
reports/results/mendeley/model_comparison_sorted.csv
reports/results/mendeley/classification_reports.csv
reports/results/mendeley/classification_report_<model>.txt
reports/figures/mendeley/confusion_matrix_<model>.png
reports/figures/mendeley/roc_curve_<model>.png
reports/figures/mendeley/model_comparison_metrics.png
```

Xem bang so sanh:

```bash
python -m src.evaluation.compare_models --results-dir reports/results/mendeley --figures-dir reports/figures/mendeley
```

Metrics gom Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, False Positive Rate, False Negative Rate va sklearn-style classification report theo tung class.

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

Runtime load cac model trong `models/saved/` va chon model kha dung co F1-score tot nhat tu:

```text
reports/results/mendeley/model_comparison.csv
```

Neu khong co metrics/model trained, demo fallback ve heuristic URL + HTML rules.

Moi lan predict duoc ghi vao:

```text
reports/results/mendeley/prediction_history.csv
```

## External Test Dataset (Khong Dung De Train)

Pipeline `collect_external_dataset.py` tao tap external test rieng gom URL sau redirect,
raw HTML va screenshot trong cung mot lan truy cap Playwright. Tap nay chi duoc dung
de danh gia external; khong dung cho train, validation, chon model hoac chinh threshold.

Setup Chromium mot lan:

```bash
pip install -r requirements.txt
playwright install chromium
```

Neu Chromium download khong kha dung nhung may da cai Chrome hoac Edge, them
`--browser-channel chrome` hoac `--browser-channel msedge`.

Vi du voi CSV co cot `url,label` va `0 = legitimate`, `1 = phishing`:

```bash
python collect_external_dataset.py ^
  --input urls.csv ^
  --output data/external_test/dataset_50 ^
  --legitimate-count 25 ^
  --phishing-count 25 ^
  --source kaggle ^
  --headless ^
  --zip
```

Dataset Kaggle `phishing_site_urls.csv` dung cot `URL,Label` va nhan `good,bad`:

```bash
python collect_external_dataset.py ^
  --input "C:\path\phishing_site_urls.csv" ^
  --url-column URL ^
  --label-column Label ^
  --source kaggle ^
  --headless ^
  --zip
```

Tranco `top-1m.csv` khong co header va co schema `rank,domain`. Tat ca domain
Tranco chi duoc gan nhan legitimate:

```bash
python collect_external_dataset.py ^
  --input "C:\path\top-1m.csv" ^
  --input-format tranco ^
  --output data/external_test/tranco_legitimate ^
  --legitimate-count 25 ^
  --phishing-count 0 ^
  --max-input-rows 50000 ^
  --source tranco ^
  --browser-channel chrome ^
  --headless
```

Tranco khong cung cap phishing. De tao tap external 25/25, can thu thap 25
phishing tu mot nguon phishing cap nhat vao output rieng hoac mot buoc merge
co kiem tra leakage; khong duoc gan nhan phishing cho domain Tranco.

Pipeline mac dinh:

- Thu thap den khi du 25 legitimate va 25 phishing; URL loi se bi ghi vao
  `rejected_samples.csv` va pipeline tiep tuc.
- Ho tro `--timeout-ms`, `--retries`, `--random-state`, `--resume` va tuy chinh
  mapping label qua `--legitimate-values`, `--phishing-values`.
- Loai URL, SHA-256 HTML va screenshot dHash trung/gan trung voi Mendeley,
  Phish360 va cac sample external da chap nhan.
- Loai domain phishing da co trong Phish360 `trainval` theo mac dinh.
- Chan popup, download, media va browser permissions khong can thiet; khong co
  buoc dang nhap.
- Browser context la tam thoi. Service worker duoc cho phep mac dinh de render
  dung cac trang IPFS; co the chan bang `--service-workers block`.
- Loai cac trang loi browser/gateway, 404, access denied va domain parking
  pho bien thay vi nhan chung thanh sample hop le.
- Ghi `collection_results.csv`, `rejected_samples.csv`, `dataset_summary.json`
  va chi tao `dataset_50.zip` khi du quota.

Smoke test gioi han, khong crawl du 50:

```bash
python collect_external_dataset.py ^
  --input urls.csv ^
  --legitimate-count 1 ^
  --phishing-count 1 ^
  --max-candidates 4 ^
  --output data/external_test/smoke
```

## External Evaluation

Dataset external chi dung inference, khong train lai, khong chon model va khong
chinh threshold:

```bash
python evaluate_external_dataset.py ^
  --dataset data/external_test/dataset_100_clean
```

Chi danh gia model dung dong thoi URL + HTML + Screenshot:

```bash
python evaluate_external_dataset.py ^
  --dataset data/external_test/dataset_100_clean ^
  --models phish360_tri_branch_cnn
```

Script danh gia tat ca model Mendeley va Phish360 da luu, dung threshold trong
checkpoint (`0.5` cho baseline joblib), va ghi ket qua rieng:

```text
reports/results/external_test/
  predictions.csv
  misclassifications.csv
  model_comparison.csv
  classification_reports.csv
  classification_report_<model>.txt
  evaluation_summary.json

reports/figures/external_test/
  confusion_matrix_<model>.png
  roc_curve_<model>.png
  external_model_comparison_metrics.png
```
