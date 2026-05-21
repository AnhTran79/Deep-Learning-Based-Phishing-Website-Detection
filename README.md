# Phishing Website Detection API

Project demo phat hien phishing website tu dataset raw URL dang `url,label`.

## Thanh phan

- `app.py`: entrypoint chay FastAPI server.
- `app/main.py`: API endpoints.
- `app/features.py`: trich xuat URL lexical features va HTML features.
- `app/model.py`: runtime detector, load CNN URL model va MLP URL/HTML model.
- `train_model.py`: train lai model tu `../balanced_dataset.csv`.
- `artifacts/`: model artifact, metrics va report.
- `dataset/`: train/test split va HTML cache neu bat fetch.

## Dataset moi

Dataset mac dinh:

```text
../balanced_dataset.csv
```

Schema:

```text
url,label
example.com,0
bad.example/login,1
```

Label:

- `0`: legitimate
- `1`: phishing

Project chi dung dataset raw URL nay lam nguon train chinh.

## Cai dat

```bash
pip install -r requirements.txt
```

## Train model tu balanced_dataset.csv

Train nhanh tren URL text va URL/HTML feature vector khong fetch HTML live:

```bash
python train_model.py
```

Mac dinh training hien tai:

- `--epochs 10`
- `--threshold 0.4`

Lenh tren train:

- `url_cnn_deep_learning`: CNN character-level hoc truc tiep chuoi URL.
- `deep_learning_url_html_mlp`: MLP hoc URL lexical features va HTML features.
- `baseline_logistic_regression`: baseline tuyen tinh de so sanh.
- `baseline_gaussian_naive_bayes`: baseline xac suat don gian de so sanh.

Output chinh:

- `artifacts/url_cnn.pt`
- `artifacts/deep_learning_model.joblib`
- `artifacts/best_model.json`
- `artifacts/classification_report.txt`
- `artifacts/model_results.csv`
- `artifacts/deep_learning_metrics.json`
- `artifacts/metadata.json`
- `dataset/train.csv`
- `dataset/test.csv`

## Train co doc HTML

Doc HTML live rat cham voi dataset 485k URL va nhieu link co the chet. Nen train theo tung dot va dung cache:

```bash
python train_model.py --fetch-html --html-cache-dir dataset/html_cache --html-max-rows 5000
```

Sau khi cache da co, co the tang `--html-max-rows` hoac bo gioi han:

```bash
python train_model.py --fetch-html --html-cache-dir dataset/html_cache --html-max-rows 0
```

Neu can test pipeline truoc:

```bash
python train_model.py --max-rows 20000 --epochs 2 --html-max-rows 1000 --fetch-html
```

## Chay API

```bash
python app.py
```

Server mac dinh:

```text
http://127.0.0.1:8000
```

Runtime mac dinh se thu fetch HTML cua URL can kiem tra. Tat fetch HTML runtime bang:

```bash
set FETCH_HTML_AT_RUNTIME=0
python app.py
```

## API

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Predict:

```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"url\":\"https://example.com/login\"}"
```

Response gom:

- `label`: `phishing` hoac `legitimate`
- `confidence`
- `phishing_probability`
- `model_source`
- `features`: URL + HTML features
- `component_probabilities`: CNN, MLP, heuristic probabilities
- `html_fetch`: trang HTML co fetch duoc hay khong
