# Giai thich `train_model.py`

File `train_model.py` la script train model phishing detection hien tai cua project. Pipeline moi dung dataset raw URL `balanced_dataset.csv`, khong train tu bang feature co san.

## 1. Input dataset

Dataset mac dinh:

```text
../balanced_dataset.csv
```

Schema:

```text
url,label
```

Label:

| Label | Y nghia |
| --- | --- |
| `0` | legitimate |
| `1` | phishing |

Ham lien quan:

- `load_url_rows(data_path, max_rows, random_state)`: doc CSV, validate cot `url,label`, validate label `0/1`, optional sample bang `--max-rows`.
- `split_rows(rows, test_size, random_state)`: chia train/test theo stratified split.
- `write_dataset_files(...)`: ghi lai `dataset/train.csv`, `dataset/test.csv`, `dataset_summary.json`, `configs.json`.

## 2. Cac model duoc train

Project hien tai train 4 model:

| Nhom | Model | Vai tro |
| --- | --- | --- |
| Baseline | `baseline_logistic_regression` | Moc so sanh tuyen tinh tren URL/HTML feature vector. |
| Baseline | `baseline_gaussian_naive_bayes` | Moc so sanh xac suat don gian. |
| Deep learning | `url_cnn_deep_learning` | CNN character-level hoc truc tiep chuoi URL. |
| Deep learning | `deep_learning_url_html_mlp` | MLP hoc URL lexical features va HTML features neu co. |

Runtime chinh uu tien 2 model deep learning:

- `url_cnn_deep_learning`
- `deep_learning_url_html_mlp`

Hai baseline chi dung de so sanh va bao cao.

## 3. URL CNN deep learning

Day la model quan trong nhat hien tai.

Luong xu ly:

1. `UrlDataset` nhan tung dong `(url, label)`.
2. `encode_url(url)` trong `app/model.py` bien URL thanh vector so theo ky tu.
3. `CharCnnUrlClassifier` hoc pattern truc tiep tu chuoi URL.
4. `train_url_cnn_model(...)` train CNN bang PyTorch.
5. `evaluate_url_cnn(...)` tinh metric tren test set.

Model CNN gom:

- `Embedding`
- `Conv1d`
- `BatchNorm1d`
- `ReLU`
- `AdaptiveMaxPool1d`
- `Linear`
- `Dropout`

Output model:

```text
artifacts/url_cnn.pt
```

File nay chua trong so da hoc cua CNN.

## 4. URL/HTML feature MLP

Model nay hoc tu feature vector.

Ham lien quan:

- `build_feature_matrix(...)`: tao matrix `X, y` tu cac URL.
- `combined_feature_dict(url, html)` trong `app/features.py`: tao dict feature URL + HTML.
- `train_tabular_models(...)`: train baseline va MLP tren feature vector.

URL features gom cac nhom:

- do dai URL/domain/path/query
- so dau cham, gach ngang, chu so, ky tu dac biet
- HTTPS, IP address, ky tu `@`
- subdomain, risky TLD, shortener
- redirect parameter, embedded URL, long token
- public hosting, brand signal, suspicious words
- entropy

HTML features gom:

- `html_available`
- `html_length`
- `title_length`
- `num_forms`
- `num_password_inputs`
- `num_iframes`
- `num_scripts`
- `num_external_links`
- `has_login_form`
- `has_meta_refresh`
- `has_javascript_redirect`
- `suspicious_html_word_count`

Luu y: lan train hien tai chua bat `--fetch-html`, nen HTML features deu bang `0`. Model da co pipeline HTML, nhung chua hoc HTML that.

Output model:

```text
artifacts/deep_learning_model.joblib
```

File nay chua MLP URL/HTML features duoc chon cho runtime.

## 5. Baseline models

Trong `train_tabular_models(...)`, project train them 2 baseline:

| Baseline | Ly do ton tai |
| --- | --- |
| `baseline_logistic_regression` | Kiem tra model tuyen tinh don gian dat duoc den dau. |
| `baseline_gaussian_naive_bayes` | Kiem tra model xac suat don gian, train nhanh. |

Baseline khong phai runtime chinh. Chung dung de tra loi cau hoi: deep learning co that su tot hon model machine learning don gian khong?

Ket qua hien tai cho thay CNN deep learning tot hon ro ret tren F1, Recall va ROC-AUC.

## 6. Metrics va evaluation

Ham lien quan:

- `evaluate_probabilities(labels, probabilities, threshold)`: tinh metric tu probability.
- `evaluate_url_cnn(model, rows, batch_size)`: evaluate CNN.
- `write_model_results(path, cnn_metrics, tabular_metrics)`: ghi bang so sanh model.
- `write_charts(chart_dir, model_results_path)`: tao chart so sanh.

Metric duoc tinh:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Trong phishing detection, `Recall` va `F1` cua class phishing rat quan trong vi bo sot phishing co rui ro cao.

## 7. Output artifacts

Sau khi train, script ghi:

| File | Y nghia |
| --- | --- |
| `artifacts/url_cnn.pt` | CNN hoc truc tiep chuoi URL. |
| `artifacts/deep_learning_model.joblib` | MLP hoc URL/HTML feature vector. |
| `artifacts/best_model.json` | Metadata artifact runtime. |
| `artifacts/model_results.csv` | Bang so sanh 4 model. |
| `artifacts/deep_learning_metrics.json` | Metrics chi tiet. |
| `artifacts/classification_report.txt` | Report precision/recall/F1. |
| `artifacts/metadata.json` | Metadata training. |
| `chart/model_comparison.png` | Chart so sanh Accuracy, Precision, Recall, F1, ROC-AUC. |
| `chart/model_comparison_f1.png` | Chart so sanh rieng F1. |
| `dataset/train.csv` | Train split. |
| `dataset/test.csv` | Test split. |
| `dataset/dataset_summary.json` | Thong ke split. |
| `dataset/configs.json` | Cau hinh train. |

## 8. CLI arguments

`parse_args()` ho tro:

| Argument | Y nghia |
| --- | --- |
| `--data` | Duong dan dataset `url,label`. Mac dinh la `../balanced_dataset.csv`. |
| `--out-dir` | Thu muc artifact. Mac dinh `artifacts`. |
| `--dataset-dir` | Thu muc split dataset. Mac dinh `dataset`. |
| `--chart-dir` | Thu muc chart. Mac dinh `chart`. |
| `--max-rows` | Sample nhanh mot phan dataset de test pipeline. |
| `--test-size` | Ti le test. |
| `--random-state` | Seed chia dataset. |
| `--epochs` | So epoch train CNN. Mac dinh `10`. |
| `--batch-size` | Batch size cho CNN. |
| `--learning-rate` | Learning rate CNN. |
| `--max-iter` | So iteration toi da cho sklearn models. |
| `--threshold` | Nguong probability de tinh phishing. Mac dinh `0.4`. |
| `--fetch-html` | Bat tai HTML live de train HTML features. |
| `--html-cache-dir` | Thu muc cache HTML. |
| `--fetch-timeout` | Timeout khi tai HTML. |
| `--html-max-rows` | Gioi han so row moi split duoc fetch HTML. |

## 9. Cach train

Train binh thuong, khong fetch HTML:

```bash
python train_model.py
```

Train nhanh de test:

```bash
python train_model.py --max-rows 20000 --epochs 2 --max-iter 50
```

Train co fetch HTML theo gioi han:

```bash
python train_model.py --fetch-html --html-cache-dir dataset/html_cache --html-max-rows 5000
```

## 10. Runtime lien quan

Sau khi train, `app/model.py` load:

- `artifacts/url_cnn.pt`
- `artifacts/deep_learning_model.joblib`

Khi API predict:

1. Nhan URL.
2. Trich xuat URL features.
3. Thu fetch HTML neu `FETCH_HTML_AT_RUNTIME` khac `0`.
4. Chay CNN tren chuoi URL.
5. Chay MLP tren URL/HTML feature vector.
6. Ket hop probability:

```text
0.65 * CNN probability + 0.35 * MLP probability
```

7. Tra ve `phishing` neu probability vuot threshold.

## 11. Training hien tai

Lan train hien tai dung:

```text
epochs = 10
threshold = 0.4
```

CNN luu model cua epoch co F1 tot nhat:

```text
best_epoch = 9
```

Thong tin nay nam trong:

- `artifacts/deep_learning_metrics.json`
- `artifacts/best_model.json`
- `artifacts/url_cnn.pt`
