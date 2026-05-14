# Giai thich file `train_model.py`

File `train_model.py` la script dung de train model phishing website detection va ghi cac artifact ra thu muc `artifacts/`, `dataset/`, `chart/`.

Script co 2 che do chinh:

| Che do | Cach chay | Muc dich |
| --- | --- | --- |
| `kaggle` | `python train_model.py` | Train tren dataset Kaggle da co san cac feature URL. Day la che do mac dinh cua project. |
| `url-cnn` | `python train_model.py --task url-cnn --data urls.csv` | Train CNN character-level tren dataset co cot `url,label`. Day la che do phu, chi dung khi co URL goc. |

## 1. Cac hang so va kieu du lieu

### `FeatureRow`

```python
FeatureRow = tuple[str, dict[str, int]]
```

Dai dien cho 1 dong du lieu feature:

- phan 1: label dang chuoi, vi du `"-1"` la phishing, `"1"` la legitimate.
- phan 2: dictionary gom ten feature va gia tri integer.

Vi du:

```python
("-1", {"URLURL_Length": -1, "SSLfinal_State": 1})
```

### `Metrics`

```python
Metrics = dict[str, float]
```

Dung cho cac metric nhu `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.

### `ModelCandidate`

```python
ModelCandidate = dict[str, str | Pipeline]
```

Dung de mo ta mot model trong danh sach so sanh:

- `name`: ten model.
- `family`: baseline hoac deep_learning.
- `model`: sklearn `Pipeline`.

### Label

```python
PHISHING_LABEL = "-1"
LEGITIMATE_LABEL = "1"
```

Dataset Kaggle dung:

- `-1`: phishing.
- `1`: legitimate.

Trong phan train sklearn, code doi lai thanh:

- `1`: phishing.
- `0`: legitimate.

## 2. Che do `url-cnn`: train CNN tu URL goc

Phan nay nam trong block `Character-level CNN training for raw URL datasets`.

### `encode_url(url)`

Chuyen URL thanh danh sach so nguyen de dua vao neural network.

Cach hoat dong:

- Lay tung ky tu trong URL.
- Doi moi ky tu thanh ma ASCII bang `ord(char)`.
- Gioi han ma ky tu nho hon `URL_CNN_VOCAB_SIZE`.
- Cat URL toi da `URL_CNN_MAX_LEN = 200`.
- Neu URL ngan hon 200 ky tu thi padding bang `0`.

Muc dich: neural network khong doc truc tiep chuoi URL, nen URL phai duoc encode thanh tensor so.

### `UrlDataset`

Class dataset cua PyTorch.

Chuc nang:

- Luu danh sach `urls` va `labels`.
- `__len__`: tra ve so mau.
- `__getitem__`: tra ve 1 mau gom:
  - URL da encode thanh tensor.
  - label thanh tensor float.

Class nay duoc `DataLoader` dung de chia batch khi train CNN.

### `load_url_label_rows(data_path)`

Doc dataset CSV dang:

```text
url,label
https://example.com,0
https://fake-login.com,1
```

Chuc nang:

- Kiem tra file co dong du lieu hay khong.
- Kiem tra bat buoc co cot `url` va `label`.
- Bo qua dong thieu URL hoac label.
- Ep label ve integer.
- Chi chap nhan label `0` hoac `1`.
- Kiem tra dataset co ca 2 class.
- Kiem tra moi class co it nhat 2 dong de stratified split.

### `evaluate_url_cnn(model, loader)`

Danh gia CNN tren test loader.

Chuc nang:

- Chuyen model sang `eval()`.
- Tat gradient bang `torch.no_grad()`.
- Lay logits tu model.
- Doi logits thanh probability bang `sigmoid`.
- Doi probability thanh prediction voi nguong `0.5`.
- Tinh:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`
  - `roc_auc`

### `train_url_cnn_model(...)`

Train model CNN character-level.

Luon thuc hien:

1. Doc dataset bang `load_url_label_rows`.
2. Chia train/test bang `train_test_split`.
3. Tao `DataLoader` cho train va test.
4. Khoi tao `CharCnnUrlClassifier`.
5. Dung optimizer `Adam`.
6. Dung loss `BCEWithLogitsLoss`.
7. Train qua nhieu epoch.
8. Sau moi epoch, danh gia bang `evaluate_url_cnn`.
9. Luu model PyTorch bang `torch.save`.

Output mac dinh:

```text
artifacts/url_cnn.pt
```

## 3. Naive Bayes fallback cho feature dataset

Phan nay dung de tao artifact fallback `best_model.json`.

### `_train_categorical_model(rows, feature_names)`

Day la model Categorical Naive Bayes tu cai dat thu cong.

Chuc nang:

- Dem so mau moi class.
- Kiem tra co du ca phishing va legitimate.
- Tim cac gia tri tung xuat hien cua moi feature.
- Dem tan suat feature theo tung class.
- Tinh prior probability cho moi class.
- Tinh likelihood cho moi feature value bang Laplace smoothing.
- Luu tat ca vao dictionary artifact.

Ly do dung Laplace smoothing: tranh xac suat bang 0 khi gap gia tri hiem hoac chua tung thay.

Output la dictionary co dang:

```json
{
  "model_type": "categorical_naive_bayes",
  "feature_names": [...],
  "priors": {...},
  "likelihoods": {...}
}
```

### `train_url_model(data_path)`

Train Naive Bayes fallback tren dataset Kaggle.

Luon thuc hien:

1. Doc Kaggle rows bang `load_kaggle_feature_rows`.
2. Train categorical model bang `_train_categorical_model`.
3. Them thong tin source dataset.

Model nay sau do duoc ghi vao `artifacts/best_model.json`.

## 4. Xu ly dataset Kaggle

### `load_kaggle_feature_rows(data_path)`

Doc CSV Kaggle co cac cot feature da duoc trich xuat san.

Chuc nang:

- Doc file CSV.
- Kiem tra file khong rong.
- Kiem tra du tat ca cot trong `KAGGLE_FEATURE_NAMES`.
- Kiem tra co cot `Result`.
- Chi lay dong co label `-1` hoac `1`.
- Doi gia tri feature sang integer.

Output:

```python
list[FeatureRow]
```

### `build_kaggle_feature_matrix(rows)`

Chuyen data tu dang dictionary sang matrix de sklearn train.

Output:

- `x`: list cac vector feature.
- `y`: list label.

Mapping label:

- Kaggle `-1` thanh `1`, nghia la phishing.
- Kaggle `1` thanh `0`, nghia la legitimate.

### `label_to_name(label)`

Doi label thanh ten de ghi file:

- `-1` hoac `1` integer phishing -> `"phishing"`.
- con lai -> `"legitimate"`.

### `write_feature_rows(path, rows)`

Ghi rows ra CSV.

File ghi gom:

- `label`
- `label_name`
- tat ca feature trong `KAGGLE_FEATURE_NAMES`

Ham nay duoc dung de sinh:

- `dataset/features.csv`
- `dataset/train.csv`
- `dataset/valid.csv`
- `dataset/test.csv`

### `split_kaggle_rows(rows, valid_size, test_size, random_state)`

Chia dataset thanh 3 tap:

- train
- validation
- test

Dung stratified split de giu ti le class phishing/legitimate tuong doi giong nhau giua cac tap.

Dieu kien:

- `valid_size` phai > 0.
- `test_size` phai > 0.
- tong `valid_size + test_size` phai < 1.

### `write_dataset_files(...)`

Ghi cac file dataset da chia.

Output:

```text
dataset/features.csv
dataset/train.csv
dataset/valid.csv
dataset/test.csv
dataset/dataset_summary.json
dataset/configs.json
```

`dataset_summary.json` luu:

- tong so dong.
- so dong train/valid/test.
- so luong moi class.
- so feature.
- danh sach feature.

`configs.json` luu cau hinh chia dataset.

## 5. Danh sach model so sanh

### `_scaled_pipeline(classifier)`

Tao sklearn `Pipeline` gom:

1. `StandardScaler`
2. classifier

Muc dich: scale feature ve cung thang do truoc khi train cac model nhay voi scale nhu Logistic Regression va MLP.

### `build_model_candidates(max_iter, random_state)`

Tra ve 4 model de so sanh:

| Model | Nhom | Ghi chu |
| --- | --- | --- |
| `baseline_logistic_regression` | baseline | Model tuyen tinh, de giai thich, train nhanh. |
| `baseline_gaussian_naive_bayes` | baseline | Model xac suat don gian, train rat nhanh. |
| `deep_learning_mlp_small` | deep learning | MLP nho voi hidden layers `(64, 32)`. |
| `deep_learning_mlp_deep` | deep learning | MLP sau hon voi hidden layers `(128, 64, 32)`. |

Hai baseline duoc dung lam moc so sanh.

Hai MLP duoc dung de chon model deep learning tot nhat cho runtime.

## 6. Ghi ket qua va bieu do

### `write_model_results(path, results)`

Ghi bang ket qua so sanh model ra CSV.

Output:

```text
artifacts/model_results.csv
```

Cot trong file:

- `name`
- `family`
- `valid_accuracy`
- `valid_f1`
- `test_accuracy`
- `test_f1`
- `selected_for_runtime`

`selected_for_runtime = True` nghia la model deep learning duoc chon de luu vao `deep_learning_model.joblib`.

### `write_evaluation_charts(...)`

Ve cac bieu do neu co cai `matplotlib`.

Output:

```text
chart/model_comparison.png
chart/model_comparison_fixed.png
chart/confusion_matrix.png
chart/cls_metrics.png
chart/cls_feature_importance.png
artifacts/score_distribution.png
```

Chuc nang tung chart:

- `model_comparison.png`: so sanh validation F1 va test F1 cua 4 model.
- `confusion_matrix.png`: confusion matrix cua model deep learning duoc chon.
- `cls_metrics.png`: accuracy va F1 cua model deep learning duoc chon.
- `score_distribution.png`: phan bo probability phishing tren test set.
- `cls_feature_importance.png`: feature importance bang permutation importance.

Neu thieu `matplotlib`, ham se bo qua phan ve chart va khong lam hong qua trinh train.

## 7. Train mode Kaggle chinh

### `train_deep_learning_url_model(...)`

Day la ham quan trong nhat cua mode `kaggle`.

Ten ham co chu `deep_learning`, nhung hien tai no lam 2 viec:

1. Train 2 baseline va 2 deep learning model de so sanh.
2. Chon MLP deep learning tot nhat de luu lam runtime model.

Luon thuc hien:

1. Doc Kaggle dataset.
2. Kiem tra co du 2 class.
3. Chia train/valid/test.
4. Ghi dataset da chia vao thu muc `dataset/`.
5. Chuyen rows thanh feature matrix.
6. Train 4 model trong `build_model_candidates`.
7. Tinh validation accuracy, validation F1, test accuracy, test F1.
8. Chon model deep learning co validation F1 cao nhat.
9. Tinh classification report va confusion matrix cho model duoc chon.
10. Luu model deep learning duoc chon vao `artifacts/deep_learning_model.joblib`.
11. Ghi report, metrics, model comparison CSV va chart.

Ly do chon bang validation F1:

- Validation set dung de chon model.
- Test set dung de danh gia sau cung.
- F1 phu hop voi phishing detection vi can can bang precision va recall.

Output chinh:

```text
artifacts/deep_learning_model.joblib
artifacts/classification_report.txt
artifacts/deep_learning_metrics.json
artifacts/model_results.csv
dataset/train.csv
dataset/valid.csv
dataset/test.csv
chart/*.png
```

## 8. Optional HTML model

Phan nay chi chay khi truyen:

```bash
python train_model.py --html-archive archive.zip
```

### `_label_from_zip_path(path)`

Lay label tu duong dan file trong zip.

Quy uoc:

- Neu path co folder `notphish` -> legitimate.
- Neu path co folder `phish` -> phishing.
- Neu khong xac dinh duoc -> bo qua.

### `train_html_model(archive_path)`

Train Naive Bayes cho HTML features.

Luon thuc hien:

1. Mo file zip.
2. Duyet cac file `.html` hoac `.htm`.
3. Lay label tu path bang `_label_from_zip_path`.
4. Doc noi dung HTML.
5. Trich xuat feature HTML bang `extract_html_features`.
6. Train `_train_categorical_model` tren `HTML_FEATURE_NAMES`.

Output la artifact HTML model, sau do duoc gan vao `best_model.json`.

## 9. Tao artifact tong hop

### `train(data_path, html_archive_path=None, deep_learning_metrics=None)`

Tao artifact JSON tong hop cho runtime fallback.

Artifact nay gom:

- `url_model`: Naive Bayes tren URL feature.
- `html_model`: Naive Bayes tren HTML feature neu co `--html-archive`.
- `deep_learning_model`: thong tin model deep learning neu co train.
- `training_summary`: tom tat dataset va model.

Output sau cung duoc ghi vao:

```text
artifacts/best_model.json
```

Runtime trong `app/model.py` se uu tien:

1. `deep_learning_model.joblib` neu ton tai.
2. Fallback ve `best_model.json`.
3. Fallback tiep ve heuristic neu khong co model.

### `write_metadata(...)`

Ghi metadata ra:

```text
artifacts/metadata.json
```

File nay ghi:

- ten project.
- source dataset.
- duong dan model.
- duong dan report.
- class counts.
- ghi chu ve cach runtime predict.

## 10. Command-line interface

### `parse_args()`

Dinh nghia cac tham so command line.

| Tham so | Mac dinh | Cong dung |
| --- | --- | --- |
| `--task` | `kaggle` | Chon `kaggle` hoac `url-cnn`. |
| `--data` | `url_dataset.csv` | Duong dan dataset CSV. |
| `--html-archive` | none | Zip HTML snapshot co folder `Phish/` va `NotPhish/`. |
| `--out-dir` | `artifacts` | Thu muc luu artifact. |
| `--dataset-dir` | `dataset` | Thu muc luu train/valid/test split. |
| `--chart-dir` | `chart` | Thu muc luu chart. |
| `--out` | none | Output path rieng cho mode `url-cnn`. |
| `--epochs` | `5` | So epoch cho mode `url-cnn`. |
| `--skip-deep-learning` | false | Chi train Naive Bayes fallback, khong train 4 model so sanh. |
| `--test-size` | `0.2` | Ti le test set. |
| `--valid-size` | `0.15` | Ti le validation set. |
| `--max-iter` | `300` | So iteration toi da cho sklearn model. |

### `run_url_cnn_task(args, data_path, output_dir)`

Chay mode `url-cnn`.

Output:

- file `.pt` cua PyTorch model.
- in metrics ra terminal.

### `run_kaggle_task(args, data_path, output_dir)`

Chay mode mac dinh.

Luon thuc hien:

1. Xac dinh `html_archive_path`, `dataset_dir`, `chart_dir`.
2. Neu khong co `--skip-deep-learning`, train 4 model so sanh va luu best MLP.
3. Train Naive Bayes fallback va optional HTML model.
4. Ghi `best_model.json`.
5. Ghi `metadata.json`.
6. In danh sach file da ghi ra terminal.

### `main()`

Entry point cua script.

Luon thuc hien:

1. Doc command-line args.
2. Tao `data_path`.
3. Tao `output_dir`.
4. Neu `--task url-cnn` thi goi `run_url_cnn_task`.
5. Nguoc lai goi `run_kaggle_task`.

Dong cuoi:

```python
if __name__ == "__main__":
    main()
```

Dam bao file chi tu dong chay khi goi truc tiep bang command line.

## 11. Tom tat luong chay mac dinh

Khi chay:

```bash
python train_model.py
```

Luong xu ly la:

```text
main()
-> parse_args()
-> run_kaggle_task()
-> train_deep_learning_url_model()
   -> load_kaggle_feature_rows()
   -> split_kaggle_rows()
   -> write_dataset_files()
   -> build_model_candidates()
   -> train 2 baseline + 2 MLP
   -> chon MLP co validation F1 cao nhat
   -> write_model_results()
   -> write_evaluation_charts()
-> train()
   -> train_url_model()
   -> optional train_html_model()
-> write best_model.json
-> write_metadata()
```

## 12. Cac file quan trong duoc tao

| File | Y nghia |
| --- | --- |
| `artifacts/deep_learning_model.joblib` | Model MLP deep learning duoc chon cho runtime. |
| `artifacts/best_model.json` | Artifact fallback gom Naive Bayes URL/HTML va summary. |
| `artifacts/model_results.csv` | Bang so sanh 2 baseline va 2 deep learning model. |
| `artifacts/deep_learning_metrics.json` | Metrics chi tiet cua qua trinh train. |
| `artifacts/classification_report.txt` | Precision, recall, F1 cua model deep learning duoc chon. |
| `artifacts/metadata.json` | Metadata ve dataset, artifact va cach runtime dung model. |
| `dataset/train.csv` | Tap train sau khi split. |
| `dataset/valid.csv` | Tap validation de chon model. |
| `dataset/test.csv` | Tap test de danh gia sau cung. |
| `chart/model_comparison.png` | Bieu do so sanh cac model. |
| `chart/confusion_matrix.png` | Confusion matrix cua model runtime. |

## 13. Ket luan ngan gon

`train_model.py` khong chi train mot model. No la pipeline train day du:

- Doc va kiem tra dataset.
- Chia train/validation/test.
- Train 2 baseline va 2 deep learning model de so sanh.
- Chon MLP tot nhat cho runtime.
- Tao Naive Bayes fallback.
- Tuy chon train HTML model neu co archive HTML.
- Xuat report, metrics, chart va artifact cho web app.

