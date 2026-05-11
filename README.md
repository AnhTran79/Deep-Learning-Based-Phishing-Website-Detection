# Phishing Website Detection Demo

Demo localhost cho de tai "Deep Learning-Based Phishing Website Detection".

Project nay duoc trinh bay theo style project cu:

- `app.py`: entrypoint chay web localhost.
- `app/`: FastAPI backend va URL feature extractor.
- `app/static/`: giao dien web.
- `artifacts/`: model artifact da train.
- `train_model.py`: train model tu dataset Kaggle hoac dataset URL dang `url,label`.

## 1. Cai dat

```bash
pip install -r requirements.txt
```

Neu may da cai Python 3.12 va package truc tiep, co the chay thang bang:

```bash
python app.py
```

## 2. Train model tu dataset Kaggle

Dataset dang dung:

```text
https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
```

Train lai artifact:

```bash
python train_model.py
```

Lenh tren se train ca:

- `MLPClassifier` deep learning model tren cac URL feature cua Kaggle.
- Naive Bayes artifact cu de lam fallback.

Train chung dataset URL voi bo HTML snapshot moi:

```bash
python train_model.py --html-archive "C:\Users\ADMIN\Downloads\archive.zip"
```

Sinh ra:

- `artifacts/best_model.json`
- `artifacts/deep_learning_model.joblib`
- `artifacts/classification_report.txt`
- `artifacts/model_results.csv`
- `artifacts/deep_learning_metrics.json`
- `artifacts/metadata.json`
- `dataset/train.csv`
- `dataset/valid.csv`
- `dataset/test.csv`
- `dataset/features.csv`
- `dataset/dataset_summary.json`
- `dataset/configs.json`
- `chart/confusion_matrix.png`
- `chart/model_comparison.png`
- `chart/cls_metrics.png`
- `chart/cls_feature_importance.png`

Artifact moi gom:

- `deep_learning_model`: neural network MLP duoc train bang `scikit-learn`.
- `url_model`: train tu file URL-feature `dataset.csv`.
- `html_model`: train tu cac file HTML trong `archive.zip` theo nhan `Phish` va `NotPhish`.
- `training_summary`: tom tat so mau, so feature va cach artifact duoc sinh ra.

File `classification_report.txt` co dang bang `precision`, `recall`, `f1-score`, `support`
de dua vao slide/bao cao nhu muc "Trained Model Results".

Neu chi muon train artifact Naive Bayes cu:

```bash
python train_model.py --skip-deep-learning
```

Luu y: `best_model.json` la artifact do `train_model.py` tao tu du lieu train, khong phai
file con nguoi tu dien xac suat bang tay. Khi co `deep_learning_model.joblib`, backend se uu tien
model MLP deep learning; neu file nay chua co thi se fallback ve Naive Bayes va heuristic.

## 3. Chay web localhost

```bash
python app.py
```

Mo trinh duyet tai:

```text
http://127.0.0.1:8000
```

Hoac chay truc tiep voi Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 4. API

```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"url\":\"https://example.com/login\"}"
```

Response gom:

- `label`: `phishing` hoac `legitimate`
- `confidence`: do tin cay cua nhan tra ve
- `phishing_probability`: xac suat phishing
- `model_source`: model dang duoc dung
- `features`: cac feature URL trich xuat duoc
- `html_fetch`: trang HTML co tai duoc hay khong
- `html_features`: cac feature HTML neu backend tai duoc noi dung trang

## 5. Luu y hoc thuat

Dataset Kaggle hien tai khong co URL goc. Dataset chi co cac feature da trich xuat san va cot `Result`.

Vi vay demo runtime se:

- trich xuat cac tin hieu co the suy ra tu URL nguoi dung nhap;
- map chung sang schema feature cua Kaggle;
- uu tien du doan bang MLP deep learning model neu da train;
- tu dong tai HTML cua URL nguoi dung nhap de trich xuat tin hieu content neu trang cho phep truy cap;
- gan gia tri trung lap `0` cho cac feature can WHOIS, DNS hoac search-engine check.

Bo HTML snapshot moi giup bo sung cac tin hieu content nhu form, input password, iframe,
meta refresh, script, link ngoai va tu khoa dang ngo. Khi backend khong tai duoc HTML vi timeout,
chan bot, loi SSL hoac trang khong phan hoi, demo van du doan bang URL model va heuristic nhu cu.

Mot nhom phishing thuc te thuong dat tren nen tang hop phap nhu Wix, Netlify, Vercel,
GitHub Pages hoac Google Sites. Vi URL/HTML tinh cua cac trang nay co the nhin kha binh
thuong, runtime heuristic co them tin hieu `has_public_hosting_platform` ket hop voi
`has_brand_impersonation` de bat cac URL gia mao brand/crypto tren public hosting.

`train_model.py` cung co che do train CNN char-level cho dataset URL goc dang:

```text
url,label
```

Chay bang:

```bash
python train_model.py --task url-cnn --data "C:\Users\ADMIN\Downloads\urls.csv"
```
