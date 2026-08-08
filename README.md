# Deep Learning Based Phishing Website Detection

Project phat hien website phishing bang deep learning da phuong thuc, su dung:

- URL
- HTML source code
- Screenshot
- Label: `0 = legitimate`, `1 = phishing`

Dataset va model chinh cua project la Phish360. Tri-Branch CNN la model chinh;
nam model con lai duoc train de baseline va ablation.

## Dataset Phish360

Tai dataset tu:

```text
https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
```

Dat dataset tai:

```text
data/external/Phish360/
```

Cau truc moi sample:

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

Dataset, external samples va model `.pt` khong duoc commit len GitHub.

## Sau Model Phish360

| Model | Input | Vai tro |
|---|---|---|
| `phish360_url_cnn` | URL | Ablation |
| `phish360_url_lstm` | URL | So sanh CNN/LSTM |
| `phish360_html_cnn` | HTML | Ablation |
| `phish360_screenshot_cnn` | Screenshot | Ablation |
| `phish360_dual_branch_cnn` | URL + HTML | Fusion baseline |
| `phish360_tri_branch_cnn` | URL + HTML + Screenshot | Model chinh |

Tri-Branch CNN gom ba nhanh:

```text
URL -> Embedding -> Conv1D -> Pooling
HTML -> Embedding -> Conv1D -> Pooling
Screenshot -> Conv2D -> Pooling
                |
        Concatenate -> Fully Connected -> Sigmoid
```

## Cai Dat

```bash
pip install -r requirements.txt
playwright install chromium
```

## Train Phish360

Train ca 6 model:

```bash
python train_phish360.py --model all
```

Chi train model chinh:

```bash
python train_phish360.py --model tri_branch_cnn
```

Train nhanh de smoke test:

```bash
python train_phish360.py --quick
```

Train index da bo sung mau audit:

```bash
python train_phish360.py ^
  --data data/processed/phish360_plus_audited.csv ^
  --model all ^
  --html-max-chars 5000 ^
  --max-html-len 5000
```

Output:

```text
data/processed/phish360_url_html_screenshot.csv
models/saved/phish360/
reports/results/phish360/
reports/figures/phish360/
```

## Web Demo

Model demo chinh:

```text
models/saved/phish360/phish360_tri_branch_cnn.pt
```

Chay:

```bash
python app.py
```

Mo:

```text
http://127.0.0.1:8000
```

Demo dung Playwright de lay rendered HTML va screenshot cua URL, sau do chay
Tri-Branch CNN. Neu checkpoint khong ton tai, demo chi dung heuristic fallback
va hien `model_source = heuristic_url_html_rules`.

Giao dien hien `phishing score` va ba muc khuyen nghi:

```text
< 40%     -> Low phishing likelihood
40% - 75% -> Suspicious - manual review
>= 75%    -> High phishing risk
```

Observed Signals la heuristic doc lap, khong duoc trinh bay nhu loi giai thich
cho Tri-Branch CNN.
Neu Tri-Branch khong chay du ca ba nhanh, heuristic fallback luon tra ve
`Limited analysis - manual review`, an phan tram va tra nhan `inconclusive`
thay vi ket luan legitimate.

## External Validation

Nguon candidate:

- OpenPhish Community Feed: phishing candidate.
- Tranco: legitimate candidate.

Playwright thu thap final URL, rendered HTML, screenshot va metadata. Collector
kiem tra URL, HTML va screenshot trung voi Phish360 de giam data leakage.

Quy trinh ngan:

```bash
python hybrid_pipeline.py collect --count 100
python hybrid_pipeline.py evaluate week_YYYY_MM_DD
```

`collect` tu tao batch theo ngay neu khong truyen ten batch. `evaluate` cham ca
6 model Phish360 va tao audit queue:

```text
data/audit/week_YYYY_MM_DD_audit_queue.csv
```

Audit status:

```text
approved  -> nhan hien tai dung
corrected -> sua nhan bang audited_label
rejected  -> khong dua vao training
uncertain -> chua du bang chung
```

Sau khi audit:

```bash
python hybrid_pipeline.py retrain week_YYYY_MM_DD
```

Batch da dua vao training khong con la external test doc lap. Model moi phai
duoc evaluate tren batch tuong lai chua tung tham gia training.

## Lenh Chi Tiet

Tao candidate:

```bash
python fetch_weekly_candidates.py ^
  --tranco-file data/raw/top-1m.csv ^
  --out data/candidates/week_YYYY_MM_DD.csv ^
  --phishing-count 100 ^
  --legitimate-count 100 ^
  --update-seen
```

Collect URL + HTML + screenshot:

```bash
python collect_external_dataset.py ^
  --input data/candidates/week_YYYY_MM_DD.csv ^
  --output data/external_test/week_YYYY_MM_DD ^
  --legitimate-count 100 ^
  --phishing-count 100 ^
  --headless ^
  --resume
```

Evaluate:

```bash
python evaluate_external_dataset.py ^
  --dataset data/external_test/week_YYYY_MM_DD ^
  --results-dir reports/results/external_test_week_YYYY_MM_DD ^
  --figures-dir reports/figures/external_test_week_YYYY_MM_DD ^
  --models phish360_url_cnn phish360_url_lstm phish360_html_cnn phish360_screenshot_cnn phish360_dual_branch_cnn phish360_tri_branch_cnn
```

## Ket Qua Hien Tai

Internal test Phish360 cua Tri-Branch CNN:

```text
Accuracy:  0.9247
Precision: 0.8993
Recall:    0.9406
F1-score:  0.9195
ROC-AUC:   0.9776
```

External batch `week_2026_07_09`, 100 phishing + 100 legitimate:

```text
Accuracy:  0.8650
Precision: 0.8288
Recall:    0.9200
F1-score:  0.8720
ROC-AUC:   0.9242
```

## Project Layout

```text
app/                         FastAPI web demo
src/data/                    Phish360 index va multimodal loader
src/models/                  6 deep learning models
src/preprocessing/           URL, HTML va image preprocessing
src/training/                Phish360 training
src/evaluation/              Metrics va model comparison
models/saved/phish360/       Checkpoint da train, khong commit
reports/results/phish360/    Internal results
reports/results/external_*   External validation results
hybrid_pipeline.py           Hybrid automation wrapper
```

## Kiem Thu

```bash
python -m pytest -q
```
