# Model Comparison

Pipeline moi trong `src/` bao cao cac model sau:

| Model | Nhom | Input | Mo ta |
| --- | --- | --- | --- |
| `baseline_tfidf_logreg` | Baseline | URL + HTML text | TF-IDF + Logistic Regression. |
| `baseline_random_forest` | Baseline | Handcrafted URL + HTML features | Random Forest tren feature co the giai thich. |
| `url_cnn` | Deep learning | URL | CNN character-level hoc raw URL. |
| `url_lstm` | Deep learning | URL | LSTM/BiLSTM hoc quan he tuan tu trong URL. |
| `html_cnn` | Deep learning | HTML | CNN character-level hoc raw HTML. |
| `dual_branch_cnn` | Deep learning | URL + HTML | URL branch + HTML branch + fusion + Dense. |

## Cach doc ket qua

- `baseline_tfidf_logreg`: moc so sanh tuyen tinh tren text URL + HTML.
- `baseline_random_forest`: moc so sanh machine learning tren feature thu cong.
- `url_cnn` va `url_lstm`: do kha nang hoc tu URL.
- `html_cnn`: do kha nang hoc tu HTML source.
- `dual_branch_cnn`: do kha nang ket hop URL + HTML bang fusion.

Trong phishing detection, `Recall` cua class phishing rat quan trong vi bo sot phishing co rui ro cao. `F1` giup can bang precision/recall. `ROC-AUC` cho biet chat luong probability truoc khi chon threshold.

## Luu y

Ket qua ghi vao:

```text
reports/results/model_comparison.csv
```

Figure ghi vao:

```text
reports/figures/confusion_matrix_<model>.png
reports/figures/roc_curve_<model>.png
```

Neu artifact chua co `models/saved/html_cnn.pt` hoac `models/saved/dual_branch_cnn.pt`, khong nen noi model da hoc HTML deep learning that su. Luc do demo chi dang dung baseline hoac heuristic fallback.

Neu artifact co `models/saved/dual_branch_cnn.pt`, co the mo ta he thong la:

```text
Dual-Branch CNN using URL and HTML source code
```
