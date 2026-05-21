# Model comparison

File `train_model.py` hien tai so sanh 4 model:

| Nhom | Model | Muc dich |
| --- | --- | --- |
| Baseline | `baseline_logistic_regression` | Moc so sanh tuyen tinh tren URL/HTML feature vector. |
| Baseline | `baseline_gaussian_naive_bayes` | Moc so sanh xac suat don gian, train nhanh, gia dinh feature gan doc lap. |
| Deep learning | `url_cnn_deep_learning` | CNN character-level hoc truc tiep chuoi URL. |
| Deep learning | `deep_learning_url_html_mlp` | MLP hoc quan he phi tuyen tu URL/HTML feature vector. |

## Ket qua test set hien tai

Dataset: `balanced_dataset.csv`  
Test rows: `97,125`
Threshold: `0.4`
URL CNN best epoch: `9`

| Model | Family | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `url_cnn_deep_learning` | Deep learning | 0.9790 | 0.9757 | 0.9776 | 0.9767 | 0.9974 |
| `deep_learning_url_html_mlp` | Deep learning | 0.9192 | 0.9274 | 0.8902 | 0.9084 | 0.9720 |
| `baseline_logistic_regression` | Baseline | 0.8390 | 0.8665 | 0.7591 | 0.8093 | 0.9107 |
| `baseline_gaussian_naive_bayes` | Baseline | 0.8237 | 0.9763 | 0.6233 | 0.7608 | 0.8735 |

## Chart

Da sinh chart so sanh:

- `chart/model_comparison.png`: so sanh Accuracy, Precision, Recall, F1, ROC-AUC.
- `chart/model_comparison_f1.png`: so sanh rieng Test F1.

## Nhan xet

`url_cnn_deep_learning` la model tot nhat hien tai. Model nay hoc truc tiep pattern trong chuoi URL nen bat duoc nhieu tin hieu ma feature thu cong co the bo sot.

`deep_learning_url_html_mlp` tot hon ca hai baseline tren Accuracy, Recall, F1 va ROC-AUC. Dieu nay cho thay MLP hoc duoc quan he phi tuyen giua cac URL features tot hon model tuyen tinh va Naive Bayes.

`baseline_logistic_regression` co Precision cao nhung Recall thap hon deep learning, nghia la khi no bao phishing thi kha chac, nhung bo sot nhieu URL phishing hon.

`baseline_gaussian_naive_bayes` co Precision rat cao nhung Recall thap nhat. Model nay qua than trong voi class phishing, nen khong phu hop lam runtime chinh trong bai toan can bat phishing.

## Ket luan

Project co 2 baseline dung de so sanh va 2 model deep learning. Runtime van uu tien 2 model deep learning:

- `url_cnn_deep_learning`
- `deep_learning_url_html_mlp`

Hai baseline chi dung cho bao cao va danh gia, khong dung lam model runtime chinh.

## Tai sao dung deep learning?

Project dung deep learning vi dataset hien tai la raw URL, khong phai chi la bang feature co san. URL la du lieu dang chuoi, trong do thu tu ky tu, cau truc domain, path, query string, token dai, tu khoa va cach cac thanh phan xuat hien lien tiep nhau deu co y nghia. `url_cnn_deep_learning` co the hoc truc tiep cac pattern nay tu chuoi URL.

Ket qua test set cho thay ly do nay ro rang:

| Model | F1 | Recall | ROC-AUC |
| --- | ---: | ---: | ---: |
| `url_cnn_deep_learning` | 0.9767 | 0.9776 | 0.9974 |
| `deep_learning_url_html_mlp` | 0.9084 | 0.8902 | 0.9720 |
| `baseline_logistic_regression` | 0.8093 | 0.7591 | 0.9107 |
| `baseline_gaussian_naive_bayes` | 0.7608 | 0.6233 | 0.8735 |

Trong bai toan phishing detection, `Recall` cua class phishing rat quan trong vi bo sot phishing la loi nguy hiem. CNN dat Recall `0.9776`, cao hon Logistic Regression `0.7591` va Gaussian Naive Bayes `0.6233`. Nghia la deep learning bat duoc nhieu URL phishing hon rat nhieu.

## Tai sao khong dung cac model machine learning khac lam chinh?

Khong phai project bo qua machine learning truyen thong. Project van train baseline de so sanh. Tuy nhien, ket qua hien tai cho thay baseline kem hon deep learning tren cac metric quan trong.

`baseline_logistic_regression` la model tuyen tinh. No chi hoc duoc quan he gan tuyen tinh giua feature va label. URL phishing thuong khong don gian nhu "co mot feature thi la phishing"; no thuong la su ket hop cua nhieu dau hieu: domain la, path dai, token dai, tu khoa login/verify, shortener, redirect parameter, entropy cao. Cac tuong tac nay phi tuyen, nen Logistic Regression co Precision `0.8665` va Recall `0.7591`, van bo sot nhieu phishing hon deep learning.

`baseline_gaussian_naive_bayes` gia dinh cac feature doc lap tuong doi. Gia dinh nay khong phu hop voi URL, vi cac feature URL thuong lien quan nhau: URL dai thuong di kem nhieu ky tu dac biet, query dai, token dai, entropy cao. Ket qua la Naive Bayes co Precision `0.9763` nhung Recall chi `0.6233`, nghia la model rat than trong: khi bao phishing thi dung nhieu, nhung bo sot qua nhieu phishing.

Vi vay, cac model machine learning truyen thong van huu ich de lam moc so sanh, nhung khong phu hop lam runtime chinh khi muc tieu la bat phishing voi Recall va F1 cao.

## Khac biet giua deep learning va machine learning trong project nay

| Diem khac biet | Baseline machine learning | Deep learning |
| --- | --- | --- |
| Du lieu hoc | Hoc tu feature da trich xuat thu cong. | CNN hoc truc tiep tu chuoi URL; MLP hoc quan he phi tuyen tu feature. |
| Kha nang hoc pattern | Tot voi quan he don gian. | Tot hon voi pattern phuc tap, thu tu ky tu va tuong tac feature. |
| Feature engineering | Phu thuoc nhieu vao feature minh tao san. | CNN tu hoc representation tu URL text. |
| Recall phishing hien tai | Thap hon: `0.7591` va `0.6233`. | Cao hon: CNN dat `0.9776`. |
| Vai tro trong project | Moc so sanh, giai thich baseline. | Runtime chinh. |

Noi ngan gon: baseline machine learning doc cac feature minh dua vao va hoc quy tac don gian hon; deep learning, dac biet la CNN, hoc truc tiep pattern trong chuoi URL nen phu hop hon voi dataset raw URL hien tai.
