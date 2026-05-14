# Model comparison plan

File `train_model.py` hien tai dung 4 model de so sanh:

| Nhom | Model | Cong dung |
| --- | --- | --- |
| Baseline | `baseline_logistic_regression` | Mo hinh tuyen tinh de kiem tra dataset co the tach phishing/legitimate bang quan he don gian hay khong. Day la baseline de so sanh toc do, do on dinh va do de giai thich. |
| Baseline | `baseline_gaussian_naive_bayes` | Mo hinh xac suat don gian, gia dinh cac feature doc lap tuong doi. Dung lam moc so sanh toi thieu vi train nhanh va de trien khai. |
| Deep learning | `deep_learning_mlp_small` | Mang neural network MLP nho voi hidden layers `(64, 32)`. Dung de hoc quan he phi tuyen giua cac feature URL. |
| Deep learning | `deep_learning_mlp_deep` | Mang neural network MLP sau hon voi hidden layers `(128, 64, 32)`. Dung de kiem tra model phuc tap hon co cai thien F1/accuracy so voi MLP nho va baseline hay khong. |

## Baseline dung de lam gi?

Baseline khong phai de bo qua. Baseline la moc so sanh bat buoc:

- Neu baseline da dat ket qua gan bang deep learning, ta biet bai toan co the duoc giai bang model don gian.
- Neu deep learning tot hon baseline tren validation/test set, ta co co so de noi rang deep learning hoc duoc quan he phuc tap hon trong du lieu.
- Baseline giup tranh viec chon deep learning chi vi ten nghe manh hon, trong khi thuc te co the khong can thiet.

## Tai sao khong chi dung baseline?

Baseline co uu diem la nhanh, de giai thich va it ton tai nguyen. Tuy nhien phishing URL thuong co nhieu tin hieu ket hop voi nhau, vi du:

- URL dai, nhieu ky tu dac biet, nhieu chu so.
- Co tu khoa dang nghi nhu `login`, `verify`, `wallet`, `secure`.
- Co redirect parameter hoac embedded URL.
- Gia mao brand tren public hosting nhu GitHub Pages, Vercel, Netlify, Google Sites.

Cac quan he nay thuong khong hoan toan tuyen tinh. Logistic Regression co the bo sot cac tuong tac phuc tap giua feature. Gaussian Naive Bayes lai gia dinh feature doc lap, trong khi cac feature URL thuong lien quan nhau. MLP co the hoc cac quan he phi tuyen va cach nhieu feature cung xuat hien de tao thanh rui ro phishing.

Vi vay, deep learning nen duoc chon khi ket qua validation/test, dac biet la `F1-score` cua class phishing, tot hon baseline hoac on dinh hon baseline. Neu baseline tot ngang hoac tot hon, nen bao cao trung thuc rang baseline la lua chon don gian va hop ly hon.

## Tieu chi so sanh

Sau khi train, xem cac file:

- `artifacts/model_results.csv`: bang so sanh 4 model.
- `artifacts/deep_learning_metrics.json`: metrics chi tiet va confusion matrix.
- `artifacts/classification_report.txt`: precision, recall, F1-score cua model deep learning duoc chon cho runtime.
- `chart/model_comparison.png`: bieu do so sanh validation/test F1.

Nen uu tien `F1-score` hon accuracy, vi phishing detection can can bang giua:

- `precision`: du doan phishing co dung khong.
- `recall`: bat duoc bao nhieu URL phishing that.

## Ket luan de tra loi bao cao

Project khong bo baseline. Project train baseline de lam moc so sanh, sau do dung deep learning neu MLP cho ket qua tot hon tren validation/test set. Ly do dung deep learning la MLP co kha nang hoc quan he phi tuyen va tuong tac giua cac feature URL, phu hop hon voi phishing website detection so voi cac baseline don gian.

