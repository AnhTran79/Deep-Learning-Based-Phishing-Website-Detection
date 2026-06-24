# Review 1 - Muc Tieu Va Phuong Phap Phishing Detection

## 1. Cau noi chinh

> Em chon phuong phap phat hien phishing dua tren URL va HTML source code vi day la hai nguon du lieu co the thu thap tu dong, nhe hon screenshot va khong phu thuoc blacklist. Em xay dung pipeline gom data cleaning, train nhieu baseline va deep learning models, sau do so sanh bang precision, recall, F1-score, ROC-AUC va classification report. Ket qua cho thay Dual Branch CNN, mo hinh ket hop ca URL va HTML, dat F1-score tot nhat la 0.9248. Dieu nay chung minh rang ket hop dac trung ben ngoai cua URL va noi dung ben trong HTML giup phat hien phishing hieu qua hon so voi chi dung mot nguon du lieu.

## 2. Muc tieu cua de tai

Muc tieu cua nhom la xay dung mot phuong phap va he thong thu nghiem de phat hien website phishing.

Cu the, he thong phan loai website thanh:

```text
legitimate
phishing
```

Nhom khong chi muon kiem tra website co nam trong blacklist hay khong, ma muon model hoc cac dau hieu bat thuong trong URL va HTML source code.

Cach trinh bay:

```text
Muc tieu cua nhom em la xay dung mot mo hinh phat hien phishing website dua tren cac dac trung lay tu URL va HTML source code, tu do phan loai website la legitimate hay phishing.
```

## 3. Vi sao chon URL

URL la thong tin dau tien nguoi dung thay khi truy cap website.

Trong phishing, URL thuong co nhieu dau hieu dang nghi:

```text
- domain la
- subdomain dai
- nhieu ky tu dac biet
- nhieu dau gach ngang
- chua tu khoa nhu login, verify, secure, account
- gia mao ten thuong hieu
- dung redirect hoac short link
```

URL la du lieu nhe, de lay, khong can render giao dien website.

Cach trinh bay:

```text
Nhom em chon URL vi day la nguon du lieu don gian nhung chua nhieu dau hieu phishing. Nhieu website phishing co tinh tao URL giong website that hoac them cac tu nhu login, verify, secure de danh lua nguoi dung.
```

## 4. Vi sao chon HTML source code

Chi dung URL doi khi chua du.

Co nhung phishing URL nhin binh thuong, nhung HTML ben trong lai co hanh vi dang nghi:

```text
- form dang nhap
- input password
- action gui du lieu sang domain khac
- iframe
- script redirect
- noi dung yeu cau nhap tai khoan/mat khau
```

HTML giup model nhin duoc cau truc va hanh vi ben trong website, khong chi nhin dia chi ben ngoai.

Cach trinh bay:

```text
HTML source code giup nhom em phan tich noi dung va cau truc cua website. Mot trang phishing thuong co form nhap thong tin, password field hoac script redirect, nen HTML cung cap them bang chung ma URL co the khong the hien ro.
```

## 5. Vi sao ket hop URL va HTML

URL va HTML bo sung cho nhau:

```text
URL  = dac trung ben ngoai
HTML = dac trung ben trong
```

Neu chi dung URL, model co the bo sot website co URL binh thuong nhung noi dung doc hai.

Neu chi dung HTML, model co the bo qua dau hieu domain/subdomain bat thuong.

Vi vay nhom ket hop ca hai nguon de tang do tin cay.

Cach trinh bay:

```text
Nhom em chon ket hop URL va HTML vi hai nguon du lieu nay bo sung cho nhau. URL the hien dau hieu ben ngoai cua website, con HTML the hien cau truc va hanh vi ben trong trang. Khi ket hop ca hai, model co nhieu thong tin hon de dua ra quyet dinh.
```

## 6. Vi sao khong phu thuoc blacklist

Blacklist la danh sach cac website phishing da duoc phat hien truoc do.

Nhuoc diem cua blacklist:

```text
- website phishing moi co the chua nam trong blacklist
- attacker thay domain lien tuc
- blacklist thuong bi tre so voi thuc te
```

Neu chi dung blacklist, he thong kho phat hien website phishing moi.

Cach trinh bay:

```text
Nhom em khong chon huong phu thuoc blacklist vi blacklist chi phat hien duoc cac website da biet. Trong thuc te, phishing website co the duoc tao moi lien tuc, nen nhom muon model hoc pattern de co kha nang phat hien ca cac truong hop chua xuat hien trong blacklist.
```

## 7. Vi sao khong dung screenshot

Screenshot co the dung de phat hien giao dien gia mao, nhung nhom chua chon huong nay o giai doan dau vi:

```text
- ton tai nguyen hon
- can render website
- thoi gian xu ly lau hon
- phu thuoc layout/giao dien
- phuc tap hon khi trien khai
```

URL va HTML nhe hon, de thu thap tu dong hon va phu hop hon voi review lan 1.

Cach trinh bay:

```text
Nhom em chua dung screenshot vi screenshot yeu cau render giao dien, ton tai nguyen va thoi gian xu ly hon. Trong pham vi hien tai, URL va HTML source code phu hop hon vi de thu thap tu dong va nhe hon.
```

## 8. Phuong phap tong quan

Pipeline nhom chon gom:

```text
1. Thu thap URL va HTML
2. Lam sach du lieu
3. Trich xuat hoac bieu dien dac trung
4. Train nhieu model
5. So sanh ket qua
6. Chon model phu hop nhat
```

Cach trinh bay:

```text
Phuong phap cua nhom em la xay dung pipeline gom data cleaning, preprocessing URL/HTML, train nhieu mo hinh khac nhau va danh gia bang cac metrics phu hop. Nhom khong chi train mot model duy nhat ma co baseline de so sanh.
```

## 9. Vi sao can data cleaning

Du lieu thuc te thuong khong sach.

Co the co:

```text
- URL rong
- label sai
- HTML khong ton tai
- HTML qua ngan
- HTML loi hoac thieu noi dung
```

Neu dua du lieu ban vao train, model co the hoc sai.

Cach trinh bay:

```text
Nhom em can data cleaning de loai bo cac mau khong du thong tin, vi du HTML rong hoac qua ngan. Dieu nay giup du lieu train dang tin cay hon va giam kha nang model hoc tu du lieu loi.
```

## 10. Vi sao train baseline models

Baseline la mo hinh nen de so sanh.

Nhom dung baseline de tra loi cau hoi:

```text
Deep learning co that su tot hon phuong phap truyen thong khong?
```

Baseline trong project:

```text
TF-IDF + Logistic Regression
Random Forest voi handcrafted features
```

Cach trinh bay:

```text
Nhom em train baseline models de co moc so sanh. Neu chi dung deep learning ma khong co baseline thi kho chung minh phuong phap deep learning that su hieu qua hon.
```

## 11. Vi sao dung deep learning

URL va HTML deu la du lieu dang chuoi.

Deep learning, dac biet CNN/LSTM, co the hoc pattern trong chuoi nhu:

```text
login
verify
secure
password
<form
script
iframe
```

Thay vi tu dinh nghia toan bo rule thu cong, model co the hoc pattern tu du lieu.

Cach trinh bay:

```text
Nhom em chon deep learning vi URL va HTML deu la du lieu dang chuoi. Cac mo hinh nhu CNN hoac LSTM co the hoc cac pattern dang ngo trong chuoi ky tu ma khong can phu thuoc hoan toan vao rule thu cong.
```

## 12. Vi sao dung Dual Branch CNN

Dual Branch CNN co hai nhanh:

```text
Nhanh 1: hoc URL
Nhanh 2: hoc HTML
```

Sau do ghep thong tin lai de du doan.

Ly do chon:

```text
- moi loai du lieu co dac trung rieng
- URL ngan hon, HTML dai hon
- khong nen tron thang hai loai du lieu ngay tu dau
- model hoc rieng tung nguon roi ket hop
```

Cach trinh bay:

```text
Nhom em chon Dual Branch CNN vi URL va HTML co ban chat khac nhau. URL thuong ngan va chua dau hieu domain/path, con HTML dai hon va chua cau truc trang. Vi vay nhom dung hai nhanh CNN rieng de hoc tung nguon du lieu, sau do ket hop lai de phan loai.
```

## 13. Vi sao can nhieu metrics

Voi phishing detection, accuracy khong du.

Neu dataset lech class, model co the accuracy cao nhung van bo sot nhieu phishing.

Nhom dung:

```text
precision
recall
F1-score
ROC-AUC
classification report
```

Cach trinh bay:

```text
Nhom em khong chi dung accuracy vi trong phishing detection, bo sot phishing la rui ro lon. Do do nhom dung them precision, recall, F1-score, ROC-AUC va classification report de danh gia can bang hon.
```

## 14. Vi sao F1-score quan trong

F1-score can bang giua:

```text
precision
recall
```

Trong phishing detection:

```text
precision = du doan phishing co dung khong
recall    = phat hien duoc bao nhieu phishing that
```

Cach trinh bay:

```text
Nhom em chu trong F1-score vi metric nay can bang giua precision va recall. Voi bai toan phishing, nhom can vua phat hien duoc nhieu phishing, vua han che canh bao sai qua nhieu.
```

## 15. Cach noi gon trong review lan 1

```text
Nhom em chon huong phat hien phishing dua tren URL va HTML source code. Ly do la vi day la hai nguon du lieu co the thu thap tu dong, nhe hon screenshot va khong phu thuoc blacklist. URL the hien cac dau hieu ben ngoai nhu domain, path, ky tu dac biet hoac tu khoa dang ngo. HTML the hien cau truc va hanh vi ben trong trang nhu form dang nhap, password input hoac script redirect.

Ve phuong phap, nhom em xay dung pipeline gom data cleaning, preprocessing URL/HTML, train baseline models va deep learning models. Nhom co baseline de so sanh, sau do thu cac mo hinh nhu URL CNN, HTML CNN va Dual Branch CNN. Dual Branch CNN la huong chinh vi no hoc rieng URL va HTML roi ket hop lai de phan loai. Nhom danh gia bang precision, recall, F1-score, ROC-AUC va classification report thay vi chi dung accuracy, vi bai toan phishing can quan tam den viec phat hien dung phishing va han che bo sot.
```

## 16. Neu giang vien hoi: Diem hop ly cua phuong phap la gi?

```text
Diem hop ly cua nhom em la khong phu thuoc vao blacklist, khong dung du lieu nang nhu screenshot, ma tan dung hai nguon du lieu nhe va giau thong tin la URL va HTML. Ngoai ra nhom khong chi dung mot mo hinh ma so sanh baseline voi deep learning de chung minh huong ket hop URL + HTML co co so hon.
```

## 17. Neu giang vien hoi: Vi sao khong chi dung URL?

```text
Neu chi dung URL thi co the bo sot nhung trang co URL nhin binh thuong nhung HTML ben trong co form danh cap thong tin hoac script redirect. Vi vay nhom dung them HTML de phan tich noi dung va hanh vi cua trang.
```

## 18. Neu giang vien hoi: Vi sao khong chi dung HTML?

```text
Neu chi dung HTML thi model co the bo qua cac dau hieu rat quan trong tu domain va URL, vi du subdomain gia mao, URL qua dai, redirect parameter hoac tu khoa dang ngo trong duong dan. Vi vay nhom ket hop ca hai.
```

## 19. Neu giang vien hoi: Vi sao dung deep learning?

```text
Vi URL va HTML deu la du lieu dang chuoi, deep learning co the hoc cac pattern lap lai trong chuoi ky tu nhu login, verify, password, form, script. Dieu nay giup giam phu thuoc vao rule thu cong va co kha nang hoc pattern tu du lieu.
```

## 20. Neu giang vien hoi: Vi sao dung classification report?

```text
Vi classification report cho thay precision, recall, F1-score va support cua tung class legitimate va phishing. Voi phishing detection, nhom can biet rieng model phat hien class phishing tot den dau, chu khong chi nhin accuracy tong the.
```
