# Feature Importance Analysis & Model Interpretation

# 💡 Latar Belakang Proyek

Proyek ini berfokus pada analisis interpretabilitas model machine learning untuk kasus prediksi churn pelanggan.
Beberapa aspek utama yang dibahas mencakup:
1. Feature importance menggunakan beberapa algoritma interpretasi model.
2. Analisis kontribusi fitur dengan SHAP dan LIME.
3. Pembuatan model baseline untuk melihat performa seluruh model hingga pemilihan model yang terbaik untuk analisis lebih lanjut
4. Visualisasi hasil interpretasi model untuk mendukung pengambilan keputusan bisnis.

# 🛠️ Tahapan Proyek
1. Data Overview :

- Jumlah data: 10.127 baris × 21 kolom
- Tipe data: Integer, float, object
- Kualitas data:
    - Tidak ada missing values atau duplikasi
    - Beberapa outlier ditangani dengan metode Capping IQR yaitu kolom total_amt_chng_q4_q1, total_ct_chng_q4_q1
    - Beberapa outlier ditangani dengan metode trimming IQR yait kolom total_trans_amt, avg_open_to_buy

2. EDA (Exploratory Data Analysis)

- Distribusi usia pelanggan dan kelompok usia yang memiliki kemungkinan churn terbesar
    - **Distribusi usia pelanggan** antara Existing Customer dan Attrited Customer sangat mirip. Dapat dilihat dari median, IQR, dan sebaran usia secara keseluruhan hampir sama terhadap dua kelompok pelanggan ini. Hal ini menunjukkan bahwa usia pelanggan bukan faktor utama membedakan atau prediksi pelanggan mana yang akan melakukan churn pada kasus ini.
    - **Distribusi Kelompok usia**

| Age Group     | Attrited Customer | Existing Customer |
|----------------|------------------|-------------------|
| Dewasa Madya   | 0.164654         | 0.835346          |
| Dewasa Muda    | 0.100457         | 0.899543          |
| Dewasa Tua     | 0.167215         | 0.832785          |
| Lansia         | 0.200000         | 0.800000          |

   Berdasarkan hasil analisis diatas, ditemukan bahwa Tingkat churn paling rendah ditemukan pada kelompok Dewasa Muda dengan persentase sekitar 10%, sementara itu, tingkat churn cenderung meningkat seiring bertambahnya usia, mencapai puncaknya pada kelompok Lansia yang menunjukkan tingkat churn tertinggi sekitar 20%. Hal ini mengindikasikan bahwa kelompok pelanggan yang lebih tua memiliki potensi lebih besar untuk churn dibandingkan kelompok pelanggan yang paling muda.

- Analisis hubungan antara tingkat pendidikan dan pendapatan, serta bagaimana korelasinya dengan churn.
    - **Hubungan tingkat pendidikan dan pendapatan (Secara Umum)** :  Berdasarkan hasil uji statistik yang diberikan, dapat disimpulkan bahwa tidak ada hubungan yang signifikan secara statistik antara tingkat pendidikan dan kategori pendapatan. Nilai p-value sebesar 0.2100, yang jauh lebih besar dari ambang batas signifikansi (0.05), artinya bahwa perbedaan pendapatan antar kelompok pendidikan tidak mencerminkan adanya hubungan secara signifikan. Hal ini lebih lanjut didukung oleh nilai Cramer's V yang sangat rendah (0.0266), yang menunjukkan kekuatan hubungan yang sangat lemah.
    - **Korelasi terhadap churn** : Berdasarkan hasil analisis, menunjukkan bahwa tidak ada hubungan yang signifikan secara statistik antara tingkat pendidikan dan pendapatan pada kelompok pelanggan yang melakukan churn. Hal ini terlihat dari nilai p-value sebesar 0.0914 yang lebih besar dari ambang batas (0.05). Selain itu, nilai Cramér's V yang sangat rendah (0.0700) artinya bahwa kekuatannya sangat lemah dan tidak memiliki efek yang berarti. Dengan kata lain, tingkat pendidikan dan pendapatan pelanggan yang churn tidak saling mempengaruhi secara signifikan, dan keduanya bukanlah faktor utama yang dapat memprediksi atau menjelaskan terjadinya churn dalam kasus ini.
 
- Identifikasi perbedaan churn berdasarkan gender.
   Hasil temuan menyatakan bahwa hubungan antara jenis kelamin dan churn, terlihat bahwa jumlah pelanggan perempuan (F) lebih banyak daripada pelanggan laki-laki (M) secara keseluruhan. Artinya, tingkat churn pelanggan perempuan lebih tinggi secara signifikan dibandingkan pelanggan laki-laki. Jumlah pelanggan perempuan yang churn (attrited) mencapai sekitar 882 pelanggan, sedangkan pelanggan laki-laki hanya sekitar 505 pelanggan. Ini menunjukkan bahwa jenis kelamin adalah salah satu faktor yang memiliki pengaruh terhadap kecenderungan pelanggan untuk melakukan churn dalam kasus ini.

3. Data Preprocessing

- Encoding : Menggunakan One Hot Encoding untuk mengubah data kategori menjadi data numerik agar dapat diproses dalam pemodelan. Kolom yang diencode adalah gender, marital status, card category, income category.
- Scaling : Dilakukan scaling menggunakan standarisasi nilai pada kolom customer age, dependent count, months on book, total relationship count, months inactive 12 mon, contacts count 12 mon, credit limit, total revolving bal, avg open to buy, total trans amt, total trans ct.

# 🤖 Pemodelan & Hasil
Dilakukan eksperimen pada model based di beberapa model klasifikasi untuk mengidentifikasi model yang memiliki performa terbaik, berikut hasilnya:

| Model               | Precision (Train) | Recall (Train) | F1-Score (Train) | ROC AUC (Train) | Precision (Test) | Recall (Test) | F1-Score (Test) | ROC AUC (Test) |
|----------------------|------------------:|----------------:|------------------:|-----------------:|------------------:|----------------:|------------------:|----------------:|
| Logistic Regression  | 0.783540 | 0.628391 | 0.697441 | 0.937491 | 0.757322 | 0.644128 | 0.696154 | 0.926499 |
| Decision Tree        | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.785479 | 0.846975 | 0.815068 | 0.900389 |
| Random Forest        | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.914980 | 0.804270 | 0.856061 | 0.987088 |
| XGBoost              | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.919118 | 0.889680 | 0.904159 | 0.994294 |
| LightGBM             | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.917563 | 0.911032 | 0.914286 | 0.994823 |
| KNN                  | 0.889989 | 0.731465 | 0.802978 | 0.979525 | 0.808612 | 0.601423 | 0.689796 | 0.910613 |
| SVM                  | 0.908709 | 0.783002 | 0.841185 | 0.982271 | 0.885106 | 0.740214 | 0.806202 | 0.973850 |
| MLP                  | 0.829240 | 0.641049 | 0.723100 | 0.950358 | 0.816143 | 0.647687 | 0.722222 | 0.939934 |

**Model terbaik : SVM**

Selanjutnya, dilakukan analisis mendalam pada model SVM 

| Model | Precision | Recall | F1-Score | ROC AUC |
|--------|------------|---------|-----------|----------|
| **Base SVM (Train)** | 0.908709 | 0.783002 | 0.841185 | 0.982271 |
| **Base SVM (Test)** | 0.885106 | 0.740214 | 0.806202 | 0.973850 |
| **Hyperparameter Tuning SVM (Train)** | 0.985294 | 0.969259 | 0.977211 | 0.998320 |
| **Hyperparameter Tuning SVM (Test)** | 0.812030 | 0.768683 | 0.789762 | 0.969805 |
| **Hyperparameter Tuning SVM menggunakan SMOTE (Train)** | 0.721345 | 0.950271 | 0.820133 | 0.980006 |
| **Hyperparameter Tuning SVM menggunakan SMOTE (Test)** | 0.697222 | 0.893238 | 0.783151 | 0.967079 |

**Hasil :**

Berdasarkan hasil model SVM baik melalui based model, hyperparameter tuning tanpa SMOTE dan hyperparameter tuning dengan SMOTE. Menghasilkan berbagai macam variasi performa. Namun jika strategi bisnis ingin menangkap sebanyak mungkin pelanggan yang akan churn maka dapat dilihat dengan hasil recall tinggi (89%) serta gap recallnya tidak terlalu jauh yaitu 6% (hal ini dapat dilihat model dengan tuned SMOTE lebih unggul dibandingkan yang lain) meskipun precision nya mencapai 69% namun gapnya tidak terlalu jauh beda (hanya 3%) dibandingkan model yang lain yang gapnya lebih dari 6%. Dengan kata lain, model ini lebih proaktif dalam mengidentifikasi pelanggan berisiko, meskipun ada peningkatan kasus di mana model salah memprediksi pelanggan yang sebenarnya tidak akan churn (false positive).

**Prediksi terhadap churn :**

Model ini berhasil mengidentifikasi 251 pelanggan yang benar-benar akan churn, dan hanya salah memprediksi 30 kasus churn sebagai pelanggan yang tidak akan churn (false negative). Selain itu, model juga salah memprediksi 109 pelanggan yang sebenarnya tidak akan churn sebagai pelanggan yang berisiko (false positive). Secara keseluruhan, dengan recall sebesar 89.3%, model ini sangat cocok untuk strategi bisnis yang memprioritaskan identifikasi sebanyak mungkin pelanggan yang berpotensi churn.

**Perspektif kurva ROC-AUC terhadap churn**

Model SVM menunjukkan performa yang sangat baik dalam memprediksi churn. Hal ini dapat dilihat dari nilai AUC mencapai 0.967 yang artinya bahwa model ini memiliki kemampuan yang baik dalam membedakan pelanggan yang berpotensi churn dan tidak.

# 📊	Metode Model Diagnostic
Berdasarkan hasil analisis feature importance pada model Support Vector Machine (SVM):
- Fitur paling berpengaruh dalam memprediksi customer churn adalah total_trans_ct (jumlah total transaksi), dengan nilai drop-out loss tertinggi sebesar +0.241.
    ➜ Ini menunjukkan bahwa frekuensi transaksi pelanggan merupakan faktor utama yang menentukan apakah pelanggan akan tetap aktif atau berhenti.
- Variabel seperti gender, status pernikahan, kategori kartu, dan kategori pendapatan memiliki pengaruh rendah terhadap keputusan model, yang terlihat dari garis horizontal datar pada grafik pentingnya fitur.
- Faktor-faktor perilaku dan transaksional memiliki kontribusi paling kuat terhadap churn, khususnya:
    - Semakin tinggi jumlah transaksi (total_trans_ct), semakin rendah kemungkinan churn.
    - Variabel total_trans_amt (total nilai transaksi) dan total_revolving_bal juga berpengaruh, tetapi tidak sebesar total_trans_ct.



  
