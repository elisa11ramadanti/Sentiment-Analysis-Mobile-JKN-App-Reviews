# Sentiment Analysis Mobile JKN App Reviews

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap ulasan aplikasi Mobile JKN yang diambil dari Google Play Store. Dataset yang digunakan merupakan hasil scraping data ulasan pengguna yang telah dilakukan sebelumnya.

Analisis ini mencakup berbagai tahapan, seperti pengumpulan data, pembersihan teks, eksplorasi data, serta klasifikasi sentimen menjadi kategori **positif**, **negatif**, atau **netral**. Dengan menggunakan teknik pemrosesan teks (text preprocessing) dan algoritma machine learning, proyek ini diharapkan dapat memberikan wawasan yang berguna bagi pengembang aplikasi untuk memahami umpan balik pengguna dan meningkatkan kualitas layanan aplikasi.

---

### **Latar Belakang**
Aplikasi Mobile JKN digunakan oleh banyak masyarakat untuk mengakses layanan BPJS Kesehatan di Indonesia. Ulasan pengguna pada Google Play Store memberikan gambaran tentang kepuasan dan masalah yang dihadapi pengguna. Analisis sentimen dari ulasan tersebut dapat membantu pengembang memahami kebutuhan pengguna dan meningkatkan kualitas layanan aplikasi.

---

### **Tujuan**
1. Mengklasifikasikan sentimen ulasan pengguna menjadi **positif**, **negatif**, atau **netral**.
2. Memberikan wawasan mengenai umpan balik pengguna untuk perbaikan aplikasi.
3. Menganalisis tren sentimen dari waktu ke waktu.

---

## **Tentang Dataset**  
Dataset yang digunakan merupakan hasil scraping ulasan aplikasi **Mobile JKN** dari **Google Play Store**. Dataset ini berisi 211.500 entri dengan 11 kolom utama yang menyimpan informasi detail tentang ulasan pengguna. Berikut adalah penjelasan dari setiap kolom:  

| **Kolom**             | **Tipe Data** |                      **Deskripsi**                                                      |
|-----------------------|---------------|-----------------------------------------------------------------------------------------|
| **reviewId**          | object        | ID unik untuk setiap ulasan.                                     |
| **userName**          | object        | Nama pengguna yang memberikan ulasan.                           |
| **userImage**         | object        | URL gambar profil pengguna.                                      |
| **content**           | object        | Teks ulasan yang ditulis oleh pengguna.                          |
| **score**             | int64         | Rating yang diberikan pengguna (skala 1-5).                     |
| **thumbsUpCount**     | int64         | Jumlah "like" atau "thumbs up" yang diterima ulasan dari pengguna lain. |
| **reviewCreatedVersion** | object        | Versi aplikasi saat ulasan dibuat (tidak semua ulasan memiliki informasi ini). |
| **at**                | object        | Timestamp kapan ulasan dibuat.                                   |
| **replyContent**      | object        | Balasan dari pengembang aplikasi (jika ada).                     |
| **repliedAt**         | object        | Timestamp kapan balasan diberikan (jika ada).                    |
| **appVersion**        | object        | Versi aplikasi yang digunakan saat ulasan dibuat (tidak semua ulasan memiliki informasi ini). |

---

## **Alur Pemrosesan Data**
1. **Memuat Dataset**: Menggunakan `pandas` untuk membaca dataset CSV hasil scraping.
2. **Pembersihan Data**:
   - Menghapus data kosong (`NaN`).
   - Menghapus karakter khusus, angka, dan URL.
3. **Preprocessing Teks**:
   - **Case Folding**: Mengubah teks menjadi huruf kecil.
   - **Tokenization**: Memecah kalimat menjadi kata-kata.
   - **Stopword Removal**: Menghapus kata-kata umum yang tidak relevan.
   - **Stemming**: Mengubah kata menjadi bentuk dasar dengan Sastrawi.
   - **Handling Slang Words**: Mengubah kata-kata gaul menjadi kata baku.
4. **Pelabelan**:
   - Menggunakan **Lexicon-Based Approach** dengan kamus kata positif dan negatif.
   - **SentimentIntensityAnalyzer** dari `NLTK` sebagai pelengkap.
5. **Klasifikasi Sentimen**:
   - Mengklasifikasikan ulasan menjadi **positif**, **negatif**, atau **netral**.
6. **Visualisasi**:
   - Menampilkan distribusi sentimen menggunakan grafik batang (`matplotlib` dan `seaborn`).
   - Membuat **WordCloud** untuk kata-kata yang sering muncul dalam ulasan.

---

## **Metodologi**

### **Lexicon-Based Approach**
Menggunakan kamus kata positif dan negatif yang diambil dari GitHub untuk menghitung skor sentimen. Jika hasil dari metode ini tidak cukup kuat, digunakan **SentimentIntensityAnalyzer** sebagai pelengkap.

### **Model Klasifikasi yang Digunakan**  

### 1. **Naive Bayes (MultinomialNB)**  
- **Metode**:  
  Naive Bayes adalah algoritma klasifikasi berbasis **Teorema Bayes** dengan asumsi bahwa setiap fitur bersifat independen satu sama lain (naive assumption). **MultinomialNB** khususnya cocok digunakan untuk data dengan distribusi multinomial, seperti data teks yang dihitung berdasarkan frekuensi kemunculan kata.  

- **Cara Kerja**:  
  - Menghitung **probabilitas posterior** dari setiap kelas berdasarkan fitur yang diberikan.  
  - Memilih kelas dengan probabilitas tertinggi sebagai hasil prediksi.  
  - **MultinomialNB** sering digunakan dalam **Text Classification** seperti **Spam Detection** dan **Sentiment Analysis**.  

### 2. **Logistic Regression**  
- **Metode**:  
  Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memprediksi **probabilitas** kejadian kelas tertentu (biner atau multiklas). Model ini menggunakan fungsi **sigmoid** untuk membatasi output dalam rentang 0 dan 1.  

- **Cara Kerja**: 
  - Menerapkan **fungsi sigmoid** untuk menghasilkan probabilitas
  - Memetakan probabilitas ke kelas menggunakan **threshold** (biasanya 0.5).  
  - Memperbarui bobot menggunakan algoritma **Gradient Descent** untuk meminimalkan **Log Loss (Binary Cross-Entropy)**.  
  - Logistic Regression efektif untuk **Binary Classification** dan **Multiclass Classification (One-vs-Rest)**.  

### 3. **Random Forest Classifier**  
- **Metode**:  
  Random Forest adalah algoritma **ensemble** yang menggunakan **Multiple Decision Trees** untuk meningkatkan akurasi dan mengurangi overfitting. Setiap tree dilatih pada subset data dan subset fitur yang dipilih secara acak.  

- **Cara Kerja**:  
  - **Bootstrap Aggregating (Bagging)**:  
    - Membuat beberapa subset data dengan pengambilan sampel acak (dengan pengembalian) dari data latih.  
    - Melatih **Decision Tree** pada setiap subset data.  
  - **Feature Randomness**:  
    - Setiap tree hanya menggunakan subset acak dari fitur yang tersedia.  
    - Ini membantu dalam membuat tree yang berbeda-beda (diversifikasi).  
  - **Voting Ensemble**:  
    - Untuk **Classification**: Hasil prediksi ditentukan dengan **Voting Mayoritas** dari hasil prediksi semua tree.  
    - Untuk **Regression**: Hasil prediksi adalah **Rata-Rata** dari output semua tree.  
  - Random Forest dikenal kuat dalam **Handling Imbalanced Data**, **High-Dimensional Data**, dan **Reducing Overfitting**.  

---

# Evaluasi Model
### Metrik Evaluasi Model

Untuk mengevaluasi performa model klasifikasi, digunakan beberapa metrik yaitu **Accuracy**, **Precision**, **Recall**, dan **F1-Score**:

1. **Accuracy** : mengukur seberapa sering model membuat prediksi yang benar secara keseluruhan.
   
   * **Kelebihan**: Mudah dipahami dan digunakan.
   * **Kekurangan**: Tidak cocok jika dataset tidak seimbang, karena kelas mayoritas akan mendominasi nilai akurasi.

3. **Precision**: mengukur ketepatan prediksi positif, yaitu seberapa banyak prediksi positif yang benar dibandingkan dengan total prediksi positif.
   
   * **Kelebihan**: Berguna saat biaya **False Positive** tinggi.
   * **Kekurangan**: Tidak memperhitungkan **False Negative**, sehingga tidak cocok jika hasil negatif yang salah juga penting.

4. **Recall** (juga dikenal sebagai **Sensitivity** atau **True Positive Rate**) mengukur kemampuan model dalam menemukan semua contoh positif yang benar.
   
   * **Kelebihan**: Berguna saat biaya **False Negative** tinggi.
   * **Kekurangan**: Tidak mempertimbangkan **False Positive**, sehingga bisa tinggi meski banyak prediksi salah sebagai positif.

5. **F1-Score** adalah rata-rata harmonis antara **Precision** dan **Recall**, memberikan keseimbangan antara keduanya.

   * **Kelebihan**: Memberikan metrik yang lebih seimbang dalam kasus data tidak seimbang.
   * **Kekurangan**: Lebih sulit untuk diinterpretasikan dibanding **Accuracy**.

---

### **Evaluasi Model Machine Learning**  
1. **Model**: MultinomialNB
   - **Training Set** - Akurasi: 0.87, F1-Score: 0.85
   - **Testing Set** - Akurasi: 0.86, F1-Score: 0.84  

  **Classification Report**:  
  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | **0** | 1.00      | 0.01   | 0.01     | 189     |
  | **1** | 0.96      | 0.42   | 0.59     | 5014    |
  | **2** | 0.85      | 1.00   | 0.92     | 17927   |
  | **Accuracy**       |         |         | 0.86     | 23130   |
  | **Macro Avg**      | 0.94    | 0.47    | 0.51     | 23130   |
  | **Weighted Avg**   | 0.88    | 0.86    | 0.84     | 23130   |


2. **Model**: RandomForestClassifier
   - **Training Set** - Akurasi: 1.00, F1-Score: 1.00
   - **Testing Set** - Akurasi: 0.98, F1-Score: 0.98  

  **Classification Report**:  
  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | **0** | 0.88      | 0.64   | 0.74     | 189     |
  | **1** | 0.96      | 0.97   | 0.97     | 5014    |
  | **2** | 0.99      | 0.99   | 0.99     | 17927   |
  | **Accuracy**       |         |         | 0.98     | 23130   |
  | **Macro Avg**      | 0.94    | 0.87    | 0.90     | 23130   |
  | **Weighted Avg**   | 0.98    | 0.98    | 0.98     | 23130   |

3. **Model**: LogisticRegression
   - **Training Set** - Akurasi: 0.99, F1-Score: 0.99
   - **Testing Set** - Akurasi: 0.99, F1-Score: 0.99  

  **Classification Report**:  
  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | **0** | 0.94      | 0.63   | 0.76     | 189     |
  | **1** | 0.95      | 0.99   | 0.97     | 5014    |
  | **2** | 1.00      | 0.99   | 0.99     | 17927   |
  | **Accuracy**       |         |         | 0.99     | 23130   |
  | **Macro Avg**      | 0.96    | 0.87    | 0.91     | 23130   |
  | **Weighted Avg**   | 0.99    | 0.99    | 0.99     | 23130   |

---

## **Hasil dan Evaluasi**
- Distribusi sentimen menunjukkan bahwa sebagian besar ulasan bersifat **netral**, diikuti oleh **negatif** dan **positif**.
- **Random Forest Classifier** memberikan performa terbaik dengan **akurasi > 85%** pada data testing.
- **WordCloud** menunjukkan kata-kata yang sering muncul dalam ulasan positif dan negatif, memberikan wawasan mengenai isu-isu utama yang dihadapi pengguna.

---

## **Cara Menjalankan**
1. **Instalasi Dependensi**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Jalankan Notebook di Google Colab atau Jupyter Notebook**:
    - Upload dataset pada folder yang sesuai.
    - Sesuaikan path dataset pada script:
      ```python
      df = pd.read_csv('/Dataset/Scraping_.csv')
      ```
---

## **Kesimpulan**
- Sebagian besar ulasan bersifat netral dengan beberapa komentar positif dan negatif.
- Isu utama yang sering muncul dalam ulasan negatif adalah **bug**, **kinerja lambat**, dan **masalah login**.
- Ulasan positif umumnya memuji **fitur-fitur yang membantu** dan **kemudahan penggunaan**.
- Dengan analisis ini, pengembang dapat fokus pada perbaikan yang relevan sesuai dengan umpan balik pengguna.

---

## **Kontak**
- **Nama**: Elisa Ramadanti  
- **LinkedIn**: [Elisa Ramadanti](https://www.linkedin.com/in/elisa-ramadanti)  

---
