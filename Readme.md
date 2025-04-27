# Predictive Analysis - Klasifikasi Kualitas dalam Proses Tambang bijih besi - Aldo Fernando Supriyadi

# 1. Domain Proyek

Industri pertambangan berperan penting dalam menyediakan bahan baku yang diperlukan untuk berbagai sektor, termasuk energi, konstruksi, dan manufaktur. Salah satu tantangan terbesar dalam industri ini adalah memastikan bahwa bahan tambang yang diekstraksi memiliki kualitas yang sesuai dengan standar pasar. Kualitas bijih besi yang tidak konsisten dapat menyebabkan kerugian finansial, pemborosan sumber daya, dan menurunnya efisiensi operasional [1]. Oleh karena itu, penting bagi perusahaan pertambangan untuk dapat memprediksi kualitas bahan tambang yang dihasilkan secara lebih akurat dan efisien. Pada umumnya, penilaian kualitas dalam proses tambang dilakukan melalui pengujian laboratorium atau pengamatan visual yang memakan waktu [2]. Proses manual seperti ini sering kali tidak dapat memberikan hasil yang cepat, yang dapat menghambat pengambilan keputusan yang penting untuk optimasi proses penambangan [3].

# 2. Business Understanding
Pengembangan model prediksi kualitas dalam proses tambang memiliki potensi besar dalam memberikan manfaat nyata bagi berbagai pemangku kepentingan dalam sektor pertambangan. Penerapan sistem klasifikasi kualitas yang akurat dapat membantu perusahaan pertambangan dalam memisahkan bahan tambang berkualitas tinggi dari yang rendah, sehingga dapat meningkatkan efisiensi operasional dan mengurangi kerugian.

Manfaat utama dari model ini adalah peningkatan efisiensi proses penambangan dan optimalisasi penggunaan sumber daya, yang pada akhirnya berdampak pada peningkatan keuntungan dan daya saing perusahaan. Dengan informasi prediktif yang dihasilkan dari model, perusahaan dapat menentukan strategi produksi dan distribusi yang lebih baik.

## 2.1 Problem Statements
Berdasarkan tujuan yang telah dipaparkan, berikut adalah masalah utama yang ingin diselesaikan dalam proyek ini:
1. **Bagaimana membangun model machine learning** yang dapat memprediksi kualitas bijih besi berdasarkan variabel seperti % Iron Feed, % Silica Feed, pH pulp bijih, dan parameter lainnya.
2. **Bagaimana mengukur kinerja model prediktif** dengan menggunakan metrik evaluasi yang tepat, seperti akurasi dan kesalahan prediksi (RMSE dan MAE), untuk memastikan model dapat digunakan dalam praktik di lapangan.
3. **Bagaimana memanfaatkan model ini untuk meningkatkan efisiensi proses flotasi**, mengurangi pemborosan material, dan mempercepat pengambilan keputusan terkait kualitas bijih.


## 2.2 Goals
Tujuan dari proyek ini adalah untuk:
1. **Membangun model prediktif berbasis machine learning** yang dapat memproses data historis tentang proses flotasi dan memprediksi kualitas bijih besi yang dihasilkan.
2. **Mengevaluasi performa model** menggunakan metrik yang relevan seperti R², RMSE, dan MAE untuk memastikan akurasi dan efektivitasnya dalam memprediksi kualitas bijih.
3. **Menyediakan solusi berbasis data** yang dapat digunakan untuk mempercepat proses analisis dan meningkatkan efisiensi operasional di sektor pertambangan.

## 2.3 Solution Statements

Beberapa langkah yang akan diambil untuk mencapai tujuan proyek ini adalah:

1. **Eksplorasi Data (EDA):** Melakukan analisis awal pada dataset untuk memahami distribusi variabel dan hubungan antar fitur serta target, untuk membantu memilih teknik machine learning yang paling tepat.
2. **Pembersihan dan Preprocessing Data:** Menangani missing values, duplikasi data, dan melakukan normalisasi fitur agar model dapat berfungsi dengan baik. Variabel seperti % Iron Feed dan % Silica Feed akan menjadi fitur utama dalam model prediktif.
3. **Pengembangan Model:** Menggunakan algoritma machine learning seperti **Linear Regression**, **Random Forest**, dan **K-Nearest Neighbors (KNN)** untuk membangun model prediksi dan mengevaluasi kinerjanya.
4. **Evaluasi Model:** Menggunakan metrik evaluasi seperti **R²**, **RMSE**, dan **MAE** untuk mengevaluasi akurasi model dan membandingkan performa berbagai model.

# 3. Data Understanding
## 3.1 EDA - Deskripsi Variabel
### **Informasi Datasets**

| Jenis   | Keterangan |
| ------- | ---------- |
| **Judul** | _Quality Prediction in a Mining Process_ |
| **Sumber data** | [Mining Process Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process/data) |
| **Pembuat** | [Eduardo Magalhães ⚡](https://www.kaggle.com/edumagalhaes) |
| **lisensi** | Other (specified in description) |
| **visibilitas** | Publik |
| **Tags** | _Mining, Data Science, Machine Learning, Flotation, Quality Prediction, Process Optimization_ |

Dataset ini memiliki total **737453 baris** data dan **24 kolom** yang berisi berbagai variabel terkait dengan proses flotasi dalam industri pertambangan. Setiap baris mewakili satu entri waktu dengan pengukuran berbagai parameter yang dilakukan dalam proses flotasi untuk memisahkan bijih besi dan silika. 

**Data ini didapatkan dari perusahaan pertambangan yang disediakan secara publik di Kaggle.**

### **Contoh Data**

| date | % Iron Feed | % Silica Feed | Starch Flow | Amina Flow | Ore Pulp Flow | Ore Pulp pH | Ore Pulp Density | Flotation Column 01 Air Flow | Flotation Column 02 Air Flow | .  .. | % Iron Concentrate | % Silica Concentrate |
|------|-------------|---------------|-------------|------------|---------------|-------------|------------------|-----------------------------|-----------------------------|-----|---------------------|-----------------------|
| 2017-03-10 01:00:00 | 55.2 | 16.98 | 3019.53 | 557.434 | 395.713 | 10.0664 | 1.74 | 249.214 | 253.235 | ... | 66.91 | 1.31 |
| 2017-03-10 01:00:00 | 55.2 | 16.98 | 3024.41 | 563.965 | 397.383 | 10.0672 | 1.74 | 249.719 | 250.532 | ... | 66.91 | 1.31 |
| 2017-03-10 01:00:00 | 55.2 | 16.98 | 3043.46 | 568.054 | 399.668 | 10.068 | 1.74 | 249.741 | 247.874 | ... | 66.91 | 1.31 |
| 2017-03-10 01:00:00 | 55.2 | 16.98 | 3047.36 | 568.665 | 397.939 | 10.0689 | 1.74 | 249.917 | 254.487 | ... | 66.91 | 1.31 |

### **Kondisi Dataset**
- Terjadi Ketidaksesuaian tipe data pada dataset ini yaitu masih berupa tipe data objek semua
- Terdapat baris-baris data yang duplikasi sebanyak 1171 data di dalam dataset Mining Process
- Memiliki distribusi yang tidak normal maka harus melakukan Penanganan Outlier pada kolom % Iron Feed, %Silica Feed, Starch Flow, Amina Flow, Ore Pulp Flow, Ore Pulp pH, Ore Pulp Density, % Iron Concentrate, dan % Silica Concentrate

### **Deskripsi Kolom:**
- **date**: Tanggal dan waktu pengukuran.
- **% Iron Feed**: Persentase kandungan besi dalam umpan bijih.
- **% Silica Feed**: Persentase kandungan silika dalam umpan bijih.
- **Starch Flow**: Aliran bahan pati yang digunakan dalam proses flotasi untuk meningkatkan selektivitas pemisahan.
- **Amina Flow**: Aliran bahan kimia amina yang digunakan dalam proses flotasi sebagai kolektor.
- **Ore Pulp Flow**: Aliran pulp bijih yang digunakan dalam kolom flotasi untuk pemisahan bijih.
- **Ore Pulp pH**: Nilai pH dari pulp bijih, yang mempengaruhi efektivitas proses flotasi.
- **Ore Pulp Density**: Kepadatan pulp bijih yang mempengaruhi efisiensi pemisahan dalam proses flotasi.
- **Flotation Column 01 Air Flow**: Aliran udara yang digunakan pada kolom flotasi 01 untuk membentuk busa yang diperlukan dalam proses flotasi.
- **Flotation Column 02 Air Flow**: Aliran udara yang digunakan pada kolom flotasi 02. Setiap kolom memiliki tingkat aliran udara yang berbeda untuk mengoptimalkan proses flotasi.
- **Flotation Column 03 Air Flow**: Aliran udara pada kolom flotasi 03, disesuaikan untuk meningkatkan selektivitas pemisahan sesuai dengan jenis bijih yang diproses.
- **Flotation Column 04 Air Flow**: Aliran udara yang digunakan pada kolom flotasi 04. Variasi aliran udara di setiap kolom membantu mengatur ketinggian busa dan pemisahan bijih.
- **Flotation Column 05 Air Flow**: Aliran udara pada kolom flotasi 05, berfungsi untuk mengatur ketinggian busa sesuai dengan karakteristik bijih yang diproses.
- **Flotation Column 06 Air Flow**: Aliran udara pada kolom flotasi 06, berfungsi untuk menghasilkan busa yang diperlukan dalam pemisahan bijih.
- **Flotation Column 07 Air Flow**: Aliran udara pada kolom flotasi 07, dengan tingkat aliran udara yang dapat disesuaikan untuk meningkatkan pemisahan bijih.
- **Flotation Column 01 Level**: Tingkat ketinggian busa yang terbentuk pada kolom flotasi 01, yang mencerminkan efektivitas pemisahan bijih.
- **Flotation Column 02 Level**: Tingkat ketinggian busa di kolom flotasi 02, yang mengindikasikan seberapa banyak bijih yang terpisah dalam kolom tersebut.
- **Flotation Column 03 Level**: Tingkat ketinggian busa di kolom flotasi 03. Busa yang lebih tinggi menunjukkan efektivitas yang lebih baik dalam proses flotasi.
- **Flotation Column 04 Level**: Tingkat ketinggian busa di kolom flotasi 04. Level busa dapat menunjukkan seberapa efektif udara yang dialirkan dalam kolom tersebut.
- **Flotation Column 05 Level**: Tingkat ketinggian busa di kolom flotasi 05, yang menunjukkan seberapa efisien busa terbentuk untuk pemisahan bijih.
- **Flotation Column 06 Level**: Tingkat ketinggian busa di kolom flotasi 06, yang menggambarkan keberhasilan proses pemisahan bijih.
- **Flotation Column 07 Level**: Tingkat ketinggian busa di kolom flotasi 07, yang berhubungan dengan keberhasilan kolom tersebut dalam proses flotasi.
- **% Iron Concentrate**: Persentase konsentrasi besi pada hasil flotasi, yang mengindikasikan keberhasilan pemisahan bijih besi dari kotoran.
- **% Silica Concentrate**: Persentase konsentrasi silika pada hasil flotasi, yang menunjukkan tingkat kotoran silika yang tersisa setelah proses flotasi.

## 3.2 EDA - Pengecekan dan Penanganan Missing Value, Data Duplikat dan Tipe data

Pada tahap ini, dilakukan pengecekan untuk memastikan bahwa tidak ada nilai yang hilang (`missing value`) atau data yang berulang (`duplicated data`). Pengecekan tipe data juga dilakukan untuk memastikan bahwa semua kolom memiliki tipe data yang sesuai dengan jenis informasi yang terkandung dalam setiap kolom.

### **Pengecekan Missing Values dan Duplikasi Data**

Pengecekan dilakukan untuk melihat apakah ada nilai yang hilang dalam dataset. Semua kolom dalam dataset memiliki nilai yang lengkap, yang berarti tidak ditemukan missing values pada dataset ini.

Selain itu, dilakukan pengecekan untuk data duplikat. Hasil pengecekan menunjukkan bahwa ada **1171 baris data yang terduplikasi**. Data yang terduplikasi perlu dihapus untuk memastikan keakuratan analisis lebih lanjut.

Baris-baris data yang terduplikasi dihapus untuk membersihkan dataset.

### **Pengecekan Tipe Data**

Saat memeriksa tipe data dataset, ditemukan bahwa semua kolom bertipe `object`.
 Hal ini menunjukkan bahwa beberapa kolom yang seharusnya bertipe numerik perlu diubah menjadi tipe data yang tepat.

### **Perubahan Tipe Data:**
- Kolom `date` diubah menjadi tipe `datetime` agar dapat dianalisis dalam konteks waktu.
- Kolom-kolom lainnya yang berisi data numerik, seperti aliran udara dan level busa, diubah menjadi tipe data `float`.

Selain itu, karakter seperti koma (`,`) diubah menjadi titik (`.`) untuk memastikan data numerik dapat diproses dengan benar.

### **Menghitung Rata-Rata Air Flow dan Froth Level**

Dalam analisis ini, kolom-kolom yang berkaitan dengan aliran udara (Air Flow) dan level busa (Froth Level) dihitung rata-ratanya untuk mendapatkan informasi yang lebih berguna. Kolom-kolom tersebut mencakup:
- Air Flow: Menghitung rata-rata aliran udara dari kolom-kolom yang tersedia.
- Froth Level: Menghitung rata-rata level busa dari kolom-kolom yang tersedia.

Kolom baru yang dihasilkan, yaitu `Average Air Flow` dan `Average Froth Level`, ditambahkan ke dalam dataset dan ditempatkan pada posisi yang sesuai.

Setelah itu, kolom-kolom individu yang berkaitan dengan Air Flow dan Froth Level dihapus, karena informasi yang relevan sudah digabungkan dalam kolom rata-rata.

 
## 3.3 EDA - Pengecekan dan Penanganan Outlier

Outlier dalam dataset ini akan diperiksa menggunakan **boxplot**, yang dapat dilihat pada gambar di bawah. Boxplot ini menggambarkan distribusi data dari kolom-kolom numerik dan menunjukkan adanya nilai ekstrim yang berpotensi mengganggu analisis lebih lanjut.


![image](assets\Gambar1.png)
Gambar 1: Pengecekan Outlier

- **Insight yang didapatkan dari Gambar boxplot pada pengecekan outlier:**
  - Berdasarkan boxplot yang ditampilkan, terlihat bahwa sebagian besar kolom numerik menunjukkan adanya outlier yang ditandai dengan simbol bulat.
  - Kolom-kolom yang menunjukkan outlier tersebut perlu penanganan lebih lanjut agar analisis tidak terganggu oleh data yang tidak normal.
  
- **Penanganan Outlier:**
  Untuk menangani outlier, dilakukan penghapusan berdasarkan metode **Interquartile Range (IQR)**. IQR dihitung dengan mengurangkan **kuartil ketiga (Q3)** dari **kuartil pertama (Q1)**. Dengan rumus berikut:

  $$ IQR = Q_3 - Q_1 $$

  - **Q1 (kuartil pertama)** adalah nilai yang berada pada persentil ke-25.
  - **Q3 (kuartil ketiga)** adalah nilai yang berada pada persentil ke-75.
  
  Data yang berada di luar rentang **(Q1 - 1.5 * IQR)** hingga **(Q3 + 1.5 * IQR)** dianggap sebagai outlier dan akan dihapus.

  Setelah melakukan penanganan outlier menggunakan metode IQR, dataset yang awalnya berjumlah **737453** baris, menjadi **sejumlah baris yang lebih sedikit** karena data yang terduplikasi atau outlier telah dihapus.

- Setelah menangani outlier jumlah data berubah yang awalnya 737453 data menjadi 577178 data.

## 3.4 EDA - Univariate Analysis

### 3.4.1 Numerical Column

Untuk analisis univariat pada kolom numerik, digunakan dua jenis visualisasi yaitu **Histogram** dan **Violin Plot** untuk mengeksplorasi distribusi data dan menemukan pola yang ada pada setiap fitur.

- **Histogram** digunakan untuk melihat distribusi frekuensi dari kolom numerik, apakah terdistribusi normal, skewed, atau distribusi lainnya.
 
Gambar 2: Histogram Persebaran Ore Pulp pH dan Ore Pulp Density

- **Insight yang didapatkan dari Gambar 2:**
  - **Ore Pulp pH** memiliki distribusi yang normal, mayoritas titik data berada di kisaran pH antara **9.0 hingga 10.0**.
  - **Ore Pulp Density** menunjukkan bahwa titik data mayoritas berada di kelompok **1.52 hingga 1.75**.

- **Violin Plot** digunakan untuk menggambarkan distribusi dan variabilitas data numerik. Di bawah ini adalah visualisasi rata-rata **Air Flow** dan **Froth Level** pada **Flotation Column 1-7**.
 
Gambar 3: Violin Plot untuk Rata-rata Air Flow dan Rata-rata Froth Level di Column 1 - 7

- **Insight yang didapatkan dari Gambar 3:**
  - **Air Flow** memiliki distribusi yang fluktuatif, dengan kisaran antara **210 hingga 330 Nm3/h**.
  - **Rata-rata Froth Level** memiliki distribusi yang lebih merata, dengan kisaran antara **400 hingga 710 mm**.

Distribusi ini menunjukkan variasi yang signifikan dalam nilai **Air Flow**, yang dapat mempengaruhi proses flotasi, sedangkan distribusi **Froth Level** cenderung lebih stabil dan menunjukkan nilai rata-rata yang lebih konsisten.

## 3.5 EDA - Bivariate Analysis

### 3.5.1 Numeric Columns

Pada tahap ini, analisis dilakukan untuk memahami hubungan antara fitur numerik yang ada dalam dataset. Fokus utama adalah untuk mengeksplorasi korelasi antara variabel-variabel penting yang dapat memberikan wawasan lebih lanjut tentang hubungan antar fitur.

#### Scatter Plot: % Iron Feed vs % Iron Concentrate

Untuk memulai, kita mengeksplorasi hubungan antara **% Iron Feed** sebagai input dan **% Iron Concentrate** sebagai output dengan menggunakan scatter plot. Visualisasi ini bertujuan untuk melihat apakah ada hubungan linier antara keduanya.


**Gambar 4**: Scatter Plot % Iron Feed vs % Iron Concentrate

- **Insight yang didapatkan dari Gambar 4:**
  - Dapat dilihat bahwa terdapat korelasi positif yang sangat kecil antara **% Iron Feed** dan **% Iron Concentrate**. 
  - Secara keseluruhan, lebih banyak **Zat Besi dalam pakan** akan menghasilkan lebih banyak **Zat Besi dalam konsentrat**, meskipun hubungan ini tidak terlalu kuat.

#### Heatmap: Korelasi antara % Iron Concentrate dan % Silica Concentrate

Selanjutnya, kami menganalisis korelasi antara **% Iron Concentrate** dan **% Silica Concentrate**. Ini akan membantu kita memahami hubungan antara konsentrasi kedua elemen tersebut dalam hasil tambang.


**Gambar 5**: Heatmap Korelasi antara % Iron Concentrate dan % Silica Concentrate

- **Insight yang didapatkan dari Gambar 5:**
  - Terlihat bahwa terdapat korelasi negatif yang cukup kuat antara **% Iron Concentrate** dan **% Silica Concentrate**. Hal ini berarti bahwa semakin tinggi **% Besi** dalam output, semakin sedikit **% Silika**, dan sebaliknya.
  - Ini menunjukkan bahwa proses penambangan dapat memisahkan kedua elemen ini dalam tingkat yang cukup baik, dimana **konsentrasi besi lebih tinggi** mengurangi **konsentrasi silika**, dan ini dapat digunakan untuk optimasi proses pengolahan.

## 3.6 EDA - Multivariate Analysis

### 3.6.1 Numeric Columns

Pada tahap ini, dilakukan analisis untuk melihat hubungan antara beberapa fitur numerik secara bersamaan untuk memahami bagaimana mereka saling berinteraksi. Analisis ini membantu dalam mengeksplorasi korelasi antara berbagai fitur numerik dan bagaimana mereka mungkin memengaruhi hasil akhir.

#### Heatmap Korelasi Antara Fitur Numerik

Kami memulai dengan mengeksplorasi korelasi antar fitur numerik yang ada di dalam dataset menggunakan **heatmap korelasi**. Visualisasi ini memberikan gambaran umum tentang seberapa kuat hubungan antara fitur-fitur numerik.

**Gambar 6**: Heatmap Korelasi antara Fitur Numerik

- **Insight yang didapatkan dari Gambar 6:**
  - **Korelasi Positif Terkuat** ditemukan antara **Ore Pulp Density** dan **Amina Flow**. Ini menunjukkan bahwa semakin tinggi kepadatan pulp ore, semakin tinggi aliran Amina yang digunakan dalam proses tersebut.
  - **Korelasi Negatif Terkuat** ditemukan antara **% Iron Concentrate** dan **% Silica Concentrate**. Hal ini mengindikasikan bahwa peningkatan konsentrasi besi dalam output berhubungan dengan penurunan konsentrasi silika, dan sebaliknya.
  - **% Iron Feed** dan **% Silica Feed** juga menunjukkan hubungan korelasi negatif. Ini berarti bahwa lebih banyak pakan besi dapat menghasilkan lebih sedikit pakan silika dalam proses.
  - **Amina Flow**, **Ore Pulp pH**, dan **Air Flow** memiliki korelasi yang lebih tinggi dengan **% Silica Concentrate** jika dibandingkan dengan variabel lainnya (termasuk **% Iron Concentrate**).

# 4. Data Preparation
Pada tahap data preparation, setelah memahami kondisi data maka dilakukanlah preprocessing agar data tersebut dapat dilatih oleh model. Tahapannya dapat diikuti seperti dibawah:

## 4.1 Mengidentifikasi Fitur dan Label

Tahap selanjutnya adalah mengidentifikasi fitur dan label dari dataset. Fitur (*features*) adalah variabel yang digunakan sebagai input untuk memprediksi target, sedangkan label (*target*) adalah variabel yang ingin diprediksi.

Pada kasus ini:

- **Fitur**:  
  - `% Iron Feed`  
  - `% Silica Feed`  
  - `Starch Flow`  
  - `Amina Flow`  
  - `Ore Pulp Flow`  
  - `Ore Pulp pH`  
  - `Ore Pulp Density`  
  - `Average Air Flow`  
  - `Average Froth Level`  
  - `% Iron Concentrate`  
  - `% Silica Concentrate`

- **Label**:  
  - `% Silica Feed`, karena merupakan target untuk diprediksi.

## 4.2 Membagi Data
Tahap berikutnya adalah membagi data menjadi data latih dan data uji dengan rasio 80:20. Pembagian ini penting untuk memastikan bahwa model tidak hanya belajar dari keseluruhan data tetapi juga diuji pada data yang belum pernah dilihat sebelumnya, sehingga performanya dapat dievaluasi secara objektif.

Jumlah data awal yang tersedia adalah 737453 baris. Setelah dilakukan pembagian data:
  - Data latih (train set): 461742 data (80%)
  - Data uji (test set): 115436 data (20%)


# 5. Modeling

Pada proyek ini, tiga algoritma machine learning yang digunakan untuk memprediksi nilai % Silica Feed berdasarkan fitur-fitur yang tersedia di dataset adalah:

## 5.1 Linear Regression
Linear Regression adalah algoritma yang digunakan untuk memprediksi hubungan linear antara variabel input (fitur) dengan variabel output (target). Dalam konteks ini, Linear Regression digunakan untuk memprediksi nilai % Silica Feed berdasarkan fitur-fitur lainnya.

Model Linear Regression diterapkan menggunakan kode berikut:

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Linear Regression -> R²: {r2_lr:.3f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
```


### Hasil Evaluasi:
- **R² (R-squared)**: 0.946
- **RMSE (Root Mean Squared Error)**: 1.54
- **MAE (Mean Absolute Error)**: 1.32

### Penjelasan Hasil:
- **R² (R-squared)**: Nilai **R²** sebesar **0.946** menunjukkan bahwa model Linear Regression ini dapat menjelaskan sekitar 94.6% variansi dalam data target (% Silica Feed). Ini berarti model ini cukup baik dalam menangkap pola data.
  
- **RMSE (Root Mean Squared Error)**: Nilai **RMSE** sebesar **1.54** menunjukkan bahwa prediksi model rata-rata menyimpang sekitar 1.54 unit dari nilai yang sebenarnya pada data uji.

- **MAE (Mean Absolute Error)**: Nilai **MAE** sebesar **1.32** menunjukkan rata-rata kesalahan absolut model dalam memprediksi nilai **% Silica Feed** adalah 1.32 unit.

### Kelebihan dari Hasil Model:
- Dengan **R²** yang tinggi, model Linear Regression memberikan indikasi yang baik bahwa model ini sangat sesuai untuk data yang memiliki hubungan linear.
- Nilai **RMSE** dan **MAE** yang relatif rendah menunjukkan bahwa model ini memiliki prediksi yang cukup akurat.

### Kekurangan dari Model:
- **Linear Regression** dapat dipengaruhi oleh **outliers** (nilai ekstrim). Oleh karena itu, untuk dataset dengan banyak data ekstrim, hasil model ini mungkin tidak seakurat yang diharapkan.
- Model ini mengasumsikan bahwa hubungan antara variabel input dan output bersifat **linear**. Jika hubungan data lebih kompleks atau non-linear, model ini mungkin tidak dapat menangkap pola yang ada.

## 5.2 Random Forest


Model **Random Forest** digunakan untuk memprediksi nilai **% Silica Feed** berdasarkan fitur-fitur lainnya pada dataset. Berikut adalah hasil evaluasi model setelah melakukan pelatihan dan prediksi:

### Hasil Evaluasi:
- **R² (R-squared)**: 1.000
- **RMSE (Root Mean Squared Error)**: 0.03
- **MAE (Mean Absolute Error)**: 0.00

### Penjelasan Hasil:
- **R² (R-squared)**: Nilai **R²** sebesar **1.000** menunjukkan bahwa model Random Forest ini dapat menjelaskan 100% variansi dalam data target (% Silica Feed). Ini menunjukkan bahwa model ini sangat baik dalam menangkap pola data dan dapat memberikan hasil prediksi yang sangat akurat.

- **RMSE (Root Mean Squared Error)**: Nilai **RMSE** sebesar **0.03** menunjukkan bahwa prediksi model rata-rata menyimpang sangat sedikit, yaitu hanya sekitar 0.03 unit dari nilai yang sebenarnya pada data uji. Ini adalah hasil yang sangat baik, menunjukkan ketepatan prediksi yang sangat tinggi.

- **MAE (Mean Absolute Error)**: Nilai **MAE** sebesar **0.00** menunjukkan bahwa model ini sangat akurat dalam memprediksi nilai **% Silica Feed**, dengan kesalahan rata-rata yang hampir tidak ada.

### Kelebihan dari Hasil Model:
- **R²** yang sangat tinggi (**1.000**) menandakan model Random Forest ini mampu memberikan prediksi yang sangat akurat, bahkan dapat menangkap hampir semua pola dalam data.
- **RMSE** dan **MAE** yang sangat rendah menunjukkan bahwa model ini memberikan prediksi yang sangat tepat tanpa banyak kesalahan.

### Kekurangan dari Model:
- **Random Forest** cenderung lebih lambat dalam proses pelatihan pada dataset besar, terutama dengan banyak pohon keputusan, karena banyaknya model yang perlu dipelajari.
- Meskipun akurat, model **Random Forest** sering kali sulit untuk diinterpretasikan karena banyaknya pohon keputusan yang digunakan, membuatnya lebih sebagai model "black-box" dibandingkan model yang lebih transparan.
- **Overfitting** bisa terjadi pada data dengan banyak noise atau variabel yang tidak relevan, meskipun teknik **bagging** yang digunakan oleh Random Forest membantu meminimalkan hal ini.

## 5.3 K-Nearest Neighbors (KNN)

Model **K-Nearest Neighbors (KNN)** digunakan untuk memprediksi nilai **% Silica Feed** berdasarkan fitur-fitur lainnya pada dataset. Berikut adalah hasil evaluasi model setelah melakukan pelatihan dan prediksi:

### Hasil Evaluasi:
- **R² (R-squared)**: 0.669
- **RMSE (Root Mean Squared Error)**: 3.82
- **MAE (Mean Absolute Error)**: 2.59

### Penjelasan Hasil:
- **R² (R-squared)**: Nilai **R²** sebesar **0.669** menunjukkan bahwa model KNN dapat menjelaskan sekitar 66.9% variansi dalam data target (% Silica Feed). Nilai ini lebih rendah dibandingkan dengan model lain seperti **Random Forest**, yang menunjukkan bahwa KNN tidak seakurat model lainnya dalam menangkap pola data.
  
- **RMSE (Root Mean Squared Error)**: Nilai **RMSE** sebesar **3.82** menunjukkan bahwa prediksi model memiliki deviasi rata-rata yang lebih besar. Kesalahan prediksi model ini lebih signifikan dibandingkan dengan model **Linear Regression** atau **Random Forest**, yang memiliki RMSE yang lebih rendah.

- **MAE (Mean Absolute Error)**: Nilai **MAE** sebesar **2.59** menunjukkan bahwa kesalahan rata-rata prediksi model KNN lebih besar dibandingkan dengan model lainnya. Ini menunjukkan bahwa model ini kurang efektif dalam memprediksi nilai target secara presisi.

### Kelebihan dari Hasil Model:
- **KNN** merupakan algoritma yang sederhana dan mudah dipahami, dan dalam beberapa kasus, dapat bekerja dengan baik tanpa banyak tuning parameter.
- Dapat menangani data yang tidak linier dengan baik, sehingga cocok untuk data dengan hubungan yang lebih kompleks.

### Kekurangan dari Model:
- **R²**, **RMSE**, dan **MAE** yang rendah menunjukkan bahwa model ini tidak terlalu cocok untuk dataset ini, dengan performa yang lebih buruk dibandingkan **Random Forest** atau **Linear Regression**.
- **KNN** cenderung memerlukan waktu lebih lama dalam prediksi jika dataset sangat besar, karena perlu menghitung jarak untuk setiap titik data.
- Sensitif terhadap **outlier** dan **noise**, yang dapat sangat mempengaruhi kinerja model KNN.
- Memilih nilai **k** yang tepat sangat penting dan memerlukan eksperimen, karena pemilihan yang salah dapat mempengaruhi akurasi model secara signifikan.

# 6. Evaluasi Model

Pada tahap evaluasi model, digunakan beberapa metrik evaluasi yang umum dalam permasalahan regresi untuk menilai performa model. Metrik-metrik yang digunakan meliputi:

## 6.1 Evaluasi Visualisasi Model

Pada evaluasi model kali ini, digunakan visualisasi untuk membandingkan hasil prediksi dengan data aktual pada setiap model yang diuji. Hasil dari **Linear Regression**, **Random Forest**, dan **K-Nearest Neighbors (KNN)** ditampilkan dengan menggunakan scatter plot.

### Visualisasi Hasil Model
- Scatter plot yang menunjukkan hubungan antara **Actual % Silica Feed** dan **Predicted % Silica Feed** pada setiap model.

- Insight yang didapatkan dari visualisasi:
  - **Linear Regression** dan **Random Forest** menunjukkan hasil prediksi yang sangat dekat dengan data aktual, dengan pola yang hampir sejajar dengan garis diagonal (red line).
  - **KNN** menunjukkan pola yang lebih tersebar dengan lebih banyak variasi dalam prediksi dibandingkan kedua model lainnya.

## 6.2 Testing pada Model

Untuk lebih mendalami performa model, berikut adalah hasil prediksi model pada satu sampel data untuk setiap model yang diuji.

### Sample Testing:
- **Actual % Silica Feed**: 25.31
- **Linear Regression Predicted**: 24.59
- **Random Forest Predicted**: 25.31
- **KNN Predicted**: 25.26

#### Keterangan:
- **Linear Regression** mendekati nilai aktual dengan selisih yang kecil.
- **Random Forest** memprediksi nilai dengan tepat, sesuai dengan nilai aktual.
- **KNN** menghasilkan prediksi yang sangat dekat dengan nilai aktual, meskipun sedikit berbeda.

```python
sample = X_test.iloc[[1]]
actual_Silica = y_test.iloc[1]

pred_lr = lr_model.predict(sample)[0]
pred_rf = rf_model.predict(sample)[0]
pred_knn = knn_model.predict(sample)[0]

print(f"% Silica Feed Actual: {actual_Silica:.2f}")
print(f"Linear Regression   : {pred_lr:.2f}")
print(f"Random Forest       : {pred_rf:.2f}")
print(f"KNN                 : {pred_knn:.2f}")
```

# Referensi
[1] V. N. Oparin, "Mining Industry and Sustainable Development: Time for Change," Journal of Mining Science, vol. 52, no. 6, pp. 1430-1441, Dec. 2016.
[2] F. Arif, N. Suryana, and B. Hussin, "A Data Mining Approach for Developing Quality Prediction Model in Multi-Stage Manufacturing," International Journal of Computer Applications, vol. 77, no. 15, pp. 1-7, Sep. 2013.
[3] D. Lieber, M. Stolpe, B. Konrad, J. Deuse, and K. Morik, "Machine Learning and Deep Learning Based Predictive Quality in Interlinked Manufacturing Processes," Procedia CIRP, vol. 79, pp. 302-307, 2019.

