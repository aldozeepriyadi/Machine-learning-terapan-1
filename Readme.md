# Predictive Analysis - Klasifikasi Kualitas dalam Proses Tambang bijih besi - Aldo Fernando Supriyadi

# Domain Proyek

Industri pertambangan berperan penting dalam menyediakan bahan baku yang diperlukan untuk berbagai sektor, termasuk energi, konstruksi, dan manufaktur. Salah satu tantangan terbesar dalam industri ini adalah memastikan bahwa bahan tambang yang diekstraksi memiliki kualitas yang sesuai dengan standar pasar. Kualitas bijih besi yang tidak konsisten dapat menyebabkan kerugian finansial, pemborosan sumber daya, dan menurunnya efisiensi operasional [1]. Oleh karena itu, penting bagi perusahaan pertambangan untuk dapat memprediksi kualitas bahan tambang yang dihasilkan secara lebih akurat dan efisien. Pada umumnya, penilaian kualitas dalam proses tambang dilakukan melalui pengujian laboratorium atau pengamatan visual yang memakan waktu [2]. Proses manual seperti ini sering kali tidak dapat memberikan hasil yang cepat, yang dapat menghambat pengambilan keputusan yang penting untuk optimasi proses penambangan [3].

# Business Understanding
Pengembangan model prediksi kualitas dalam proses tambang memiliki potensi besar dalam memberikan manfaat nyata bagi berbagai pemangku kepentingan dalam sektor pertambangan. Penerapan sistem klasifikasi kualitas yang akurat dapat membantu perusahaan pertambangan dalam memisahkan bahan tambang berkualitas tinggi dari yang rendah, sehingga dapat meningkatkan efisiensi operasional dan mengurangi kerugian.

Manfaat utama dari model ini adalah peningkatan efisiensi proses penambangan dan optimalisasi penggunaan sumber daya, yang pada akhirnya berdampak pada peningkatan keuntungan dan daya saing perusahaan. Dengan informasi prediktif yang dihasilkan dari model, perusahaan dapat menentukan strategi produksi dan distribusi yang lebih baik.

## Problem Statements
Berdasarkan tujuan yang telah dipaparkan, berikut adalah masalah utama yang ingin diselesaikan dalam proyek ini:
1. **Bagaimana membangun model machine learning** yang dapat memprediksi kualitas bijih besi berdasarkan variabel seperti % Iron Feed, % Silica Feed, pH pulp bijih, dan parameter lainnya.
2. **Bagaimana mengukur kinerja model prediktif** dengan menggunakan metrik evaluasi yang tepat, seperti akurasi dan kesalahan prediksi (RMSE dan MAE), untuk memastikan model dapat digunakan dalam praktik di lapangan.
3. **Bagaimana memanfaatkan model ini untuk meningkatkan efisiensi proses flotasi**, mengurangi pemborosan material, dan mempercepat pengambilan keputusan terkait kualitas bijih.


## Goals
Tujuan dari proyek ini adalah untuk:
1. **Membangun model prediktif berbasis machine learning** yang dapat memproses data historis tentang proses flotasi dan memprediksi kualitas bijih besi yang dihasilkan.
2. **Mengevaluasi performa model** menggunakan metrik yang relevan seperti R², RMSE, dan MAE untuk memastikan akurasi dan efektivitasnya dalam memprediksi kualitas bijih.
3. **Menyediakan solusi berbasis data** yang dapat digunakan untuk mempercepat proses analisis dan meningkatkan efisiensi operasional di sektor pertambangan.

## Solution Statements

Beberapa langkah yang akan diambil untuk mencapai tujuan proyek ini adalah:

1. **Eksplorasi Data (EDA):** Melakukan analisis awal pada dataset untuk memahami distribusi variabel dan hubungan antar fitur serta target, untuk membantu memilih teknik machine learning yang paling tepat.
2. **Pembersihan dan Preprocessing Data:** Menangani missing values, duplikasi data, dan melakukan normalisasi fitur agar model dapat berfungsi dengan baik. Variabel seperti % Iron Feed dan % Silica Feed akan menjadi fitur utama dalam model prediktif.
3. **Pengembangan Model:** Menggunakan algoritma machine learning seperti **Linear Regression**, **Random Forest**, dan **K-Nearest Neighbors (KNN)** untuk membangun model prediksi dan mengevaluasi kinerjanya.
4. **Evaluasi Model:** Menggunakan metrik evaluasi seperti **R²**, **RMSE**, dan **MAE** untuk mengevaluasi akurasi model dan membandingkan performa berbagai model.

# Data Understanding
## Deskripsi Variabel
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

## Pengecekan Missing Value, Data Duplikat dan Tipe data

Pada tahap ini, dilakukan pengecekan untuk memastikan bahwa tidak ada nilai yang hilang (`missing value`) atau data yang berulang (`duplicated data`). Pengecekan tipe data juga dilakukan untuk memastikan bahwa semua kolom memiliki tipe data yang sesuai dengan jenis informasi yang terkandung dalam setiap kolom.

### **Pengecekan Missing Values dan Duplikasi Data**

Pengecekan dilakukan untuk melihat apakah ada nilai yang hilang dalam dataset. Semua kolom dalam dataset memiliki nilai yang lengkap, yang berarti tidak ditemukan missing values pada dataset ini.

Selain itu, dilakukan pengecekan untuk data duplikat. Hasil pengecekan menunjukkan bahwa ada **1171 baris data yang terduplikasi**. Data yang terduplikasi perlu dihapus untuk memastikan keakuratan analisis lebih lanjut.

Baris-baris data yang terduplikasi dihapus untuk membersihkan dataset.

### **Pengecekan Tipe Data**

Saat memeriksa tipe data dataset, ditemukan bahwa semua kolom bertipe `object`.
 Hal ini menunjukkan bahwa beberapa kolom yang seharusnya bertipe numerik perlu diubah menjadi tipe data yang tepat.
## **Perubahan Tipe Data:**
- Kolom `date` diubah menjadi tipe `datetime` agar dapat dianalisis dalam konteks waktu.
- Kolom-kolom lainnya yang berisi data numerik, seperti aliran udara dan level busa, diubah menjadi tipe data `float`.

Selain itu, karakter seperti koma (`,`) diubah menjadi titik (`.`) untuk memastikan data numerik dapat diproses dengan benar.


 
## Pengecekan Outlier

Outlier dalam dataset ini akan diperiksa menggunakan **boxplot**, yang dapat dilihat pada gambar di bawah. Boxplot ini menggambarkan distribusi data dari kolom-kolom numerik dan menunjukkan adanya nilai ekstrim yang berpotensi mengganggu analisis lebih lanjut.


![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar1.png)

Gambar 1: Pengecekan Outlier

- **Insight yang didapatkan dari Gambar boxplot pada pengecekan outlier:**
  - Berdasarkan boxplot yang ditampilkan, terlihat bahwa sebagian besar kolom numerik menunjukkan adanya outlier yang ditandai dengan simbol bulat.
  - Kolom-kolom yang menunjukkan outlier tersebut perlu penanganan lebih lanjut agar analisis tidak terganggu oleh data yang tidak normal.


## EDA - Univariate Analysis

Untuk analisis univariat pada kolom numerik, digunakan dua jenis visualisasi yaitu **Histogram** dan **Violin Plot** untuk mengeksplorasi distribusi data dan menemukan pola yang ada pada setiap fitur.

- **Histogram** digunakan untuk melihat distribusi frekuensi dari kolom numerik, apakah terdistribusi normal, skewed, atau distribusi lainnya.
  
![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar2.png)

Gambar 2: Histogram Persebaran Ore Pulp pH dan Ore Pulp Density

- **Insight yang didapatkan dari Gambar 2:**
  - **Ore Pulp pH** memiliki distribusi yang normal, mayoritas titik data berada di kisaran pH antara **9.0 hingga 10.0**.
  - **Ore Pulp Density** menunjukkan bahwa titik data mayoritas berada di kelompok **1.52 hingga 1.75**.

- **Violin Plot** digunakan untuk menggambarkan distribusi dan variabilitas data numerik. Di bawah ini adalah visualisasi rata-rata **Air Flow** dan **Froth Level** pada **Flotation Column 1-7**.

![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar3.png)

Gambar 3: Violin Plot untuk Rata-rata Air Flow dan Rata-rata Froth Level di Column 1 - 7

- **Insight yang didapatkan dari Gambar 3:**
  - **Air Flow** memiliki distribusi yang fluktuatif, dengan kisaran antara **210 hingga 330 Nm3/h**.
  - **Rata-rata Froth Level** memiliki distribusi yang lebih merata, dengan kisaran antara **400 hingga 710 mm**.

Distribusi ini menunjukkan variasi yang signifikan dalam nilai **Air Flow**, yang dapat mempengaruhi proses flotasi, sedangkan distribusi **Froth Level** cenderung lebih stabil dan menunjukkan nilai rata-rata yang lebih konsisten.

## EDA - Bivariate Analysis


Pada tahap ini, analisis dilakukan untuk memahami hubungan antara fitur numerik yang ada dalam dataset. Fokus utama adalah untuk mengeksplorasi korelasi antara variabel-variabel penting yang dapat memberikan wawasan lebih lanjut tentang hubungan antar fitur.

#### Scatter Plot: % Iron Feed vs % Iron Concentrate

Untuk memulai, kita mengeksplorasi hubungan antara **% Iron Feed** sebagai input dan **% Iron Concentrate** sebagai output dengan menggunakan scatter plot. Visualisasi ini bertujuan untuk melihat apakah ada hubungan linier antara keduanya.


![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar4.png)

**Gambar 4**: Scatter Plot % Iron Feed vs % Iron Concentrate

- **Insight yang didapatkan dari Gambar 4:**
  - Dapat dilihat bahwa terdapat korelasi positif yang sangat kecil antara **% Iron Feed** dan **% Iron Concentrate**. 
  - Secara keseluruhan, lebih banyak **Zat Besi dalam pakan** akan menghasilkan lebih banyak **Zat Besi dalam konsentrat**, meskipun hubungan ini tidak terlalu kuat.

#### Heatmap: Korelasi antara % Iron Concentrate dan % Silica Concentrate

Selanjutnya, kami menganalisis korelasi antara **% Iron Concentrate** dan **% Silica Concentrate**. Ini akan membantu kita memahami hubungan antara konsentrasi kedua elemen tersebut dalam hasil tambang.


![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar5.png)

**Gambar 5**: Heatmap Korelasi antara % Iron Concentrate dan % Silica Concentrate

- **Insight yang didapatkan dari Gambar 5:**
  - Terlihat bahwa terdapat korelasi negatif yang cukup kuat antara **% Iron Concentrate** dan **% Silica Concentrate**. Hal ini berarti bahwa semakin tinggi **% Besi** dalam output, semakin sedikit **% Silika**, dan sebaliknya.
  - Ini menunjukkan bahwa proses penambangan dapat memisahkan kedua elemen ini dalam tingkat yang cukup baik, dimana **konsentrasi besi lebih tinggi** mengurangi **konsentrasi silika**, dan ini dapat digunakan untuk optimasi proses pengolahan.

## EDA - Multivariate Analysis


Pada tahap ini, dilakukan analisis untuk melihat hubungan antara beberapa fitur numerik secara bersamaan untuk memahami bagaimana mereka saling berinteraksi. Analisis ini membantu dalam mengeksplorasi korelasi antara berbagai fitur numerik dan bagaimana mereka mungkin memengaruhi hasil akhir.

#### Heatmap Korelasi Antara Fitur Numerik

Kami memulai dengan mengeksplorasi korelasi antar fitur numerik yang ada di dalam dataset menggunakan **heatmap korelasi**. Visualisasi ini memberikan gambaran umum tentang seberapa kuat hubungan antara fitur-fitur numerik.


![image](https://github.com/aldozeepriyadi/Machine-learning-terapan-1/blob/main/assets/Gambar6.png)

**Gambar 6**: Heatmap Korelasi antara Fitur Numerik

- **Insight yang didapatkan dari Gambar 6:**
  - **Korelasi Positif Terkuat** ditemukan antara **Ore Pulp Density** dan **Amina Flow**. Ini menunjukkan bahwa semakin tinggi kepadatan pulp ore, semakin tinggi aliran Amina yang digunakan dalam proses tersebut.
  - **Korelasi Negatif Terkuat** ditemukan antara **% Iron Concentrate** dan **% Silica Concentrate**. Hal ini mengindikasikan bahwa peningkatan konsentrasi besi dalam output berhubungan dengan penurunan konsentrasi silika, dan sebaliknya.
  - **% Iron Feed** dan **% Silica Feed** juga menunjukkan hubungan korelasi negatif. Ini berarti bahwa lebih banyak pakan besi dapat menghasilkan lebih sedikit pakan silika dalam proses.
  - **Amina Flow**, **Ore Pulp pH**, dan **Air Flow** memiliki korelasi yang lebih tinggi dengan **% Silica Concentrate** jika dibandingkan dengan variabel lainnya (termasuk **% Iron Concentrate**).

# Data Preparation
Pada tahap data preparation, setelah memahami kondisi data maka dilakukanlah preprocessing agar data tersebut dapat dilatih oleh model. Tahapannya dapat diikuti seperti dibawah:


## Penanganan Data Duplikasi

Pada dataset ini, ditemukan adanya **1171 baris data duplikat**. Data duplikasi sering terjadi karena berbagai alasan, seperti kesalahan saat pengumpulan data, penggabungan beberapa sumber data, atau kesalahan dalam proses input data. Duplikasi ini dapat mengganggu analisis dan menyebabkan hasil yang tidak akurat. Oleh karena itu, penting untuk menangani data duplikasi dengan hati-hati untuk memastikan analisis yang dilakukan tetap valid dan akurat. Penanganan data duplikasi dilakukan dengan menggunakan **Python** dan **pandas**, yang memungkinkan untuk mengecek dan menghapus data yang terduplikasi, serta mempertahankan hanya data yang unik.
Berikut adalah kode yang digunakan untuk menangani data duplikasi:

```python
df_mining = df_mining.drop_duplicates() 
```

## Penanganan Outlier
  Untuk menangani outlier, dilakukan penghapusan berdasarkan metode **Interquartile Range (IQR)**. IQR dihitung dengan mengurangkan **kuartil ketiga (Q3)** dari **kuartil pertama (Q1)**. Dengan rumus berikut:

  $$ IQR = Q_3 - Q_1 $$

  - **Q1 (kuartil pertama)** adalah nilai yang berada pada persentil ke-25.
  - **Q3 (kuartil ketiga)** adalah nilai yang berada pada persentil ke-75.
  
  Data yang berada di luar rentang **(Q1 - 1.5 * IQR)** hingga **(Q3 + 1.5 * IQR)** dianggap sebagai outlier dan akan dihapus.

  Setelah melakukan penanganan outlier menggunakan metode IQR, dataset yang awalnya berjumlah **737453** baris, menjadi **sejumlah baris yang lebih sedikit** karena data yang terduplikasi atau outlier telah dihapus. Setelah menangani outlier jumlah data berubah yang awalnya 737453 data menjadi 426602 data.
  


## Mengidentifikasi Fitur dan Label

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

## Membagi Data
Tahap berikutnya adalah membagi data menjadi data latih dan data uji dengan rasio 80:20. Pembagian ini penting untuk memastikan bahwa model tidak hanya belajar dari keseluruhan data tetapi juga diuji pada data yang belum pernah dilihat sebelumnya, sehingga performanya dapat dievaluasi secara objektif.

Jumlah data awal yang tersedia adalah 737453 baris. Setelah dilakukan pembagian data:
  - Data latih (train set): 461742 data (80%)
  - Data uji (test set): 115436 data (20%)


# Model Development

Pada proyek ini, tiga algoritma machine learning yang digunakan untuk memprediksi nilai % Silica Feed berdasarkan fitur-fitur yang tersedia di dataset adalah:

## Linear Regression

### Deskripsi
**Linear Regression** adalah algoritma regresi yang digunakan untuk memodelkan hubungan linier antara variabel input (fitur) dan variabel output (target). Dalam kode ini, model **Linear Regression** digunakan untuk memprediksi nilai **% Silica Feed** berdasarkan fitur-fitur yang ada dalam dataset. Model ini mengasumsikan bahwa ada hubungan linier antara variabel input dan target.

### Cara Kerja Linear Regression
Model **Linear Regression** bekerja dengan mencari hubungan linier antara input (fitur) dan target (output). Proses ini dilakukan dengan menghitung koefisien regresi untuk setiap fitur, di mana koefisien ini meminimalkan kesalahan prediksi. Model ini berusaha meminimalkan **Mean Squared Error (MSE)** antara nilai yang diprediksi dan nilai yang sebenarnya.

Secara matematis, hubungan ini digambarkan dengan persamaan:

\[
y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n
\]

Dimana:
- \( y \) adalah nilai target yang diprediksi,
- \( b_0 \) adalah intercept (titik potong dengan sumbu y),
- \( b_1, b_2, \dots, b_n \) adalah koefisien regresi untuk masing-masing fitur \( x_1, x_2, \dots, x_n \).

### Parameter Model Linear Regression yang Digunakan:
#### **`fit_intercept`** (default: `True`):
   - Menentukan apakah model akan menghitung intercept (titik potong dengan sumbu y). Jika `True`, model akan mencoba mencari nilai intercept terbaik untuk data. Jika diatur ke `False`, tidak ada intercept yang digunakan, dan model akan memaksa garis regresi melalui asal (0,0).


### Kode Implementasi Linear Regression:
Kode berikut digunakan untuk membangun dan melatih model Linear Regression, serta melakukan evaluasi menggunakan metrik evaluasi yang umum digunakan:

```python
lr_model = LinearRegression(fit_intercept=True))

# Melatih model menggunakan data pelatihan
lr_model.fit(X_train, y_train)

# Membuat prediksi dengan data uji
y_pred_lr = lr_model.predict(X_test)

# Evaluasi model menggunakan metrik yang umum
r2_lr = r2_score(y_test, y_pred_lr)  # Menghitung R²
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # Menghitung RMSE
mae_lr = mean_absolute_error(y_test, y_pred_lr)  # Menghitung MAE

# Menampilkan hasil evaluasi
print(f"Linear Regression -> R²: {r2_lr:.3f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
```

## Random Forest 

### Deskripsi
**Random Forest** adalah algoritma ensemble yang digunakan untuk memprediksi nilai numerik. Algoritma ini bekerja dengan membangun banyak pohon keputusan (decision trees) pada subset acak dari data pelatihan dan menggabungkan hasil prediksi dari masing-masing pohon untuk menghasilkan prediksi akhir. Random Forest sangat berguna dalam menangkap hubungan yang lebih kompleks antara fitur dan target, serta mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal.

### Cara Kerja Random Forest
**Random Forest** bekerja dengan membangun banyak pohon keputusan secara acak, dengan memilih subset acak dari data dan fitur untuk setiap pohon. Setiap pohon dilatih secara independen, dan hasil prediksi dari semua pohon digabungkan untuk menghasilkan prediksi final. Dalam regresi, hasil prediksi dari setiap pohon dijumlahkan atau dirata-ratakan untuk menghasilkan prediksi akhir.

Model ini sangat kuat karena dapat menangkap kompleksitas yang lebih besar dalam data dan memiliki mekanisme untuk menghindari overfitting.

### Parameter Model Random Forest 

Dalam implementasi ini, hanya satu parameter yang digunakan, yaitu **`random_state`**:

#### **`random_state`** (default: `None`):
   - Menentukan nilai seed acak untuk memastikan bahwa pembagian data dan pembuatan pohon keputusan dapat direproduksi. Ini sangat berguna ketika Anda ingin memastikan bahwa model yang dibangun menghasilkan hasil yang konsisten di setiap eksekusi. Menetapkan `random_state` ke nilai tetap, seperti `42`, akan menghasilkan hasil yang konsisten setiap kali model dijalankan.

   Pada kode ini, `random_state` diatur ke nilai `42` untuk memastikan bahwa hasil model dapat direproduksi dan konsisten saat dijalankan berulang kali.

### Kode Implementasi Random Forest 

Berikut adalah kode untuk mengimplementasikan **Random Forest** dengan parameter **`random_state`**:

```python
rf_model = RandomForestRegressor(random_state=42)

# Melatih model dengan data pelatihan
rf_model.fit(X_train, y_train)

# Membuat prediksi dengan data uji
y_pred_rf = rf_model.predict(X_test)

# Evaluasi model
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Menampilkan hasil evaluasi
print(f"Random Forest -> R²: {r2_rf:.3f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")
```

## K-Nearest Neighbors (KNN)

### Deskripsi
**K-Nearest Neighbors (KNN)** adalah algoritma yang digunakan untuk regresi dan klasifikasi, di mana prediksi dilakukan berdasarkan kedekatan titik data yang ingin diprediksi dengan data yang sudah diketahui (tetangga terdekat). Untuk regresi, KNN menghitung rata-rata dari **k** tetangga terdekat untuk menghasilkan prediksi nilai target.

### Cara Kerja KNN
Model **K-Nearest Neighbors** melakukan prediksi dengan mencari **k** tetangga terdekat berdasarkan jarak antara data yang ingin diprediksi dengan data lainnya dalam dataset. Biasanya, jarak yang digunakan adalah jarak Euclidean. Setelah menemukan tetangga terdekat, model menghitung rata-rata nilai target dari tetangga tersebut untuk memberikan prediksi.

Proses utamanya:
1. **Perhitungan Jarak**: Menghitung jarak antara data yang ingin diprediksi dan semua data dalam dataset pelatihan.
2. **Pemilihan K Tetangga Terdekat**: Memilih **k** tetangga terdekat berdasarkan jarak terpendek.
3. **Prediksi**: Menghitung rata-rata nilai target dari **k** tetangga terdekat untuk memberikan prediksi.

### Parameter Model K-Nearest Neighbors yang Digunakan

#### **`n_neighbors`** (default: `5`):
   - Menentukan jumlah tetangga yang digunakan untuk melakukan prediksi. Dalam implementasi ini, parameter **`n_neighbors`** diatur ke **`10`**, yang berarti model akan mempertimbangkan 10 tetangga terdekat untuk memprediksi nilai target. Nilai **k** yang optimal seringkali harus dicari melalui eksperimen, namun nilai **k = 10** cukup sering digunakan sebagai nilai awal untuk mencari keseimbangan antara bias dan varians.

### Kode Implementasi K-Nearest Neighbors

Berikut adalah kode untuk mengimplementasikan **K-Nearest Neighbors** dengan parameter **`n_neighbors=10`**:

```python

knn_model = KNeighborsRegressor(n_neighbors=10)

# Melatih model dengan data pelatihan
knn_model.fit(X_train, y_train)

# Membuat prediksi dengan data uji
y_pred_knn = knn_model.predict(X_test)

# Evaluasi model
r2_knn = r2_score(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
mae_knn = mean_absolute_error(y_test, y_pred_knn)

# Menampilkan hasil evaluasi
print(f"KNN -> R²: {r2_knn:.3f}, RMSE: {rmse_knn:.2f}, MAE: {mae_knn:.2f}")
```

# Evaluation

Pada tahap evaluasi model, beberapa metrik evaluasi digunakan untuk menilai performa model dalam memprediksi kualitas **% Silica Feed** pada proses flotasi. Metrik yang digunakan dalam evaluasi meliputi **R² (R-squared)**, **RMSE (Root Mean Squared Error)**, dan **MAE (Mean Absolute Error)**. Metrik-metrik ini dipilih karena relevansinya dalam menilai akurasi dan kesalahan prediksi untuk model regresi.

## Metrik Evaluasi yang Digunakan

### 1. **R² (R-squared)**
   **R²** adalah ukuran seberapa baik model dapat menjelaskan variansi dalam data target. Nilai **R²** yang lebih tinggi menunjukkan bahwa model memiliki kemampuan yang baik dalam menjelaskan variansi dalam data yang diprediksi. Nilai **R²** yang mendekati 1 menunjukkan prediksi yang sangat akurat.

### 2. **RMSE (Root Mean Squared Error)**
   **RMSE** mengukur seberapa besar rata-rata kesalahan kuadrat antara nilai yang diprediksi dan nilai yang sebenarnya. Semakin kecil nilai **RMSE**, semakin akurat model dalam memprediksi data target. RMSE yang lebih rendah menunjukkan model yang lebih presisi.

### 3. **MAE (Mean Absolute Error)**
   **MAE** mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai yang sebenarnya. Seperti RMSE, semakin rendah nilai **MAE**, semakin baik model dalam melakukan prediksi.

## Hasil Evaluasi Model

Tiga model yang diuji dalam proyek ini adalah **Linear Regression**, **Random Forest**, dan **K-Nearest Neighbors (KNN)**. Hasil evaluasi berdasarkan metrik **R²**, **RMSE**, dan **MAE** adalah sebagai berikut:

### 1. **Linear Regression**
   - **R²**: 0.950
   - **RMSE**: 1.49
   - **MAE**: 1.23

### 2. **Random Forest**
   - **R²**: 1.000
   - **RMSE**: 0.04
   - **MAE**: 0.00

### 3. **K-Nearest Neighbors (KNN)**
   - **R²**: 0.671
   - **RMSE**: 3.82
   - **MAE**: 2.59

## Komparasi Model
Berdasarkan hasil evaluasi, model **Random Forest** menunjukkan performa terbaik dengan **R² = 1.000**, yang berarti model ini dapat sepenuhnya menjelaskan variansi dalam data target dan memberikan prediksi yang sangat akurat. Nilai **RMSE = 0.04** dan **MAE = 0.00** menunjukkan bahwa model ini sangat baik dalam memprediksi nilai yang sebenarnya.

**Linear Regression** juga memberikan hasil yang baik dengan **R² = 0.950**, yang menunjukkan bahwa model ini dapat menjelaskan sekitar 94.6% variansi dalam data target. Meskipun demikian, RMSE dan MAE yang sedikit lebih besar (1.49 dan 1.23) menunjukkan bahwa prediksi model ini sedikit lebih tidak akurat dibandingkan dengan Random Forest.

**K-Nearest Neighbors (KNN)** memiliki performa yang lebih rendah, dengan **R² = 0.671**, yang menunjukkan bahwa model ini hanya dapat menjelaskan sekitar 66.9% variansi dalam data target. **RMSE = 3.82** dan **MAE = 2.59** menunjukkan bahwa prediksi model ini memiliki kesalahan yang lebih besar dibandingkan dengan dua model lainnya.

## Visualisasi Hasil Model
Untuk memberikan wawasan yang lebih dalam mengenai performa model, dilakukan visualisasi yang menunjukkan hubungan antara **Actual % Silica Feed** dan **Predicted % Silica Feed** pada setiap model. Visualisasi ini membantu untuk melihat seberapa baik model-model tersebut dalam mencocokkan nilai prediksi dengan nilai aktual.

- **Linear Regression** dan **Random Forest** menunjukkan pola yang sangat dekat dengan garis diagonal (garis merah), menunjukkan prediksi yang hampir sempurna.
- **KNN** menunjukkan pola yang lebih tersebar, dengan variasi yang lebih besar dalam prediksi dibandingkan dengan kedua model lainnya. Ini menunjukkan bahwa **KNN** kurang stabil dalam memprediksi nilai target.

## Sample Testing

Untuk lebih mendalami performa model, berikut adalah hasil prediksi untuk **% Silica Feed** pada satu sampel data yang diuji:


- % Silica Feed Actual: 6.26
- Linear Regression   : 5.27
- Random Forest       : 6.26

Dari hasil testing pada sampel tersebut, terlihat bahwa:
- **Linear Regression** memberikan prediksi yang sangat dekat dengan nilai aktual, meskipun ada sedikit selisih.
- **Random Forest** memberikan prediksi yang tepat, sesuai dengan nilai aktual.



### Kesesuaian dengan Problem Statement

Model yang telah dibangun memberikan solusi yang sangat relevan terhadap masalah yang ingin diselesaikan dalam proyek ini. Berikut adalah bagaimana masing-masing problem statement dapat dijawab:

- **Problem Statement 1**: Membangun model machine learning untuk memprediksi kualitas bijih besi berdasarkan variabel seperti **% Iron Feed**, **% Silica Feed**, dan parameter lainnya.
  
  Model **Random Forest** menunjukkan hasil yang sangat baik dalam memprediksi kualitas bijih besi, dengan **R² = 1.000** dan **MAE = 0.00**, yang menunjukkan bahwa model ini dapat memisahkan dengan akurat bijih besi berkualitas tinggi dan rendah. Hasil ini menunjukkan kemampuan model untuk mengatasi masalah prediksi kualitas bijih secara efektif.

- **Problem Statement 2**: Mengukur kinerja model dengan metrik evaluasi yang tepat, seperti **R²**, **RMSE**, dan **MAE**, untuk memastikan keandalan dan aplikasi model dalam lapangan.
  
  Berdasarkan evaluasi metrik, **Random Forest** berhasil menunjukkan performa luar biasa dengan **R² = 1.000**, **RMSE = 0.04**, dan **MAE = 0.00**. Metrik-metrik ini menunjukkan bahwa model ini sangat dapat diandalkan dan cocok untuk digunakan di lapangan, memenuhi standar keakuratan yang diperlukan untuk aplikasi praktis di sektor pertambangan.

- **Problem Statement 3**: Memanfaatkan model untuk meningkatkan efisiensi proses flotasi, mengurangi pemborosan material, dan mempercepat pengambilan keputusan terkait kualitas bijih.
  
  Dengan hasil yang sangat akurat dari **Random Forest**, model ini dapat membantu dalam mengoptimalkan proses flotasi, mengurangi pemborosan material, dan mempercepat pengambilan keputusan terkait kualitas bijih. Prediksi yang akurat memungkinkan perusahaan untuk membuat keputusan yang lebih tepat waktu dan berbasis data.

### Pencapaian Tujuan Proyek

Proyek ini berhasil mencapai setiap tujuan yang ditetapkan, dengan hasil yang menunjukkan bahwa model yang dibangun dapat memenuhi harapan yang telah ditentukan:

- **Goal 1**: Membangun model prediktif berbasis machine learning untuk memproses data historis dan memprediksi kualitas bijih besi.
  
  **Random Forest** berhasil memproses data historis dan memberikan prediksi dengan akurasi tinggi, memenuhi tujuan pertama proyek untuk membangun model prediktif yang efektif dan dapat diandalkan.

- **Goal 2**: Mengevaluasi performa model dengan menggunakan metrik yang relevan seperti **R²**, **RMSE**, dan **MAE** untuk memastikan akurasi dan efektivitasnya dalam memprediksi kualitas bijih.
  
  Evaluasi yang dilakukan dengan menggunakan metrik-metrik tersebut mengonfirmasi bahwa **Random Forest** adalah model terbaik untuk memprediksi kualitas bijih besi, sesuai dengan tujuan evaluasi proyek.

- **Goal 3**: Menyediakan solusi berbasis data yang dapat digunakan untuk mempercepat proses analisis dan meningkatkan efisiensi operasional di sektor pertambangan.
  
  Model ini dapat mempercepat analisis kualitas bijih besi, yang memungkinkan peningkatan efisiensi operasional, serta mendukung pengambilan keputusan yang lebih tepat dalam waktu yang lebih singkat.


# Referensi
1. V. N. Oparin, "Mining Industry and Sustainable Development: Time for Change," Journal of Mining Science, vol. 52, no. 6, pp. 1430-1441, Dec. 2016.
2. F. Arif, N. Suryana, and B. Hussin, "A Data Mining Approach for Developing Quality Prediction Model in Multi-Stage Manufacturing," International Journal of Computer Applications, vol. 77, no. 15, pp. 1-7, Sep. 2013.
3. D. Lieber, M. Stolpe, B. Konrad, J. Deuse, and K. Morik, "Machine Learning and Deep Learning Based Predictive Quality in Interlinked Manufacturing Processes," Procedia CIRP, vol. 79, pp. 302-307, 2019.

