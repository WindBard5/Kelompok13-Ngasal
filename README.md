# Kelompok 13 : Ngasal
- Abu Bakar Siddiq Siregar - 122140164
- Bayu Praneswara Haris - 122140219
- Jason Surya Padantya - 1221402137

## Perbandingan performa antara Plain-34 dan ResNet-34

### 1. Tabel perbandingan metrik pada epoch terakhir 

| Metrik | Plain-34 (Tahap 1 - Baseline) | **ResNet-34 (Tahap 2 - Residual)** |
| :--- | :---: | :---: |
| **Akurasi Validasi Tertinggi** | 35.68% | **44.59%** |
| Akurasi Training Akhir | 45.92% | 77.03% |
| Loss Validasi Terendah | 1.1 | **0.59** |
| Loss Training Akhir | 1.22 | 0.64 |
| Konvergensi | Stagnan / Degradasi | Cepat dan Stabil |

### 2. Grafik sederhana yang menunjukkan kurva training

Plain-34

![alt text](outputPlain34.png)

Resnet-34

![alt text](outputResnet34.png)

![alt text](output2Resnet34.png)

### 3. Analisis singkat (2-3 paragraf) mengenai perbedaan performa dan dampak residual connection

Perbedaan kinerja antara Plain-34 dan ResNet-34 menunjukkan dampak signifikan dari koneksi residual dalam mengatasi masalah fundamental pada pelatihan jaringan dalam. Model Plain-34 mengalami masalah degradasi yang nyata; model gagal mempelajari fungsi kompleks secara memadai, yang ditunjukkan oleh Akurasi Training yang rendah (45.92%) dan Akurasi Validasi yang stagnan di 35.68%. Kinerja yang rendah ini adalah manifestasi dari vanishing gradient, di mana gradien menghilang selama backpropagation karena bertambahnya kedalaman lapisan, sehingga bobot pada lapisan awal tidak dapat diperbarui secara efektif.

Sebaliknya, ResNet-34 menunjukkan peningkatan kemampuan belajar internal yang drastis, membuktikan bahwa masalah degradasi telah teratasi. Akurasi Training meningkat tajam dari 45.92% menjadi 77.03% (peningkatan lebih dari 31 persentase poin), dan yang paling krusial, Loss Validasi turun signifikan dari 1.1 menjadi 0.59. Penurunan loss yang besar ini adalah bukti kuat bahwa koneksi shortcut memungkinkan optimizer menemukan solusi bobot yang jauh lebih baik dan stabil. Kenaikan Akurasi Validasi menjadi 44.59% mengindikasikan bahwa model sekarang mampu bergeneralisasi, meskipun masih menyisakan ruang untuk optimasi hyperparameter di Tahap 3.

Secara keseluruhan, dampak koneksi residual adalah menciptakan jalur identitas H(x)=F(x)+x yang stabil, yang memungkinkan gradien mengalir tanpa terhambat. Hal ini membuat proses optimasi menjadi jauh lebih mudah, memungkinkan jaringan untuk berkonvergen ke loss yang lebih rendah, dan memastikan bahwa penambahan lapisan kedalaman (34 lapisan) dapat dilakukan tanpa merusak kinerja. Keberhasilan ini adalah fondasi untuk Tahap 3, di mana upaya dapat difokuskan pada peningkatan akurasi generalisasi (validasi) lebih lanjut.

### 4. Konfigurasi hyperparameter yang digunakan

| Hyperparameter | Nilai yang Digunakan |
| :--- | :--- |
| **Dataset** | 5 Makanan Indonesia |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-3 (0.001) |
| **Batch Size** | 8 |
| **Jumlah Epoch** | 15 |
| **Loss Function** | Cross Entropy Loss |
