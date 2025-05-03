📘 MLP untuk Prediksi Regresi & Klasifikasi pada Dataset RegresiUTSTelkom.csv
Proyek ini membangun dan mengevaluasi dua model deep learning berbasis Multilayer Perceptron (MLP) menggunakan PyTorch untuk menyelesaikan dua tugas: regresi dan klasifikasi pada dataset RegresiUTSTelkom.csv. Pipeline mencakup preprocessing, pelatihan, evaluasi metrik, dan visualisasi performa model.

📂 Struktur Notebook
Import Library & Load Dataset

Preprocessing & Split Dataset

Normalisasi fitur

One-hot encoding target klasifikasi

Split 80-20 untuk train-test

Model MLP Regresi

3 hidden layers (128 → 64 → 32)

Fungsi aktivasi ReLU

Loss function: MSELoss

Optimizer: Adam

Model MLP Klasifikasi

3 hidden layers (128 → 64 → 32)

Output layer: Sigmoid (binary classification)

Loss function: BCEWithLogitsLoss

Evaluasi Model

Regresi: MAE, RMSE, R²

Klasifikasi: Accuracy, Precision, Recall, F1, AUC-ROC

Visualisasi & Interpretasi

ROC Curve

Bar Chart perbandingan metrik klasifikasi

Analisis Teoritis

Bias-variance, pemilihan fungsi loss, normalisasi fitur, interpretasi fitur, tuning hyperparameter

✅ Hasil Evaluasi
🔢 Model Regresi
Metrik	Nilai
MAE	5.9296
RMSE	9.2720
R²	0.2757

🔍 Interpretasi: Model masih underfitting (R² rendah), perlu eksplorasi arsitektur lain, regularisasi, atau penambahan fitur baru.

✅ Model Klasifikasi
Metrik	Nilai
Accuracy	0.7344
Precision	0.7237
Recall	0.7151
F1 Score	0.7194
AUC-ROC	0.8086

🔍 Interpretasi: Model cukup seimbang antara presisi dan recall. AUC-ROC menunjukkan kemampuan baik dalam membedakan kelas.

🧠 Insight Analisis Teori
Underfitting pada MLP regresi (R² rendah) → bisa diatasi dengan menambah kompleksitas model (misal: lebih banyak neuron/layer) atau teknik regularisasi.

Fungsi Loss Alternatif: MAE lebih robust terhadap outlier dibanding MSE, cocok saat data memiliki noise tinggi.

Skala Fitur penting untuk konvergensi efisien. Perbedaan range (misal 0–1 vs 100–1000) menyebabkan gradien tidak stabil.

Interpretasi Fitur: Dapat menggunakan permutation importance atau analisis bobot, namun memiliki keterbatasan jika fitur saling berkorelasi.

Tuning Hyperparameter seperti learning rate & batch size dilakukan dengan grid search atau random search, sambil memonitor kurva loss & stabilitas gradien.
