import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import os
import random
from datetime import datetime

pd.set_option('display.max_columns', 100)

# Membuka file dataset
app_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(app_dir, 'Epileptic Seizure Recognition.csv')
data = pd.read_csv(file)

# Mengubah target kolom y menjadi biner
dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
data['y'] = data['y'].map(dic)

# Menghilangkan kolom unamed
data = data.drop('Unnamed', axis=1)
data = shuffle(data)

# Fungsi untuk melakukan flipping
def flip_data(data):
    return -data

# Fungsi untuk menambahkan noise
def add_noise(data, noise_level=0.5):
    noise = np.random.normal(0, noise_level, len(data))
    return data + noise

# Jumlah augmentasi yang diinginkan
num_augmentations = 1

random_decision = random.randint(0, 1)

# Menentukan alamat folder root di dalam folder 'uas'
root_folder = os.path.join(app_dir, 'root')

# Membuat folder 'root' jika belum ada
os.makedirs(root_folder, exist_ok=True)

for i in range(num_augmentations):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#F0F0F0')

    # Set warna latar belakang di dalam area grafik
    ax.patch.set_facecolor('#F0F0F0')  # Ganti warna latar belakang di sini

    plt.ylim(-1500, 1500)

    # Memilih data untuk di-augmentasi
    sample_data = data[data['y'] == random_decision].iloc[i][:-1]

    # Menerapkan flipping
    augmented_data = flip_data(sample_data)

    # Menerapkan penambahan noise
    augmented_data = add_noise(augmented_data)

    # Plot grafik hasil augmentasi
    plt.plot(augmented_data)

    # Mendapatkan tanggal dan waktu saat ini
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Membuat nama file sesuai format yang diinginkan
    file_name = os.path.join(root_folder, f"{current_time}_{i+1}.png")

    # Menyimpan gambar plot dengan nama file yang baru dibuat
    plt.savefig(file_name)

    # Menutup plot setelah disimpan (opsional)
    plt.close()

print("Plots telah disimpan di dalam folder 'root'")