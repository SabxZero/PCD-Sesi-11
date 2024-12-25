import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar grayscale
image = img.imread("Pewdiepie1.jpg", mode='F')

# Kernel Sobel
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Padding pada gambar
imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# Menginisialisasi Gx dan Gy
Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

# Operasi konvolusi manual
for y in range(1, imgPad.shape[0]-1):
    for x in range(1, imgPad.shape[1]-1):
        region = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = (region * sobelX).sum()
        Gy[y-1, x-1] = (region * sobelY).sum()

# Menghitung magnitudo gradien
G = np.sqrt(Gx**2 + Gy**2)
G = (G / G.max()) * 255
G = np.clip(G, 0, 255)
G = G.astype(np.uint8)

# Menyimpan hasil gambar setelah Sobel
output_path = "pewdiepiesobel.jpg"
img.imwrite(output_path, G)
print(f"Hasil gambar setelah Sobel disimpan di {output_path}")

# Menampilkan hasil
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Gradient X (Gx)")
plt.imshow(Gx, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Gradient Y (Gy)")
plt.imshow(Gy, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Gradient Magnitude (G)")
plt.imshow(G, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Program segmentasi dengan thresholding
import imageio as img2

# Membaca gambar hasil sobel
segmented_image = img2.imread(output_path, mode='F')

# Threshold untuk segmentasi
threshold = 0.3
binary_segmented_image = segmented_image > (threshold * 255)

# Menampilkan hasil segmentasi
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Sobel Result Image")
plt.imshow(segmented_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(binary_segmented_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
