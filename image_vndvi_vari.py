import numpy as np
import cv2
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def compute_vndvi_vari(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype('float32') / 255.0
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    epsilon = 1e-6
    vndvi = (G - R) / (G + R + epsilon)
    vari = (G - R) / (G + R - B + epsilon)
    vari = np.clip(vari, -1, 1)
    return np.mean(vndvi), np.mean(vari), image_rgb

def show_interface():
    def open_image():
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])
        if not file_path:
            return
        try:
            vndvi, vari, image_rgb = compute_vndvi_vari(file_path)
            img = Image.fromarray((image_rgb * 255).astype(np.uint8))
            img = img.resize((256, 256))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            result_label.config(text=f"vNDVI: {vndvi:.4f}\nVARI: {vari:.4f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("vNDVI & VARI Image Analyzer")
    root.geometry("300x400")
    btn = tk.Button(root, text="Open Image", command=open_image)
    btn.pack(pady=10)
    image_label = tk.Label(root)
    image_label.pack(pady=10)
    result_label = tk.Label(root, text="vNDVI: --\nVARI: --", font=("Arial", 12))
    result_label.pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_interface()
    else:
        image_path = sys.argv[1]
        vndvi, vari, _ = compute_vndvi_vari(image_path)
        print(f"vNDVI: {vndvi:.4f}")
        print(f"VARI: {vari:.4f}")
