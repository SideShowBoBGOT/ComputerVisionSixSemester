import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import numpy as np

def load_image():
    global img, tk_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('RGB')
        tk_img = ImageTk.PhotoImage(img)
        canvas.itemconfig(image_container, image=tk_img)

def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        img.save(file_path)

def update_image(new_img):
    global img, tk_img
    img = new_img
    tk_img = ImageTk.PhotoImage(new_img)
    canvas.itemconfig(image_container, image=tk_img)

def change_brightness(factor):
    # Ensure img is a numpy array
    img_array = np.array(img, dtype=np.float32)
    img_array *= factor
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    update_image(Image.fromarray(img_array))

def add_noise(sigma):
    img_array = np.array(img)
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    update_image(Image.fromarray(noisy_img_array))

def gaussian_blur(radius):
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    update_image(blurred_img)

def make_negative():
    img_array = np.array(img)
    negative_img_array = 255 - img_array
    update_image(Image.fromarray(negative_img_array))

app = tk.Tk()
app.title("Редактор")

canvas = tk.Canvas(app, width=600, height=400)
canvas.pack()
image_container = canvas.create_image(20, 20, anchor="nw")

# Buttons for Image Operations
button_frame = tk.Frame(app)
button_frame.pack()

load_button = tk.Button(button_frame, text="Завантажити", command=load_image)
load_button.pack(side=tk.LEFT)

save_button = tk.Button(button_frame, text="Зберегти", command=save_image)
save_button.pack(side=tk.LEFT)

brightness_inc_button = tk.Button(button_frame, text="Збільшити яскравість", command=lambda: change_brightness(1.5))
brightness_inc_button.pack(side=tk.LEFT)

brightness_dec_button = tk.Button(button_frame, text="Зменшити яскравість", command=lambda: change_brightness(0.5))
brightness_dec_button.pack(side=tk.LEFT)

noise_button = tk.Button(button_frame, text="Додати шум", command=lambda: add_noise(25))
noise_button.pack(side=tk.LEFT)

blur_button = tk.Button(button_frame, text="Гаусівський блюр", command=lambda: gaussian_blur(2))
blur_button.pack(side=tk.LEFT)

negative_button = tk.Button(button_frame, text="Негатив", command=make_negative)
negative_button.pack(side=tk.LEFT)

app.mainloop()