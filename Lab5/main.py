import cv2
import numpy as np
from matplotlib import pyplot as plt


# зчитування та відображення зображення
def image_read(FileIm):
    img = cv2.imread(FileIm)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img


# опрацювання зображення
def image_processing(img, is_lowres=False):
    # розмиття гауса з більшим ядром
    img = cv2.GaussianBlur(img, (7, 7), 3)

    # перехід до HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # розширений діапазон кольорів для водойм
    if is_lowres:
        # для low-res зображення включаємо більш темно-сині відтінки
        lower_range = (70, 0, 0)
        upper_range = (180, 255, 100)
    else:
        # для high-res зображення залишаємо все як є
        lower_range = (0, 0, 0)
        upper_range = (180, 255, 50)

    # маска частин зображення що задовольняють потреби
    mask = cv2.inRange(hsv_img, lower_range, upper_range)

    # морфологічне закриття для видалення шумів
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    plt.imshow(mask)
    plt.show()

    # змінюємо колір водойм на блакитний для кращої видимості
    img[mask > 0] = (255, 255, 0)

    # підрахунок водойм у кадрі
    res = img.shape
    water = np.sum(mask == 255)
    print(f"Водойми займають = {(water / (res[0] * res[1]) * 100):.1f}% кадру")

    # відображення
    plt.imshow(img)
    plt.show()
    return


# головна частина скрипту
if __name__ == '__main__':
    # зчитування високоточного зображення
    image_entrance = image_read("highres.png")

    # його опрацювання
    image_processing(image_entrance)

    # зчитування оперативного зображення
    image_entrance = image_read("lowres.png")

    # його опрацювання з урахуванням низької роздільної здатності
    image_processing(image_entrance, is_lowres=True)