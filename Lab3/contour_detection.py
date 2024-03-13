import cv2
import numpy as np

# Завантаження зображення
image = cv2.imread('Fx9307uXwAEwyVR.jpg')

# Перетворення зображення на відтінки сірого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Застосування фільтра для зменшення шуму
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Бінаризація зображення
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Виділення контурів
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Вибір контуру з найбільшою площею (припускаємо, що це об'єкт)
max_contour = max(contours, key=cv2.contourArea)

# Намалювати контур на оригінальному зображенні
cv2.drawContours(image, [max_contour], 0, (0, 255, 0), 2)

# Відобразити результат
cv2.imshow('Contour Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()