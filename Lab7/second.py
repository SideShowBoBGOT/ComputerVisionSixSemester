import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Завантаження попередньо навченої моделі ResNet50
model = ResNet50(weights='imagenet')


# Функція для передбачення наявності зброї на зображенні
def detect_weapon(image_path):
    # Завантаження і препроцесинг зображення
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Отримання передбачень моделі
    preds = model.predict(image)

    # Декодування передбачень
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=5)[0]

    # Перевірка наявності класу "гвинтівка" або "пістолет" в топ-5 передбачених класах
    weapon_detected = False
    for a, class_name, b in decoded_preds:
        if class_name in ['rifle', 'revolver']:
            weapon_detected = True
            break

    return weapon_detected

for image_path in ['armed_soldiers.png', 'unarmed_soldiers.png']:
    # Виявлення наявності зброї на зображенні
    if detect_weapon(image_path):
        print(f"На зображенні {image_path} виявлено зброю!")
    else:
        print(f"На зображенні {image_path} не виявлено зброї.")