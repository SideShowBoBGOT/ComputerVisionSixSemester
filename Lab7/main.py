"""
Розробити програмний скрипт, який забезпечує ідентифікацію бінарних зображень 4 спеціальних символів,
заданих растровою матрицею. Для ідентифікації синтезувати, навчити та застосувати штучну нейронну мережу
в "сирому" вигляді реалізації матричних операцій.
Вибрані символи: @, #, ?, &.
"""

import numpy as np
import matplotlib.pyplot as plt


# Вхідні дані DataSet масиву
def create_input_data():
    """
    Вхідна частина навчального DataSet масиву.
    Формування вхідних бінарних даних графічних примітивів.
    :return: x - np.array
    """

    characters = [
        np.array([
            0, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 1,
            1, 0, 0, 1, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            0, 1, 1, 1, 1, 0,
        ]).reshape(8, 6),
        np.array([
            0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0,
        ]).reshape(8, 6),
        np.array([
            0, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 1, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]).reshape(8, 6),
        np.array([
            0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 1,
            1, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 1, 1,
            0, 1, 1, 1, 0, 0,
        ]).reshape(8, 6)
    ]

    # Візуалізація
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(characters[i], cmap='binary')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Вхідна частина навчального DataSet масиву
    input_data = [char.reshape(1, 48) for char in characters]

    return input_data


def create_output_data():
    """
    Вихідна частина навчального DataSet масиву - відповідь.
    Формування кодових комбінацій бінарних відповідей у просторі 4 значень.
    :return: y - np.array
    """

    # Вихідна частина навчального DataSet масиву
    output_data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return np.array(output_data)


# Побудова нейронної мережі

# Функція активації - сигмоїда
def sigmoid(x):
    """
    :param x: - np.array DataSet in
    :return: функція активації - сигмоїда
    """

    return 1 / (1 + np.exp(-x))


# Побудова нейронної мережі
def forward_propagation(x, weights1, weights2):
    """
    Головний компонент побудови нейронної мережі.
    Це та наступне відрізняє цей приклад від звичайного перцептрона.
    Архітектурні залежності:
    1-й рівень: вхідний рівень (1, 48);
    2-й шар: прихований шар (1, 4);
    3-й шар: вихідний рівень (4, 4).
    Графічне представлення архітектури дивись Neural_Networks_numpy_2.jpg

    :param x: np.array -
    :param weights1: початкові вагові коефіцієнти шару 1 (вхідного)
    :param weights2: початкові вагові коефіцієнти шару 2 (прихованого)
    :return: output_vals - вектор вихідних параметрів мережі - 4 компоненти
    """

    # Структура вхідного шару визначається простором вхідних параметрів x

    # Прихований шар
    hidden_inputs = x.dot(weights1)  # зважені вхідні параметри вхідного шару 1
    hidden_outputs = sigmoid(hidden_inputs)  # адитивна згортка - вихід з шару 1 - вхід до шару 2

    # Вихідний шар
    final_inputs = hidden_outputs.dot(weights2)  # зважені вхідні параметри шару 2 до вихідного шару
    final_outputs = sigmoid(final_inputs)  # вихідні параметри нейронної мережі

    return final_outputs


# Ініціалізація початкових значень вагових коефіцієнтів мережі методом рандомізації
def initialize_weights(rows, cols):
    weights = []
    for _ in range(rows * cols):
        weights.append(np.random.randn())
    return np.array(weights).reshape(rows, cols)


# Контроль навчання мережі за допомогою середньоквадратичної помилки (MSE)
def calculate_loss(output, target):
    squared_error = np.square(output - target)
    loss = np.sum(squared_error) / len(target)
    return loss


# Зворотне поширення похибки
def backpropagation(x, y, weights1, weights2, learning_rate):
    # Прихований шар
    hidden_inputs = x.dot(weights1)  # зважені вхідні параметри вхідного шару 1
    hidden_outputs = sigmoid(hidden_inputs)  # адитивна згортка - вихід з шару 1 - вхід до шару 2

    # Вихідний шар
    final_inputs = hidden_outputs.dot(weights2)  # зважені вхідні параметри шару 2 до вихідного шару
    final_outputs = sigmoid(final_inputs)  # вихідні параметри нейронної мережі

    # Похибка на вихідному шарі
    output_errors = final_outputs - y
    hidden_errors = np.multiply((weights2.dot((output_errors.transpose()))).transpose(),
                                (np.multiply(hidden_outputs, 1 - hidden_outputs)))

    # Градієнт для weights1 та weights2
    weights1_gradients = x.transpose().dot(hidden_errors)
    weights2_gradients = hidden_outputs.transpose().dot(output_errors)

    # Оновлення параметрів з контролем помилки learning_rate
    weights1 -= learning_rate * weights1_gradients
    weights2 -= learning_rate * weights2_gradients

    return weights1, weights2


# Навчання мережі з контролем помилки learning_rate на епоху
def train_network(x, y, weights1, weights2, learning_rate=0.01, num_epochs=10):
    def update_weights(inputs, targets, w1, w2, lr):
        output = forward_propagation(inputs, w1, w2)
        loss = calculate_loss(output, targets)
        updated_w1, updated_w2 = backpropagation(inputs, targets, w1, w2, lr)
        return loss, updated_w1, updated_w2

    def train_epoch(epoch, data, labels, w1, w2, lr):
        epoch_loss, updated_w1, updated_w2 = zip(*[update_weights(x, y, w1, w2, lr) for x, y in zip(data, labels)])
        avg_loss = sum(epoch_loss) / len(data)
        accuracy = (1 - avg_loss) * 100
        print(f"Епоха: {epoch + 1}, Точність: {accuracy:.2f}%")
        return accuracy, avg_loss, updated_w1[-1], updated_w2[-1]

    accuracies, losses, trained_weights1, trained_weights2 = zip(*[train_epoch(epoch, x, y, weights1, weights2, learning_rate) for epoch in range(num_epochs)])
    return accuracies, losses, trained_weights1[-1], trained_weights2[-1]


# Ідентифікація символів / прогнозування
def predict_symbol(x, weights1, weights2):
    """
    Функція прогнозування приймає наступні аргументи:
    :param x: матриця зображення
    :param weights1: натреновані ваги
    :param weights2: натреновані ваги
    :return: відображає ідентифікований символ - графічну форму
    """

    def get_predicted_class(output):
        return max(range(len(output[0])), key=lambda i: output[0][i])

    def get_symbol(predicted_class):
        symbols = ["@", "#", "?", "&"]
        return symbols[predicted_class]

    output = forward_propagation(x, weights1, weights2)
    predicted_class = get_predicted_class(output)
    symbol = get_symbol(predicted_class)

    print(f"Зображення символу {symbol}.\n")
    plt.imshow(x.reshape(8, 6), cmap='binary')
    plt.show()

    return


if __name__ == '__main__':
    # Вхідні дані
    input_data = create_input_data()
    output_data = create_output_data()
    print('Масив DataSet: навчальна пара для навчання з учителем')
    print('Вхідні дані:', input_data, '\n')
    print('Вихідні дані:', output_data, '\n')

    # Ініціалізація вагових коефіцієнтів для 2 шарів
    layer_sizes = [(48, 4), (4, 4)]
    weights = [initialize_weights(*size) for size in layer_sizes]
    print('Ініціалізація вагових коефіцієнтів для 2 шарів')

    # Навчання мережі з контролем помилки learning_rate на епоху
    print('Навчання мережі з контролем помилки learning_rate на епоху')
    accuracies, losses, *trained_weights = train_network(input_data, output_data, *weights, 0.1, 70)

    # Контроль / візуалізація параметрів навчання
    training_metrics = [
        ('Точність', accuracies),
        ('Втрати', losses)
    ]
    for metric, data in training_metrics:
        plt.figure()
        plt.plot(data)
        plt.ylabel(metric)
        plt.xlabel("Епохи")
        plt.show()

    # Ідентифікація символів / прогнозування
    symbols = ["@", "#", "?", "&"]
    for i, symbol in enumerate(symbols):
        print(f'Вхідні параметри відповідають символу "{symbol}"')
        print('Результат ідентифікації:')
        predict_symbol(input_data[i], *trained_weights)