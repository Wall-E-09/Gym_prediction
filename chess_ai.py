import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

# Завантаження даних
data = pd.read_csv('games.csv')

# Обробка даних
def preprocess_moves(moves):
    move_list = moves.split(' ')
    # Перетворення кожного ходу у координати
    coordinates = []
    for move in move_list:
        start = ord(move[0]) - ord('a')
        end = ord(move[1]) - ord('a')
        coordinates.append([start, end])
    return np.array(coordinates)

# Створення моделі
model = Sequential([
    Dense(128, input_dim=64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # Останній шар передбачає оцінку ходу
])

model.compile(optimizer='adam', loss='mse')

# Підготовка вхідних та вихідних даних
X = np.array([preprocess_moves(m) for m in data['moves']])
y = np.array([get_best_move(m) for m in data['moves']])  # Найкращий хід

# Навчання моделі
model.fit(X, y, epochs=10, batch_size=32)

# Прогнозування
def predict_best_move(board_state):
    return model.predict(np.array([board_state]))

# Реалізація консолі для введення ситуації
while True:
    board_state = input("Введіть поточний стан гри (наприклад, 'e2 e4 e7 e5'): ")
    best_move = predict_best_move(board_state)
    print(f"Найкращий хід: {best_move}")
