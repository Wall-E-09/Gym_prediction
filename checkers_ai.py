import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Глобальні змінні для поточного гравця та вибраної шашки
selected_piece = None
player_turn = 1  # Початок з чорних фігур

# Створення нейронної мережі
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Модель вибирає хід
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Перевірка, чи можливий хід (по діагоналі)
def is_valid_move(board, start, end, player):
    if abs(start[0] - end[0]) != abs(start[1] - end[1]):
        return False
    if board[end[0], end[1]] != 0:
        return False
    if player == 1 and end[0] < start[0]:
        return False
    if player == 2 and end[0] > start[0]:
        return False
    return True

# Перевірка, чи можливе биття
def can_capture(board, start, end, player):
    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2
    if board[mid_x, mid_y] == (3 - player) and board[end[0], end[1]] == 0:
        return True
    return False

# Функція для обробки ходу користувача
def make_move(i, j):
    global selected_piece, player_turn
    if selected_piece is not None:
        start_x, start_y = selected_piece
        if is_valid_move(board, (start_x, start_y), (i, j), player_turn):
            board[i, j] = player_turn
            board[start_x, start_y] = 0
            player_turn = 3 - player_turn  # Змінюємо гравця
            visualize_board(board)
            if player_turn == 2:  # Якщо хід зробив білий гравець, тоді настає хід нейронної мережі
                calculate_best_move()
        elif can_capture(board, (start_x, start_y), (i, j), player_turn):  # Перевірка можливості биття
            mid_x = (start_x + i) // 2
            mid_y = (start_y + j) // 2
            board[i, j] = player_turn
            board[start_x, start_y] = 0
            board[mid_x, mid_y] = 0  # Биття шашки
            player_turn = 3 - player_turn  # Змінюємо гравця
            visualize_board(board)
            if player_turn == 2:
                calculate_best_move()
        selected_piece = None
    else:
        if board[i, j] == player_turn:  # Якщо клітинка належить поточному гравцеві
            selected_piece = (i, j)

# Нейронна мережа для вибору найкращого ходу
def calculate_best_move():
    global player_turn, model
    board_input = np.expand_dims(board, axis=0)
    prediction = model.predict(board_input)
    
    # Для демонстрації модель просто вибирає випадковий хід
    move_index = int(prediction[0][0] * 64)  # Перетворення оцінки у хід
    move_x = move_index // 8
    move_y = move_index % 8
    messagebox.showinfo("Нейромережа", f"Нейромережа обрала хід на ({move_x}, {move_y})")
    make_move(move_x, move_y)

# Створення порожньої дошки
def create_board():
    board = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(i % 2, 8, 2):
            if i < 3:
                board[i, j] = 1  # Чорні фігури
            elif i > 4:
                board[i, j] = 2  # Білі фігури
    return board

# Візуалізація дошки
def visualize_board(board):
    for i in range(8):
        for j in range(8):
            color = 'black' if (i + j) % 2 == 0 else 'white'
            if board[i, j] == 1:
                color = 'red'  # Чорні фігури
            elif board[i, j] == 2:
                color = 'blue'  # Білі фігури
            buttons[i][j].config(bg=color)

# Графічний інтерфейс
root = tk.Tk()
root.title("Шашки")

# Створення кнопок для дошки
buttons = [[None for _ in range(8)] for _ in range(8)]
board = create_board()

# Створення нейронної мережі
model = create_model()

# Встановлення кнопок для кожної клітинки дошки
for i in range(8):
    for j in range(8):
        buttons[i][j] = tk.Button(root, width=8, height=4, command=lambda i=i, j=j: make_move(i, j))
        buttons[i][j].grid(row=i+1, column=j+1)

# Додавання нумерації (літер для колонок і цифр для рядків)
for i in range(8):
    # Літери для колонок (A, B, C, ...)
    tk.Label(root, text=chr(65 + i), width=4).grid(row=0, column=i + 1)
    # Цифри для рядків (1, 2, 3, ...)
    tk.Label(root, text=str(8 - i), width=4).grid(row=i + 1, column=0)

# Кнопка для підрахунку найкращого ходу
calc_button = tk.Button(root, text="Підрахувати найкращий хід", command=calculate_best_move)
calc_button.grid(row=9, column=0, columnspan=8)

# Візуалізація початкової дошки
visualize_board(board)

root.mainloop()
