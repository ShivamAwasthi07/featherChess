import os
import chess
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense
from keras import callbacks
from tensorflow.keras.models import model_from_json

csv = pd.read_csv('fen_analysis.csv')
fens = csv['FEN'].values
evaluations = csv['Evaluation'].values

# Map chess pieces to 12D one-hot vectors
chess_dict = {
    'p': [1,0,0,0,0,0,0,0,0,0,0,0], 'P': [0,0,0,0,0,0,1,0,0,0,0,0],
    'n': [0,1,0,0,0,0,0,0,0,0,0,0], 'N': [0,0,0,0,0,0,0,1,0,0,0,0],
    'b': [0,0,1,0,0,0,0,0,0,0,0,0], 'B': [0,0,0,0,0,0,0,0,1,0,0,0],
    'r': [0,0,0,1,0,0,0,0,0,0,0,0], 'R': [0,0,0,0,0,0,0,0,0,1,0,0],
    'q': [0,0,0,0,1,0,0,0,0,0,0,0], 'Q': [0,0,0,0,0,0,0,0,0,0,1,0],
    'k': [0,0,0,0,0,1,0,0,0,0,0,0], 'K': [0,0,0,0,0,0,0,0,0,0,0,1],
    '.': [0]*12
}

def make_matrix(board):
    """Convert a chess board to 8x8 character matrix."""
    pgn = board.epd().split(' ')[0]
    rows = pgn.split('/')
    matrix = []

    for row in rows:
        expanded_row = []
        for char in row:
            if char.isdigit():
                expanded_row.extend(['.'] * int(char))
            else:
                expanded_row.append(char)
        matrix.append(expanded_row)
    
    return matrix

def translate(matrix, piece_dict):
    """Translate a character matrix to a 8x8x12 numerical tensor."""
    return [[piece_dict[square] for square in row] for row in matrix]

# Limit the number of samples
length = 10000
X = []
y = evaluations[:length]

# Process each FEN to numerical data
for i in range(length):
    board = chess.Board(fens[i])
    matrix = make_matrix(board)
    translated = translate(matrix, chess_dict)
    X.append(translated)

# Handle special cases in evaluation values (e.g. checkmate)
for i in range(length):
    if isinstance(y[i], str) and '#' in y[i]:
        y[i] = float(y[i][-1]) * 1000

# Convert to float32 and normalize y
y = np.array(y, dtype='float32')
y_min, y_max = y.min(), y.max()
y = (y - y_min) / (y_max - y_min)

# Convert input data to numpy array
X = np.array(X)

# Split into training and testing sets
test_size = 1000
X_train, X_test = X[test_size:], X[:test_size]
y_train, y_test = y[test_size:], y[:test_size]

# Build CNN model
model = Sequential([
    Conv2D(10, kernel_size=1, activation='relu', input_shape=(8, 8, 12)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

# Set up model saving callbacks
model_json_path = 'chess_best_model.json'
model_weights_path = 'chess_best_model.weights.h5'

# Save model architecture
with open(model_json_path, 'w') as json_file:
    json_file.write(model.to_json())

# Define callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    model_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0
)
early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss', patience=500, verbose=1, mode='min'
)

# Train the model
print("Training network...")

model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=2
)

def load_model_from_files(json_path='chess_best_model.json', weights_path='chess_best_model.weights.h5'):
    with open(json_path, 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='mse')
    return model

