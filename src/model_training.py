import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self):
        pass

    def create_model(self, model_name):
        if model_name == 'model_one':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='tanh', input_shape=(4,)),
                tf.keras.layers.Dense(32, activation='tanh'),
                tf.keras.layers.Dense(2)  # Output layer with 2 units for left and right motor adjustments
            ])
        elif model_name == 'model_two':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(2)  # Output layer with 2 units for left and right motor adjustments
            ])
        else:
            raise ValueError(f"Unknown model name '{model_name}'")

        return model

    def train_model(self, model_name, X_train_path, y_train_path):
        # Create the model
        model = self.create_model(model_name)

        # Load training data from CSV files
        X = np.loadtxt(X_train_path, delimiter=',')
        y = np.loadtxt(y_train_path, delimiter=',')

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
        
        
        val_loss, val_mae = model.evaluate(X_val, y_val)
        print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

        # Evaluate the model
        loss, accuracy = model.evaluate(X_train, y_train)
        print(f'Training Loss: {loss}, Training Accuracy: {accuracy}')

        # Save the trained model
        model.save(os.path.join(os.path.abspath('../models'), f'robot_navigation_model_{model_name}.h5'))

if __name__ == "__main__":
    X_train_path = open(os.path.abspath('../data/X_train.csv'))
    y_train_path = open(os.path.abspath('../data/y_train.csv'))

    trainer = ModelTrainer()

    # Train model1
    trainer.train_model('model_two', X_train_path, y_train_path)

    # Train model2
    trainer.train_model('model_one', X_train_path, y_train_path)
