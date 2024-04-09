import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# Function to generate random sensor readings
def generate_sensor_readings(num_samples):
    # Generate random sensor readings
    sensor_readings = np.random.rand(num_samples, 3)  #Assuming 3 sensors
    return sensor_readings




# Function to generate training data based on severity conditions

def generate_training_data(num_samples, min_speed=0.1, max_speed=0.9):
    # Generate random sensor readings and speed values
    sensor_readings = generate_sensor_readings(num_samples)
    speeds = np.random.uniform(min_speed, max_speed, num_samples)
    
    # Initialize arrays to store input features and severity labels
    X = np.zeros((num_samples, 4))  # Input features (sensor readings + speed)
    y = np.zeros((num_samples, 2))  # Severity labels (left motor, right motor adjustments)
    
    # Categorize severity levels based on sensor readings and speed
    for i, (reading, speed) in enumerate(zip(sensor_readings, speeds)):
        # Calculate intensity of sensor readings (average of all sensor readings)
        intensity = np.mean(reading)
        
        # Adjust severity labels based on intensity and speed
        adjust_value = -0.2 - 0.1 * intensity - 0.1 * speed
        
        # Case 1: Only middle sensor senses obstacle
        if reading[1] < 0.5 and (reading[0] >= 0.5 or reading[2] >= 0.5):
            y[i] = [adjust_value, adjust_value]  # Decrease speed from both motors and turn right
        # Case 2: Only left or right side detects obstacle
        elif reading[0] < 0.5 and reading[1] >= 0.5 and reading[2] >= 0.5:
            y[i] = [0, adjust_value]  # Decrease speed from right motor to turn right
        elif reading[2] < 0.5 and reading[0] >= 0.5 and reading[1] >= 0.5:
            y[i] = [adjust_value, 0]  # Decrease speed from left motor to turn left
        # Case 3: Middle and left or right senses obstacle
        elif reading[1] < 0.5 and (reading[0] < 0.5 or reading[2] < 0.5):
            y[i] = [0, adjust_value]  # Decrease speed from right motor and both to slow down the motor
        elif reading[1] < 0.5 and (reading[0] < 0.5 or reading[2] < 0.5):
            y[i] = [adjust_value, 0]  # Decrease speed from left motor and both to slow down the motor
    
        # Store sensor readings and speed in input features
        X[i, :3] = reading
        X[i, 3] = speed
    
    # Return sensor readings, speed, and severity labels
    return X, y

# Generate training data
num_samples = 1000
X_train, y_train = generate_training_data(num_samples)

# Print sample data
print("Sample Input Features (Sensor Readings + Speed):\n", X_train[:5])
print("Sample Severity Labels:\n", y_train[:5])
