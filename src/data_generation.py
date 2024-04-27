import os
import numpy as np

class RobotNavigationModel:
    
    def __init__(self):
        pass

    def generate_sensor_readings(self, num_samples):

        min_value = 0  # Minimum value of the range
        max_value = 1  # Maximum value of the range

        # Generate random sensor readings
        sensor_readings = np.random.rand(num_samples, 3)  # Generating random values between 0 and 1

        # Scale and shift the generated values to the desired range
        sensor_readings = (max_value - min_value) * sensor_readings + min_value
        return sensor_readings

    def generate_training_data(self, num_samples, min_speed=4, max_speed=10):
        # Generate random sensor readings and speed values
        sensor_readings = self.generate_sensor_readings(num_samples)
        speeds = np.random.uniform(min_speed, max_speed, num_samples)

        # Initialize arrays to store input features and severity labels
        X = np.zeros((num_samples, 4))  # Input features (sensor readings + speed)
        y = np.zeros((num_samples, 2))  # Severity labels (left motor, right motor adjustments)

        # Categorize severity levels based on sensor readings and speed
        for i, (reading, speed) in enumerate(zip(sensor_readings, speeds)):
            # Calculate intensity of sensor readings (average of all sensor readings)
            intensity = np.mean(reading)

            # Adjust severity labels based on intensity and speed
            adjust_value = -0.2 - 0.1 * (1 -intensity) - 0.1 * speed

            # Case 1: Only middle sensor senses obstacle
            if reading[1] < 0.5 and (reading[0] >= 0.5 or reading[2] >= 0.5):
                y[i] = [adjust_value, adjust_value]  # Decrease speed from both motors and turn right
            # Case 2: Only left or right side detects obstacle
            elif reading[0] < 0.5 and reading[1] >= 0.5 and reading[2] >= 0.5:
                y[i] = [0, adjust_value]  # Decrease speed from right motor to turn right
            elif reading[2] < 0.5 and reading[0] >= 0.5 and reading[1] >= 0.5:
                y[i] = [adjust_value, 0]  # Decrease speed from left motor to turn left
            # Case 3: Middle and left or right senses obstacle
            elif reading[1] < 0.5 and reading[0] < 0.5 and reading[2] >= 0.5:
                y[i] = [- 0.1 * (1-intensity), adjust_value]  # Decrease speed from right motor to slow down
            elif reading[1] < 0.5 and reading[0] > 0.5 or reading[2] < 0.5:
                y[i] = [adjust_value, - 0.1 * (1-intensity)]  # Decrease speed from left motor to slow down

            # Store sensor readings and speed in input features
            X[i, :3] = reading
            X[i, 3] = speed

        # Return sensor readings, speed, and severity labels
        return X, y

    # Save the training Data to a folder
    def save_training_data(self, X, y, folder_path):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save data to CSV files
        np.savetxt(os.path.join(folder_path, 'X_train.csv'), X, delimiter=',')
        np.savetxt(os.path.join(folder_path, 'y_train.csv'), y, delimiter=',')

model = RobotNavigationModel()
X_train, y_train = model.generate_training_data(num_samples=10000)

# Specify the folder path where you want to save the data
folder_path = '../data'

# Save the training data
model.save_training_data(X_train, y_train, folder_path)


