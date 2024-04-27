import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AutonomousVehicleUI:
    """This is a class for UI for interacting witht the trained model"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Vehicle Correction")
        
        # Label and Entry for Sensor 1
        ttk.Label(root, text="Sensor 1 Reading:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.sensor1_entry = ttk.Entry(root)
        self.sensor1_entry.grid(row=0, column=1, padx=5, pady=5)

        # Label and Entry for Sensor 2
        ttk.Label(root, text="Sensor 2 Reading:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.sensor2_entry = ttk.Entry(root)
        self.sensor2_entry.grid(row=1, column=1, padx=5, pady=5)

        # Label and Entry for Sensor 3
        ttk.Label(root, text="Sensor 3 Reading:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.sensor3_entry = ttk.Entry(root)
        self.sensor3_entry.grid(row=2, column=1, padx=5, pady=5)

        # Label and Entry for Speed
        ttk.Label(root, text="Speed:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.speed_entry = ttk.Entry(root)
        self.speed_entry.grid(row=3, column=1, padx=5, pady=5)

        # Button to calculate correction
        ttk.Button(root, text="Calculate Correction", command=self.calculate_correction).grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # Label to display correction
        self.correction_label = ttk.Label(root, text="")
        self.correction_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        # Figure for sensor readings plot
        self.fig_sensor_readings = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax_sensor_readings = self.fig_sensor_readings.add_subplot(111)
        self.ax_sensor_readings.set_xlabel('Sensor')
        self.ax_sensor_readings.set_ylabel('Reading')
        self.ax_sensor_readings.set_title('Sensor Readings')
        self.canvas_sensor_readings = FigureCanvasTkAgg(self.fig_sensor_readings, master=root)
        self.canvas_sensor_readings.get_tk_widget().grid(row=0, column=2, rowspan=4, padx=5, pady=5)

        # Figure for correction plot
        self.fig_correction = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax_correction = self.fig_correction.add_subplot(111)
        self.ax_correction.set_xlabel('Motor')
        self.ax_correction.set_ylabel('Adjustment')
        self.ax_correction.set_title('Predicted Correction')
        self.canvas_correction = FigureCanvasTkAgg(self.fig_correction, master=root)
        self.canvas_correction.get_tk_widget().grid(row=4, column=2, padx=5, pady=5)

    def calculate_correction(self):
        try:
            # Get sensor readings and speed from entries
            sensor1 = float(self.sensor1_entry.get())
            sensor2 = float(self.sensor2_entry.get())
            sensor3 = float(self.sensor3_entry.get())
            speed = float(self.speed_entry.get())

            # Load the trained model
            model = load_model(os.path.abspath('../models/robot_navigation_model_model_one.h5'))
            input_data = np.array([[sensor1, sensor2, sensor3, speed]])
            input_data_scaled = input_data / 1.0  # Scale input data if needed

            # Predict correction using the model
            correction = model.predict(input_data_scaled)

            # Update correction label
            self.correction_label.config(text=f"Left Motor Adjustment: {correction[0][0]}, Right Motor Adjustment: {correction[0][1]}")

            # Update sensor readings plot
            sensors = ['Sensor 1', 'Sensor 2', 'Sensor 3']
            readings = [sensor1, sensor2, sensor3]
            self.ax_sensor_readings.clear()
            self.ax_sensor_readings.bar(sensors, readings, color='blue')
            self.ax_sensor_readings.set_ylim(0, 1)
            self.canvas_sensor_readings.draw()

            # Update correction plot
            motors = ['Left Motor', 'Right Motor']
            adjustments = [correction[0][0], correction[0][1]]
            self.ax_correction.clear()
            self.ax_correction.bar(motors, adjustments, color='green')
            self.ax_correction.set_ylim(-0.5, 0.5)
            self.canvas_correction.draw()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for sensor readings and speed.")

    
    def execute(self):
            self.root.mainloop()

# Create a Tkinter root window
root = tk.Tk()

# Create an instance of AutonomousVehicleUI
ui = AutonomousVehicleUI(root)

# Start the event loop
ui.execute()