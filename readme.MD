### Autonomous Collision Avoidance System Documentation

#### Overview
The Autonomous Collision Avoidance System is designed to enable a ground vehicle to autonomously navigate through an environment by detecting obstacles using three front-facing sensors. The system utilizes a fully connected feed-forward neural network to predict correction values for the left and right motors of the vehicle, allowing it to maneuver around obstacles effectively. This documentation provides an overview of the project, including data generation, model training, testing, and the development of a user interface for visualization and prediction.

#### Project Structure
The project is organized into several key components:

1. **Data Generation**:
   - Purpose: Generate sensor data and corresponding correction labels based on obstacle detection and vehicle speed.
   - Implementation: Randomly generate sensor readings and speed values within specified ranges. Calculate correction labels proportional to sensor intensity and vehicle speed.
   - Output: Dataset containing sensor data and corresponding correction labels.

2. **Model Training**:
   - Purpose: Train a fully connected deep neural network and a regression model using the generated dataset to predict correction values.
   - Implementation: Preprocess the data and split it into training and testing sets. Train the models using supervised learning techniques.
   - Output: Trained models capable of predicting motor correction values based on sensor data and speed.

3. **User Interface Development**:
   - Purpose: Develop a user interface to visualize sensor data, input speed, and predict vehicle motion based on the trained models.
   - Implementation: Use Tkinter library to create a graphical user interface. Design interface with entry fields for sensor readings and speed inputs. Implement functionality to calculate and display predicted motor adjustments.
   - Output: User-friendly interface for interacting with the autonomous collision avoidance system.

#### Conclusion
The Autonomous Collision Avoidance System project aims to create an effective solution for autonomous navigation in various environments. By generating realistic sensor data, training accurate prediction models, and developing an intuitive user interface, the system provides a comprehensive approach to collision avoidance. The project's modular structure allows for scalability and adaptation to different vehicle platforms and environments, making it a versatile tool for autonomous vehicle development.

#### Getting Started
To get started with the Autonomous Collision Avoidance System:
1. Clone the repository: `git clone https://github.com/your_username/autonomous-collision-avoidance.git`
2. Navigate to the project directory: `cd AGV-Collision-Avoidance_SL`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the data generation script to generate sensor data: `cd src --> `python data_generation/generate_data.py`
5. Train the neural network models using the generated data: `python model_training.py`
6. Launch the user interface to visualize sensor data and predict vehicle motion: `python user_interface.py`

#### Contributors
- [Natanel Kebede](https://github.com/Natnael- k)


#### License
This project is licensed under the MIT License 

#### Acknowledgements
- Special thanks to Dr.Meha Saadeh's Lecture and insights to the project.
- Inspired by Protok Lab for the inspiration for autonomous vehicle development.
