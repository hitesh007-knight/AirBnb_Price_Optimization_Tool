# AirBnb_Price_Optimization_Tool
Project Implementation & Dataset Description--

#1. Technologies and Components

This project was built using a specific technology stack designed for efficient machine learning development and deployment. The key technologies and components are outlined below.
Core Technologies
•	Python: The primary programming language used for all scripting and application logic.
•	Scikit-learn: A robust and widely-used machine learning library in Python. It was used for data preprocessing (e.g., OneHotEncoder, SimpleImputer), model building (RandomForestRegressor), and evaluation (mean_absolute_error, r2_score).
•	Pandas: A powerful library for data manipulation and analysis, used for loading the dataset, handling dataframes, and feature selection.
•	NumPy: A fundamental library for scientific computing in Python, used for numerical operations.
•	Joblib: A library for serializing and deserializing Python objects. It was used to save the trained machine learning model to a file, which allows the Streamlit application to load and use it without retraining.
•	Streamlit: An open-source Python library used to create and deploy interactive web applications for machine learning and data science. It provides a simple and fast way to build the user interface for our price prediction tool.
Project Components--

The project is structured into three main components that work together to form the final application:

1.	airbnb_model_trainer.py: This is the backend script responsible for all the machine learning logic. Its functions include:
o	Loading and cleaning the raw data.
o	Splitting the data into training and testing sets.
o	Creating a scikit-learn pipeline that handles all preprocessing and model training.
o	Training the RandomForestRegressor model.
o	Evaluating the model's performance on the test data.
o	Saving the entire trained pipeline to a file (airbnb_price_model.joblib) for later use.

2.	app.py: This is the frontend script that runs the Streamlit web application. Its functions include:
o	Loading the pre-trained model from the airbnb_price_model.joblib file.
o	Creating a user-friendly form with interactive widgets for users to input listing details.
o	Passing the user's input to the loaded model for prediction.
o	Displaying the predicted price and other relevant information to the user in a visually appealing interface.

3. Requirements.txt File: The requirements.txt file is a crucial part of the project's setup. It is a plain text file that specifies all the Python packages and their versions that are necessary to run the project.
       Purpose of requirements.txt:
•	Dependency Management: It ensures that anyone who runs this project, including deployment platforms like Streamlit Cloud, installs the exact same dependencies. This prevents compatibility issues and guarantees the code will run as expected.
•	Reproducibility: It makes the project environment reproducible. By running a simple command (pip install -r requirements.txt), a user can set up the correct environment without manually installing each library.
      Packages Used in this Project:
•	streamlit: For building the interactive web application.
•	pandas: For data manipulation and analysis.
•	scikit-learn: For all machine learning operations, including data preprocessing and model training.
•	joblib: For saving and loading the trained machine learning model.


