🎓 High-Accuracy Student Performance Predictor
This is an advanced, interactive web application built with Streamlit that uses a powerful Stacked Ensemble machine learning model to predict a student's final exam score with high accuracy. The model focuses on three key input features: weekly hours studied, previous exam scores, and class attendance.

🌟 Features
1.High-Accuracy Predictions: Utilizes a stacked ensemble model (combining XGBoost and Random Forest) to achieve a high degree of prediction accuracy.

2.Interactive Controls: Users can easily adjust a student's study hours, previous scores, and attendance to see real-time score predictions.

3.Advanced Data Processing: The app includes built-in, toggleable options for outlier removal and feature normalization to optimize model performance.

4.Personalized Recommendations: The app provides specific, data-driven advice based on the user's inputs to help students identify areas for improvement.

5.Rich Data Visualization: Interactive charts from Plotly are used to visualize the data and the model's predictions, making it easy to understand the relationships between the features.

⚙️ How to Run the Application
To run this application on your local machine, please follow these steps.

Prerequisites
Python 3.8 or newer

pip for package installation

1. Clone the Repository
First, clone this repository to your local machine (or simply download the files into a new folder).

git clone <your-repository-url>
cd <repository-folder>

2. Install Dependencies
Install the required Python libraries using the requirements.txt file. This file includes all necessary packages, including xgboost and statsmodels

pip install -r requirements.txt

3. Run the Streamlit App
Once the dependencies are installed, you can run the app with a single command:

streamlit run app.py

Your web browser should open a new tab with the running application.

📂 File Structure
The project is organized as follows:

.
├── app.py                           # The main Streamlit application script
├── StudentPerformanceFactors.csv    # The dataset used for model training
├── requirements.txt                 # A list of Python dependencies for the project
└── README.md                        # This file

🤖 Modeling Logic: Stacked Ensembling
To achieve the highest possible accuracy, this project uses a state-of-the-art technique called Stacked Ensembling. Instead of relying on a single model, this approach works in two stages:

Base Models: Two powerful and distinct models, a Random Forest and an XGBoost regressor, are trained on the data. Each model learns the patterns in the data differently and makes its own predictions.

Meta-Model: A final, simpler model (a LinearRegression model in this case) is then trained. Its job is not to learn from the original data, but to learn from the predictions of the base models. By learning how to best combine the predictions from the Random Forest and XGBoost models, it can make a final prediction that is more accurate and robust than any single model could achieve on its own.

This entire process, including data cleaning and normalization, is wrapped in a single function to ensure the best possible performance.