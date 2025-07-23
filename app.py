import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath):
    """Loads the student performance data and selects relevant columns."""
    try:
        df = pd.read_csv(filepath)
        required_columns = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Exam_Score']
        return df[required_columns]
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please ensure it's in the correct directory. Using sample data instead.")
        data = {
            'Hours_Studied': np.random.randint(1, 45, 100),
            'Attendance': np.random.randint(60, 100, 100),
            'Previous_Scores': np.random.randint(50, 100, 100),
            'Exam_Score': np.random.randint(50, 100, 100)
        }
        return pd.DataFrame(data)

# --- Outlier Handling Function ---
def handle_outliers(df):
    """Removes outliers from the dataframe's key numerical columns using the IQR method."""
    df_cleaned = df.copy()
    cols_to_check = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Exam_Score']
    
    for col in cols_to_check:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned

# --- Model Training and Caching ---
@st.cache_resource
def train_regression_model(df):
    """
    Performs hyperparameter tuning on an XGBoost model to find the best configuration.
    """
    features = ['Hours_Studied', 'Previous_Scores', 'Attendance']
    target = 'Exam_Score'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'colsample_bytree': [0.7, 1.0]
    }
    
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(estimator=xgbr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    return best_model, scaler, r2

# --- Main Application ---
df_original = load_data('StudentPerformanceFactors.csv')

# --- UI Layout ---
st.title("🎓 Optimized Student Performance Predictor")
st.markdown("This tool uses a powerful, **hyperparameter-tuned XGBoost model** to predict a student's final exam score with high accuracy.")

# --- Sidebar for User Input and Model Control ---
with st.sidebar:
    st.header("Model Optimization")
    improve_accuracy = st.checkbox("Improve Accuracy", value=True, 
                                  help="Applies advanced techniques to improve model accuracy.")

if improve_accuracy:
    df_for_model = handle_outliers(df_original)
else:
    df_for_model = df_original.copy()

model, scaler, r2 = train_regression_model(df_for_model)

with st.sidebar:
    st.header("Student Input Features")
    st.markdown("Enter the student's details to get a score prediction.")
    
    user_hours_studied = st.slider("Weekly Hours Studied:", 
                                   min_value=int(df_for_model['Hours_Studied'].min()), 
                                   max_value=int(df_for_model['Hours_Studied'].max()), 
                                   value=int(df_for_model['Hours_Studied'].mean()))
                                   
    user_previous_scores = st.slider("Previous Exam Scores:", 
                                     min_value=int(df_for_model['Previous_Scores'].min()), 
                                     max_value=int(df_for_model['Previous_Scores'].max()), 
                                     value=int(df_for_model['Previous_Scores'].mean()))

    user_attendance = st.slider("Attendance (%):", 
                                min_value=int(df_for_model['Attendance'].min()), 
                                max_value=int(df_for_model['Attendance'].max()), 
                                value=int(df_for_model['Attendance'].mean()))
    
    st.markdown("---")
    st.header("Model Performance")
    st.markdown("The R-squared score measures how well the model's predictions match the actual data. A score closer to 1 is better.")
    st.info(f"**Optimized R-squared Score:** {r2:.2f}")

# --- Prediction and Insights ---
input_features = np.array([[user_hours_studied, user_previous_scores, user_attendance]])
input_features_scaled = scaler.transform(input_features)
predicted_score = model.predict(input_features_scaled)[0]

# --- Display Predictions and Insights ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    st.metric(label="Predicted Exam Score", value=f"{predicted_score:.2f}")

    st.subheader("Personalized Recommendations")
    
    # --- NEW: Dynamic Recommendation Logic ---
    recommendations = []
    if predicted_score < 70:
        # Analyze inputs if the score is low
        if user_hours_studied < df_for_model['Hours_Studied'].mean():
            recommendations.append("<li><strong>Increase Study Time:</strong> The model indicates that more weekly study hours could significantly improve the score.</li>")
        if user_attendance < df_for_model['Attendance'].mean():
            recommendations.append("<li><strong>Improve Attendance:</strong> Consistent class attendance is strongly linked to better performance.</li>")
        if user_previous_scores < df_for_model['Previous_Scores'].mean():
             recommendations.append("<li><strong>Review Past Material:</strong> Strengthening foundational knowledge from previous topics could help.</li>")
        
        # Generic advice if inputs are already good but score is low
        if not recommendations:
            recommendations.append("<li><strong>Seek Support:</strong> The student's inputs are strong, but the predicted score is low. This might be a good time to offer tutoring or a review session on difficult subjects.</li>")
        
        # Build the HTML for the recommendation box
        rec_html = """
        <div style="background-color:#F8D7DA; color: black; padding: 15px; border-radius: 10px; border-left: 5px solid #DC3545;">
        <p>⚠️ <strong>Action Recommended.</strong> This student may be at risk.</p>
        <ul>
        """ + "".join(recommendations) + """
        </ul>
        </div>
        """
        st.markdown(rec_html, unsafe_allow_html=True)

    elif predicted_score < 85:
        st.markdown("""
            <div style="background-color:#FFF3CD; color: black; padding: 15px; border-radius: 10px; border-left: 5px solid #FFC107;">
            <p>👍 <strong>Good work!</strong> Solid performance is expected.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color:#D4EDDA; color: black; padding: 15px; border-radius: 10px; border-left: 5px solid #28A745;">
            <p>✅ <strong>Excellent!</strong> This student is on track for a great score.</p>
            </div>
            """, unsafe_allow_html=True)


with col2:
    st.subheader("Score Distribution")
    fig_hist = px.histogram(df_for_model, x='Exam_Score', nbins=20, title="Distribution of Exam Scores")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# --- Data Visualizations ---
st.header("Visual Analysis")
st.markdown("The charts below show the relationship between different factors and the final exam score.")

tab1, tab2, tab3 = st.tabs(["Hours Studied vs. Score", "Previous Scores vs. Score", "Attendance vs. Score"])

with tab1:
    fig1 = px.scatter(df_for_model, x='Hours_Studied', y='Exam_Score', 
                      title="Weekly Hours Studied vs. Exam Score",
                      trendline="ols", trendline_color_override="red")
    fig1.add_trace(go.Scatter(x=[user_hours_studied], y=[predicted_score], name='Your Prediction',
                              mode='markers', marker=dict(color='orange', size=15, symbol='star')))
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(df_for_model, x='Previous_Scores', y='Exam_Score', 
                      title="Previous Scores vs. Exam Score",
                      trendline="ols", trendline_color_override="red")
    fig2.add_trace(go.Scatter(x=[user_previous_scores], y=[predicted_score], name='Your Prediction',
                              mode='markers', marker=dict(color='orange', size=15, symbol='star')))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.scatter(df_for_model, x='Attendance', y='Exam_Score', 
                      title="Attendance vs. Exam Score",
                      trendline="ols", trendline_color_override="red")
    fig3.add_trace(go.Scatter(x=[user_attendance], y=[predicted_score], name='Your Prediction',
                              mode='markers', marker=dict(color='orange', size=15, symbol='star')))
    st.plotly_chart(fig3, use_container_width=True)

# --- Display Raw Data ---
with st.expander("View the Original Raw Dataset"):
    st.dataframe(df_original)
