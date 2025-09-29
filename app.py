import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the datasets
@st.cache_data
def load_data():
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    # Combine the two dataframes
    df = pd.merge(calories, exercise, on='User_ID')
    return df

df = load_data()

# Preprocess the data and train the model
@st.cache_resource
def train_model():
    # Label encode the 'Gender' column
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # Define features and target
    features = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    target = 'Calories'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the XGBoost Regressor model
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    return model, le

model, le = train_model()

# Streamlit App
st.title('Fitness Application')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Go to', ['Calorie Prediction', 'BMI Calculator'])

if options == 'Calorie Prediction':
    st.header('Predict Your Calorie Burn')
    
    st.write("Enter your details to predict the number of calories burned.")
    
    # Input fields for user
    gender = st.selectbox('Gender', ['male', 'female'])
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Weight (kg)', min_value=20, max_value=200, value=70)
    duration = st.number_input('Exercise Duration (minutes)', min_value=1, max_value=180, value=30)
    heart_rate = st.number_input('Heart Rate (bpm)', min_value=60, max_value=220, value=120)
    body_temp = st.number_input('Body Temperature (°C)', min_value=35.0, max_value=45.0, value=38.0, format="%.1f")
    
    # Prediction button
    if st.button('Predict Calories Burned'):
        # Prepare input for prediction
        gender_encoded = le.transform([gender])[0]
        input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        st.success(f'Predicted Calories Burned: {prediction[0]:.2f} kcal')

elif options == 'BMI Calculator':
    st.header('BMI (Body Mass Index) Calculator')
    
    st.write("Enter your height and weight to calculate your BMI.")
    
    # Input fields for BMI calculation
    height_bmi = st.number_input('Height (cm)', min_value=50, max_value=250, value=170, key='bmi_height')
    weight_bmi = st.number_input('Weight (kg)', min_value=20, max_value=200, value=70, key='bmi_weight')
    
    if st.button('Calculate BMI'):
        if height_bmi > 0 and weight_bmi > 0:
            height_m = height_bmi / 100.0
            bmi = weight_bmi / (height_m ** 2)
            st.success(f'Your BMI is: {bmi:.2f}')
            
            # BMI interpretation
            if bmi < 18.5:
                st.warning('You are Underweight.')
            elif 18.5 <= bmi < 24.9:
                st.success('You have a Normal weight.')
            elif 25 <= bmi < 29.9:
                st.warning('You are Overweight.')
            else:
                st.error('You are in the Obese range.')
        else:
            st.error("Please enter valid height and weight.")