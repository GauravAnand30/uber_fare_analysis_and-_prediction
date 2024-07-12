import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# CSS styles
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-title {
        color: #1f77b4;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .section-title {
        color: #ff7f0e;
        font-size: 28px;
        font-weight: bold;
    }
    .button {
        background-color: #ff7f0e;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
    }
    .note {
        font-size: 16px;
        font-weight: bold;
    }
    .fade-in {
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .sidebar .sidebar-content {
        background-color: #006400;
        color: white;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h3 {
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content p {
        color: white;
    }
    .sidebar .sidebar-content button {
        background-color: #32CD32;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        width: 100%;
    }
    .prediction-section {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-input {
        text-align: center;
        font-size: 20px;
    }
    .prediction-button {
        background-color: #ff7f0e;
        color: white;
        font-weight: bold;
        font-size: 24px;
        padding: 15px 30px;
        border: none;
        border-radius: 5px;
        width: 50%;
        margin: 10px auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<p class="sidebar-title">Uber Data Analysis</p>', unsafe_allow_html=True)
st.sidebar.write("""
### Steps to Follow:
1. Upload the dataset (CSV file).
2. Analyze the data.
3. Visualize the data.
4. Predict fare price.
""")

# Function to render the data analysis page
def render_data_analysis_page():
    # Main title
    st.markdown('<p class="main-title fade-in">Uber Data Analysis</p>', unsafe_allow_html=True)

    # Upload Dataset
    if 'data' not in st.session_state:
        st.session_state.data = None

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.write("Dataset uploaded successfully!")
        st.write(data.head())

    # Analyze Data
    if st.session_state.data is not None and st.sidebar.button("Analyze Data", key='analyze_data'):
        st.markdown('<p class="section-title fade-in">Data Analysis Results:</p>', unsafe_allow_html=True)

        # Assuming the target variable is 'fare_amount' or similar
        if 'fare_amount' in st.session_state.data.columns:
            target = 'fare_amount'
        else:
            st.error("The dataset does not contain a 'fare_amount' column.")
            target = None

        if target:
            # Prepare the data for feature importance analysis
            X = st.session_state.data.drop(columns=[target])
            y = st.session_state.data[target]

            # Use target encoding for categorical variables
            encoder = TargetEncoder()
            X = encoder.fit_transform(X, y)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit a Random Forest model
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Save the model and encoder to session state
            st.session_state.model = model
            st.session_state.encoder = encoder

            # Get feature importances
            importances = model.feature_importances_
            feature_names = X.columns
            feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importances = feature_importances.sort_values(by='importance', ascending=False)

            # Display the top 5 most important results
            st.write("### Top 5 Most Important Features:")
            top_5_features = feature_importances.head(5)
            for i, row in top_5_features.iterrows():
                st.write(f"{i + 1}. **{row['feature']}**: {row['importance']:.4f}")

            # Save top features to session state
            st.session_state.top_features = feature_importances

    # Graph Analysis
    if st.session_state.data is not None and 'top_features' in st.session_state and st.sidebar.button("Graph Analysis", key='graph_analysis'):
        st.markdown('<p class="section-title fade-in">Data Visualization:</p>', unsafe_allow_html=True)

        # Graph 1: Distribution of the most important feature
        st.write(f"#### Distribution of {st.session_state.top_features.iloc[0]['feature']}")
        fig1, ax1 = plt.subplots()
        sns.histplot(st.session_state.data[st.session_state.top_features.iloc[0]['feature']], kde=True, ax=ax1)
        st.pyplot(fig1)

        # Graph 2: Pairplot of top 3 features
        st.write("#### Pairplot of Top 3 Features")
        fig2 = sns.pairplot(st.session_state.data[st.session_state.top_features.head(3)['feature']])
        st.pyplot(fig2)

        # Button to navigate to prediction page
        if st.button("Next", key='next'):
            st.session_state.page = 'predict_fare'
            st.experimental_rerun()

# Function to render the fare prediction page
def render_fare_prediction_page():
    st.markdown('<p class="main-title fade-in">Predict Uber Fare Price</p>', unsafe_allow_html=True)

    st.markdown('<div class="prediction-section fade-in">Enter the details below to predict the fare price:</div>', unsafe_allow_html=True)

    # Input fields for prediction
    distance = st.number_input("Enter Distance (in miles):", min_value=0.0, step=0.1, key='distance')
    car_model = st.selectbox("Select Car Model:", st.session_state.data['car_model'].unique(), key='car_model')
    time = st.number_input("Enter Time (in minutes):", min_value=0.0, step=1.0, key='time')

    # Create input data for prediction
    input_data = pd.DataFrame({
        'distance': [distance],
        'car_model': [car_model],
        'time': [time]
    })

    # Encode the input data
    input_data = st.session_state.encoder.transform(input_data)

    # Make prediction
    if st.button("Predict Fare", key='predict_fare'):
        prediction = st.session_state.model.predict(input_data)
        st.markdown(f'<div class="prediction-section fade-in">### Predicted Fare Price: ${prediction[0]:.2f}</div>', unsafe_allow_html=True)

    # Button to navigate back to the analysis page
    if st.button("Back", key='back'):
        st.session_state.page = 'data_analysis'
        st.experimental_rerun()

# Main logic to render pages based on state
if 'page' not in st.session_state:
    st.session_state.page = 'data_analysis'

if st.session_state.page == 'data_analysis':
    render_data_analysis_page()
elif st.session_state.page == 'predict_fare':
    render_fare_prediction_page()

# Footer
st.markdown('<p class="note">Notes:<br>- Ensure the dataset contains a <code>fare_amount</code> column for target variable.<br>- The app analyzes the top 10 most important features based on a Random Forest model.</p>', unsafe_allow_html=True)
