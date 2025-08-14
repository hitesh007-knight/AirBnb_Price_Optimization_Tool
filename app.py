import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model pipeline
try:
    model_pipeline = joblib.load('airbnb_price_model.joblib')
except FileNotFoundError:
    st.error("Error: The model file 'airbnb_price_model.joblib' was not found. Please run the model trainer script first.")
    st.stop()

# Get the list of neighborhood groups and room types from the model's preprocessor
try:
    one_hot_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot']
    neighborhood_groups = sorted(one_hot_encoder.categories_[0].tolist())
    room_types = sorted(one_hot_encoder.categories_[1].tolist())
except KeyError:
    st.error("Could not extract categorical features from the loaded pipeline.")
    st.stop()


# --- Streamlit UI Components ---
st.set_page_config(page_title="Airbnb Price Optimization Tool", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 90%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF5A5F;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    .stMarkdown h1 {
        color: #FF5A5F;
        text-align: center;
    }
    .stMarkdown h2 {
        color: #484848;
        text-align: center;
    }
    .stTextInput label, .stSelectbox label, .stSlider label {
        color: #484848;
        font-size: 16px;
    }
    .prediction-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f7f7f7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #FF5A5F;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #484848;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Airbnb Price Optimization Tool")
st.markdown("## Predict the best price for your NYC listing")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Listing Details")
    form = st.form(key='my_form')
    with form:
        location = st.selectbox(
            "Select Neighborhood Group",
            options=neighborhood_groups
        )
        room_type = st.selectbox(
            "Select Room Type",
            options=room_types
        )
        minimum_nights = st.slider(
            "Minimum Nights",
            min_value=1, max_value=30, value=3
        )
        number_of_reviews = st.number_input(
            "Number of Reviews",
            min_value=0, max_value=1000, value=50, step=1
        )
        reviews_per_month = st.number_input(
            "Reviews per Month",
            min_value=0.0, max_value=100.0, value=1.0, step=0.1
        )
        calculated_host_listings_count = st.number_input(
            "Number of Listings for this Host",
            min_value=1, max_value=100, value=1
        )
        availability_365 = st.slider(
            "Availability (days/year)",
            min_value=0, max_value=365, value=90
        )
        
        submitted = st.form_submit_button("Suggest Price")

with col2:
    st.header("Suggested Price")
    if submitted:
        input_data = pd.DataFrame([{
            'neighbourhood_group': location,
            'room_type': room_type,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365
        }])

        prediction = model_pipeline.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="prediction-container">
                <p class="prediction-label">Based on your inputs, the optimal price for your listing is:</p>
                <p class="prediction-value">${prediction:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Prediction successful! Use this as a guide for your pricing strategy.")

    else:
        st.info("Fill out the form on the left and click 'Suggest Price' to get a price recommendation.")
