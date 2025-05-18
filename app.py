import streamlit as st
import pandas as pd
import joblib
import os

st.title("🍽️ Restaurant Rating Predictor")

# Load model and encoders
model = joblib.load("restaurant_rating_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Show current working directory and files for debugging (optional)
# st.write("Working directory:", os.getcwd())
# st.write("Files:", os.listdir())

# Load dataset only if it exists
dataset_path = "Dataset.csv"
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    st.error("❌ 'Dataset.csv' not found in the app directory. Please ensure it is present before running the app.")
    st.stop()

# Dropdown values
all_cities = sorted(df['City'].dropna().unique())
all_country_codes = sorted(df['Country Code'].dropna().unique())
all_cuisines = sorted(df['Cuisines'].dropna().unique())

# Optional: Display label encoder classes
if st.checkbox("Show encoder classes"):
    for col, le in label_encoders.items():
        st.write(f"{col} classes: {list(le.classes_)}")

# Encoded choices
has_table_booking_choices = list(label_encoders["Has Table booking"].classes_)
has_online_delivery_choices = list(label_encoders["Has Online delivery"].classes_)
is_delivering_now_choices = list(label_encoders["Is delivering now"].classes_)
switch_to_order_menu_choices = list(label_encoders["Switch to order menu"].classes_)
currency_choices = list(label_encoders["Currency"].classes_)

# Inputs
country_code = st.selectbox("Country Code", all_country_codes)
city = st.selectbox("City", all_cities)
longitude = st.number_input("Longitude", value=77.0, format="%.6f")
latitude = st.number_input("Latitude", value=28.0, format="%.6f")
cuisines = st.selectbox("Cuisines", all_cuisines)
avg_cost = st.number_input("Average Cost for two", min_value=0, step=1, value=500)
currency = st.selectbox("Currency", currency_choices)
has_table_booking = st.selectbox("Has Table booking", has_table_booking_choices)
has_online_delivery = st.selectbox("Has Online delivery", has_online_delivery_choices)
is_delivering_now = st.selectbox("Is delivering now", is_delivering_now_choices)
switch_to_order_menu = st.selectbox("Switch to order menu", switch_to_order_menu_choices)
price_range = st.selectbox("Price Range", [1, 2, 3, 4])
votes = st.number_input("Votes", min_value=0, step=1, value=0)

# Create input dataframe
input_dict = {
    'Country Code': [country_code],
    'City': [city],
    'Longitude': [longitude],
    'Latitude': [latitude],
    'Cuisines': [cuisines],
    'Average Cost for two': [avg_cost],
    'Currency': [currency],
    'Has Table booking': [has_table_booking],
    'Has Online delivery': [has_online_delivery],
    'Is delivering now': [is_delivering_now],
    'Switch to order menu': [switch_to_order_menu],
    'Price range': [price_range],
    'Votes': [votes]
}
input_df = pd.DataFrame(input_dict)

# Encode categorical features
for col in input_df.columns:
    if col in label_encoders:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"Encoding error in column '{col}': {e}")
            st.stop()

# Predict rating
if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]
    st.success(f"⭐ Predicted Restaurant Rating: {round(prediction, 2)} ⭐")
