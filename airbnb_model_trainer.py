import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Starting Phase 2: Data Preprocessing and Phase 3: Model Building...")

# 1. Load the dataset
# You can download the 'AB_NYC_2019.csv' file from Kaggle.
try:
    df = pd.read_csv('AB_NYC_2019.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'AB_NYC_2019.csv' was not found. Please download it from Kaggle and place it in the same directory.")
    exit()

# 2. Select relevant features
# We'll select features that are likely to influence the price.
# 'name', 'id' are dropped as they are not useful for prediction.
# 'reviews_per_month', 'last_review' have missing values that need to be handled.
# 'host_id', 'host_name' are also dropped.
features = ['neighbourhood_group', 'room_type', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
            'availability_365']
target = 'price'

# Drop rows where the target variable 'price' is zero, as this is likely an error.
df = df[df['price'] > 0]

X = df[features]
y = df[target]

# 3. Handle missing values and preprocess data
# We'll use a preprocessing pipeline for both numerical and categorical data.
# This ensures that our preprocessor is ready to handle new, unseen data in the Streamlit app.

# Identify numerical and categorical features
numerical_features = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365']
categorical_features = ['neighbourhood_group', 'room_type']

# Create a preprocessing pipeline
# The SimpleImputer for numerical data will fill missing values with the mean.
# The OneHotEncoder will convert categorical text data into a numerical format
# that the model can understand.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a ColumnTransformer to apply the right transformations to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training set (80%) and testing set (20%).")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 5. Build the model pipeline
# We'll create a full pipeline that first preprocesses the data and then trains the model.
# This ensures consistency between training and prediction.
# RandomForestRegressor is a great choice for this problem due to its robustness.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
print("Model pipeline created: Preprocessor + RandomForestRegressor.")

# 6. Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 7. Evaluate the model
print("Evaluating the model on the test data...")
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
# Corrected: calculate RMSE by taking the square root of the mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics (on Test Set) ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
print("---------------------------------------------")

# 8. Save the trained model and the preprocessor
# We save the entire pipeline so we can use it directly in our Streamlit app.
# This ensures that the same preprocessing steps are applied to the user's input.
joblib.dump(model_pipeline, 'airbnb_price_model.joblib')
print("\nModel saved as 'airbnb_price_model.joblib'.")

print("End of Phase 2 & 3. You can now proceed to building the Streamlit UI.")

