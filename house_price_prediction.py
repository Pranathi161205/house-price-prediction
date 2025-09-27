import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load the dataset
filename = 'House Price Prediction Dataset.csv'
data = pd.read_csv(filename)

# Select features and target
X = data[['Area', 'Bedrooms', 'Location']]
X = pd.get_dummies(X, columns=['Location'], drop_first=True)
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# --- User Input Section ---
location_options = data['Location'].unique().tolist()
print(f"Available locations: {location_options}")

area_input = float(input("Enter area (e.g., 2500): "))
bedrooms_input = int(input("Enter number of bedrooms (e.g., 4): "))
location_input = input(f"Enter location (choose from {location_options}): ")

# Prepare user input as a DataFrame with all columns in the same order as X
user_dict = {'Area': area_input, 'Bedrooms': bedrooms_input}

# Add one-hot encoded location columns
for col in X.columns:
    if col.startswith('Location_'):
        loc_name = col.split('Location_')[1]
        user_dict[col] = int(location_input == loc_name)

# If the first location (alphabetically) was dropped by drop_first, make sure it's all zeros if selected
if f'Location_{location_options[0]}' not in X.columns and location_input == location_options[0]:
    for col in X.columns:
        if col.startswith('Location_'):
            user_dict[col] = 0

# Ensure all columns are present and in the correct order
user_df = pd.DataFrame([user_dict], columns=X.columns)

# Predict price
predicted_price = model.predict(user_df)[0]
print(f"Predicted house price: {predicted_price:.2f}")



# Calculate metrics
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2*100:.2f}%")

# Print some actual vs. predicted values
print("\nSample Actual vs. Predicted Prices:")
for actual, pred in zip(y_test[:5], model.predict(X_test)[:5]):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# --- Visualization ---
plt.scatter(y_test, model.predict(X_test))
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

