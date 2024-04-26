# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("1553768847-housing.csv")

# Data preprocessing
# Fill missing values in 'total_bedrooms' column with median
data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)

# Feature selection/engineering
# Select relevant features
features = ['total_rooms', 'total_bedrooms', 'housing_median_age']
X = data[features]
y = data['median_house_value']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Loop to predict house values until 'q' is pressed
while True:
    # Take user input for features
    print("Enter the details of the house to predict its value:")
    total_rooms = float(input("Total Rooms: "))
    total_bedrooms = float(input("Total Bedrooms: "))
    housing_median_age = float(input("Housing Median Age: "))

    # Predict price for the user input
    example_house = [[total_rooms, total_bedrooms, housing_median_age]]
    predicted_price = model.predict(example_house)
    print("Predicted Median House Value for the Example House:", predicted_price[0])

    # Ask user if they want to continue or quit
    choice = input("Press 'q' to quit, or any other key to continue: ")
    if choice.lower() == 'q':
        break
