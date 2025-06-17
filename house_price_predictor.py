# 📦 Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 📂 Load your dataset (make sure house_prices.csv is in the same folder)
df = pd.read_csv("house_prices.csv")

# 🔍 Select the input (X) and output (y) for the model
X = df[['sqft_living']].values  # house size
y = df['price'].values          # house price

# 🧠 Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# 📈 Generate values for the best-fit line
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

# 💡 Ask the user for the house size and predict the price
try:
    sqft_example = float(input("Enter the house size in sqft: "))
except ValueError:
    print("Invalid input. Using default value of 2000 sqft.")
    sqft_example = 2000
predicted_price = model.predict([[sqft_example]])[0]

# 🎨 Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='skyblue', alpha=0.4, label="Actual Data")
plt.plot(x_line, y_line, color='red', label="Best Fit Line")
plt.scatter(sqft_example, predicted_price, color='green', s=100,
            label=f"Predicted ({sqft_example:.0f} sqft): £{predicted_price:,.0f}")

plt.title("House Price Prediction Based on Size")
plt.xlabel("Size (sqft)")
plt.ylabel("Price (£)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧾 Print slope, intercept, and prediction
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"📈 Predicted price for a {sqft_example:.0f} sqft house: £{predicted_price:,.2f}")
