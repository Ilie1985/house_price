"""Simple house price predictor using linear regression."""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path: str = "house_prices.csv"):
    """Load the CSV dataset and return the feature and target arrays."""
    df = pd.read_csv(path)
    X = df[["sqft_living"]].values  # house size
    y = df["price"].values          # house price
    return X, y


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Train a linear regression model and return it."""
    model = LinearRegression()
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Plotting Helper
# ---------------------------------------------------------------------------
def plot_results(X: np.ndarray, y: np.ndarray, model: LinearRegression,
                 sqft_example: int = 2000) -> float:
    """Plot the data, best-fit line, and prediction."""
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    predicted_price = model.predict([[sqft_example]])[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="skyblue", alpha=0.4, label="Actual Data")
    plt.plot(x_line, y_line, color="red", label="Best Fit Line")
    plt.scatter(sqft_example, predicted_price, color="green", s=100,
                label=f"Predicted (2000 sqft): £{predicted_price:,.0f}")

    plt.title("House Price Prediction Based on Size")
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price (£)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predicted_price


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the house price prediction workflow."""
    # Load and prepare data
    X, y = load_data()

    # Train the model
    model = train_model(X, y)

    # Visualize the results and obtain prediction for a sample size
    predicted_price = plot_results(X, y, model, sqft_example=2000)

    # Display model parameters and prediction
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Slope: {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print(
        f"📈 Predicted price for a 2000 sqft house: £{predicted_price:,.2f}"
    )


if __name__ == "__main__":
    main()

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

