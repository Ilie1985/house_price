import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(path: str = "house_prices.csv", features=None):
    """Load dataset and return feature matrix, target vector, and feature names."""
    if features is None:
        features = ["sqft_living", "bedrooms", "bathrooms", "floors"]
    df = pd.read_csv(path)
    X = df[features].values
    y = df["price"].values
    return X, y, features


def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Train and return a linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def prompt_for_features(feature_names):
    """Prompt the user for feature values in order."""
    values = []
    for name in feature_names:
        val = input(f"Enter {name.replace('_', ' ')}: ")
        try:
            values.append(float(val))
        except ValueError:
            values.append(0.0)
    return np.array([values])


def main() -> None:
    X, y, feature_names = load_data()
    model = train_model(X, y)
    user_values = prompt_for_features(feature_names)
    predicted = model.predict(user_values)[0]

    coefs = ", ".join(f"{n}: {c:.2f}" for n, c in zip(feature_names, model.coef_))
    print(f"Coefficients -> {coefs}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Predicted price: £{predicted:,.2f}")


if __name__ == "__main__":
    main()
