# House Price Predictors

This repository demonstrates two simple linear regression examples for predicting
house prices using the data stored in `house_prices.csv`.

## Dataset

`house_prices.csv` contains basic housing information including house size,
number of bedrooms, bathrooms and additional features such as latitude and
longitude. The `price` column is used as the target variable when training the
models.

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install the dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Single Input Example

`house_price_predictor.py` trains a simple linear regression model using only
the `sqft_living` column (house size) as the input feature. Running the script
plots the regression line alongside the raw data and prints the predicted price
for a 2000 square foot house.

Execute:

```bash
python house_price_predictor.py
```

Follow the interactive prompt to provide a house size if desired.

## Multiple Input Example

`multi_input_regression.py` extends the approach by training on several
features by default: `sqft_living`, `bedrooms`, `bathrooms` and `floors`. When
run, the script asks you to supply values for these features and then displays
the coefficients, intercept and resulting predicted price.

Execute:

```bash
python multi_input_regression.py
```

Enter the requested feature values to obtain a predicted price for that
combination.

## Repository Structure

```
.
├── house_price_predictor.py   # single input example
├── multi_input_regression.py  # multiple input example
└── house_prices.csv           # dataset
```

Each script is self‑contained and can be run directly after installing the
required packages.

