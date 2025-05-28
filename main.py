import joblib
from utils.fetch_data import fetch_stock_data
from utils.indicators import compute_indicators

model = joblib.load("models/stock_model.pkl")

df = fetch_stock_data("AAPL", period="1mo")
df = compute_indicators(df)

latest_data = df.iloc[-1][['SMA_14', 'EMA_14', 'RSI', 'Volume']].values.reshape(1, -1)
prediction = model.predict(latest_data)

print("Prediction for next day:", "Up" if prediction[0] == 1 else "Down")
