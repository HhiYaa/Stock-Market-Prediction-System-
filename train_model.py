from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils.fetch_data import fetch_stock_data
from utils.indicators import compute_indicators

df = fetch_stock_data("AAPL")
df = compute_indicators(df)

features = ['SMA_14', 'EMA_14', 'RSI', 'Volume']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

joblib.dump(model, "models/stock_model.pkl")
