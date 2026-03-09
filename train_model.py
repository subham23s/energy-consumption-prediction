import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────
print("Loading dataset...")
df = pd.read_excel("D:\\Energy_Prediction\\ENB2012_data.xlsx")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────
# STEP 2 — Separate Features and Target
# ─────────────────────────────────────────
X = df[['X1','X2','X3','X4','X5','X6','X7','X8']].values  # 8 input features
y = df['Y1'].values  # Target → Heating Load

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ─────────────────────────────────────────
# STEP 3 — Normalize Features
# ─────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for dashboard use later
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("\nScaler saved as scaler.pkl")

# ─────────────────────────────────────────
# STEP 4 — Train Test Split (80/20)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ─────────────────────────────────────────
# STEP 5 — Build ANN Model
# ─────────────────────────────────────────
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(8,)))  # Input + Hidden Layer 1
model.add(Dropout(0.2))                                     # Prevent overfitting
model.add(Dense(32, activation='relu'))                     # Hidden Layer 2
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))                     # Hidden Layer 3
model.add(Dense(1))                                          # Output Layer (regression)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ─────────────────────────────────────────
# STEP 6 — Train Model
# ─────────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Save trained model
model.save("energy_model.keras")
print("\nModel saved as energy_model.keras")

# ─────────────────────────────────────────
# STEP 7 — Evaluate Model
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n========== Model Performance ==========")
print(f"MAE  (Mean Absolute Error) : {mae:.4f}")
print(f"RMSE (Root Mean Sq Error)  : {rmse:.4f}")
print(f"R²   (Accuracy Score)      : {r2:.4f}")
print("========================================")

# ─────────────────────────────────────────
# STEP 8 — Plot Results
# ─────────────────────────────────────────

# Plot 1 — Training vs Validation Loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
print("Loss curve saved as loss_curve.png")

# Plot 2 — Actual vs Predicted
plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label='Actual', color='green', marker='o', markersize=4)
plt.plot(y_pred[:50], label='Predicted', color='red', marker='x', markersize=4)
plt.title('Actual vs Predicted Heating Load (First 50 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("Actual vs Predicted chart saved as actual_vs_predicted.png")
