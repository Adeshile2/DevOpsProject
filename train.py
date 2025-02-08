import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Generate Dummy Data
data = {
    "Age": [22, 35, 50, 28, 40],
    "Fare": [7.25, 71.28, 30.50, 8.05, 50.00],
    "Pclass": [3, 1, 2, 3, 1],
    "Sex": ["male", "female", "male", "male", "female"],
    "Embarked": ["S", "C", "Q", "S", "C"],
    "Survived": [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Define feature names
feature_columns = ["Age", "Fare", "Pclass", "Sex", "Embarked"]

# Preprocessing: Encode categorical features
categorical_features = ["Sex", "Embarked"]
encoder = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
    remainder="passthrough"
)

# Transform the dataset
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_transformed = encoder.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model and Encoder
joblib.dump(model, "ml_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("✅ Model and encoder saved as ml_model.pkl and encoder.pkl")
