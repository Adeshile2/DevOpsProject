import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("synthetic_titanic_dataset.csv")

# Define features and target variable
features = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked']
target = 'Survived'

X = df[features]  # Feature matrix
y = df[target]  # Target variable

# Handle missing values
df.dropna(subset=['Age', 'Embarked'], inplace=True)

# Define categorical features for encoding
categorical_features = ['Sex', 'Embarked']

# Apply One-Hot Encoding
encoder = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')

# Transform data
X_transformed = encoder.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save the model and encoder
joblib.dump(model, "ml_model.pkl")
joblib.dump(encoder, "encoder.pkl")
