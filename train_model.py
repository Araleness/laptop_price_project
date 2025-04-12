import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib

# Загрузка данных
file_path = 'data/Laptop_price.csv'  # убедись, что путь корректный
df = pd.read_csv(file_path)

# Разделение данных: признаки и целевая переменная
X = df.drop(columns=['Price'])
y = df['Price']

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение числовых и категориальных признаков
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Создание пайплайна для числовых данных
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Создание пайплайна для категориальных данных
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Объединение преобразований в ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Создание полного пайплайна с моделью
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Создание папки для модели, если её ещё нет
os.makedirs("models", exist_ok=True)

# Сохранение обученной модели
model_file = 'models/laptop_price_model.pkl'
joblib.dump(pipeline, model_file)
print(f"✅ Модель обучена и сохранена в {model_file}")