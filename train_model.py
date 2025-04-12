import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
import os

# Задайте путь к файлу. Файл CSV должен находиться в папке data
file_path = os.path.join("data", "Laptop_price.csv")

# Загрузка данных
df = pd.read_csv(file_path)

# Разделение на признаки и целевую переменную
X = df.drop(columns=['Price'])
y = df['Price']

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение колонок с числовыми и категориальными данными
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Пайплайны для предобработки числовых и категориальных данных
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Объединение трансформеров в единый ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Формирование полного пайплайна для обучения модели
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Обучение пайплайна на обучающих данных
pipeline.fit(X_train, y_train)

# Создание папки для модели, если её еще нет
model_dir = os.path.join("models")
os.makedirs(model_dir, exist_ok=True)

# Сохранение обученного пайплайна в виде pickle-файла
model_path = os.path.join(model_dir, "laptop_price_model.pkl")
joblib.dump(pipeline, model_path)

print("Обучение завершено, модель сохранена по пути:", model_path)