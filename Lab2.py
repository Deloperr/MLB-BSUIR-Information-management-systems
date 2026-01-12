import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns
import os

# Создаем папку для сохранения графиков
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# 1. Импорт датасета
df = pd.read_csv("/home/ashryae/Загрузки/PriceCarPrediction/scrap price.csv")

print("Первые 5 строк датасета:")
print(df.head(), "\n")

print("Информация о данных:")
print(df.info(), "\n")

# 2. Проверка на пропуски и аномальные значения
print("Пропущенные значения:")
print(df.isnull().sum(), "\n")

# Визуализация пропусков
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap пропущенных значений')
plt.tight_layout()
plt.savefig('graphs/heatmap_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Heatmap сохранен: graphs/heatmap_missing_values.png")

# Boxplot для выявления выбросов (показывает медиану, квартили и выбросы)
# Выбросы — точки за пределами интервала [Q1 - 1.5 * IQR, Q3 + 1.5*IQR].
numerical_cols_original = df.select_dtypes(include=["int64", "float64"]).columns
main_numerical = ['price', 'enginesize', 'horsepower', 'curbweight', 'citympg', 'highwaympg']

if len(main_numerical) > 0:
    plt.figure(figsize=(14, 8))
    df[main_numerical].boxplot()
    plt.title('Boxplot для выявления выбросов (основные признаки)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graphs/boxplot_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Boxplot сохранен: graphs/boxplot_outliers.png")

# 3. Кодирование категориальных признаков
categorical_cols = ['fueltypes', 'aspiration', 'carbody', 'drivewheels', 'enginetype']
print("Категориальные признаки:", categorical_cols)

encoder = OneHotEncoder(sparse_output=False, drop="first") # Чтобы избежать мультиколлинеарность
encoded = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols)
)

# Объединение
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

print("\nДанные после кодирования категориальных признаков:")
print(df_encoded.head(), "\n")
print(f"Размерность после кодирования: {df_encoded.shape}")

# 3.1 Генерация новых признаков
df_encoded['power_to_weight'] = df_encoded['horsepower'] / df_encoded['curbweight'] # Отношение мощности к весу
df_encoded['mpg_combined'] = (df_encoded['citympg'] + df_encoded['highwaympg']) / 2 # Комбинированный расход топлива
df_encoded['size_ratio'] = df_encoded['carlength'] * df_encoded['carwidth'] * df_encoded['carheight'] # Объем автомобиля

print("Новые созданные признаки:")
new_features = ['power_to_weight', 'mpg_combined', 'size_ratio']
print(new_features)
print(df_encoded[new_features].head(), "\n")

# Визуализация новых признаков
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feature in enumerate(new_features):
    axes[i].hist(df_encoded[feature], bins=20, alpha=0.7, color='purple')
    axes[i].set_title(f'Распределение {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('graphs/new_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ График новых признаков сохранен: graphs/new_features_distribution.png")

# 4. Стандартизация и нормализация
main_numerical_cols = ['price', 'enginesize', 'horsepower', 'curbweight', 'citympg', 'highwaympg',
                       'power_to_weight', 'mpg_combined', 'size_ratio']
main_numerical_cols = [col for col in main_numerical_cols if col in df_encoded.columns]

print("Основные числовые признаки для масштабирования:", main_numerical_cols)

if len(main_numerical_cols) > 0:
    # Стандартизация (привел данные к нормальному распределению с μ=0 и σ=1)
    scaler_std = StandardScaler()
    df_std = df_encoded.copy()
    df_std[main_numerical_cols] = scaler_std.fit_transform(df_encoded[main_numerical_cols])

    # Нормализация
    scaler_minmax = MinMaxScaler()
    df_normalized = df_encoded.copy()
    df_normalized[main_numerical_cols] = scaler_minmax.fit_transform(df_encoded[main_numerical_cols])

    print("\nПосле StandardScaler (первые 5 строк):")
    print(df_std[main_numerical_cols].head(), "\n")

    print("После MinMaxScaler (первые 5 строк):")
    print(df_normalized[main_numerical_cols].head(), "\n")

    # Визуализация после обработки
    # Гистограммы после StandardScaler
    n_cols = min(3, len(main_numerical_cols))
    n_rows = (len(main_numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.ravel()

    for i, col in enumerate(main_numerical_cols):
        axes[i].hist(df_std[col], bins=20, alpha=0.7, color='blue')
        axes[i].set_title(f'Std: {col}')
        axes[i].tick_params(axis='x', rotation=45)

    for i in range(len(main_numerical_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Распределение после StandardScaler', fontsize=16)
    plt.tight_layout()
    plt.savefig('graphs/histograms_standard_scaler.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Гистограммы StandardScaler сохранены: graphs/histograms_standard_scaler.png")

    # Гистограммы после MinMaxScaler
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.ravel()

    for i, col in enumerate(main_numerical_cols):
        axes[i].hist(df_normalized[col], bins=20, alpha=0.7, color='green')
        axes[i].set_title(f'MinMax: {col}')
        axes[i].tick_params(axis='x', rotation=45)

    for i in range(len(main_numerical_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Распределение после MinMaxScaler', fontsize=16)
    plt.tight_layout()
    plt.savefig('graphs/histograms_minmax_scaler.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Гистограммы MinMaxScaler сохранены: graphs/histograms_minmax_scaler.png")

    # Boxplot после обработки
    plt.figure(figsize=(12, 8))
    df_std[main_numerical_cols].boxplot()
    plt.title('Boxplot после StandardScaler')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graphs/boxplot_after_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Boxplot после масштабирования сохранен: graphs/boxplot_after_scaling.png")

else:
    print("Нет числовых признаков для масштабирования")

# Дополнительная визуализация: сравнение исходных и преобразованных данных
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Исходное распределение цены
axes[0, 0].hist(df['price'], bins=20, alpha=0.7, color='red')
axes[0, 0].set_title('Исходное распределение цены')
axes[0, 0].set_xlabel('Price')
axes[0, 0].set_ylabel('Frequency')

# После StandardScaler
axes[0, 1].hist(df_std['price'], bins=20, alpha=0.7, color='blue')
axes[0, 1].set_title('Цена после StandardScaler')
axes[0, 1].set_xlabel('Standardized Price')
axes[0, 1].set_ylabel('Frequency')

# После MinMaxScaler
axes[1, 0].hist(df_normalized['price'], bins=20, alpha=0.7, color='green')
axes[1, 0].set_title('Цена после MinMaxScaler')
axes[1, 0].set_xlabel('Normalized Price')
axes[1, 0].set_ylabel('Frequency')

# Scatter plot нового признака
axes[1, 1].scatter(df_encoded['power_to_weight'], df_encoded['price'], alpha=0.6)
axes[1, 1].set_title('Мощность/Вес vs Цена')
axes[1, 1].set_xlabel('Power to Weight Ratio')
axes[1, 1].set_ylabel('Price')

plt.tight_layout()
plt.savefig('graphs/comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сравнительный анализ сохранен: graphs/comparison_analysis.png")