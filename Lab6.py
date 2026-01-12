# Градиентный бустинг
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# 1. Загрузка данных
print("1. Загрузка и первичный анализ данных")
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
print(f"Размерность исходных данных: {X.shape}")
print("Целевые классы:", list(data.target_names))
print()

# 2. Предобработка данных
print("2. Предобработка данных (импутация, масштабирование)")
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print("Пропуски после обработки:", X_scaled.isna().sum().sum())
print("Масштабирование завершено.\n")

#  3. Разделение данных
print(" 3. Разделение выборки на обучающую и тестовую ")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Размеры: обучающая {X_train.shape}, тестовая {X_test.shape}\n")

#  4. Обучение базовой модели RandomForest
print(" 4. Базовая модель: Случайный лес (Random Forest) ")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

metrics_rf = {
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf),
    "recall": recall_score(y_test, y_pred_rf),
    "f1": f1_score(y_test, y_pred_rf)
}
print("Результаты (Random Forest):")
for k, v in metrics_rf.items():
    print(f"{k.capitalize():<10}: {v:.4f}")
print()

#  5. Базовая модель Gradient Boosting
print(" 5. Базовая модель: Градиентный бустинг (по умолчанию) ")
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

metrics_gb = {
    "accuracy": accuracy_score(y_test, y_pred_gb),
    "precision": precision_score(y_test, y_pred_gb),
    "recall": recall_score(y_test, y_pred_gb),
    "f1": f1_score(y_test, y_pred_gb)
}
print("Результаты (Gradient Boosting):")
for k, v in metrics_gb.items():
    print(f"{k.capitalize():<10}: {v:.4f}")
print()

#  6. Матрица ошибок
print(" 6. Матрица ошибок для Gradient Boosting ")
cm = confusion_matrix(y_test, y_pred_gb)
print("Матрица ошибок (строки — истинные классы, столбцы — предсказанные):")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Матрица ошибок — Gradient Boosting")
plt.xlabel("Предсказанные классы")
plt.ylabel("Истинные классы")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Матрица ошибок сохранена в файл 'confusion_matrix.png'\n")

#  7. Подбор гиперпараметров (GridSearchCV)
print(" 7. Подбор гиперпараметров с помощью GridSearchCV ")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.05, 0.1]
}
gb_base = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gb_base, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nЛучшие параметры:")
print(grid.best_params_)
print(f"Лучший средний F1 по кросс-валидации: {grid.best_score_:.4f}\n")

best_gb = grid.best_estimator_
y_pred_best = best_gb.predict(X_test)

metrics_best = {
    "accuracy": accuracy_score(y_test, y_pred_best),
    "precision": precision_score(y_test, y_pred_best),
    "recall": recall_score(y_test, y_pred_best),
    "f1": f1_score(y_test, y_pred_best)
}
print("Результаты модели с подобранными параметрами:")
for k, v in metrics_best.items():
    print(f"{k.capitalize():<10}: {v:.4f}")
print()

# 8. Исследование влияния learning_rate и n_estimators
print(" 8. Влияние скорости обучения и количества деревьев ")
lr_values = [0.01, 0.05, 0.1]
n_est_values = [50, 100, 200]

results = []
for lr in lr_values:
    for n in n_est_values:
        model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        results.append({"learning_rate": lr, "n_estimators": n, "f1": f1})

df_results = pd.DataFrame(results)
print("Таблица влияния параметров:")
print(df_results.pivot(index='n_estimators', columns='learning_rate', values='f1'))
print()

for lr in lr_values:
    subset = df_results[df_results["learning_rate"] == lr]
    plt.figure(figsize=(7, 4))
    plt.plot(subset["n_estimators"], subset["f1"], marker='o')
    plt.title(f"Зависимость F1 от n_estimators (learning_rate={lr})")
    plt.xlabel("n_estimators")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.savefig(f"f1_vs_estimators_lr{lr}.png")
    plt.close()
print("Графики сохранены: f1_vs_estimators_lr*.png\n")

# 9. Сводная таблица
summary = pd.DataFrame([
    {"model": "RandomForest (default)", **metrics_rf},
    {"model": "GradientBoosting (default)", **metrics_gb},
    {"model": "GradientBoosting (best params)", **metrics_best}
])
print("9. Сравнение моделей")
print(summary)
summary.to_csv("gb_lab_summary_metrics.csv", index=False)
df_results.to_csv("gb_lab_lr_nestim_results.csv", index=False)
print("\nРезультаты сохранены в файлы:")
print(" - gb_lab_summary_metrics.csv")
print(" - gb_lab_lr_nestim_results.csv")