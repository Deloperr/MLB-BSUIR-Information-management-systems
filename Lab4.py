import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
import warnings

warnings.filterwarnings('ignore')

# Создаем папку для сохранения графиков
os.makedirs('graphs', exist_ok=True)

# Загрузка датасета Breast Cancer Wisconsin
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"Размер датасета: {X.shape}")
print(f"Классы: {data.target_names}")
print(f"Распределение классов:\n{y.value_counts()}")
print(f"Класс 0: {data.target_names[0]} - {y.value_counts()[0]} samples")
print(f"Класс 1: {data.target_names[1]} - {y.value_counts()[1]} samples")

# ПРЕДОБРАБОТКА ДАННЫХ
print("\n" + "=" * 60)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Среднее после стандартизации: {np.mean(X_scaled, axis=0)[:3].values}")
print(f"Стд. отклонение после стандартизации: {np.std(X_scaled, axis=0)[:3].values}")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
y_train_np = y_train.values
X_train_np = X_train.values

print(f"\nРазделение данных:")
print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

# РЕАЛИЗАЦИЯ МОДЕЛЕЙ

# ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
print("\n" + "=" * 60)
print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("=" * 60)

# Реализация сигмоидной функции
def sigmoid(x):
    """Сигмоидная функция - преобразует линейную комбинацию в вероятность σ(z) = e^z / (1 + e^z)"""
    return np.exp(x) / (1 + np.exp(x))

# Демонстрация сигмоидной функции
z_values = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z_values)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(z_values, sigmoid_values, linewidth=2)
plt.title('Сигмоидная функция\n(логистическая функция)')
plt.xlabel('z = wᵀx + b')
plt.ylabel('σ(z)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 0.5')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='z=0')
plt.legend()

# Обучение логистической регрессии
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print(f"Коэффициенты модели: {lr_model.coef_[0][:5]}...")
print(f"Смещение (bias): {lr_model.intercept_[0]:.4f}")

# МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)
print("\n" + "=" * 60)
print("МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)")
print("=" * 60)
print("Ищет гиперплоскость с максимальным зазором между классами")
print("Используем RBF ядро для нелинейного разделения")

svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_np, y_train_np)
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)[:, 1]

print(f"Количество опорных векторов: {len(svm_model.support_vectors_)}")

# Правильный способ подсчета опорных векторов по классам
support_indices = svm_model.support_
support_labels = y_train_np[support_indices]

print(f"Опорные векторы класса 0: {np.sum(support_labels == 0)}")
print(f"Опорные векторы класса 1: {np.sum(support_labels == 1)}")
print("Опорные векторы - это точки, определяющие границу решения")

# ДЕРЕВО РЕШЕНИЙ (CART)
print("\n" + "=" * 60)
print("ДЕРЕВО РЕШЕНИЙ (CART)")
print("=" * 60)
print("Строит иерархическую структуру правил 'если-то'")

dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# Визуализация важности признаков для дерева
feature_importance = dt_model.feature_importances_
important_features_idx = np.argsort(feature_importance)[-10:]
important_features = X.columns[important_features_idx]

plt.subplot(2, 3, 2)
plt.barh(range(len(important_features)), feature_importance[important_features_idx])
plt.yticks(range(len(important_features)), important_features)
plt.title('Важность признаков (Decision Tree)')
plt.xlabel('Важность')

# МЕТРИКИ КАЧЕСТВА
print("\n" + "=" * 60)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("=" * 60)

models = {
    'Logistic Regression': (lr_pred, lr_prob),
    'SVM': (svm_pred, svm_prob),
    'Decision Tree': (dt_pred, dt_prob)
}

# Матрицы ошибок
print("\nМАТРИЦЫ ОШИБОК")
print("TP - True Positive (верно предсказанные класс 1)")
print("FP - False Positive (ложные срабатывания)")
print("FN - False Negative (пропущенные класс 1)")
print("TN - True Negative (верно предсказанные класс 0)")

for i, (name, (pred, prob)) in enumerate(models.items()):
    plt.subplot(2, 3, i + 3)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'{name}\nConfusion Matrix')

    TN, FP, FN, TP = cm.ravel()
    print(f"\n{name}:")
    print(f"  TP={TP} (Истинно положительные)")
    print(f"  FP={FP} (Ложно положительные)")
    print(f"  FN={FN} (Ложно отрицательные)")
    print(f"  TN={TN} (Истинно отрицательные)")

# ROC-кривые
print("\nROC-КРИВЫЕ")
print("Показывают качество разделения классов при разных порогах")

plt.subplot(2, 3, 6)
for name, (pred, prob) in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный классификатор')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend()
plt.grid(True, alpha=0.3)

# Сохраняем первый блок графиков
plt.tight_layout()
plt.savefig('graphs/основные_графики.png', dpi=300, bbox_inches='tight')
plt.savefig('graphs/основные_графики.pdf', bbox_inches='tight')
plt.show()

# Основные метрики
print("\nОСНОВНЫЕ МЕТРИКИ КАЧЕСТВА")

results = []
for name, (pred, prob) in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc_roc = roc_auc_score(y_test, prob)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    })

    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f} - доля верных предсказаний")
    print(f"  Precision: {precision:.4f} - точность положительных предсказаний")
    print(f"  Recall:    {recall:.4f} - полнота (сколько нашли из реальных класса 1)")
    print(f"  F1-Score:  {f1:.4f} - гармоническое среднее Precision и Recall")
    print(f"  AUC-ROC:   {auc_roc:.4f} - площадь под ROC-кривой")

# СРАВНИТЕЛЬНЫЙ АНАЛИЗ
print("\n" + "=" * 60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\nСводная таблица метрик:")
print(results_df.round(4))

# Определение лучшей модели по F1-Score
best_model_idx = results_df['F1-Score'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_f1 = results_df.loc[best_model_idx, 'F1-Score']

print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model} (F1-Score = {best_f1:.4f})")

# Дополнительная визуализация - сравнение метрик
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics_to_plot):
    row, col = i // 2, i % 2
    values = [results_df[metric].iloc[j] for j in range(len(results_df))]
    bars = ax[row, col].bar(results_df['Model'], values, color=colors[i], alpha=0.7)
    ax[row, col].set_title(f'Сравнение {metric}')
    ax[row, col].set_ylabel(metric)
    ax[row, col].set_ylim(0.8, 1.0)

    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax[row, col].text(bar.get_x() + bar.get_width() / 2., height,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/сравнение_метрик.png', dpi=300, bbox_inches='tight')
plt.savefig('graphs/сравнение_метрик.pdf', bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("СОХРАНЕНИЕ ГРАФИКОВ")
print("=" * 60)

# 1. Сигмоидная функция отдельно
plt.figure(figsize=(8, 6))
plt.plot(z_values, sigmoid_values, linewidth=3)
plt.title('Сигмоидная функция', fontsize=14)
plt.xlabel('z = wᵀx + b', fontsize=12)
plt.ylabel('σ(z)', fontsize=12)
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Порог 0.5')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z=0')
plt.legend()
plt.savefig('graphs/сигмоидная_функция.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Важность признаков отдельно
plt.figure(figsize=(10, 6))
plt.barh(range(len(important_features)), feature_importance[important_features_idx], color='lightblue')
plt.yticks(range(len(important_features)), important_features)
plt.title('Важность признаков (Decision Tree)', fontsize=14)
plt.xlabel('Важность', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('graphs/важность_признаков.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. ROC-кривые отдельно
plt.figure(figsize=(8, 6))
for name, (pred, prob) in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайный классификаator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('graphs/roc_кривые.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Матрицы ошибок отдельно для каждой модели
for name, (pred, prob) in models.items():
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Матрица ошибок: {name}', fontsize=14)
    plt.savefig(f'graphs/матрица_ошибок_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Все графики сохранены в папку 'graphs':")
print("   - основные_графики.png/pdf")
print("   - сравнение_метрик.png/pdf")
print("   - сигмоидная_функция.png")
print("   - важность_признаков.png")
print("   - roc_кривые.png")
print("   - матрица_ошибок_для_каждой_модели.png")

# Сохраняем таблицу с результатами
results_df.to_csv('graphs/результаты_моделей.csv', index=False)
print("   - результаты_моделей.csv")