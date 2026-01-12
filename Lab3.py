import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Генерация датасета
print("1. Генерация датасета")
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

# Сохранение исходного датасета в CSV
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df_original = pd.DataFrame(X, columns=feature_names)
df_original['target'] = y
df_original.to_csv('dataset_original.csv', index=False)
print("Исходный датасет сохранен в 'dataset_original.csv'")

# Добавляем свободный член (столбец из единиц)
X = np.column_stack([np.ones(X.shape[0]), X])

# Масштабирование данных (кроме столбца единиц)
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Сохранение масштабированного датасета в CSV
feature_names_scaled = ['bias'] + [f'feature_{i}_scaled' for i in range(X.shape[1] - 1)]
df_scaled = pd.DataFrame(X, columns=feature_names_scaled)
df_scaled['target_scaled'] = y
df_scaled.to_csv('dataset_scaled.csv', index=False)
print("Масштабированный датасет сохранен в 'dataset_scaled.csv'")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение train/test выборок
train_df = pd.DataFrame(X_train, columns=feature_names_scaled)
train_df['target'] = y_train
train_df.to_csv('train_dataset.csv', index=False)
print("Обучающая выборка сохранена в 'train_dataset.csv'")

test_df = pd.DataFrame(X_test, columns=feature_names_scaled)
test_df['target'] = y_test
test_df.to_csv('test_dataset.csv', index=False)
print("Тестовая выборка сохранена в 'test_dataset.csv'")

print(f"Размерность данных: X_train {X_train.shape}, y_train {y_train.shape}")


# 2. Функция градиентного спуска
def gradient_descent(X, y, learning_rate=0.01, max_iter=1000, epsilon=1e-6, regularization=None, alpha=0):
    """
    Градиентный спуск для линейной регрессии
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    history = {'weights': [], 'mse': []}

    for i in range(max_iter):
        # Предсказание
        y_pred = X @ weights

        # Вычисление градиента
        error = y_pred - y
        gradient = (X.T @ error) / n_samples #  ∇L(w_old) = 2/n * Xᵀ(X · w_old - y)

        # Добавление регуляризации
        if regularization == 'l2':
            gradient += alpha * weights / n_samples
        elif regularization == 'l1':
            gradient += alpha * np.sign(weights) / n_samples

        # Сохранение предыдущих весов
        old_weights = weights.copy()

        # Обновление весов
        weights = weights - learning_rate * gradient

        # Сохранение истории
        history['weights'].append(weights.copy())
        history['mse'].append(mean_squared_error(y, X @ weights))

        # Проверка критерия остановки
        if np.linalg.norm(weights - old_weights) < epsilon:
            print(f"Сходимость достигнута на итерации {i + 1}")
            break

    return weights, history


# 3. Эксперименты со скоростью обучения
print("\n3. Эксперименты со скоростью обучения...")
learning_rates = [0.001, 0.01, 0.1, 0.5]
plt.figure(figsize=(12, 8))

# Сохраним результаты экспериментов в CSV
lr_results = []

for lr in learning_rates:
    weights, history = gradient_descent(X_train, y_train, learning_rate=lr, max_iter=1000)
    plt.plot(history['mse'][:100], label=f'LR={lr}')

    # Сохраняем результаты
    final_mse = history['mse'][-1]
    lr_results.append({
        'learning_rate': lr,
        'final_mse': final_mse,
        'iterations': len(history['mse'])
    })

# Сохраняем результаты экспериментов
lr_df = pd.DataFrame(lr_results)
lr_df.to_csv('learning_rate_experiments.csv', index=False)
print("Результаты экспериментов со скоростью обучения сохранены в 'learning_rate_experiments.csv'")

plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.title('Влияние скорости обучения на сходимость')
plt.legend()
plt.grid(True)
plt.savefig('learning_rates.png')  # Сохраняем график
plt.close()  # Закрываем figure чтобы избежать warning


# 4. Стохастический градиентный спуск
def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iter=1000, epsilon=1e-6,
                                regularization=None, alpha=0, batch_size=1):
    """
    Стохастический градиентный спуск
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    history = {'weights': [], 'mse': []}

    for i in range(max_iter):
        # Случайный выбор батча
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        # Предсказание для батча
        y_pred = X_batch @ weights

        # Вычисление градиента для батча
        error = y_pred - y_batch
        gradient = (X_batch.T @ error) / batch_size

        # Добавление регуляризации
        if regularization == 'l2': # ограничение весов, штраф за сложность модели
            gradient += alpha * weights / n_samples
        elif regularization == 'l1': # может занулять веса некоторых признаков
            gradient += alpha * np.sign(weights) / n_samples

        # Сохранение предыдущих весов
        old_weights = weights.copy()

        # Обновление весов
        weights = weights - learning_rate * gradient

        # Вычисление MSE на всей выборке для мониторинга
        full_mse = mean_squared_error(y, X @ weights)
        history['mse'].append(full_mse)
        history['weights'].append(weights.copy())

        # Проверка критерия остановки
        if np.linalg.norm(weights - old_weights) < epsilon:
            print(f"Сходимость достигнута на итерации {i + 1}")
            break

    return weights, history


# 4. Сравнение градиентного и стохастического градиентного спуска
print("\n4. Сравнение градиентного и стохастического градиентного спуска")

# Градиентный спуск
print("Запуск градиентного спуска")
weights_gd, history_gd = gradient_descent(X_train, y_train, learning_rate=0.01, max_iter=500)

# Стохастический градиентный спуск
print("Запуск стохастического градиентного спуска")
weights_sgd, history_sgd = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01,
                                                       max_iter=500, batch_size=32)

# Сохраняем историю обучения в CSV
history_df = pd.DataFrame({
    'iteration': range(len(history_gd['mse'])),
    'gd_mse': history_gd['mse'],
    'sgd_mse': history_sgd['mse'][:len(history_gd['mse'])]  # Обрезаем до одинаковой длины
})
history_df.to_csv('training_history.csv', index=False)
print("История обучения сохранена в 'training_history.csv'")

plt.figure(figsize=(12, 8))
plt.plot(history_gd['mse'], label='Градиентный спуск', linewidth=2)
plt.plot(history_sgd['mse'], label='Стохастический градиентный спуск', linewidth=2)
plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.title('Сравнение скорости сходимости')
plt.legend()
plt.grid(True)
plt.savefig('gd_vs_sgd.png')
plt.close()

# Предсказания на тестовой выборке
y_pred_gd = X_test @ weights_gd
y_pred_sgd = X_test @ weights_sgd

mse_gd = mean_squared_error(y_test, y_pred_gd)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)

print(f"\nРезультаты на тестовой выборке:")
print(f"MSE градиентного спуска: {mse_gd:.6f}")
print(f"MSE стохастического градиентного спуска: {mse_sgd:.6f}")

# Сохраняем предсказания
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred_gd': y_pred_gd,
    'y_pred_sgd': y_pred_sgd
})
predictions_df.to_csv('predictions.csv', index=False)
print("Предсказания моделей сохранены в 'predictions.csv'")

# 5. L2 регуляризация (для нечетных вариантов)
print("\n6. L2 регуляризация...")

# Коэффициенты регуляризации
alphas = np.logspace(-3, 3, 20)
weights_history = []
mse_history = []

for alpha in alphas:
    weights, history = gradient_descent(X_train, y_train, learning_rate=0.01,
                                        max_iter=1000, regularization='l2', alpha=alpha)
    weights_history.append(weights)
    mse_history.append(history['mse'][-1])

weights_history = np.array(weights_history)

# Сохраняем результаты регуляризации
regularization_results = pd.DataFrame(weights_history, columns=feature_names_scaled)
regularization_results['alpha'] = alphas
regularization_results['final_mse'] = mse_history
regularization_results.to_csv('regularization_results.csv', index=False)
print("Результаты регуляризации сохранены в 'regularization_results.csv'")

# График изменения весов в зависимости от коэффициента регуляризации
plt.figure(figsize=(12, 8))
for i in range(weights_history.shape[1]):
    plt.plot(alphas, weights_history[:, i], label=f'w_{i}', linewidth=2)

plt.xscale('log')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('Значение веса')
plt.title('Изменение весов при L2 регуляризации')
plt.legend()
plt.grid(True)
plt.savefig('l2_regularization.png')
plt.close()

# Сравнение с sklearn для проверки
print("\nСравнение с LinearRegression из sklearn:")
sklearn_model = LinearRegression()
sklearn_model.fit(X_train[:, 1:], y_train)
sklearn_weights = np.concatenate([[sklearn_model.intercept_], sklearn_model.coef_])

print("Веса из sklearn:", [f"{w:.6f}" for w in sklearn_weights])
print("Веса из градиентного спуска:", [f"{w:.6f}" for w in weights_gd])

# Сохраняем веса моделей
weights_comparison = pd.DataFrame({
    'feature': feature_names_scaled,
    'sklearn_weights': sklearn_weights,
    'gd_weights': weights_gd,
    'sgd_weights': weights_sgd
})
weights_comparison.to_csv('weights_comparison.csv', index=False)
print("Сравнение весов сохранено в 'weights_comparison.csv'")

# Анализ разницы в весах
weight_diff = np.abs(sklearn_weights - weights_gd)
print(f"Максимальная разница в весах: {np.max(weight_diff):.6f}")

# Дополнительный анализ: влияние размера батча на SGD
print("\nДополнительно: влияние размера батча на SGD...")
batch_sizes = [1, 10, 32, 100]
plt.figure(figsize=(12, 8))

batch_results = []

for batch_size in batch_sizes:
    print(f"Обучение с batch_size={batch_size}...")
    weights, history = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01,
                                                   max_iter=500, batch_size=batch_size)
    plt.plot(history['mse'][:200], label=f'Batch size={batch_size}')

    # Сохраняем результаты
    batch_results.append({
        'batch_size': batch_size,
        'final_mse': history['mse'][-1],
        'iterations': len(history['mse'])
    })

# Сохраняем результаты экспериментов с batch size
batch_df = pd.DataFrame(batch_results)
batch_df.to_csv('batch_size_experiments.csv', index=False)
print("Результаты экспериментов с batch size сохранены в 'batch_size_experiments.csv'")

plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.title('Влияние размера батча на сходимость SGD')
plt.legend()
plt.grid(True)
plt.savefig('batch_sizes.png')
plt.close()

print("\nВсе файлы сохранены:")
print("- dataset_original.csv: исходный датасет")
print("- dataset_scaled.csv: масштабированный датасет")
print("- train_dataset.csv: обучающая выборка")
print("- test_dataset.csv: тестовая выборка")
print("- learning_rate_experiments.csv: эксперименты со скоростью обучения")
print("- training_history.csv: история обучения GD и SGD")
print("- predictions.csv: предсказания на тестовой выборке")
print("- regularization_results.csv: результаты регуляризации")
print("- weights_comparison.csv: сравнение весов моделей")
print("- batch_size_experiments.csv: эксперименты с размером батча")
print("\nГрафики сохранены в файлы:")
print("- learning_rates.png: влияние скорости обучения")
print("- gd_vs_sgd.png: сравнение GD и SGD")
print("- l2_regularization.png: L2 регуляризация")
print("- batch_sizes.png: влияние размера батча")