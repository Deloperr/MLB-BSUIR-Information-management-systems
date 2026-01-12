import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Создаем папку для графиков
os.makedirs('plots', exist_ok=True)


# Реализуем класс узла
class Node:
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t # пороговое значение
        self.true_branch = true_branch
        self.false_branch = false_branch


# Класс терминального узла (листа)
class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        classes = {}
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        prediction = max(classes, key=classes.get)
        return prediction


# 1) Расчет критерия Джини H(X) = 1 - ∑_{k=1}^{K} p_k^2
def gini(labels):
    if len(labels) == 0:
        return 0
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
    gini_val = 1.0
    for count in classes.values():
        probability = count / len(labels)
        gini_val -= probability ** 2
    return gini_val


# 2) Расчет прироста информации
def gain(left_labels, right_labels, root_gini):
    n = len(left_labels) + len(right_labels)
    p_left = len(left_labels) / n
    p_right = len(right_labels) / n
    weighted_gini = p_left * gini(left_labels) + p_right * gini(right_labels) # взвешенный критерий Джини
    information_gain = root_gini - weighted_gini
    return information_gain


# Разбиение датасета в узле
def split(data, labels, column_index, t):
    left_mask = data.iloc[:, column_index] <= t
    right_mask = data.iloc[:, column_index] > t
    true_data = data[left_mask]
    false_data = data[right_mask]
    true_labels = labels[left_mask]
    false_labels = labels[right_mask]
    return true_data, false_data, true_labels, false_labels


# 3) Критерии останова
def should_stop(labels, depth, max_depth=5, min_samples_leaf=3, min_samples_split=6):
    if depth >= max_depth:
        return True
    if len(labels) < min_samples_leaf:
        return True
    if len(labels) < min_samples_split:
        return True
    if len(np.unique(labels)) == 1:
        return True
    return False


# Нахождение наилучшего разбиения
def find_best_split(data, labels):
    min_samples_leaf = 3
    root_gini = gini(labels)
    best_gain = 0
    best_t = None
    best_index = None
    n_features = data.shape[1]

    for index in range(n_features):
        t_values = data.iloc[:, index].unique()
        for t in t_values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            if len(true_labels) < min_samples_leaf or len(false_labels) < min_samples_leaf:
                continue
            current_gain = gain(true_labels, false_labels, root_gini)
            if current_gain > best_gain:
                best_gain, best_t, best_index = current_gain, t, index
    return best_gain, best_t, best_index


# Построение дерева с помощью рекурсивной функции
def build_tree(data, labels, depth=0):
    if should_stop(labels, depth):
        return Leaf(data, labels)
    gain_val, t, index = find_best_split(data, labels)
    if gain_val == 0:
        return Leaf(data, labels)
    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
    true_branch = build_tree(true_data, true_labels, depth + 1)
    false_branch = build_tree(false_data, false_labels, depth + 1)
    return Node(index, t, true_branch, false_branch)


def classify_object(obj, node):
    if isinstance(node, Leaf):
        return node.prediction
    if isinstance(obj, pd.Series):
        value = obj.iloc[node.index]
    else:
        value = obj[node.index]
    if value <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)


def predict(data, tree):
    classes = []
    for i in range(len(data)):
        if isinstance(data, pd.DataFrame):
            obj = data.iloc[i]
        else:
            obj = data[i]
        prediction = classify_object(obj, tree)
        classes.append(prediction)
    return np.array(classes)


# 4) Функция подсчета точности
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual)


# Напечатаем ход нашего дерева
def print_tree(node, spacing="", feature_names=None):
    if isinstance(node, Leaf):
        print(spacing + "Прогноз:", node.prediction)
        return
    if feature_names is not None:
        feature_name = feature_names[node.index]
    else:
        feature_name = f"Feature_{node.index}"
    print(spacing + f'{feature_name} <= {node.t:.3f}')
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ", feature_names)
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ", feature_names)


# Функции для визуализации (сохранение в файлы)
def plot_feature_importance(tree, feature_names, title, filename):
    """Визуализация важности признаков"""
    if hasattr(tree, 'feature_importances_'):
        importances = tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
    else:
        importances = calculate_feature_importance(tree, feature_names)
        feature_importance_pairs = sorted(zip(feature_names, importances),
                                          key=lambda x: x[1], reverse=True)
        sorted_features = [pair[0] for pair in feature_importance_pairs]
        sorted_importances = [pair[1] for pair in feature_importance_pairs]

    plt.figure(figsize=(10, 6))
    plt.title(f"Важность признаков - {title}")
    bars = plt.bar(range(len(sorted_importances)), sorted_importances,
                   color='skyblue', alpha=0.7)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
    plt.ylabel('Важность')

    for bar, importance in zip(bars, sorted_importances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{importance:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: plots/{filename}.png")


def calculate_feature_importance(tree, feature_names):
    """Эвристический расчет важности признаков для нашего дерева"""
    importance_dict = {i: 0 for i in range(len(feature_names))}

    def traverse_tree(node):
        if isinstance(node, Leaf):
            return
        importance_dict[node.index] += 1
        traverse_tree(node.true_branch)
        traverse_tree(node.false_branch)

    traverse_tree(tree)
    total = sum(importance_dict.values())
    if total > 0:
        return [importance_dict[i] / total for i in range(len(feature_names))]
    else:
        return [0] * len(feature_names)


def plot_confusion_matrix_comparison(y_true, my_pred, sk_pred):
    """Сравнение матриц ошибок нашего и sklearn дерева"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cm_my = confusion_matrix(y_true, my_pred)
    sns.heatmap(cm_my, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
    ax1.set_title('Матрица ошибок - Наше дерево')
    ax1.set_xlabel('Предсказанный класс')
    ax1.set_ylabel('Истинный класс')

    cm_sk = confusion_matrix(y_true, sk_pred)
    sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title('Матрица ошибок - Sklearn дерево')
    ax2.set_xlabel('Предсказанный класс')
    ax2.set_ylabel('Истинный класс')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/confusion_matrix_comparison.png")


def plot_data_distribution(X, y, feature_names):
    """Визуализация распределения данных"""
    n_features = X.shape[1]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.ravel()

    for i in range(n_features):
        if n_rows > 1:
            ax = axes[i]
        else:
            ax = axes[i] if n_cols > 1 else axes

        for class_label in np.unique(y):
            mask = y == class_label
            ax.hist(X[mask, i], alpha=0.7, label=f'Class {class_label}', bins=15)
        ax.set_title(f'Распределение {feature_names[i]}')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Частота')
        ax.legend()

    for i in range(n_features, n_rows * n_cols):
        if n_rows > 1:
            fig.delaxes(axes[i])
        else:
            if n_cols > 1:
                fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('plots/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/data_distribution.png")


def plot_accuracy_comparison(my_train_acc, my_test_acc, sk_train_acc, sk_test_acc):
    """Сравнение точности на обучающей и тестовой выборках"""
    labels = ['Обучающая', 'Тестовая']
    my_scores = [my_train_acc, my_test_acc]
    sk_scores = [sk_train_acc, sk_test_acc]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, my_scores, width, label='Наше дерево', alpha=0.7, color='lightcoral')
    rects2 = ax.bar(x + width / 2, sk_scores, width, label='Sklearn дерево', alpha=0.7, color='lightblue')

    ax.set_ylabel('Точность')
    ax.set_title('Сравнение точности моделей')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('plots/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/accuracy_comparison.png")


def plot_tree_depth_analysis(my_tree, sk_tree):
    """Анализ глубины деревьев"""

    def get_tree_depth(tree):
        if isinstance(tree, Leaf):
            return 0
        return 1 + max(get_tree_depth(tree.true_branch), get_tree_depth(tree.false_branch))

    my_depth = get_tree_depth(my_tree)
    sk_depth = sk_tree.get_depth()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Наше дерево', 'Sklearn дерево'], [my_depth, sk_depth],
                  color=['lightcoral', 'lightblue'], alpha=0.7)

    ax.set_ylabel('Глубина дерева')
    ax.set_title('Сравнение глубины деревьев')

    for bar, depth in zip(bars, [my_depth, sk_depth]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{depth}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/tree_depth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/tree_depth.png")

    return my_depth, sk_depth


def save_sklearn_tree_visualization(sk_tree, feature_names):
    """Сохранение визуализации sklearn дерева"""
    plt.figure(figsize=(20, 10))
    plot_tree(sk_tree, feature_names=feature_names, filled=True,
              class_names=['Class_0', 'Class_1'], rounded=True, fontsize=10)
    plt.title("Дерево решений (Sklearn)", fontsize=16)
    plt.savefig('plots/sklearn_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/sklearn_tree.png")


# Главная функция
if __name__ == "__main__":
    print("=" * 60)
    print("ДЕРЕВЬЯ РЕШЕНИЙ")
    print("=" * 60)
    print("Создана папка 'plots' для сохранения графиков")

    # Генерируем датасет
    X, y = make_classification(n_samples=200, n_features=4, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)

    feature_names = [f'Признак_{i}' for i in range(4)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series,
                                                        test_size=0.3,
                                                        random_state=42)

    print("\n1. ИНФОРМАЦИЯ О ДАННЫХ:")
    print("=" * 30)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"Классы в обучающей выборке: {np.unique(y_train, return_counts=True)}")
    print(f"Классы в тестовой выборке: {np.unique(y_test, return_counts=True)}")

    # 1. Визуализация распределения данных
    print("\n2. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ДАННЫХ:")
    print("=" * 30)
    plot_data_distribution(X_train.values, y_train.values, feature_names)

    # Обучаем модели
    print("\n3. ОБУЧЕНИЕ МОДЕЛЕЙ:")
    print("=" * 30)
    print("Обучаем наше дерево...")
    my_tree = build_tree(X_train, y_train)
    print("Обучаем DecisionTreeClassifier из sklearn...")
    sk_tree = DecisionTreeClassifier(random_state=42, min_samples_leaf=3, min_samples_split=6)
    sk_tree.fit(X_train, y_train)

    # Предсказания
    my_train_pred = predict(X_train, my_tree)
    my_test_pred = predict(X_test, my_tree)
    sk_train_pred = sk_tree.predict(X_train)
    sk_test_pred = sk_tree.predict(X_test)

    # Расчет точности
    my_train_acc = accuracy_metric(y_train.values, my_train_pred)
    my_test_acc = accuracy_metric(y_test.values, my_test_pred)
    sk_train_acc = accuracy_score(y_train, sk_train_pred)
    sk_test_acc = accuracy_score(y_test, sk_test_pred)

    print("\n4. РЕЗУЛЬТАТЫ:")
    print("=" * 30)
    print(f"Наше дерево - Обучающая: {my_train_acc:.4f}, Тестовая: {my_test_acc:.4f}")
    print(f"Sklearn дерево - Обучающая: {sk_train_acc:.4f}, Тестовая: {sk_test_acc:.4f}")

    # Графики
    print("\n5. СОЗДАНИЕ ГРАФИКОВ:")
    print("=" * 30)
    plot_accuracy_comparison(my_train_acc, my_test_acc, sk_train_acc, sk_test_acc)
    my_depth, sk_depth = plot_tree_depth_analysis(my_tree, sk_tree)
    plot_confusion_matrix_comparison(y_test.values, my_test_pred, sk_test_pred)
    plot_feature_importance(my_tree, feature_names, "Наше дерево", "our_tree_feature_importance")
    plot_feature_importance(sk_tree, feature_names, "Sklearn дерево", "sklearn_tree_feature_importance")
    save_sklearn_tree_visualization(sk_tree, feature_names)

    print("\n6. ДЕТАЛЬНЫЙ АНАЛИЗ:")
    print("=" * 30)
    print("Наше дерево (тестовая выборка):")
    print(classification_report(y_test, my_test_pred))
    print("Sklearn дерево (тестовая выборка):")
    print(classification_report(y_test, sk_test_pred))

    print("\n7. СТРУКТУРА НАШЕГО ДЕРЕВА:")
    print("=" * 30)
    print_tree(my_tree, feature_names=feature_names)

    print("\n8. СРАВНЕНИЕ:")
    print("=" * 30)
    print(f"Глубина нашего дерева: {my_depth}")
    print(f"Глубина sklearn дерева: {sk_depth}")
    print(f"Совпадают предсказания (тест): {np.array_equal(my_test_pred, sk_test_pred)}")
    print(f"Совпадают предсказания (train): {np.array_equal(my_train_pred, sk_train_pred)}")