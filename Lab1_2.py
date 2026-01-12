import math

# 1
def solve(a: int, b: int, c: int):
    d = b ** 2 - 4 * a * c
    if d < 0:
        return []
    elif d == 0:
        return [-b / (2 * a)]
    else:
        x1 = (-b - math.sqrt(d)) / (2 * a)
        x2 = (-b + math.sqrt(d)) / (2 * a)
        return sorted([x1, x2])


def task1():
    a, b, c = map(int, input("Введите коэффициенты a b c: ").split())
    roots = solve(a, b, c)
    print("Корни:", roots if roots else "Нет действительных корней")


# 2
def universal(arg):
    vowels = "аеёиоуыэюяaeiou"
    consonants = "бвгджзйклмнпрстфхцчшщbcdfghjklmnpqrstvwxyz"

    if isinstance(arg, list):
        product = 1
        for i in range(1, len(arg), 2):
            product *= arg[i]
        print("Произведение элементов с нечетными номерами:", product)
        if arg:
            max_el = max(arg)
            arg.remove(max_el)
            print("Список без максимального элемента:", arg)
        return

    elif isinstance(arg, dict):
        sorted_dict = dict(sorted(arg.items()))
        print("Отсортированный словарь:", sorted_dict)
        return

    elif isinstance(arg, int):
        if arg < 2:
            print(f"{arg} не является простым")
            return
        for i in range(2, int(math.sqrt(arg)) + 1):
            if arg % i == 0:
                print(f"{arg} не является простым")
                return
        print(f"{arg} простое число")
        return

    elif isinstance(arg, str):
        v_count = sum(1 for ch in arg.lower() if ch in vowels)
        c_count = sum(1 for ch in arg.lower() if ch in consonants)
        print("Гласных:", v_count)
        print("Согласных:", c_count)
        return

    else:
        print("Тип аргумента не поддерживается")


def task2():
    print("Проверка на всех типах:")
    universal([1, 2, 3, 4, 5])
    universal({3: "a", 1: "b", 2: "c"})
    universal(17)
    universal("Hello World!")


# 3
def task3():
    n, m = map(int, input("Введите размеры матрицы n m: ").split())
    matrix = [list(map(int, input().split())) for _ in range(n)]

    zero_row = -1
    for i in range(n):
        if all(x == 0 for x in matrix[i]):
            zero_row = i
            break

    if zero_row == -1:
        print("Нет строки, где все элементы равны 0")
        return

    print("Первая строка с нулями:", zero_row)

    for i in range(n):
        matrix[i][zero_row] //= 2

    print("Матрица после изменений:")
    for row in matrix:
        print(*row)

# 4
def task4():
    try:
        x = int(input("Введите число: "))
        print("Результат деления 100 на x:", 100 / x)
    except ZeroDivisionError:
        print("Ошибка: деление на ноль!")
    except ValueError:
        print("Ошибка: введено не число!")
    finally:
        print("Блок finally выполнен")


if __name__ == "__main__":
    tasks = {
        "1": task1,
        "2": task2,
        "3": task3,
        "4": task4
    }

    while True:
        print("\nЗадания:")
        for i in range(1, 5):
            print(i)
        print("0. Выход")

        choice = input("Выберите задание: ")
        if choice == "0":
            break
        if choice in tasks:
            tasks[choice]()
        else:
            print("Неверный выбор")
