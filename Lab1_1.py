def task1():
    num = input("Введите натуральное число: ")
    digits = [int(d) for d in num]
    print("Максимальная цифра:", max(digits))
    print("Обратное число:", num[::-1])


def task2():
    text = input("Введите текст: ").lower()
    vowels = "аеёиоуыэюяaeiou"
    consonants = "бвгджзйклмнпрстфхцчшщbcdfghjklmnpqrstvwxyz"
    v_count = sum(1 for ch in text if ch in vowels)
    c_count = sum(1 for ch in text if ch in consonants)
    words = len(text.split())

    print("Гласных:", v_count)
    print("Согласных:", c_count)
    print("Количество слов:", words)

    if v_count == c_count:
        print("Гласные буквы:", ", ".join(sorted(set(ch for ch in text if ch in vowels))))


def task3():
    nums = list(map(int, input("Введите список чисел через пробел: ").split()))
    product = 1
    for i in range(1, len(nums), 2):
        product *= nums[i]
    print("Произведение элементов с нечетными номерами:", product)

    max_el = max(nums)
    nums.remove(max_el)
    print("Список после удаления наибольшего:", nums)

    print("Три наибольших элемента:", sorted(nums, reverse=True)[:3])


def task4():
    scores = {
        1: "AEIOULNSTR",
        2: "DG",
        3: "BCMP",
        4: "FHVWY",
        5: "K",
        6: "JX",
        10: "QZ"
    }
    points = {}
    for k, letters in scores.items():
        for l in letters:
            points[l] = k

    word = input("Введите слово (английское): ").upper()
    total = sum(points.get(ch, 0) for ch in word)
    print("Стоимость слова:", total)


def task5():
    store = {
        "Кольцо": ["золото", 5000, 10],
        "Браслет": ["серебро", 2000, 5],
        "Серьги": ["платина", 7000, 3],
        "Цепочка": ["золото", 6000, 4]
    }

    while True:
        print("""
        1. Просмотр описания
        2. Просмотр цены
        3. Просмотр количества
        4. Вся информация
        5. Покупка
        6. До свидания
        """)
        choice = input("Выберите пункт меню: ")
        match choice:
            case "1":
                for k, v in store.items():
                    print(f"{k} – {v[0]}")
            case "2":
                for k, v in store.items():
                    print(f"{k} – {v[1]} руб.")
            case "3":
                for k, v in store.items():
                    print(f"{k} – {v[2]} шт.")
            case "4":
                for k, v in store.items():
                    print(f"{k}: материал={v[0]}, цена={v[1]}, кол-во={v[2]}")
            case "5":
                total_cost = 0
                while True:
                    item = input("Введите название изделия (или 'n' для выхода): ")
                    if item.lower() == 'n':
                        break
                    if item in store:
                        qty = int(input("Введите количество: "))
                        if qty <= store[item][2]:
                            cost = store[item][1] * qty
                            total_cost += cost
                            store[item][2] -= qty
                            print(f"Вы купили {qty} шт. {item} за {cost} руб.")
                        else:
                            print("Недостаточно товара!")
                    else:
                        print("Товар не найден!")
                print("Общая сумма покупки:", total_cost)
            case "6":
                print("До свидания!")
                break
            case _:
                print("Неверный пункт меню")


def task6():
    nums = tuple(map(int, input("Введите числа через пробел: ").split()))
    print("Максимум:", max(nums))
    print("Минимум:", min(nums))


if __name__ == "__main__":
    tasks = {
        "1": task1,
        "2": task2,
        "3": task3,
        "4": task4,
        "5": task5,
        "6": task6
    }

    while True:
        print("\nЗадания:")
        for i in range(1, 7):
            print(i)
        print("0. Выход")

        choice = input("Выберите задание: ")
        if choice == "0":
            break
        if choice in tasks:
            tasks[choice]()
        else:
            print("Неверный выбор")
