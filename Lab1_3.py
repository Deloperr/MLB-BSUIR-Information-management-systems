import math

# 1
class Circle:
    def __init__(self, radius):
        if radius <= 0:
            raise ValueError("Радиус должен быть положительным")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def circumference(self):
        return 2 * math.pi * self.radius


# 2
class Worker:
    def __init__(self, name, surname, position, wage, bonus):
        self.name = name
        self.surname = surname
        self.position = position
        self._income = {"wage": wage, "bonus": bonus}


class Position(Worker):
    def get_full_name(self):
        return f"{self.name} {self.surname}"

    def get_total_income(self):
        return self._income["wage"] + self._income["bonus"]


# 3
class Stationery:
    def __init__(self, title):
        self.title = title

    def draw(self):
        print("Запуск отрисовки")


class Pen(Stationery):
    def draw(self):
        print(f"{self.title}: рисуем ручкой")


class Pencil(Stationery):
    def draw(self):
        print(f"{self.title}: рисуем карандашом")


class Handle(Stationery):
    def draw(self):
        print(f"{self.title}: выделяем маркером")


# 4
class BankAccount:
    bank_name = "Python Bank"  # атрибут класса

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} пополнил счёт на {amount}. Баланс: {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Недостаточно средств")
        else:
            self.balance -= amount
            print(f"{self.owner} снял {amount}. Баланс: {self.balance}")

    @classmethod
    def change_bank_name(cls, new_name):
        cls.bank_name = new_name
        print("Новое название банка:", cls.bank_name)

    @staticmethod
    def info():
        print("Банковский счёт хранит деньги и позволяет снимать и пополнять баланс.")


if __name__ == "__main__":
    print("1")
    try:
        rad = float(input("Введите радиус круга: "))
        c = Circle(rad)
        print("Площадь:", c.area())
        print("Длина окружности:", c.circumference())
    except ValueError as e:
        print("Ошибка:", e)

    print("\n2")
    p = Position("Иван", "Иванов", "Инженер", 50000, 10000)
    print("ФИО:", p.get_full_name())
    print("Общий доход:", p.get_total_income())

    print("\n3")
    pen = Pen("Parker")
    pencil = Pencil("Koh-i-Noor")
    handle = Handle("Sharpie")
    pen.draw()
    pencil.draw()
    handle.draw()

    print("\n4")
    acc = BankAccount("Анна", 1000)
    acc.deposit(500)
    acc.withdraw(300)
    BankAccount.change_bank_name("SuperBank")
    BankAccount.info()
