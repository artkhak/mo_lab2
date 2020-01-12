def simplex_method(function, x_count, start_x, delta, ex, ef, alpha, calculation_data=None):
    """
    Функция реализующая ПСМ.
    @param function: Целевая функция.
    @param x_count: Размерность функции.
    @param start_x: Начальная точка.
    @param delta: Длина ребра симплекса.
    @param ex: Точность по аргументу.
    @param ef: Точность по функции.
    @param alpha: Коэффициент сжатия.
    @param calculation_data: Объект для сбора данных по поиску.
    @return: Точка оптимума.
    """

    # Получаем коэффициенты для начального расчета симплекса.
    p = delta * (((x_count + 1) ** 1 / 2 + x_count - 1) / (x_count * 2 ** 1 / 2))
    g = delta * (((x_count + 1) ** 1 / 2 - 1) / (x_count * 2 ** 1 / 2))

    # Получаем начальный симплекс.
    simplex = [[start_x[j] if i == 0 else
                start_x[j] + p if j == i - 1 else
                start_x[j] + g
                for j in range(0, x_count, 1)]
               for i in range(0, x_count + 1, 1)]

    if calculation_data is not None:
        calculation_data.add_simplex(simplex)

    # Получаем начальное приблежение точки оптимума.
    optimum_approximation = approximate_optimum(x_count, simplex)

    if calculation_data is not None:
        calculation_data.add_optimum_approximation(optimum_approximation)

    # Получаем значение функции в начальном приблежении точки оптимума.
    f_optimum_approximation = calculate_function_and_add_function_call(function,
                                                                       optimum_approximation,
                                                                       calculation_data)

    # Начинаем цикл
    while True:
        if calculation_data is not None:
            calculation_data.add_iteration()

        # Вычисляем значение функции в точках симплекса.
        f_simplex = [calculate_function_and_add_function_call(function, xs, calculation_data)
                     for xs in simplex]

        # Получаем максимальное значение функции в точках симлекса и индекс точки в симплексе.
        max_f = max(f_simplex)
        max_f_index = f_simplex.index(max_f)

        # Получаем зеркальное отражение максимальной точки симплекса и значение функции в этой точке
        c = 2 / x_count
        column_sums = vector_sum(simplex)
        max_simplex = simplex[max_f_index]
        reflection = [c * (column_sums[index] - max_simplex[index]) - max_simplex[index] for index in
                      range(0, x_count)]
        f_reflection = calculate_function_and_add_function_call(function, reflection, calculation_data)

        # Получаем новый симплекс.
        if f_reflection <= max_f:
            simplex_next = [reflection
                            if index == max_f_index
                            else value
                            for index, value in enumerate(simplex)]
        else:
            min_f = min(f_simplex)
            min_f_index = f_simplex.index(min_f)
            min_f_simplex_vector = simplex[min_f_index]
            simplex_next = [[alpha * simplex_value + (1 - alpha) * min_f_simplex_vector[index]
                             for index, simplex_value in enumerate(simplex_vector)]
                            for simplex_vector in simplex]

        if calculation_data is not None:
            calculation_data.add_simplex(simplex_next)

        # Получаем новое приблежение точки оптимума.
        optimum_approximation_next = approximate_optimum(x_count, simplex_next)

        if calculation_data is not None:
            calculation_data.add_optimum_approximation(optimum_approximation_next)

        # Получаем значение функции в новом приблежении точки оптимума.
        f_optimum_approximation_next = calculate_function_and_add_function_call(function,
                                                                                optimum_approximation_next,
                                                                                calculation_data)

        # Проверяем условие выхода. Если условие не выполняется, то продолжаем.
        if ((optimum_approximation_next[0] - optimum_approximation[0]) ** 2 +
            (optimum_approximation_next[1] - optimum_approximation[1]) ** 2) ** 1 / 2 <= ex and \
                abs(f_optimum_approximation - f_optimum_approximation_next) <= ef:
            return optimum_approximation_next
        else:
            simplex = simplex_next
            optimum_approximation = optimum_approximation_next
            f_optimum_approximation = f_optimum_approximation_next


def vector_sum(matrix):
    """
    Суммирует векторы в матрице.
    @param matrix: Матрица.
    @return: Вектор, являющийся суммой векторов в матрице.
    """
    result = []
    for vector in matrix:
        for index, value in enumerate(vector):
            if len(result) <= index:
                result.append(value)
            else:
                result[index] += value

    return result


def calculate_function_and_add_function_call(function, xs, calculation_data=None):
    """
    Получает значение функции и обновляет счетчик вызова функции.
    @param function: Функция.
    @param xs: Аргументы.
    @param calculation_data: Объект для сбора данных по поиску.
    @return: Значение функции.
    """
    result = function(*xs)

    if calculation_data is not None:
        calculation_data.add_function_call()

    return result


def approximate_optimum(x_count, simplex):
    """
    Получает приближение точки оптимума из точек симплекса.
    @param x_count: Размерность.
    @param simplex: Симплекс.
    @return: Приблежение точки оптимума.
    """
    c = 1 / (x_count + 1)
    return [c * vector_sum_value for vector_sum_value in vector_sum(simplex)]


class CalculationData:
    """
    Класс для сбора данных расчета.
    """

    def __init__(self):
        """
        Конструктор.
        """
        self.iteration_count = 0
        self.function_call_count = 0
        self.simplexes = []
        self.optimum_approximations = []

    def add_iteration(self):
        """
        Добавляет итерацию.
        """
        self.iteration_count += 1

    def add_function_call(self):
        """
        Добавляет вызов функции.
        """
        self.function_call_count += 1

    def add_simplex(self, simplex):
        """
        Добавляет симплекс.
        @param simplex: Симплекс.
        """
        self.simplexes.append(simplex)

    def add_optimum_approximation(self, optimum_approximation):
        """
        Добавляет приблежение точки оптимума.
        @param optimum_approximation: Приблежение точки оптимума.
        """
        self.optimum_approximations.append(optimum_approximation)


def f(x1, x2):
    """
    Заданая функция.
    :param x1: Первый аргумент функции.
    :param x2: Второй аргумент функции.
    :return: Значение функции.
    """
    return 3 * (x1 - 5) ** 2 + (x2 - 4) ** 2


# Подключаем библиотеки, необходимые для формирования и вывода графика
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

start_point = [0, 0]
ex = 0.01
ef = 0.01
calculation_data = CalculationData()
delta = 5
alpha = 0.5
result_x1, result_x2 = simplex_method(f, 2, start_point, delta, ex, ef, alpha, calculation_data)

# Вывод графика
lin_space = np.linspace(0, 10, 100)
x1s, x2s = np.meshgrid(lin_space, lin_space)
f_mesh_grid = [f(x1, x2) for x1, x2 in zip(x1s, x2s)]
plt.contour(x1s, x2s, f_mesh_grid)

for simplex in calculation_data.simplexes:
    polygon = Polygon(simplex, fill=False, color='b')
    plt.gca().add_patch(polygon)

for optimum_approximation in calculation_data.optimum_approximations:
    plt.scatter(optimum_approximation[0], optimum_approximation[1], marker='o', c='g', s=20)

plt.scatter(result_x1, result_x2, marker='x', c='r', s=200)

ticks = range(0, 11)
plt.xlabel('x1')
plt.xticks(ticks)
plt.ylabel('x2')
plt.yticks(ticks)

plt.show()

# Вывод выходных данных
print(f'Точка оптимума - [{result_x1}; {result_x2}]')
print(f'Оптимальное значение ЦФ - {f(result_x1, result_x2)}')
print(f'Количество итераций - {calculation_data.iteration_count}')
print(f'Количество вызовов расчета функции - {calculation_data.function_call_count}')

index = 1
for simplex, optimum_approximation in zip(calculation_data.simplexes, calculation_data.optimum_approximations):
    print(f'{index}) Точки симплекса:\n'f'\t[{simplex[0][0]}; {simplex[0][1]}]\n'
          f'\t[{simplex[1][0]}; {simplex[1][1]}]\n'
          f'\t[{simplex[2][0]}; {simplex[2][1]}]\n'
          f'Приближение точки оптимума - [{optimum_approximation[0]}; {optimum_approximation[1]}]')
    index += 1
