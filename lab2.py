from simplex import simplex_method
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from CalculationData import CalculationData


def f(x1, x2):
    """
    Заданая функция.
    :param x1: Первый аргумент функции.
    :param x2: Второй аргумент функции.
    :return: Значение функции.
    """
    return 3 * (x1 - 5) ** 2 + (x2 - 4) ** 2


start_point = [0, 0]
ex = 0.0001
ef = 0.0001
calculation_data = CalculationData()

result_x1, result_x2 = simplex_method(f, 2, start_point, 1, ex, ef, 0.5, calculation_data)

lin_space = np.linspace(0, 10, 100)
x1s, x2s = np.meshgrid(lin_space, lin_space)
f_mesh_grid = [f(x1, x2) for x1, x2 in zip(x1s, x2s)]
plt.contour(x1s, x2s, f_mesh_grid)

for simplex in calculation_data.simplexes:
    polygon = Polygon(simplex, fill=False, color='b')
    plt.gca().add_patch(polygon)

plt.scatter(result_x1, result_x2, marker='x', c='r', s=200)

ticks = range(0, 11)
plt.xlabel('x1')
plt.xticks(ticks)
plt.ylabel('x2')
plt.yticks(ticks)
plt.show()

print(f'x1 = {result_x1}')
print(f'x2 = {result_x2}')
print(f'f = {f(result_x1, result_x2)}')
print(f'Количество итераций - {calculation_data.iteration_count}')
print(f'Количество вызовов расчета функции - {calculation_data.function_call_count}')




