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
