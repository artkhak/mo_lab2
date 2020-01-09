import math


def simplex_method(function, x_count, zero_x, delta, ex, ef, alpha, calculation_data=None):

    pn = delta * ((math.sqrt(x_count + 1) + x_count - 1) / (x_count * math.sqrt(2)))
    gn = delta * ((math.sqrt(x_count + 1) - 1) / (x_count * math.sqrt(2)))

    simplex = [[zero_x[j] if i == 0 else
                zero_x[j] + pn if j == i - 1 else
                zero_x[j] + gn
                for j in range(0, x_count, 1)]
               for i in range(0, x_count + 1, 1)]

    if calculation_data is not None:
        calculation_data.add_simplex(simplex)

    optimum_approximation = approximate_optimum(x_count, simplex)

    if calculation_data is not None:
        calculation_data.add_optimum_approximation(optimum_approximation)

    while True:
        if calculation_data is not None:
            calculation_data.add_iteration()

        f_simplex = [calculate_function_and_add_function_call(function, xs, calculation_data)
                     for xs in simplex]

        max_f = max(f_simplex)
        max_f_index = f_simplex.index(max_f)

        reflection = get_reflection(x_count, simplex, max_f_index)
        f_reflection = calculate_function_and_add_function_call(function, reflection, calculation_data)

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

        f_optimum_approximation = calculate_function_and_add_function_call(function,
                                                                           optimum_approximation,
                                                                           calculation_data)

        if calculation_data is not None:
            calculation_data.add_simplex(simplex_next)

        optimum_approximation_next = approximate_optimum(x_count, simplex_next)

        if calculation_data is not None:
            calculation_data.add_optimum_approximation(optimum_approximation_next)

        f_optimum_approximation_next = calculate_function_and_add_function_call(function,
                                                                                optimum_approximation_next,
                                                                                calculation_data)

        if distance(optimum_approximation_next, optimum_approximation) <= ex and \
                abs(f_optimum_approximation - f_optimum_approximation_next) <= ef:
            return optimum_approximation_next
        else:
            simplex = simplex_next
            optimum_approximation = optimum_approximation_next


def distance(x1, x2):
    return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def get_reflection(x_count, simplex, max_f_index):
    c = 2 / x_count
    column_sums = vector_sum(simplex)
    max_f_simplex = simplex[max_f_index]
    return [c * (column_sums[index] - max_f_simplex[index]) - max_f_simplex[index] for index in range(0, x_count)]


def vector_sum(matrix):
    result = []
    for vector in matrix:
        for index, value in enumerate(vector):
            if len(result) <= index:
                result.append(value)
            else:
                result[index] += value

    return result


def get_key_second_element(t):
    return t[1]


def calculate_function_and_add_function_call(function, xs, calculation_data=None):
    result = function(*xs)

    if calculation_data is not None:
        calculation_data.add_function_call()

    return result


def approximate_optimum(x_count, simplex):
    c = 1 / (x_count + 1)
    return [c * vector_sum_value for vector_sum_value in vector_sum(simplex)]
