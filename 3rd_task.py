import numpy as np
from gauss_method import gauss_solve
from simple_iterations_method import simple_iterations_solve
from seidel_method import seidel_solve
from gradient_descent_method import gradient_descent_solve
from min_residuals_method import minimal_residuals_solve
from conjugate_gradient_method import conjugate_gradient_solve

# from testing import gauss_solve_v1
# Матрица коэффициентов расхода сырья для каждого вида продукции
matrix_A = np.array([[6, 4, 5],
              [4, 3, 1],
              [5, 2, 3]], dtype=float)

# Вектор имеющегося в запасе сырья для каждого типа
vector_b = np.array([2400, 1450, 1550], dtype=float)
A_orig = matrix_A.copy()
B_orig = vector_b.copy()
# Решение системы линейных уравнений встроенным в numpy методом(он, кстати, использует метод сопряженных градиентов)
x = np.linalg.solve(matrix_A, vector_b)
# Вывод плана выпуска каждого вида продукции
print("План выпуска П1: {0}\nПлан выпуска П2: {1}\nПлан выпуска П3: {2}\n".format(x[0], x[1], x[2]))

# Далее решение реализованными самостоятельно методами

result_vector = gauss_solve(matrix_A, vector_b)
print("\n 1) Решение методом Гаусса:", result_vector)

matrix_A = A_orig
vector_b = B_orig
result_vector, iterations = simple_iterations_solve(matrix_A, vector_b, 100, 1e-5)
print("\n 2) Решение методом простых итераций:", result_vector, iterations)

matrix_A = A_orig
vector_b = B_orig
initial_x = np.array([120, 260, 90]) # Метод Зейделя не сходится для приведенной недиагонально-преобладающей и
#  несимметричной матрицы. Поэтому для хотя бы какого-то решения, приходится давать максимально близкое начальное приближение
# к действительному решению этой СЛАУ
result_vector, eps, num_iterations_for_Seidel = seidel_solve(matrix_A, vector_b, 1e-6, initial_x)
print("\n 3) Решение методом Зейделя :", result_vector)

matrix_A = A_orig
vector_b = B_orig
result_vector, num_iterations_for_gradient_descent = gradient_descent_solve(matrix_A, vector_b, 1e-6, 6)
print("\n 4) Решение методом наискорейшего спуска: ", result_vector)

matrix_A = A_orig
vector_b = B_orig
initial_x = np.array([100, 200, 90])
result_vector, num_iterations_for_min_residuals = minimal_residuals_solve(matrix_A, vector_b, initial_x, 1e-5)
print("\n 5) Решение методом минимальных невязок: ", result_vector)

matrix_A = A_orig
vector_b = B_orig
result_vector, num_iterations_for_conjugate_gradient = conjugate_gradient_solve(matrix_A, vector_b, 1e-6, 1000)
print("\n 6) Решение методом сопряженных градиентов: ", result_vector)


