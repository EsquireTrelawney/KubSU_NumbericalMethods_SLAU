import numpy as np
from scipy.linalg import solve
from gradient_descent_method import gradient_descent_solve
from seidel_method import seidel_solve
from simple_iterations_method import simple_iterations_solve
from gauss_method import gauss_solve
from min_residuals_method import minimal_residuals_solve
from conjugate_gradient_method import conjugate_gradient_solve


# функция для задания матрицы из 11го задания по формуле
def create_matrix_and_vector(dimension):
    matrix_A = np.zeros((dimension, dimension))
    vector_b = np.zeros(dimension)

    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                matrix_A[i][j] = 2 * dimension
            else:
                matrix_A[i][j] = 1

        vector_b[i] = dimension * (dimension + 1) / 2 + (i + 1) * (2 * dimension - 1)

    return matrix_A, vector_b


n = 3  # Меняем на нужное значение. Это размерность создаваемой матрицы для 11го задания
matrix_A, vector_b = create_matrix_and_vector(n)  # Создаём матрицу и вектор для 11го задания
A_orig = matrix_A.copy()  # на ВСЯЯЯЯЯКИй случай сохраняем созданные матрицу и вектор, ибо тот же
# метод Гаусса будет их менять в ходе вычислений
B_orig = vector_b.copy()



print("#11 Задание. Реализация всех методов и их проверка на тестовой СЛАУ, заданной формулой:")
print("Матрица коэффициентов A: \n", A_orig)
print("Вектор-столбец свободных коэффициентов: \n", B_orig)
# Решение с помощью Метода Гаусса
result_vector = gauss_solve(matrix_A, vector_b)
print("\n 1) Решение методом Гаусса:", result_vector)
# Числа очень близкие к нулю в экспоненциальном представлении говорят, что решение найдено верно
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)

# Решение с помощью метода простых итераций
matrix_A = A_orig
vector_b = B_orig
epsilon = 1e-5  # погрешность для условия завершения( 1/10^5 -> 0,00001 )
max_iterations = 1000  # кол-во итераций
result_vector, num_iterations_for_simple_iterations = simple_iterations_solve(matrix_A, vector_b, max_iterations,
                                                                              epsilon)
print("\n 2) Решение методом простых итераций:", result_vector)
print("Кол-во итераций: ", num_iterations_for_simple_iterations)
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)

# Решение с помощью метода Зейделя
matrix_A = A_orig
vector_b = B_orig
initial_epsilon = 1e-6  # начальная погрешность для метода Зейделя. задаём её не меньше чем 0,000001
initial_x_for_Seidel = np.zeros_like(vector_b)
result_vector, eps, num_iterations_for_Seidel = seidel_solve(matrix_A, vector_b, initial_epsilon, initial_x_for_Seidel)
print("\n 3) Решение методом Зейделя :", result_vector)
print("Полученная погрешность при работе метода: ", eps)
print("Кол-во итераций: ", num_iterations_for_Seidel)
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)

# Решение с помощью метода наискорейшего спуска (градиентного спуска)
matrix_A = A_orig
vector_b = B_orig
epsilon = 1e-6  # погрешность для условия завершения( 1/10^5 -> 0,00001 )
max_iterations = 1000  # кол-во итераций
result_vector, num_iterations_for_gradient_descent = gradient_descent_solve(matrix_A, vector_b, epsilon, max_iterations)
print("\n 4) Решение методом наискорейшего спуска: ", result_vector)
print("Кол-во итераций: ", num_iterations_for_gradient_descent)
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)

# Решение с помощью метода минимальных невязок
matrix_A = A_orig
vector_b = B_orig
epsilon = 1e-5  # погрешность для условия завершения( 1/10^5 -> 0,00001 )
initial_x = np.zeros_like(vector_b)
result_vector, num_iterations_for_min_residuals = minimal_residuals_solve(matrix_A, vector_b, initial_x, epsilon)
print("\n 5) Решение методом минимальных невязок: ", result_vector)
print("Кол-во итераций: ", num_iterations_for_min_residuals)
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)

# Решение с помощью метода сопряженных градиентов
matrix_A = A_orig
vector_b = B_orig
epsilon = 1e-5  # погрешность для условия завершения( 1/10^5 -> 0,00001 )
max_iterations = 1000  # кол-во итераций
result_vector, num_iterations_for_conjugate_gradient = conjugate_gradient_solve(matrix_A, vector_b, epsilon, max_iterations)
print("\n 6) Решение методом сопряженных градиентов: ", result_vector)
print("Кол-во итераций: ", num_iterations_for_conjugate_gradient)
print("Результат: [a]{x} - b =", np.dot(A_orig, result_vector) - B_orig)


# Встроенный в numpy метод для проверки всех решений
X = np.linalg.solve(A_orig, B_orig)
print("\n \nРешение встроенным в numpy методом solve: X= ", X)
# Встроенный в scipy метод для проверки всех решений
X = solve(A_orig, B_orig)
print("Решение встроенным в scipy методом solve: X= ", X)
