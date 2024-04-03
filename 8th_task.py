import numpy as np
from numpy.linalg import solve
from gauss_method import gauss_solve
from simple_iterations_method import simple_iterations_solve
from seidel_method import seidel_solve
from gradient_descent_method import gradient_descent_solve
from min_residuals_method import minimal_residuals_solve
from conjugate_gradient_method import conjugate_gradient_solve

lam = np.array([
    [34.27, 0, -22.22, 0, -12.05, 0, 0, 0],
    [0, 45.57, 0, -26.05, 0, -19.52, 0, 0],
    [-22.22, 0, 66.22, -4, 0, 0, 0, -40],
    [0, -26.05, -4, 39.49, 0, 0, -8.333, -1.103],
    [-12.05, 0, 0, 0, 22.05, 0, -10, 0],
    [0, -19.52, 0, 0, 0, 33.81, -14.29, 0],
    [0, 0, 0, -8.333, -10, -14.29, 39.29, -6.667],
    [0, 0, -40, -1.103, 0, 0, -6.667, 69.77]
])

P_nominal = np.array([272.25, 103.95, 312.567, 70.567, 332.75, 127.05, 0, 0])
P_0_9U1n = np.array([254.7, 150.3, 260, 99.187, 311.3, 183.7, 0, 0])



# Решение СЛАУ для номинального входного напряжения встроенным в numpy методом solve(метод сопряженных градиентов)
T_nominal = solve(lam, P_nominal)
# Решение СЛАУ для входного напряжения 0.9U1н
T_0_9U1n = solve(lam, P_0_9U1n)
print("Установившаяся температура в пазовой и лобовой обмотке статора при номинальном входном напряжении:")
print("1) Пазовая обмотка(t): ", T_nominal[0])  # Печать первого элемента для температуры в пазовой обмотке статора)
print("2) Лобовая обмотка (t): ", T_nominal[4]) # ... и 5 элемента для лобовой обмотки статора
print("Установившаяся температура в пазовой и лобовой обмотке статора при входном напряжении 0.9U1н:")
print("1) Пазовая обмотка(t): ", T_0_9U1n[0])  # Печать первого элемента для температуры в пазовой обмотке статора)
print("2) Лобовая обмотка(t): ", T_0_9U1n[4]) # ... и 5 элемента для лобовой обмотки статора

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
T_nominal = gauss_solve(lam_copy, P_nominal_copy)
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n = gauss_solve(lam_copy, P_0_9U1n_copy)
print("\n Решение методом Гаусса:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
T_nominal, iterations = simple_iterations_solve(lam_copy, P_nominal_copy, 100, 1e-5)
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n, iterations = simple_iterations_solve(lam_copy, P_0_9U1n_copy, 100, 1e-5)
print("\n Решение методом простых итераций:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
T_nominal, eps, iterations = seidel_solve(lam_copy, P_nominal_copy, 1e-6, np.zeros_like(P_nominal))
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n, eps, iterations = seidel_solve(lam_copy, P_0_9U1n_copy, 1e-6, np.zeros_like(P_0_9U1n_copy))
print("\n Решение методом Зейделя:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
T_nominal, iterations = gradient_descent_solve(lam_copy, P_nominal_copy, 1e-6, 1000)
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n, iterations = gradient_descent_solve(lam_copy, P_0_9U1n_copy, 1e-6, 1000)
print("\n Решение методом наискорейшего спуска:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
initial_t = np.zeros_like(P_nominal_copy)
T_nominal, iterations = minimal_residuals_solve(lam_copy, P_nominal_copy,initial_t, 1e-6, 1000)
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n, iterations = minimal_residuals_solve(lam_copy, P_0_9U1n_copy, initial_t, 1e-6, 1000)
print("\n Решение методом минимальных невязок:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")

lam_copy = lam.copy()
P_nominal_copy = P_nominal.copy()
initial_t = np.zeros_like(P_nominal_copy)
T_nominal, iterations = conjugate_gradient_solve(lam_copy, P_nominal_copy, 1e-6, 1000)
lam_copy = lam.copy()
P_0_9U1n_copy = P_0_9U1n.copy()
T_0_9U1n, iterations = conjugate_gradient_solve(lam_copy, P_0_9U1n_copy, 1e-6, 1000)
print("\n Решение методом сопряженных градиентов:")
print("Температура при номинальном U пазовой обмотки =", T_nominal[0], "и", T_nominal[4], "для лобовой обмотки")
print("Температура при 0,9 U пазовой обмотки =", T_0_9U1n[0], "и", T_0_9U1n[4], "для лобовой обмотки")