import numpy as np


def conjugate_gradient_solve(A, b, tol=1e-8, max_iter=1000):
    iterations = 0
    x = np.zeros_like(b)  # начальное приближение решения, вектор нулей
    r = b - A @ x  # вектор невязки
    z = r  # вектор направления спуска
    r_k_prev = np.dot(r, r)  # квадрат вектора невязки

    for i in range(max_iter):
        iterations += 1
        Ap = A @ z  # произведение матрицы A на вектор направления спуска
        alpha = r_k_prev / np.dot(z, Ap)  # шаг минимизации
        x += alpha * z  # обновление приближения решения
        r -= alpha * Ap  # обновление вектора невязки
        r_k_curr = np.dot(r, r)  # новый квадрат вектора невязки

        if np.sqrt(r_k_curr) < tol:  # проверка условия окончания (норма невязки)
            break

        beta = r_k_curr / r_k_prev  # коэффициент для обновления направления
        z = r + beta * z  # обновление направления спуска
        r_k_prev = r_k_curr

    return x, iterations

