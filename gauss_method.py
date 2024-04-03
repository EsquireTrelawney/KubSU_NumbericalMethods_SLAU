import numpy as np


# Метод Гаусса: На вход подаётся значения матрицы и значения вектора коэфф СЛАУ
def gauss_solve(matrix, column_vector):
    # кол-во строк в столбце свободных коэфф
    num_of_rows = len(column_vector)
    # Каждый коэфф i-го уравнения делится на первый ненулевой коэфф этого уравнения
    for k in range(0, num_of_rows):
        # Поиск максимального элемента в столбце для предотвращения деления на ноль
        max_elem_line = np.argmax(np.abs(matrix[k:, k])) + k
        # Обмен строк в матрице и векторе свободных членов
        matrix[[k, max_elem_line]] = matrix[[max_elem_line, k]]
        column_vector[[k, max_elem_line]] = column_vector[[max_elem_line, k]]
        for i in range(k + 1, num_of_rows):
            # Если этот коэфф не нулевой, то определяем скаляр λ (2 фундаментальная операция эквивалентных
            # преобразований СЛАУ)
            if 0.0 != matrix[i, k]:
                lam = matrix[i, k] / matrix[k, k]
                # Новая строка матрицы рассчитывается путём вычитания из неё строки, умноженной на λ
                # так, чтобы элементы под диагональю стали равны нулю
                matrix[i, k + 1:num_of_rows] = matrix[i, k + 1:num_of_rows] - lam * matrix[k, k + 1:num_of_rows]
                # Происходит также обновление столбца свободных членов в соответствии с преобразованием в матрице коэфф
                column_vector[i] = column_vector[i] - lam * column_vector[k]
    # Здесь выполняет обратных ход для подстановки и нахождения значений неизвестных
    for k in range(num_of_rows - 1, -1, -1):
        # Значения столбца свободных коэфф вычисляется как разность между текущим значением и скалярным произведением
        # соответствующей строки матрицы коэффициентов и уже найденных значений переменных.
        column_vector[k] = (column_vector[k] - np.dot(matrix[k, k + 1:num_of_rows], column_vector[k + 1:num_of_rows])) / \
                           matrix[k, k]
    # Возвращаем измененый столбец свободных коэфф, который и представляет решение СЛАУ
    return column_vector