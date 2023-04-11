import pandas as pd
import numpy as np
import math
import scipy.stats as st

chat_id = 541133397 # Ваш chat ID, не меняйте название переменной


def solution(control_data, test_data) -> bool:

    alpha=0.1

    # Вычисляем средние значения и размеры выборок
    mean_control = np.mean(control_data)
    mean_test = np.mean(test_data)
    n_control = len(control_data)
    n_test = len(test_data)

    # Вычисляем стандартные отклонения для контроля и теста
    std_control = np.std(control_data)
    std_test = np.std(test_data)

    # Вычисляем стандартную ошибку разности средних значений
    se = math.sqrt((std_control**2 / n_control) + (std_test**2 / n_test))

    # Вычисляем Z-статистику
    z_stat = (mean_test - mean_control) / se

    # Вычисляем p-значение
    p_value = 1 - st.norm.cdf(z_stat)

    # Сравниваем p-значение с уровнем значимости и возвращаем результат
    return p_value < alpha
