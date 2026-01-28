# project_base/geometry.py
import math

def angle_from_minimum(point, center, minimum):
    """
    Вычисляет угол между вектором center-point и вектором center-minimum.
    Параметры:
    - point: кортеж (x, y) координат точки
    - center: кортеж (x, y) координат центра
    - minimum: кортеж (x, y) координат минимальной точки
    
    Возвращает:
    - угол в градусах, растущий по часовой стрелке от минимального вектора
    """
    # Вычисляем вектор center-point
    v_point = (point[0] - center[0], point[1] - center[1])
    
    # Вычисляем вектор center-minimum
    v_min = (minimum[0] - center[0], minimum[1] - center[1])
    
    # Скалярное произведение векторов
    dot_product = v_point[0] * v_min[0] + v_point[1] * v_min[1]
    
    # Длины векторов
    length_v_point = math.sqrt(v_point[0]**2 + v_point[1]**2)
    length_v_min = math.sqrt(v_min[0]**2 + v_min[1]**2)
    
    # Косинус угла между векторами
    cos_angle = dot_product / (length_v_point * length_v_min)
    
    # Вычисляем угол в радианах
    angle_rad = math.acos(cos_angle)
    
    # Векторное произведение для определения направления вращения
    cross_product = v_point[0] * v_min[1] - v_point[1] * v_min[0]
    
    # Если векторное произведение отрицательно, угол считается против часовой стрелки
    if cross_product > 0:
        angle_rad = 2 * math.pi - angle_rad
    
    # Приводим угол к диапазону [0, 2π)
    angle_rad = angle_rad % (2 * math.pi)
    
    # Переводим угол в градусы
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def normalize_angle(angle_tip, angle_max):
    """
    Приводит угол стрелки к [0..1]
    """
    if angle_tip < 0 or angle_tip > angle_max:
        return None
    return angle_tip / angle_max
