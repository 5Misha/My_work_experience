import pytest
from A_functions import bin_search, quick_sort, dfs, bfs
# cd "C:\Стажировочка\VS_Code\DZ_1"
# запуск через консоль pytest A_test_functions.py
# запуск одного из тестов pytest A_test_functions.py::test_bin_search


# тест для бинарного поиска
def test_bin_search():
    assert bin_search([1, 2, 3, 4, 5], 3) == 2, 'Индекс найден неверно'
    assert bin_search([1, 7, 33, 39], 39) == 3, 'Индекс найден неверно'
    assert bin_search([0, 3, 3, 9], 0) == 0, 'Индекс найден неверно'
    assert bin_search([0, 3, 3, 9], 3) == 1, 'Индекс найден неверно'
#    assert bin_search([0, 3, 3, 9], 3) == 2, 'Индекс найден неверно'



# тест для сортировки
def test_quick_sort():
    assert quick_sort([4, 3, 5, 1, 2]) == [1, 2, 3, 4, 5], 'Ошибка сортировки'
    assert quick_sort([0, 1, 0, -1, 2]) == [-1, 0, 0, 1, 2], 'Ошибка сортировки'
    assert quick_sort([0]) == [0], 'Ошибка сортировки'



# тест для поиска в глубину dfs
def test_dfs():
    assert dfs({
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': [],
        'D': [],
        'E': []
    }, 'A') == ['A', 'B', 'D', 'E', 'C']

    assert dfs({
        'A': ['B'],
        'B': ['D', 'E' ,'C'],
        'C': [],
        'D': [],
        'E': []
    }, 'A') == ['A', 'B', 'D', 'E', 'C']





#bfs
def test_bfs():
    assert bfs({
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': [],
    'D': [],
    'E': []
    }, 'A') == ['A', 'B', 'C', 'D', 'E']
    
    assert bfs({
    'A': ['B', 'C', 'D'],
    'B': ['E'],
    'C': [],
    'D': [],
    'E': []
    }, 'A') == ['A', 'B', 'C', 'D', 'E']

    assert bfs({
    'A': ['D', 'B'],
    'B': ['C'],
    'C': ['E'],
    'D': [],
    'E': []
    }, 'A') == ['A', 'D', 'B', 'C', 'E']
