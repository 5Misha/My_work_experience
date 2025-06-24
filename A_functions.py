import random

# Функция бинарного поиска
def bin_search(arr: list, target: int) -> int:
    ''' Поиск значения с помощью бинарного поиска
        Parametrs: 
            arr (list) - список из чисел среди которых нужно найти target
            target (int) - число, которое нужно найти
        Reterns:
            mid (int) - индекс найденного значения
    '''
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return  None

# Функция быстрой сортировки 
def quick_sort(arr: list) -> list:
    ''' Функция быстрой сортировки
        Parametrs:
            arr (list) - список, который нужно отсортировать 
        Reterns:
            arr (list) - отсортированный список
    '''
    if len(arr) <= 1:
        return arr
    else:
        q = random.choice(arr)
        more_arr = []
        equally_arr = []
        less_arr = []
        for i in arr:
            if i < q:
                less_arr.append(i)
            elif i == q:
                equally_arr.append(i)
            else:
                more_arr.append(i)
        return quick_sort(less_arr) + equally_arr + quick_sort(more_arr)
    



# Функция поиска в глубину
def dfs(graph: dict, node: list) -> list:
    ''' Функция поиска в глубину
        Parametrs:
            graph (dict) - словарь, в котором хранятся всевозможные пути, 
                где ключ - вершина, а значение - список из вершин, куда можно попасть
                пример записи {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': [], 'D': [], 'E': []}
            node (list) - начальная вершина для решения задачи
        Returns:
            itog (list) - список, состоящий из упорядоченной записи вершин графа в глубину 
    '''
    itog = []
    itog.append(node)
    for nodik in graph[node]:
        if len(nodik) > 0:
            itog.extend(dfs(graph, nodik))
    return itog




#bfs
def bfs(graph: dict, node: list) -> list:
    ''' Функция поиска в шиирну
        Parametrs:
            graph (dict) - словарь, в котором хранятся всевозможные пути, 
                где ключ - вершина, а значение - список из вершин, куда можно попасть
                пример записи {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': [], 'D': [], 'E': []}
            node (list) - начальная вершина для решения задачи
        Returns:
            itog (list) - список, состоящий из упорядоченной записи вершин графа в его ширину 
    '''
    itog = []
    itog.append(node)
    for nodik in graph:
        for podnodik in graph[nodik]:
            if not podnodik in itog:
                itog.append(podnodik)
    return itog
