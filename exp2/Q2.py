from queue import PriorityQueue as PQ
import numpy as np


# 启发函数为当前层所有通道中最小的那个通道长度
def heuristic(floor, matrix):
    return min(matrix[floor][floor + 1:], default=np.inf)

def Astar(N,M,K,matrix):
    # 优先队列
    openset = PQ()

    result = []
    # 加入初始节点，第一个元素为f值，第二个为g值，第三个为层数
    openset.put([0 + heuristic(1, matrix), 0, 1])

    while not openset.empty():
        # 选出f值最小的节点/层
        current_floor = openset.get()

        g = current_floor[1]
        floor = current_floor[2]

        if floor == N - 1:
            result.append(g)

        if len(result) == K:
            break

        # 加入子节点
        for i in range(floor + 1, N):
            path = matrix[floor][i]
            if path != np.inf:
                f = g + path + heuristic(i, matrix)
                openset.put([f, g + path, i])

    result += [-1] * (K - len(result))
    for num in result:
        print(num)


N, M, K = map(int, input().split())
N = N + 1
# 用于存储图之间的关系的矩阵
matrix = [[np.inf for i in range(N)] for i in range(N)]

# 读入数据
for _ in range(M):
    a, b, c = map(int, input().split())
    # 如果b大于a，认为不是下行通道，则跳过此数据
    if a > b:
        continue
    matrix[a][b] = c

Astar(N,M,K,matrix)
