# 启发函数:不在正确位置的数字的个数
def heuristic(state):
    count = 0
    for i in range(9):
        if state[i] != goal_state[i]:
            count += 1
    return count


def Astar(initial_state):
    # 定义一个普通的列表open_list，用来存储待遍历的节点
    open_list = [{"f": heuristic(initial_state), "g": 0, "state": initial_state}]
    # 定义一个集合closed_set，用来存储已经遍历过的节点
    closed_set = set()

    # 定义移动方向
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 进入一个循环，直到open_list为空或者找到目标状态为止
    while open_list:
        # 从open_list中选择f值最小的状态，记为当前状态，并将其从列表中删除
        current_node = min(open_list, key=lambda x: x["f"])
        open_list.remove(current_node)
        g, current_state = current_node["g"], current_node["state"]
        # 将当前节点加入closed_set中，表示已经遍历过
        closed_set.add(tuple(current_state))

        # 如果当前节点的状态等于目标状态，说明已经找到最短路径，返回当前节点的移动代价
        if current_state == goal_state:
            return g

        # 找到当前节点中空格（数字0）的位置，记为(row, col)
        zero_idx = current_state.index(0)
        row, col = zero_idx // 3, zero_idx % 3

        # 遍历当前节点的所有邻近节点，也就是将空格与其上下左右四个方向的数字交换所得到的状态
        for dr, dc in moves:
            # 计算新的空格位置，记为(new_row, new_col)
            new_row, new_col = row + dr, col + dc
            # 如果新的空格位置在九宫格的范围内，说明是一个合理的移动
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # 计算新的空格的索引，记为new_zero_idx
                new_zero_idx = new_row * 3 + new_col
                # 复制当前节点的状态，记为new_state
                new_state = current_state[:]
                # 将当前节点的0和新的0的数字交换，得到新的状态
                new_state[zero_idx], new_state[new_zero_idx] = new_state[new_zero_idx], new_state[zero_idx]

                # 如果新的状态不在closed_set中，说明没有被遍历过
                if tuple(new_state) not in closed_set:
                    # 计算新的状态的启发函数的值，记为h
                    h = heuristic(new_state)
                    # 将新的状态加入open_list中，其启发值为g + h，其移动代价为g + 1，其数据结构为字典
                    open_list.append({"f": g + h, "g": g + 1, "state": new_state})


# 定义目标状态
goal_state = [1, 3, 5, 7, 0, 2, 6, 8, 4]


input_str = input()
arr = list(input_str)
input_state = []

for i in arr:
    input_state.append(int(i))

# 调用A*算法
result = Astar(input_state)

# 输出结果
print(result)
