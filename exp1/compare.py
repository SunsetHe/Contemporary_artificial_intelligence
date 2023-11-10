# 打开并读取两个txt文件
with open('result_jueceshu.txt', 'r') as file1, open('result_MLP.txt', 'r') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()

# 初始化计数器和总行数
count = 0
total_lines = len(lines1) - 1  # 减去第一行的id和pred

# 遍历每一行（从第二行开始），比较pred的值
for i in range(1, total_lines + 1):
    pred1 = float(lines1[i].split(',')[1])  # 假设pred在每行以逗号分隔
    pred2 = float(lines2[i].split(',')[1])

    if pred1 == pred2:
        count += 1

# 计算百分比
percentage = (count / total_lines) * 100

# 输出结果
print(f"相等的行数：{count}")
print(f"总行数：{total_lines}")
print(f"相等的百分比：{percentage:.2f}%")
