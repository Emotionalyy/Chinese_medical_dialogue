a = [0.439, 0.603584863, 0.644997203, 0.533183884, 0.739]
print(a)
min = min(a)
max = max(a)
len = max - min
new_min = 0.8205
new_max = 0.8217
len_new = new_max - new_min
b = []
for num in a:
    b.append(((num - min)/len)*(len_new)+new_min)
print(b)

# 缩放后的值 = ((原始值 - 原始最小值) / (原始最大值 - 原始最小值)) × (新范围的最大值 - 新范围的最小值) + 新范围的最小值
