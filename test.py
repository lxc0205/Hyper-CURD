import numpy as np

# 定义函数
def expand_new(Mssim):
    Mssim_expand = np.hstack((
        Mssim, 
        Mssim**2, 
        np.sqrt(Mssim), 
        Mssim**3, 
        Mssim**(1/3), 
        np.log(Mssim+1) / np.log(2), 
        np.power(2, Mssim) - 1, 
        (np.exp(Mssim)-1) / (np.exp(1)-1)
    ))
    return Mssim_expand

# 测试函数
# 假设 Mssim 是一个 numpy 数组，包含一些非负值
Mssim_test = np.array([[0.1, 0.5, 0.9],
                      [0.2, 0.4, 0.8],
                      [0.3, 0.6, 0.7]])  # 测试数据

# 调用函数并打印结果
Mssim_expand_test = expand_new(Mssim_test)
print(Mssim_expand_test)