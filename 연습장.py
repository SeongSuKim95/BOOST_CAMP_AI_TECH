import torch

# a = torch.zeros(3,2)
# b = a.view(2,3)
# a.fill_(1)
# print(a,b)

# a = torch.zeros(3,2)
# b = a.t().reshape(6)
# a.fill_(1)
# print(a,b)

# a = torch.tensor([[1,2,3],[4,5,6]])
# print(a)
# print(a.is_contiguous())

# a = a.transpose(0,1)
# print(a)
# print(a.is_contiguous())

# a = a.contiguous()
# print(a)
# print(a.is_contiguous())

# a = torch.tensor([[1,2,3],[4,5,6]])
# b = a.t()

# for i in range(2):
#     for j in range(3):
#         print(a[i][j].data_ptr())
# # 94066690072064
# # 94066690072072
# # 94066690072080
# # 94066690072088
# # 94066690072096
# # 94066690072104 32bit float, 8 byte씩 증가
# for i in range(3):
#     for j in range(2):
#         print(b[i][j].data_ptr())
# # 94066690072064 1
# # 94066690072088 4
# # 94066690072072 2 
# # 94066690072096 5
# # 94066690072080 3 
# # 94066690072104 6
# b = b.contiguous()
# for i in range(3):
#     for j in range(2):
#         print(b[i][j].data_ptr())
# # 94395224670592
# # 94395224670600
# # 94395224670608
# # 94395224670616
# # 94395224670624
# # 94395224670632 8 byte씩 증가 하도록 align된 모습
# import ctypes
# print(ctypes.cast(id(a), ctypes.py_object).value)
# # tensor([[1,2,3],[4,5,6]])

# a = torch.zeros(3,2)
# for i in range(3):
#     for j in range(2):
#         print(a[i][j].data_ptr())
# b = a.view(2,3)
# a.fill_(1)
# print(a,b)
# a = torch.zeros(3,2)
# for i in range(3):
#     for j in range(2):
#         print(a[i][j].data_ptr())
# a = a.t()
# for i in range(2):
#     for j in range(3):
#         print(a[i][j].data_ptr())
# b = a.reshape((3,2))
# a.fill_(1)
# print(a,b) # [[1,1,1],[1,1,1]], [0,0,0,0,0,0]
# for i in range(3):
#     for j in range(2):
#         print(b[i][j].data_ptr())

a = torch.zeros(3,2)
a = a.t()
print(a.is_contiguous())
# b = a.view(6)
b = a.reshape(6)
