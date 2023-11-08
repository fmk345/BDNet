import torch

# 设置随机种子
torch.manual_seed(0)

urandom_tensor_1 = torch.rand(3, 3)
print("第一次随机数生成结果:")
print(urandom_tensor_1)
torch.manual_seed(0)

# 再次随机生成，第二次结果和第一次是一样的
random_tensor_2 = torch.rand(3, 3)
print(random_tensor_2)
# 重新设置不的随机种子
torch.manual_seed(456)

# 再次创建一个随机数张量：因为设置了不同的随机数种子，这次生成的结果不和之前两次不同
random_tensor_3 = torch.rand(3, 3)
print("\n第三次随机数生成结果:")
print(random_tensor_3)
