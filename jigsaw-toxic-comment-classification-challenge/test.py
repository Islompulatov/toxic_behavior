import torch

test = torch.randint(0, 1, (4, 6))

# print(test)

pred = torch.rand((4, 6))

# print(pred)

classes = pred > 0.5

# print ( classes == test)

# result = torch.sum(classes == test, dim= 1) == len(test)
result = (classes == test).all(dim=1)
accuracy = sum(result)/len(result)

print(result, accuracy)