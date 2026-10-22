import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



#准备数据
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

#数据集处理功能
class ReviewDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        return self.x[item], self.y[item]

# 划分训练集和测试集
x_train = x[:800]
y_train = y[:800]

x_test = x[800:]
y_test = y[800:]
train_dataset = ReviewDataset(x_train, y_train)
test_dataset = ReviewDataset(x_test, y_test)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)




#模型建立
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
model = Model()

#loss功能
loss_fn = nn.CrossEntropyLoss()

#优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)


# 训练模型
for epoch in range(10):
    model.train
    for x_batch, y_batch in train_dataloader:
       logits = model(x_batch)
       loss = loss_fn(logits, y_batch)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
#保存模型参数   
torch.save(
    model.state_dict(),
    "linear_model.pth"
)



# 测试原来的模型
def evaluate_model(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

    accuracy = correct / total

    return accuracy


# 测试
accuracy1 = evaluate_model(model, test_loader)
print("原模型准确率:", accuracy1)
# 创建一个全新的模型
new_model = Model()
# 加载之前保存的参数
new_model.load_state_dict(
    torch.load("linear_model.pth")
)
# 测试加载后的模型
accuracy2 = evaluate_model(new_model, test_loader)
print("加载后模型准确率:", accuracy2)