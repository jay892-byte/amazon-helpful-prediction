import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Data
text = torch.randn(1000, 100)
image = torch.randn(1000, 200)
y = torch.randint(0, 2, (1000,))
text_train = text[:800]
image_train = image[:800]
y_train = y[:800]
text_val = text[800:900]
image_val = image[800:900]
y_val = y[800:900]
text_test = text[900:1000]
image_test = image[900:1000]
y_test = y[900:1000]



# Dataset
class MultiModalDataset(Dataset):
    def __init__(self,text_feature,image_feature,y):
        self.text_feature = text_feature
        self.image_feature = image_feature
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.text_feature[index], self.image_feature[index], self.y[index]
dataset_train = MultiModalDataset(text_train, image_train, y_train)
dataset_val = MultiModalDataset(text_val, image_val, y_val)
dataset_test = MultiModalDataset(text_test, image_test, y_test)
dataloader_train = DataLoader(
    dataset_train,
    batch_size = 25,
    shuffle = False
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size = 25,
    shuffle = False
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size = 25,
    shuffle = False
)

# model
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 文本编码器
        self.text_encoder = nn.Linear(100,64)
        # 图片编码器
        self.image_encoder = nn.Linear(200,64)
        # 分类器
        self.classifier = nn.Linear(128, 2)
    def forward(self, text, image):
        # 1. 文本特征
        text_feature = self.text_encoder(text)
        # 2. 图片特征
        image_feature = self.image_encoder(image)
        # 3. 融合
        fused = torch.cat([text_feature, image_feature], dim = 1)
        # 4. 分类
        logits = self.classifier(fused)
        return logits
model = MultiModalModel()

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

# Evaluate Model
def evaluate_model(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for text_batch, image_batch, y_batch in dataloader:
            logits = model(text_batch, image_batch)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)
    accuracy = correct / total
    return accuracy


# Train
best_accuracy = 0
for epoch in range(10):
    model.train()
    total_loss = 0
    for text_batch, image_batch, y_batch in dataloader_train:
        logits = model(text_batch, image_batch)
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader_train)
    val_accuracy = evaluate_model(
        model,
        dataloader_val
    )
    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {average_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}"
    )
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            "best_MultiModalModel.pth"
        )

        print("保存最佳模型")

#加载最佳模型
best_model = MultiModalModel()
best_model.load_state_dict(
    torch.load(
        "best_MultiModalModel.pth"
    )
)



# 测试
accuracy_test = evaluate_model(best_model, dataloader_test)
print("模型准确率:", accuracy_test)
