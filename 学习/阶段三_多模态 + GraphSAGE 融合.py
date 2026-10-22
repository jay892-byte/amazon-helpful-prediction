import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Data
text = torch.randn(1000, 100)
image = torch.randn(1000, 200)
y = torch.randint(0, 2, (1000,))


# 创建图的边
source = torch.arange(1000)
target = torch.roll(source, shifts=-1)

edge_index = torch.stack([
    torch.cat([source, target]),
    torch.cat([target, source])
], dim=0)


# 创建训练/验证/测试 mask
train_mask = torch.zeros(1000, dtype=torch.bool)
val_mask = torch.zeros(1000, dtype=torch.bool)
test_mask = torch.zeros(1000, dtype=torch.bool)

train_mask[:800] = True
val_mask[800:900] = True
test_mask[900:1000] = True


# 创建 Dataset
data = Data(
    text=text,
    image=image,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)


# model
class MultiModalGraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(100, 64)
        self.image_encoder = nn.Linear(200, 64)
        self.conv1 = SAGEConv(128,16)
        self.conv2 = SAGEConv(16,2)
    def forward(self, text, image, edge_index):
        text_feature = self.text_encoder(text)
        image_feature = self.image_encoder(image)
        fused = torch.cat(
            [text_feature, image_feature],
            dim=1
        )
        x = self.conv1(fused,edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
model = MultiModalGraphSAGE()


# 损失和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# 检验功能
def evaluate_model(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(
            data.text,
            data.image,
            data.edge_index
        )
        # 只取验证/测试节点
        logits = logits[mask]
        y = data.y[mask]
        predictions = logits.argmax(dim=1)
        correct = (
            predictions == y
        ).sum().item()
        total = len(y)
        accuracy = correct / total
    return accuracy



# Train
best_accuracy = 0
for epoch in range(50):
    model.train()
    logits = model(
        data.text,
        data.image,
        data.edge_index
    )
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_accuracy = evaluate_model(
        model,
        data,
        data.val_mask)
    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss.item():.4f}, "
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
best_model = MultiModalGraphSAGE()
best_model.load_state_dict(
    torch.load(
        "best_MultiModalModel.pth"
    )
)

# 测试
test_accuracy = evaluate_model(
    best_model,
    data,
    data.test_mask
)

print(
    "Test Accuracy:",
    test_accuracy
)