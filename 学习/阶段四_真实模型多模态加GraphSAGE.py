import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from transformers import CLIPProcessor, CLIPModel
from torch_geometric.nn import SAGEConv
from torch import nn

# Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


# Data
texts = [
    "This product is very good.",
    "The quality is terrible.",
    "Very useful product.",
    "I don't recommend it."
]
image_paths = [
    "test.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg"
]
image = [Image.open(path).convert("RGB")
         for path in image_paths
]
y = torch.tensor([1, 0, 1, 0])


# 文本处理
text_inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True
)
print(text_inputs)
with torch.no_grad():
    text_outputs = text_model(**text_inputs)
text_feature = text_outputs.last_hidden_state.mean(dim=1)


# 图片处理
image_inputs = processor(
    images=image,
    return_tensors="pt"
)
with torch.no_grad():
    image_outputs = image_model.get_image_features(**image_inputs)
image_feature = image_outputs.pooler_output


# 多模态融合
fused_feature = torch.cat(
    [text_feature, image_feature],
    dim=1
)


# 图结构
edge_index = torch.tensor([
    [0, 1, 0, 2, 1, 3, 2, 3],
    [1, 0, 2, 0, 3, 1, 3, 2]
])


# GraphSAGE
class MultiModalGraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sage1 = SAGEConv(896,128)
        self.sage2 = SAGEConv(128,2)
    def forward(self,x,edge_index):
        x = self.sage1(x,edge_index)
        x = torch.relu(x)
        x = self.sage2(x,edge_index)
        return x
model = MultiModalGraphSAGE()


logits = model(fused_feature,edge_index)
print("logits:", logits.shape)


predictions = logits.argmax(dim=1)
print("predictions:", predictions)



loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(
    logits,
    y
)

print("loss:", loss.item())