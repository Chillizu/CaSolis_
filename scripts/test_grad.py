"""诊断 backward 图问题"""
import torch, sys
sys.path.insert(0, '.')
from arch.native import NativeCore

model = NativeCore(hidden_dim=64)
h = model.init_state()

print("测试 1: 单步 backward")
tok = torch.tensor([10], dtype=torch.long)
h_new, outputs = model.step(h, tok)
loss = torch.nn.functional.cross_entropy(outputs["char_logits"], torch.tensor([20]))
loss.backward()
print("  OK")

print("测试 2: 循环 + detach")
h = model.init_state()
for i in range(3):
    tok = torch.tensor([i + 30], dtype=torch.long)
    h_new, outputs = model.step(h.detach(), tok)
    l = torch.nn.functional.cross_entropy(outputs["char_logits"], torch.tensor([i + 40]))
    l.backward()
    h = h_new.detach()
    print(f"  step {i}: OK")

print("测试 3: 循环 + 好奇心 loss")
h = model.init_state()
for i in range(3):
    tok = torch.tensor([i + 50], dtype=torch.long)
    h_new, outputs = model.step(h.detach(), tok)
    
    # 世界模型 loss
    char_loss = torch.nn.functional.cross_entropy(
        outputs["char_logits"], torch.tensor([i + 60])
    )
    # 好奇心 loss  
    target = torch.sigmoid(char_loss.detach() - 1.0)
    curio_loss = torch.nn.functional.mse_loss(
        outputs["curiosity"].squeeze(-1), target.expand(1)
    )
    
    total = char_loss + 0.05 * curio_loss
    total.backward()
    h = h_new.detach()
    print(f"  step {i}: OK")

print("全部通过!")
