import os

from torch import nn
import torch

class imagePrompts(nn.Module):
    def __init__(self):
        super().__init__()

        self.meta_dropout = torch.nn.Dropout(0.1)
        self.meta_dropout_2 = torch.nn.Dropout(0.1)
        self.prompt_patch = 8  # 修改为32，以适应512x512的输入
        n = self.prompt_patch
        h = 64  # 增加隐藏层大小以适应更大的输入
        self.meta_net = nn.Sequential(
            nn.Linear(3 * n * n, h),
            nn.ReLU(),
            # nn.Linear(h, h),
            # nn.ReLU(),
            nn.Linear(h, 3 * n * n)
        )

        self.meta_net_2 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(6, 3, 5, stride=1, padding=2),
        )

    def get_local_prompts(self, x):
        # [32, 3, 512, 512]
        B = x.shape[0]
        n = self.prompt_patch
        n_patch = int(512 / n)
        x = x.reshape(B, 3, n_patch, n, n_patch, n)  # [32, 3, 16, 32, 16, 32]
        x = x.permute(0, 2, 4, 1, 3, 5)  # [32, 16, 16, 3, 32, 32]
        x = x.reshape(B, n_patch*n_patch, 3*n*n)
        x = x.reshape(B*n_patch*n_patch, 3*n*n)
        x = self.meta_net(x)
        x = x.reshape(B, n_patch, n_patch, 3, n, n)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [32, 3, 16, 32, 16, 32]
        x = x.reshape(B, 3, 512, 512)  # 修改为512x512
        return x

    def get_prompts(self, x):
        prompts_1 = self.get_local_prompts(x)
        x =  self.meta_net_2(x)
        return x 

    def forward(self, x):
        prompts = self.get_prompts(x)
        x = x + prompts
        return x


def test_image_prompts():
    model = imagePrompts().cuda()
    batch_size = 32
    channels = 3
    height = 512
    width = 512
    input_tensor = torch.rand(batch_size, channels, height, width).cuda()
    model.eval()
    # 运行模型
    with torch.no_grad():
        output = model(input_tensor)
    print(output.shape)
    assert output.shape == input_tensor.shape, f"输出形状 {output.shape} 不匹配预期形状 {input_tensor.shape}"
    assert torch.any(output != input_tensor), "输出与输入相同，模型可能没有执行任何操作"
    assert torch.all(torch.isfinite(output)), "输出包含 NaN 或 Inf 值"



if __name__ == "__main__":
    test_image_prompts()