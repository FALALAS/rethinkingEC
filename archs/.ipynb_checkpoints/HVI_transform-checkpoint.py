import torch
import torch.nn as nn

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    def HVIT(self, img):
        """
                RGB → HSL
                输入: img [B, 3, H, W] 取值范围[0,1]
                输出: [B, 3, H, W] (H,S,L)
                """
        eps = 1e-8
        device, dtypes = img.device, img.dtype
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        maxc, _ = img.max(1)
        minc, _ = img.min(1)
        delta = maxc - minc

        # Lightness
        L = (maxc + minc) / 2.0  # [B,H,W]

        # Hue
        H = torch.zeros_like(L)
        mask = delta > eps
        # 各通道对应公式
        H_r = ((g - b) / (delta + eps)  ) % 6.0
        H_g = ((b - r) / (delta + eps)) + 2
        H_b = ((r - g) / (delta + eps)) + 4

        H[(maxc == r) & mask] = H_r[(maxc == r) & mask]
        H[(maxc == g) & mask] = H_g[(maxc == g) & mask]
        H[(maxc == b) & mask] = H_b[(maxc == b) & mask]

        H = (H / 6.0) % 1.0  # 归一化到[0,1]
        H[~mask] = 0  # 灰色区域 hue 不定义，设为0

        # Saturation
        S = torch.zeros_like(L)
        denom = 1.0 - torch.abs(2 * L - 1)
        S[mask] = delta[mask] / (denom[mask] + eps)
        S = torch.clamp(S, 0, 1)

        H = H.unsqueeze(1)  # bhw
        S = S.unsqueeze(1)
        L = L.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        #color_sensitive = torch.where( L<0.5 , ((L * 0.5 * pi).sin() + eps).pow(k) , L)
        color_sensitive = ((torch.abs(L-0.5) * 1 * pi).sin() + eps).pow(k)

        h = torch.cos(H * torch.pi * 2) * S*1
        s = torch.sin(H * torch.pi * 2) * S*1

        return torch.cat([h, s, L], dim=1)  # [B,3,H,W]

    def PHVIT(self, img):
        """
               HSL → RGB
               输入: [B, 3, H, W]
               输出: [B, 3, H, W] (RGB)
               """
        eps = 1e-8
        hx, hy, L = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        k = self.this_k
        #color_sensitive = torch.where( L<0.5 , ((L * 0.5 * pi).sin() + eps).pow(k) , L)
        color_sensitive = ((torch.abs(L-0.5) * 1 * pi).sin() + eps).pow(k)

        S = torch.sqrt(hx ** 2 + hy ** 2 + eps)

        H = torch.atan2(hy+eps, hx+eps) / (2 * torch.pi)
        H = (H % 1.0)  # 把范围归一化到 [0,1)
        #if self.gated:
        #    S = S * self.alpha_s
        S = torch.clamp(S, 0, 1)
        L = torch.clamp(L, 0, 1)

        C = (1 - torch.abs(2 * L - 1)) * S
        Hp = H * 6.0

        zeros = torch.zeros_like(H)
        r = torch.zeros_like(H)
        g = torch.zeros_like(H)
        b = torch.zeros_like(H)

        # 不同区间
        cond0 = (0 <= Hp) & (Hp < 1)
        cond1 = (1 <= Hp) & (Hp < 2)
        cond2 = (2 <= Hp) & (Hp < 3)
        cond3 = (3 <= Hp) & (Hp < 4)
        cond4 = (4 <= Hp) & (Hp < 5)
        cond5 = (5 <= Hp) & (Hp < 6)

        r = torch.where(cond0, L + C / 2, r)
        b = torch.where(cond0, L - C / 2, b)
        g = torch.where(cond0, b + torch.abs(Hp - 0) * C, g)

        g = torch.where(cond1, L + C / 2, g)
        b = torch.where(cond1, L - C / 2, b)
        r = torch.where(cond1, b + torch.abs(Hp - 2) * C, r)

        g = torch.where(cond2, L + C / 2, g)
        r = torch.where(cond2, L - C / 2, r)
        b = torch.where(cond2, r + torch.abs(Hp - 2) * C, b)

        b = torch.where(cond3, L + C / 2, b)
        r = torch.where(cond3, L - C / 2, r)
        g = torch.where(cond3, r + torch.abs(Hp - 4) * C, g)

        b = torch.where(cond4, L + C / 2, b)
        g = torch.where(cond4, L - C / 2, g)
        r = torch.where(cond4, g + torch.abs(Hp - 4) * C, r)

        r = torch.where(cond5, L + C / 2, r)
        g = torch.where(cond5, L - C / 2, g)
        b = torch.where(cond5, g + torch.abs(Hp - 6) * C, b)

        rgb = torch.stack([r, g, b], dim=1)
        #if self.gated2:
         #   rgb = rgb * self.alpha
        return torch.clamp(rgb, 0, 1)
'''
model=RGB_HVI()
x = torch.rand(1, 3, 5, 5)
hsl=model.HVIT(x)
y=model.PHVIT(hsl)
print(x)
z=torch.abs(x-y)>1e-4
print(z)
print(torch.abs(x-y).max())
'''
