"""
Axis 4 (technical hardening pass) - bottleneck self-attention variant of
NAFNetSR, per the pre-registered decision rule
(reports/axis4_architecture_decision_rule_PREREGISTERED.md, written and
committed before this file or any comparison run).

Reuses NAFBlock/LayerNorm2d from nafnet.py directly rather than
duplicating them - the baseline nafnet.py is NOT modified in place,
matching this project's established pattern (see stageB_composite.py's
own docstring for the same rationale) of adding variants alongside a
validated module instead of editing it.

Adds exactly ONE lightweight multi-head self-attention block at the
bottleneck (after the existing `middle` NAFBlocks, at the lowest spatial
resolution - 8x8 for a 128x128 input). Cheap by construction: at 8x8=64
tokens, the attention matrix is 64x64, trivial next to the convolutional
cost of the rest of the network. No positional encoding - at this
resolution and with only one attention layer, the convolutional stem
already provides ample local position information via padding, and the
bottleneck's job here is long-range mixing, not position-sensitive
detail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nafnet import LayerNorm2d, NAFBlock


class BottleneckSelfAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init residual scale, same spirit as NAFBlock's beta/gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        tokens = y.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out * self.gamma


class NAFNetSRWithAttention(nn.Module):
    def __init__(self, img_channel: int = 1, width: int = 32,
                 enc_blk_nums=(1, 1, 1, 2), middle_blk_num: int = 2,
                 dec_blk_nums=(1, 1, 1, 1), upscale: int = 2, attn_heads: int = 4):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        self.bottleneck_attn = BottleneckSelfAttention(chan, n_heads=attn_heads)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.up_head = nn.Sequential(
            nn.Conv2d(chan, img_channel * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
        )

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        mod = self.padder_size
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        x = self._pad_to_multiple(inp)

        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)
        x = self.bottleneck_attn(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        out = self.up_head(x)

        base = F.interpolate(inp, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        out = out[:, :, : h * self.upscale, : w * self.upscale] + base
        return out
