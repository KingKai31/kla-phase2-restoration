"""
Item 1 (new technique classes): one additional lightweight residual block
inserted in the decoder's FINAL stage, immediately before the pixel-
shuffle head - not the encoder, not the bottleneck (different location
and purpose than src/models/nafnet_attention.py's bottleneck self-
attention test). Tests whether the decoder's capacity to represent fine
boundaries, not the loss pushing it toward them, is the limiting factor
behind the Axis 5 edge-preservation gap.

Near-identity initialization is automatic, not a separate step: NAFBlock
(src/models/nafnet.py) already initializes both its residual gates
(`beta`, `gamma`) to exact zero, so a freshly-added NAFBlock contributes
NOTHING to the residual stream at init, regardless of its (randomly
initialized) internal conv weights - forward(x) == x at step 0. Training
then lets it learn a real, non-trivial contribution from a stable
starting point identical to the shipped model's current behavior.
"""
import torch
import torch.nn.functional as F

from src.models.nafnet import NAFNetSR, NAFBlock


class NAFNetSRDecoderCapacity(NAFNetSR):
    def __init__(self, img_channel: int = 1, width: int = 32,
                 enc_blk_nums=(1, 1, 1, 2), middle_blk_num: int = 2,
                 dec_blk_nums=(1, 1, 1, 1), upscale: int = 2):
        super().__init__(img_channel, width, enc_blk_nums, middle_blk_num, dec_blk_nums, upscale)
        # after the decoder loop, channel count returns to `width` - same
        # tap point item3_confidence_head.py's frozen_forward_with_feature
        # reads from, right before up_head
        self.extra_decoder_block = NAFBlock(width)

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

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.extra_decoder_block(x)  # the one new capacity-increase step

        out = self.up_head(x)

        base = F.interpolate(inp, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        out = out[:, :, : h * self.upscale, : w * self.upscale] + base
        return out


def load_from_shipped_checkpoint(checkpoint_path, device, width: int = 32, upscale: int = 2):
    """Loads all compatible weights from a plain NAFNetSR checkpoint;
    the new extra_decoder_block is left at its fresh (near-identity via
    zero beta/gamma) initialization since no matching key exists in the
    shipped checkpoint - strict=False handles this correctly."""
    model = NAFNetSRDecoderCapacity(img_channel=1, width=width, upscale=upscale).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert unexpected == [], f"unexpected keys not in new model: {unexpected}"
    assert missing == [f"extra_decoder_block.{n}" for n, _ in
                        NAFBlock(width).named_parameters()] or all(m.startswith("extra_decoder_block") for m in missing), \
        f"unexpected missing keys (should only be the new block's own params): {missing}"
    return model
