import math
from copy import deepcopy
from typing import Optional, Tuple, Union, cast

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

from .dataops.pipelines.s1_s2_era5_srtm import BANDS_GROUPS_IDX
from .model import FinetuningHead, FineTuningModel, Seq2Seq
from .utils import default_model_path, device


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # a value of True indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).to(device)


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        channel_embed_ratio: float = 0.5,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX
        self.embedding_size = embedding_size

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx for idx, (group_name, _) in enumerate(self.band_groups.items())
        }

        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = int(embedding_size * (1 - channel_embed_ratio))
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups), embedding_dim=channel_embedding_size
        )

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def mask_tokens(x, mask):
        mask = mask.bool()

        # Move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)

        x = x.gather(1, indices[:, :, None].expand_as(x))

        # Set masked values to 0
        x = x * sorted_mask.unsqueeze(-1)

        # Cut off to the length of the longest unmasked sequence
        max_length = sorted_mask.sum(-1).max()

        x = x[:, :max_length]

        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_task: bool = True,
    ):
        device = x.device

        # Initialize mask if None
        if mask is None:
            mask = torch.zeros_like(x, device=device).float()

        # Create positional embeddings expanded to match batch size
        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :], "b t d -> (repeat b) t d", repeat=x.shape[0]
        )

        all_tokens, all_masks = [], []

        # Process each channel group
        for channel_group, channel_idxs in self.band_groups.items():
            # Extract tokens via patch embedding for this channel group
            tokens = self.eo_patch_embed[channel_group](x[:, :, channel_idxs])
            print(f"[ENCODER] x shape: {x.shape}, channel_idxs shape: {channel_idxs.shape if hasattr(channel_idxs, 'shape') else f'len={len(channel_idxs)}'}, tokens shape: {tokens.shape}")

            # Get channel-specific embedding and expand to match batch/timesteps
            channel_embed = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(device)
            )
            channel_embedding = repeat(channel_embed, "d -> b t d", b=x.shape[0], t=x.shape[1])

            # Combine channel and positional embeddings
            channel_wise_positional_embedding = torch.cat(
                (channel_embedding, positional_embedding), dim=-1
            )

            # Add combined embeddings to tokens
            tokens += channel_wise_positional_embedding

            # Compute mask for this group (max over channels)
            group_mask = torch.max(mask[:, :, channel_idxs], dim=-1)[0]

            all_tokens.append(tokens)
            all_masks.append(group_mask)

        # Concatenate tokens and masks across channel groups
        x = torch.cat(all_tokens, dim=1)
        mask = torch.cat(all_masks, dim=1)

        # Apply token masking and get indices
        x, orig_indices, upd_mask = self.mask_tokens(x, mask)

        # orig_indices = torch.cat(
        #     (torch.zeros(x.shape[0])[:, None].to(device).int(), orig_indices + 1),
        #     dim=1,
        # )

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=~upd_mask.bool())

        if eval_task:
            # Compute mean of unmasked tokens
            x_for_mean = x * (1 - upd_mask.unsqueeze(-1))

            x_mean = x_for_mean.sum(dim=1)

            x_mean = x_mean / torch.sum(1 - upd_mask, -1, keepdim=True)

            # Apply final layer norm
            output = self.norm(x_mean)
            return output

        # Return full sequence if not in eval mode
        output = self.norm(x)
        return output, orig_indices, upd_mask


class Decoder(nn.Module):
    def __init__(
        self,
        channel_embeddings: nn.Embedding,
        encoder_embed_dim=128,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=2,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx for idx, (group_name, _) in enumerate(self.band_groups.items())
        }

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.eo_decoder_pred = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embed_dim, len(group))
                for group_name, group in self.band_groups.items()
            }
        )
        self.channel_embeddings = channel_embeddings

        channel_embedding_dims = channel_embeddings.weight.shape[-1]

        remaining_embeddings = decoder_embed_dim - channel_embedding_dims

        # Save max sequence length
        self.max_sequence_length = max_sequence_length

        # Positional embedding size is half of remaining
        pos_embed_shape = (1, max_sequence_length, remaining_embeddings)
        self.pos_embed = nn.Parameter(
            torch.zeros(pos_embed_shape),
            requires_grad=False,
        )

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_masked_tokens(self, x, orig_indices, x_mask):

        all_masked = repeat(self.mask_token, "d -> b t d", b=x.shape[0], t=orig_indices.shape[1])

        mask = torch.cat(
            (
                x_mask,
                torch.ones((x.shape[0], orig_indices.shape[1] - x.shape[1]), device=x.device),
            ),
            dim=-1,
        )

        out = all_masked.clone()

        # Insert real tokens into correct spots (temporarily at the start of each row)
        out[~mask.bool()] = x[~x_mask.bool()]

        # Scatter to original positions
        out = out.scatter(1, orig_indices[:, :, None].expand_as(out), out)

        return out

    def add_embeddings(self, x):
        num_channel_groups = len(self.band_group_to_idx)

        num_timesteps = int(x.shape[1] / num_channel_groups)

        remove_mask = torch.full(size=(num_timesteps * num_channel_groups,), fill_value=False)

        positional_embedding = repeat(
            self.pos_embed[:, :num_timesteps, :],
            "b t d -> (b2 b) (t2 t) d",
            b2=x.shape[0],
            t2=num_channel_groups,
        )

        positional_embedding = positional_embedding[:, ~remove_mask]

        channel_embeddings = torch.repeat_interleave(
            self.channel_embeddings.weight, repeats=num_timesteps, dim=0
        )

        channel_embeddings = repeat(channel_embeddings, "c d -> b c d", b=x.shape[0])

        channel_embeddings = channel_embeddings[:, ~remove_mask]

        positional_embedding = torch.cat(
            (channel_embeddings, positional_embedding), dim=-1
        )

        x += positional_embedding

        return x

    def reconstruct_inputs(self, x) -> Tuple[torch.Tensor]:
        # Split into channel groups
        num_channel_groups = len(self.band_group_to_idx)

        num_timesteps = int(x.shape[1] / num_channel_groups)

        mask = torch.full((x.shape[1],), True, device=x.device)

        x = x[:, mask]

        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        eo_output = []
        for group_name, idx in self.band_group_to_idx.items():
            group_tokens = x[:, idx]
            decoded = self.eo_decoder_pred[group_name](group_tokens)
            eo_output.append(decoded)

        output = torch.cat(eo_output, dim=-1)

        return output

    def forward(self, x, orig_indices, x_mask):

        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, orig_indices, x_mask)
        x = self.add_embeddings(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.reconstruct_inputs(x)


class PrestoFineTuningModel(FineTuningModel):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder: Encoder = deepcopy(encoder)
        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # but don't unfreeze the position encoder, which
        # shouldn't be trainable
        self.encoder.pos_embed.requires_grad_(False)
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.head(
            self.encoder(
                x=x,
                mask=mask,
                eval_task=True,
            )
        )


class PrestoFinetuningWithAggregates(FineTuningModel):
    def __init__(
        self,
        encoder,
        num_outputs: int,
        regression: bool,
        aggregate: str,
    ):
        super().__init__()
        self.encoder: Encoder = deepcopy(encoder)
        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # but don't unfreeze the position encoder, which
        # shouldn't be trainable
        self.encoder.pos_embed.requires_grad_(False)

        aggregate_to_multiplier = {"mean": 2, "quantiles": 5}
        if aggregate not in aggregate_to_multiplier.keys():
            raise ValueError(f"Unsupported aggregate {aggregate}")
        self.aggregate = aggregate

        self.head = FinetuningHead(
            num_outputs=num_outputs,
            hidden_size=self.encoder.embedding_size * aggregate_to_multiplier[aggregate],
            regression=regression,
        )

    @staticmethod
    def reshape_for_aggregate(
        encodings: torch.Tensor, aggregate: str, outputs_per_images: int
    ) -> torch.Tensor:
        encodings_im = rearrange(encodings, "(img p) h_dim -> img p h_dim", p=outputs_per_images)
        if aggregate == "quantiles":
            return torch.cat(
                [
                    torch.quantile(encodings_im, 0.25, dim=1),
                    torch.mean(encodings_im, dim=1),
                    torch.quantile(encodings_im, 0.75, dim=1),
                    # the unbiased (default) estimate divides by (n-1) giving NaN
                    #   for self.outputs_per_image == 1
                    torch.std(encodings_im, dim=1, correction=int(encodings_im.shape[1] > 1)),
                    torch.quantile(encodings_im, q=0.5, dim=1),  # median
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    torch.mean(encodings_im, dim=1),
                    torch.std(encodings_im, dim=1, correction=int(encodings_im.shape[1] > 1)),
                ],
                dim=-1,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # inputs are expected to be with 2 batch dimensions
        # (batches of images) (patches within an image) ...
        # vmap doesn't work with data dependent flows (yet)
        outputs_per_image = x.shape[1]
        encodings = self.encoder(
            x=rearrange(x, "b bp t d -> (b bp) t d"),
            # masking is created by the _mask_to_batch_tensor, which
            # doesn't know about this extra dimension
            mask=repeat(mask, "b t d -> (repeat b) t d", repeat=outputs_per_image),
        )
        encodings = self.reshape_for_aggregate(encodings, self.aggregate, outputs_per_image)
        return self.head(encodings)


class Presto(Seq2Seq):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, orig_indices, x_mask = self.encoder(
            x=x,
            mask=mask,
            eval_task=False,
        )

        return self.decoder(x, orig_indices, x_mask)

    @classmethod
    def construct(
        cls,
        encoder_embedding_size: int = 128,
        channel_embed_ratio: float = 0.5,
        encoder_depth=2,
        mlp_ratio=4,
        encoder_num_heads=8,
        decoder_embedding_size=128,
        decoder_depth=2,
        decoder_num_heads=8,
        max_sequence_length=60,
    ):
        encoder = Encoder(
            embedding_size=encoder_embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            num_heads=encoder_num_heads,
            max_sequence_length=max_sequence_length,
        )
        decoder = Decoder(
            channel_embeddings=encoder.channel_embed,
            encoder_embed_dim=encoder_embedding_size,
            decoder_embed_dim=decoder_embedding_size,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
        )
        return cls(encoder, decoder)

    def construct_finetuning_model(
        self,
        num_outputs: int,
        regression: bool = False,
    ):
        head = FinetuningHead(
            num_outputs=num_outputs,
            hidden_size=self.encoder.embedding_size,
            regression=regression,
        )
        model = PrestoFineTuningModel(self.encoder, head).to(device)
        model.train()
        return model

    @classmethod
    def load_pretrained(cls):
        model = cls.construct()
        model.load_state_dict(torch.load(default_model_path, map_location=device))
        return model
