from functools import partial

import torch
from torch import nn

from hydra.modules.hydra import Hydra
from tabpfn.hydra import HydraBlock, _init_weights
from tabpfn.layer import TransformerEncoderLayer
from tabpfn.transformer import TransformerModel
from tabpfn.utils import SeqBN, bool_mask_to_att_mask


class HydraEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        layer_idx,
        dropout=0.0,
        layer_norm_eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Hydra, layer_idx=layer_idx, **factory_kwargs)
        norm_cls = partial(nn.LayerNorm, eps=layer_norm_eps, **factory_kwargs)
        self.block = HydraBlock(
            d_model,
            mixer_cls,
            nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=False,
            residual_in_fp32=False,
        )
        self.norm_f = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        hidden_states = src.transpose(0, 1)
        hidden_states, residual = self.block(hidden_states, None)
        hidden_states = self.dropout(hidden_states)
        residual = hidden_states + residual
        return self.norm_f(residual).transpose(0, 1)


class HybridEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout=0.0,
        activation="gelu",
        pre_norm=False,
        recompute_attn=False,
        layer_norm_eps=1e-5,
        layer_types=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if layer_types is None:
            layer_types = ["hydra" if i % 2 == 0 else "transformer" for i in range(num_layers)]
        if len(layer_types) != num_layers:
            raise ValueError(f"Expected {num_layers} hybrid layer types, got {len(layer_types)}.")

        layers = []
        for i, layer_type in enumerate(layer_types):
            if layer_type == "hydra":
                layers.append(
                    HydraEncoderLayer(
                        d_model,
                        dim_feedforward,
                        layer_idx=i,
                        dropout=dropout,
                        layer_norm_eps=layer_norm_eps,
                        **factory_kwargs,
                    )
                )
            elif layer_type == "transformer":
                layers.append(
                    TransformerEncoderLayer(
                        d_model,
                        nhead,
                        dim_feedforward,
                        dropout,
                        activation=activation,
                        pre_norm=pre_norm,
                        recompute_attn=recompute_attn,
                        **factory_kwargs,
                    )
                )
            else:
                raise ValueError(f"Unknown hybrid layer type: {layer_type}")

        self.layers = nn.ModuleList(layers)
        self.layer_types = layer_types
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class HybridHydraTransformerModel(nn.Module):
    def __init__(
        self,
        encoder,
        n_out,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.0,
        style_encoder=None,
        y_encoder=None,
        pos_encoder=None,
        decoder=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        full_attention=False,
        efficient_eval_masking=True,
        layer_types=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.model_type = "HybridHydraTransformer"
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.style_encoder = style_encoder
        self.pos_encoder = pos_encoder
        self.decoder = decoder(ninp, nhid, n_out) if decoder is not None else nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_out),
        )
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.hybrid_encoder = HybridEncoder(
            ninp,
            nhead,
            nhid,
            nlayers,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
            recompute_attn=recompute_attn,
            layer_types=layer_types,
            device=device,
            dtype=dtype,
        )
        self.ninp = ninp
        self.n_out = n_out
        self.nhid = nhid
        self.init_method = init_method
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking
        self.init_weights()

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        return TransformerModel.generate_D_q_matrix(sz, query_size)

    def init_weights(self):
        if self.init_method is not None:
            self.apply(self.init_method)
        self.apply(partial(_init_weights, n_layer=self.hybrid_encoder.num_layers, n_residuals_per_layer=2))
        for layer in self.hybrid_encoder.layers:
            if isinstance(layer, TransformerEncoderLayer):
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"

        if len(src) == 2:
            src = (None,) + src

        style_src, x_src, y_src = src
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        style_src = (
            self.style_encoder(style_src).unsqueeze(0)
            if self.style_encoder
            else torch.tensor([], device=x_src.device)
        )

        if src_mask is None:
            full_len = len(x_src) + len(style_src)
            if self.full_attention:
                src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_src.device)
            elif self.efficient_eval_masking:
                src_mask = single_eval_pos + len(style_src)
            else:
                src_mask = self.generate_D_q_matrix(full_len, len(x_src) - single_eval_pos).to(x_src.device)

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.hybrid_encoder(src, src_mask)
        output = self.decoder(output)
        return output[single_eval_pos + len(style_src):]
