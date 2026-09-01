from typing import Optional, Sequence

from torch import Tensor
from torch.nn import Module, ModuleList

from tabpfn.transformer import TransformerModel


class LoopedTransformerEncoder(Module):
    """Transformer encoder with untied edge blocks and a tied middle core."""

    def __init__(
        self,
        layers,
        warmup_layers: int,
        core_layers: int,
        exit_layers: int,
        effective_layers: int,
        core_repeat_pattern: Optional[Sequence[int]] = None,
        norm=None,
    ):
        super().__init__()
        self.layers = ModuleList(layers)
        self.warmup_layers = warmup_layers
        self.core_layers = core_layers
        self.exit_layers = exit_layers
        self.num_layers = effective_layers
        self.norm = norm

        physical_layers = warmup_layers + core_layers + exit_layers
        if len(self.layers) != physical_layers:
            raise ValueError(
                f"Expected {physical_layers} physical layers, got {len(self.layers)}."
            )

        if core_repeat_pattern is None:
            looped_layers = effective_layers - warmup_layers - exit_layers
            if looped_layers <= 0 or looped_layers % core_layers != 0:
                raise ValueError(
                    "effective_layers must equal warmup_layers + "
                    "core_layers * repeats + exit_layers."
                )
            self.core_repeats = looped_layers // core_layers
            self.core_repeat_pattern = None
        else:
            if len(core_repeat_pattern) != core_layers:
                raise ValueError(
                    "core_repeat_pattern must have one repeat count per core layer."
                )
            self.core_repeat_pattern = tuple(int(repeats) for repeats in core_repeat_pattern)
            if any(repeats <= 0 for repeats in self.core_repeat_pattern):
                raise ValueError("core_repeat_pattern repeat counts must be positive.")
            if effective_layers != warmup_layers + sum(self.core_repeat_pattern) + exit_layers:
                raise ValueError(
                    "effective_layers must equal warmup_layers + "
                    "sum(core_repeat_pattern) + exit_layers."
                )
            self.core_repeats = None

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        num_loops: Optional[int] = None,
    ) -> Tensor:
        output = src

        warmup_end = self.warmup_layers
        core_end = warmup_end + self.core_layers

        for mod in self.layers[:warmup_end]:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        core = self.layers[warmup_end:core_end]
        if num_loops is not None:
            num_loops = int(num_loops)
            if num_loops <= 0:
                raise ValueError("num_loops must be positive.")

        if self.core_repeat_pattern is None:
            repeats = self.core_repeats if num_loops is None else num_loops
            for _ in range(repeats):
                for mod in core:
                    output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        else:
            if num_loops is not None and self.core_layers != 1:
                raise ValueError(
                    "num_loops can only override core_repeat_pattern when there is "
                    "one physical core layer."
                )
            repeat_pattern = (
                self.core_repeat_pattern if num_loops is None else (num_loops,)
            )
            for mod, repeats in zip(core, repeat_pattern):
                for _ in range(repeats):
                    output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        for mod in self.layers[core_end:]:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LoopedTransformerModel(TransformerModel):
    def __init__(
        self,
        *args,
        looped_warmup_layers: int = 1,
        looped_core_layers: int = 2,
        looped_exit_layers: int = 1,
        looped_core_repeat_pattern: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        effective_layers = args[5] if len(args) > 5 else kwargs["nlayers"]
        physical_layers = looped_warmup_layers + looped_core_layers + looped_exit_layers

        if len(args) > 5:
            args = list(args)
            args[5] = physical_layers
            args = tuple(args)
        else:
            kwargs["nlayers"] = physical_layers

        super().__init__(*args, **kwargs)
        original_encoder = self.transformer_encoder
        self.transformer_encoder = LoopedTransformerEncoder(
            list(original_encoder.layers),
            warmup_layers=looped_warmup_layers,
            core_layers=looped_core_layers,
            exit_layers=looped_exit_layers,
            effective_layers=effective_layers,
            core_repeat_pattern=looped_core_repeat_pattern,
            norm=original_encoder.norm,
        )
        self.model_type = "LoopedTransformer"
        self.looped_effective_layers = effective_layers
        self.looped_physical_layers = physical_layers
