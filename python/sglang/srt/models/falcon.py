import math
import torch
import torch.nn as nn
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes

def reorder_falcon_qkv_bias(bias, num_heads, head_dim):
    bias = bias.view(
        num_heads,
        3,
        head_dim,
    )

    q = bias[:, 0, :].reshape(-1)
    k = bias[:, 1, :].reshape(-1)
    v = bias[:, 2, :].reshape(-1)

    return torch.cat([q, k, v], dim=0)

def reorder_falcon_qkv_weight(weight, num_heads, head_dim):
            in_dim = weight.shape[1]
    
            
            weight = weight.view(
                num_heads,
                3,
                head_dim,
                in_dim,
            )
    
    
            q = weight[:, 0, :, :].reshape(
                num_heads * head_dim,
                in_dim,
            )
    
            
            k = weight[:, 1, :, :].reshape(
                num_heads * head_dim,
                in_dim,
            )
    
          
            v = weight[:, 2, :, :].reshape(
                num_heads * head_dim,
                in_dim,
            )
    
            return torch.cat([q, k, v], dim=0)

class FalconDecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.LayerNorm = LayerNorm
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(
            config,
            "num_kv_heads",
            None,
        ) or self.num_heads

        alibi_slopes_tensor = _get_alibi_slopes(self.num_heads).cuda()

        self.Attention = RadixAttention(
            num_kv_heads=self.num_kv_heads,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            scaling=(config.hidden_size // config.num_attention_heads) ** -0.5,
            layer_id=layer_id,
        )
        self.Attention.alibi_slopes = alibi_slopes_tensor

        self.attn_dense = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.attn_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=getattr(config, "bias", False),
        )

        self.gelu = nn.GELU()

        intermediate_size = getattr(
            config, "intermediate_size", config.hidden_size * 4
        )

        self.UP_projection = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=intermediate_size,
            bias=getattr(config, "bias", False),
        )

        self.Undo_wide = RowParallelLinear(
            input_size=intermediate_size,
            output_size=self.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.input_layernorm = self.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.post_attention_layernorm = self.LayerNorm(
            self.hidden_size,
                eps=config.layer_norm_epsilon,
                )



    def forward(self, X, positions, forward_batch):
        X_cleaned = self.input_layernorm(X)

        qkv, _ = self.attn_proj(X_cleaned)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = qkv.split(
            [q_size, kv_size, kv_size],
            dim=-1,
        )

        v = v.contiguous()

        att_output = self.Attention(q, k, v, forward_batch)
        
        att_output, _ = self.attn_dense(att_output)

        X = X + att_output

        mlp_input = self.post_attention_layernorm(X)

        X_wide, _ = self.UP_projection(mlp_input)
        X_gelu = self.gelu(X_wide)
        mlp_output, _ = self.Undo_wide(X_gelu)

        output = X + mlp_output
        return output


class FalconModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                FalconDecoderLayer(config, layer_id=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(self, input_ids, positions, forward_batch):
        hidden_States = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_States = layer(hidden_States, positions, forward_batch)

        hidden_States = self.final_layernorm(hidden_States)
        return hidden_States


class FalconForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()

        self.logits_processor = LogitsProcessor(config)
        self.transformer = FalconModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    

    @torch.no_grad
    def load_weights(self, weights_iterable):
        for name, loaded_weight in weights_iterable:

            if "transformer.word_embeddings.weight" in name:
                self.transformer.embed_tokens.weight.copy_(loaded_weight)
                weight_loader = getattr(
                    self.lm_head.weight, "weight_loader", default_weight_loader
                )
                weight_loader(self.lm_head.weight, loaded_weight)

            elif "transformer.ln_f.weight" in name:
                self.transformer.final_layernorm.weight.copy_(loaded_weight)
            elif "transformer.ln_f.bias" in name:
                self.transformer.final_layernorm.bias.copy_(loaded_weight)

            elif "lm_head.weight" in name:
                weight_loader = getattr(
                    self.lm_head.weight, "weight_loader", default_weight_loader
                )
                weight_loader(self.lm_head.weight, loaded_weight)

            elif "transformer.h." in name:
                layer_id = int(name.split(".")[2])
                layer = self.transformer.layers[layer_id]

                name_inside_layer = ".".join(name.split(".")[3:])

                if name_inside_layer == "input_layernorm.weight":
                    layer.input_layernorm.weight.copy_(loaded_weight)

                elif name_inside_layer == "input_layernorm.bias":
                    layer.input_layernorm.bias.copy_(loaded_weight)

                elif name_inside_layer == "post_attention_layernorm.weight":
                    layer.post_attention_layernorm.weight.copy_(loaded_weight)

                elif name_inside_layer == "post_attention_layernorm.bias":
                    layer.post_attention_layernorm.bias.copy_(loaded_weight)

                elif name_inside_layer == "self_attention.query_key_value.weight":

                    loaded_weight = reorder_falcon_qkv_weight(
                        loaded_weight,
                        layer.num_heads,
                        layer.head_dim,
                    )

                    weight_loader = getattr(
                        layer.attn_proj.weight,
                        "weight_loader",
                        default_weight_loader,
                    )

                    weight_loader(
                        layer.attn_proj.weight,
                        loaded_weight,
                    )

                elif name_inside_layer == "self_attention.query_key_value.bias":

                    loaded_weight = reorder_falcon_qkv_bias(
                        loaded_weight,
                        layer.num_heads,
                        layer.head_dim,
                    )

                    weight_loader = getattr(
                        layer.attn_proj.bias,
                        "weight_loader",
                        default_weight_loader,
                    )

                    weight_loader(
                        layer.attn_proj.bias,
                        loaded_weight,
                    )


                elif name_inside_layer == "self_attention.dense.bias":
                    weight_loader = getattr(
                        layer.attn_dense.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(layer.attn_dense.bias, loaded_weight)


                elif name_inside_layer == "mlp.dense_h_to_4h.bias":
                    weight_loader = getattr(
                        layer.UP_projection.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(layer.UP_projection.bias, loaded_weight)


                elif name_inside_layer == "mlp.dense_4h_to_h.bias":
                    weight_loader = getattr(
                        layer.Undo_wide.bias,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(layer.Undo_wide.bias, loaded_weight)

                elif name_inside_layer == "mlp.dense_h_to_4h.weight":
                    weight_loader = getattr(
                        layer.UP_projection.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(layer.UP_projection.weight, loaded_weight)

                elif name_inside_layer == "mlp.dense_4h_to_h.weight":
                    weight_loader = getattr(
                        layer.Undo_wide.weight,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(layer.Undo_wide.weight, loaded_weight)

                elif name_inside_layer == "self_attention.dense.weight":
                    weight_loader = getattr(
                        layer.attn_dense.weight,
                        "weight_loader",
                        default_weight_loader,  
                    )
                    weight_loader(layer.attn_dense.weight, loaded_weight)

               


EntryClass = [FalconForCausalLM]