    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding import get_rope_wrapper
    from sglang.srt.layers.attention import RadixAttention
    from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
    import torch 
    import torch.nn as nn 
    from sglang.srt.layers.linear import ParallelLMHead
    from sglang.srt.layers.logits_processor import LogitsProcessor




    class FalconDecoderLayer(nn.Module):
        def __init__(self, config, layer_id):
            super().__init__()
            self.Norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            self.Attention = RadixAttention(
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                layer_id=layer_id 
            )

            self.attn_proj = QKVParallelLinear(
                input_size=config.hidden_size,
                output_size=(config.num_attention_heads + 2 * config.num_kv_heads) * (config.hidden_size // config.num_attention_heads),
                bias=config.bias,
            )

            self.gelu = nn.GELU()
            self.Undo_wide = RowParallelLinear(
                input_size=config.intermediate_size,
                output_size=config.hidden_size,
                bias=config.bias,
            )

            self.UP_projection = QKVParallelLinear(
                input_size=config.hidden_size,
                output_size=config.intermediate_size,
                bias=config.bias
    )




        def forward(self, X, forward_batch, positions, ):
            X_cleaned = self.Norm(X,)
            qkv, _ = self.attn_proj(X_cleaned)
            att_output = self.Attention(qkv, positions, forward_batch)

            X_wide,_ = self.UP_projection(X_cleaned)
            X_gelu = self.gelu(X_wide)
            mlp_output,_ = self.Undo_wide(X_gelu)

            output = X + mlp_output + att_output
            return output 
        

        
    class FalconModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.Norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            self.layers = nn.ModuleList(
                [FalconDecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)]
            )
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        def forward(self, config, positions, forward_batch, input_ids):
            hidden_States = self.embed_tokens(input_ids)

            for layer in self.layers:
                hidden_States = layer(hidden_States, forward_batch, positions)

            hidden_States = self.Norm(hidden_States)

            return hidden_States
        

    class FalconForCausalLM(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.logits_processor = LogitsProcessor(config)
            self.transformer = FalconModel(config)
            self.lm_head = ParallelLMHead(
                vocab_size=config.vocab_size,
                embedding_dim=config.hidden_size,
                bias=False
            )

        def forward(forward_batch, positions, input_ids, self):
            hidden_states = self.transformer(input_ids, forward_batch, input_ids)
            logits,_ = self.lm_head(hidden_states)
            input_id = self.logits_processor(logits, forward_batch)


        def load_weights(self, weights_iterable):
            for name, loaded_weight in weights_iterable:
                # 1. Global Token Embeddings
                if "transformer.word_embeddings.weight" in name:
                    self.transformer.embed_tokens.weight.copy_(loaded_weight)
                    
        
                elif "transformer.ln_f.weight" in name:
                    self.transformer.Norm.weight.copy_(loaded_weight)
                elif "transformer.ln_f.bias" in name:
                    self.transformer.Norm.bias.copy_(loaded_weight)
                    
                # 3. Final Causal LM Head Vocabulary Projection
                elif "lm_head.weight" in name:
                    self.lm_head.weight.loader(loaded_weight)
                    
            
                elif "transformer.h." in name:
                    layer_id = int(name.split(".")[2])
                    layer = self.transformer.layers[layer_id]
                    
                
                    name_inside_layer = ".".join(name.split(".")[3:])
                    
                    if name_inside_layer == "input_layernorm.weight":
                        layer.Norm.weight.copy_(loaded_weight)
                    elif name_inside_layer == "input_layernorm.bias":
                        layer.Norm.bias.copy_(loaded_weight)
                        
                
                    elif name_inside_layer == "self_attention.query_key_value.weight":
                        layer.attn_proj.weight.loader(loaded_weight)
                        
                
                    elif name_inside_layer == "mlp.dense_h_to_4h.weight":
                        layer.UP_projection.weight.loader(loaded_weight)
                        
            
                    elif name_inside_layer == "mlp.dense_4h_to_h.weight":
                        layer.Undo_wide.weight.loader(loaded_weight)
            

    EntryClass = [FalconForCausalLM]

        
