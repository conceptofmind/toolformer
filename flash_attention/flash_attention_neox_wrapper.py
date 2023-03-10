# From: https://github.com/kyleliang919/Long-context-transformers
import torch
from flash_attn.modules.mha import FlashSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb


class FlashAttentionWrapper(torch.nn.Module):
    def __init__(self, attention, max_seqlen=8192):
        super().__init__()
        self.attention = attention
        self.max_seqlen = max_seqlen
        self.flash_self_attention = FlashSelfAttention(causal=True)
        self.dropout_p = 0.0

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.attention.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (
            self.attention.num_attention_heads,
            3 * self.attention.head_size,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.attention.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.attention.head_size : 2 * self.attention.head_size].permute(
            0, 2, 1, 3
        )
        value = qkv[..., 2 * self.attention.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.attention.rotary_ndims]
        query_pass = query[..., self.attention.rotary_ndims :]
        key_rot = key[..., : self.attention.rotary_ndims]
        key_pass = key[..., self.attention.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.attention.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        qkv = (
            torch.concat(
                [query.unsqueeze(2), key.unsqueeze(2), value.unsqueeze(2)], dim=2
            )
            .permute(0, 3, 2, 1, 4)
            .half()
        )
        attn_output = self.flash_self_attention(qkv)
        attn_weights = None

        # Reshape outputs
        attn_output = attn_output.view(
            attn_output.size(0),
            attn_output.size(1),
            self.attention.num_attention_heads * self.attention.head_size,
        )
        attn_output = self.attention.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
