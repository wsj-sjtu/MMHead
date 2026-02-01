import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from typing import Dict


from retrieval_models.mld_vae.tools.embeddings import TimestepEmbedding, Timesteps
from retrieval_models.mld_vae.operator import PositionalEncoding
from retrieval_models.mld_vae.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from retrieval_models.mld_vae.operator.position_encoding import build_position_encoding


from einops import repeat


class ACTORStyleEncoder(nn.Module):
    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = build_position_encoding(
            latent_dim, position_embedding='learned')


        seq_trans_encoder_layer = TransformerEncoderLayer(
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            False,
        )
        encoder_norm = nn.LayerNorm(latent_dim)
        self.seqTransEncoder = SkipTransformerEncoder(seq_trans_encoder_layer, num_layers+1,
                                              encoder_norm)

        
        

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq.permute(1, 0, 2), src_key_padding_mask=~aug_mask)

        final = final.permute(1, 0, 2)
        return final[:, : self.nbtokens]


class ACTORStyleDecoder(nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats

        # self.sequence_pos_encoding = PositionalEncoding(
        #     latent_dim, dropout, batch_first=True
        # )


        self.sequence_pos_encoding = build_position_encoding(
            latent_dim, position_embedding='learned')

        seq_trans_decoder_layer =  TransformerDecoderLayer(
                latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                False,
            )
        
        decoder_norm = nn.LayerNorm(latent_dim)

        self.seqTransDecoder = SkipTransformerDecoder(seq_trans_decoder_layer, num_layers+1,
                                                  decoder_norm)


        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z_dict: Dict) -> Tensor:
        z = z_dict["z"]
        mask = z_dict["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        # print("z shape000", z.shape) # torch.Size([72, 256])
        z = z[:, None]  # sequence of 1 element for the memory
        # print("z shape", z.shape) #torch.Size([72, 1, 256])
        z = z.permute(1, 0, 2) 

        # Construct time queries
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = time_queries.permute(1, 0, 2)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )
        # print("output1", output.shape, time_queries.shape, z.shape, mask.shape)
        # torch.Size([300, 72, 256]) torch.Size([300, 72, 256]) torch.Size([1, 72, 256]) torch.Size([72, 300])
        

        output = self.final_layer(output)

        # print("output2", output.shape)
        # torch.Size([300, 72, 371])

        output = output.permute(1, 0, 2)
        # zero for padded area
        output[~mask] = 0
        return output
