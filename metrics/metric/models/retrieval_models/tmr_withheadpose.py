import torch
import torch.nn as nn

from metric.models.utils.losses import InfoNCE_with_filtering, KLLoss
from typing import List, Dict, Optional
from torch import Tensor

def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask



class tmr(nn.Module):
    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        temperature: float = 0.1,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
    ) -> None:
        super().__init__()

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.motion_decoder = motion_decoder

        self.fact = 1
        self.lmd = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1, "headpose_recons": 1.0}

        # all losses, adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()


    def encode_text(self, inputs, sample_mean = False):
        encoded = self.text_encoder(inputs)

        dists = encoded.unbind(1)
        mu, logvar = dists
        if sample_mean:
            latent_vectors = mu

        return latent_vectors
    

    def encode_motion(self, inputs, sample_mean = False):
        encoded = self.motion_encoder(inputs)

        dists = encoded.unbind(1)
        mu, logvar = dists
        if sample_mean:
            latent_vectors = mu

        return latent_vectors


    def decode(
        self,
        latent_vectors: Tensor,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
    ):
        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.motion_decoder(z_dict)
        return motions


    def forward_motion(self, inputs,  mask, lengths = None, sample_mean=False):
        # Encoding the inputs and sampling if needed
        encoded = self.motion_encoder(inputs)

        if True:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + self.fact * eps * std

        # Decoding the latent vector: generating motions
        motions , headposes = self.decode(latent_vectors, lengths, mask)

        return motions , headposes , latent_vectors, dists

    def forward_text(self, inputs, mask, lengths = None,  sample_mean=False):
        # Encoding the inputs and sampling if needed
        encoded = self.text_encoder(inputs)

        if True:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + self.fact * eps * std

        # Decoding the latent vector: generating motions
        motions, headposes = self.decode(latent_vectors, lengths, mask)

        return motions, headposes, latent_vectors, dists


    def compute_loss(self, motion_x_dict, text_x_dict, return_all=False):
        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]
        ref_headposes = motion_x_dict['headpose']

        # text -> motion
        t_motions,  t_headposes, t_latents, t_dists = self.forward_text(text_x_dict, mask=mask)

        # motion -> motion
        m_motions, m_headposes, m_latents, m_dists = self.forward_motion(motion_x_dict, mask=mask)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )  
        
        losses["headpose_recons"] = self.reconstruction_loss_fn(t_headposes, ref_headposes) + self.reconstruction_loss_fn(m_headposes, ref_headposes) 

        # losses["recons"] = 0


        # VAE losses
        if True:
        # if False:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # losses["kl"] = 0

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, None)
        # losses["contrastive"] = self.contrastive_loss_fn(t_latents, m_latents, sent_emb)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents

        return losses








