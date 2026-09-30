# VERBATIM copy of inference/model.py from huggingface.co/manavdhamecha77/iSign-t5-pose-to-text
# (paper arXiv 2609.12993). Do not edit: this is the exact PoseEncoder + PoseT5 architecture.
"""
PoseT5: T5 model with pose keypoint encoder.
Maps pose keypoints -> T5 embedding space -> T5 encoder-decoder.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class PoseEncoder(nn.Module):
    """
    Encodes pose keypoints into T5 embedding space.
    
    Input: (batch_size, seq_length, pose_dim)
    Output: (batch_size, seq_length, embedding_dim)
    """
    
    def __init__(self, pose_dim: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.pose_dim = pose_dim
        self.embedding_dim = embedding_dim
        
        # Simple MLP: pose_dim -> hidden_dim -> embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (batch_size, seq_length, pose_dim)
        
        Returns:
            embeddings: (batch_size, seq_length, embedding_dim)
        """
        return self.encoder(pose)


class PoseT5(nn.Module):
    """
    T5 model with pose keypoint input.
    
    Architecture:
    - PoseEncoder: pose keypoints -> T5 embeddings
    - T5ForConditionalGeneration: standard encoder-decoder
    """
    
    def __init__(
        self,
        model_name: str,
        pose_dim: int,
        local_model_path: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pose_dim = pose_dim
        
        # Load T5 model
        if local_model_path:
            logger.info(f"Loading T5 from local path: {local_model_path}")
            self.t5 = T5ForConditionalGeneration.from_pretrained(local_model_path)
        else:
            logger.info(f"Loading T5 from HuggingFace: {model_name}")
            self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Get embedding dimension from T5
        embedding_dim = self.t5.config.d_model
        
        # Create pose encoder
        self.pose_encoder = PoseEncoder(pose_dim, embedding_dim)
        
        logger.info(f"PoseT5 initialized: pose_dim={pose_dim}, embedding_dim={embedding_dim}")
    
    def forward(
        self,
        pose: torch.Tensor,
        pose_length: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ):
        """
        Forward pass for training.
        
        Args:
            pose: (batch_size, seq_length, pose_dim) - pose keypoints
            pose_length: (batch_size,) - actual length of each pose sequence
            input_ids: (batch_size, tgt_seq_length) - target text token ids
            attention_mask: (batch_size, tgt_seq_length) - target attention mask
            labels: (batch_size, tgt_seq_length) - target labels for loss computation
            decoder_input_ids: (batch_size, tgt_seq_length) - optional decoder input
        
        Returns:
            outputs: T5 model output (loss, logits, etc.)
        """
        # Encode pose to embeddings
        pose_embeddings = self.pose_encoder(pose)  # (batch_size, seq_length, d_model)
        
        # Create pose attention mask (1 for real tokens, 0 for padding)
        batch_size, seq_length, _ = pose_embeddings.shape
        device = pose_embeddings.device
        
        pose_attention_mask = (torch.arange(seq_length, device=device)[None, :] < pose_length[:, None]).long()
        
        # Forward through T5
        # Use pose embeddings as encoder input instead of text embeddings
        outputs = self.t5(
            inputs_embeds=pose_embeddings,  # Use pose embeddings directly
            attention_mask=pose_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        
        return outputs
    
    def generate(
        self,
        pose: torch.Tensor,
        pose_length: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
    ):
        """
        Generate text from pose input.
        
        Args:
            pose: (batch_size, seq_length, pose_dim)
            pose_length: (batch_size,)
            max_length: Maximum generation length
            num_beams: Beam search size
            early_stopping: Stop early if done
        
        Returns:
            generated_ids: (batch_size, max_length)
        """
        # Encode pose
        pose_embeddings = self.pose_encoder(pose)
        
        # Create attention mask
        batch_size, seq_length, _ = pose_embeddings.shape
        device = pose_embeddings.device
        
        pose_attention_mask = (torch.arange(seq_length, device=device)[None, :] < pose_length[:, None]).long()
        
        # Generate
        generated_ids = self.t5.generate(
            inputs_embeds=pose_embeddings,
            attention_mask=pose_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=2.0,
        )
        
        return generated_ids
    
    def save_pretrained(self, save_path: str):
        """Save model and encoder."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        self.t5.save_pretrained(os.path.join(save_path, "t5"))
        torch.save(self.pose_encoder.state_dict(), os.path.join(save_path, "pose_encoder.pt"))
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(
        self,
        load_path: str,
        load_pose_encoder: bool = True,
    ):

        import os

        self.t5 = T5ForConditionalGeneration.from_pretrained(
            os.path.join(
                load_path,
                "t5",
            )
        )

        if load_pose_encoder:

            pose_encoder_path = os.path.join(
                load_path,
                "pose_encoder.pt",
            )

            self.pose_encoder.load_state_dict(
                torch.load(
                    pose_encoder_path,
                    map_location="cpu",
                )
            )

        logger.info(
            f"Model loaded from {load_path}"
        )


def create_model(
    model_name: str,
    pose_dim: int,
    local_model_path: str = None,
):
    """Factory function to create PoseT5 model."""
    return PoseT5(
        model_name=model_name,
        pose_dim=pose_dim,
        local_model_path=local_model_path,
    )
