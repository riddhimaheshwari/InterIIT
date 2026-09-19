import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union

class FeatureDistillationLoss:
    """
    Feature-level (hidden-state) distillation loss on replay sequences.
    Matches intermediate representations of the student model to the frozen teacher model.
    Supports both PyTorch tensors (with autograd) and numpy arrays.
    """
    def __init__(self, hidden_dim: int = 896):
        self.hidden_dim = hidden_dim

    def compute_loss(
        self,
        student_hidden_states: Union[torch.Tensor, np.ndarray],
        teacher_hidden_states: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, float]:
        """
        Mean Squared Error between normalized hidden state vectors.
        L_distill = (1 / N) * sum(||h_s - h_t||_2^2)
        """
        if isinstance(student_hidden_states, torch.Tensor):
            if not isinstance(teacher_hidden_states, torch.Tensor):
                teacher_hidden_states = torch.tensor(teacher_hidden_states, device=student_hidden_states.device, dtype=student_hidden_states.dtype)
            else:
                teacher_hidden_states = teacher_hidden_states.to(student_hidden_states.device)

            if student_hidden_states.shape != teacher_hidden_states.shape:
                min_dim = min(student_hidden_states.shape[-1], teacher_hidden_states.shape[-1])
                student_hidden_states = student_hidden_states[..., :min_dim]
                teacher_hidden_states = teacher_hidden_states[..., :min_dim]

            diff = student_hidden_states - teacher_hidden_states
            return torch.mean(diff ** 2)
        else:
            if student_hidden_states.shape != teacher_hidden_states.shape:
                min_dim = min(student_hidden_states.shape[-1], teacher_hidden_states.shape[-1])
                student_hidden_states = student_hidden_states[..., :min_dim]
                teacher_hidden_states = teacher_hidden_states[..., :min_dim]

            diff = student_hidden_states - teacher_hidden_states
            mse = np.mean(np.square(diff))
            return float(mse)

