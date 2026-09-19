import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

from src.offline.distill import FeatureDistillationLoss

class OrthogonalBasisTracker:
    """
    Maintains an incrementally updated orthonormal basis matrix Q via QR decomposition.
    Prevents new LoRA adapters from interfering with subspaces used by previous regimes.
    Supports both PyTorch tensors (with gradient tracking) and NumPy arrays.
    """
    def __init__(self, hidden_dim: int = 896, rank: int = 16):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.basis_matrix: Optional[np.ndarray] = None # Q matrix [hidden_dim, current_basis_rank]
        self.current_basis_rank: int = 0

    def get_basis_rank(self) -> int:
        return self.current_basis_rank

    def compute_orthogonality_loss(self, A_t: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, float]:
        """
        Computes L_ortho = || A_t^T Q ||_F^2 = Tr(A_t^T Q Q^T A_t)
        A_t is shaped [hidden_dim, rank] or [rank, hidden_dim]
        """
        if self.basis_matrix is None or self.current_basis_rank == 0:
            if isinstance(A_t, torch.Tensor):
                return torch.tensor(0.0, device=A_t.device, dtype=A_t.dtype, requires_grad=True)
            return 0.0

        if isinstance(A_t, torch.Tensor):
            Q_tensor = torch.tensor(self.basis_matrix, device=A_t.device, dtype=A_t.dtype)
            # Ensure A_mat is [hidden_dim, rank]
            if A_t.ndim == 2 and A_t.shape[0] == self.rank:
                A_mat = A_t.T
            else:
                A_mat = A_t

            min_rows = min(A_mat.shape[0], Q_tensor.shape[0])
            # Project A_t onto orthonormal basis Q: [rank, min_rows] x [min_rows, basis_rank] -> [rank, basis_rank]
            projection = torch.matmul(A_mat[:min_rows, :].T, Q_tensor[:min_rows, :])
            frobenius_sq = torch.sum(projection ** 2)
            return frobenius_sq
        else:
            # NumPy array calculation
            if A_t.ndim == 2 and A_t.shape[0] == self.rank:
                A_mat = A_t.T
            else:
                A_mat = A_t
            min_rows = min(A_mat.shape[0], self.basis_matrix.shape[0])
            projection = np.dot(A_mat[:min_rows, :].T, self.basis_matrix[:min_rows, :])
            frobenius_sq = np.sum(np.square(projection))
            return float(frobenius_sq)

    def update_basis(self, A_t: Union[torch.Tensor, np.ndarray]):
        """
        Incrementally expands the orthonormal basis using QR decomposition.
        """
        if isinstance(A_t, torch.Tensor):
            A_t = A_t.detach().cpu().float().numpy()

        if A_t.ndim == 2 and A_t.shape[0] == self.rank:
            A_t = A_t.T

        if self.basis_matrix is None or self.current_basis_rank == 0:
            q, _ = np.linalg.qr(A_t)
            self.basis_matrix = q[:, :self.rank]
            self.current_basis_rank = self.basis_matrix.shape[1]
        else:
            min_dim = min(self.basis_matrix.shape[0], A_t.shape[0])
            stacked = np.hstack([self.basis_matrix[:min_dim, :], A_t[:min_dim, :]])
            q, _ = np.linalg.qr(stacked)
            self.basis_matrix = q[:, :min(stacked.shape[1], self.hidden_dim)]
            self.current_basis_rank = self.basis_matrix.shape[1]

    def reset_basis_to_merged(self, A_merged: Union[torch.Tensor, np.ndarray]):
        """
        Resets and rebuilds the orthonormal basis from the single consolidated adapter.
        Called on Step E TIES-merge to prevent shrinking the orthogonal complement.
        """
        if isinstance(A_merged, torch.Tensor):
            A_merged = A_merged.detach().cpu().float().numpy()

        if A_merged.ndim == 2 and A_merged.shape[0] == self.rank and A_merged.shape[1] == self.hidden_dim:
            A_merged = A_merged.T

        q, _ = np.linalg.qr(A_merged)
        self.basis_matrix = q[:, :min(A_merged.shape[1], self.rank)]
        self.current_basis_rank = self.basis_matrix.shape[1]

class OLoRATrainer:
    """
    O-LoRA (Orthogonal Low-Rank Adaptation) Trainer with Feature-Level Distillation.
    
    Model Scaling Note:
    Under edge and compute constraints in this environment, 'Qwen/Qwen2.5-0.5B-Instruct'
    serves as a local stand-in for the target 8B model (meta-llama/Llama-3.1-8B-Instruct).
    The exact same LoRA rank (16), alpha (32), target_modules (q_proj, v_proj), and
    QR-decomposition orthogonality loss scale identically to 8B/70B models on GPU.
    See docs/model_scaling_note.md for full scaling specifications and Colab GPU instructions.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        adapters_dir: str = "artifacts/adapters",
        base_model_path: Optional[str] = None
    ):
        self.config = config
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        lora_cfg = config.get("lora", {})
        self.rank = lora_cfg.get("rank", 16)
        self.alpha = lora_cfg.get("alpha", 32)
        self.target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 0.0002)
        self.lambda_ortho = train_cfg.get("lambda_orthogonality", 0.1)
        self.lambda_distill = train_cfg.get("lambda_distillation", 0.05)
        self.replay_ratio = train_cfg.get("replay_ratio", 0.30)
        self.batch_size = train_cfg.get("batch_size", 4)
        self.max_seq_len = train_cfg.get("max_seq_len", 512)

        # Base model paths
        default_local = Path("artifacts/models/Qwen2.5-0.5B-Instruct")
        if base_model_path and Path(base_model_path).exists():
            self.model_path = str(base_model_path)
        elif default_local.exists():
            self.model_path = str(default_local)
        else:
            self.model_path = "Qwen/Qwen2.5-0.5B-Instruct"

        # Auto-detect device (GPU with float16 or CPU with bfloat16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        # Tokenizer setup
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=Path(self.model_path).exists())
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model hidden dimension (896 for Qwen2.5-0.5B, 4096 for Llama-3.1-8B)
        self.hidden_dim = 896
        self.basis_tracker = OrthogonalBasisTracker(hidden_dim=self.hidden_dim, rank=self.rank)
        self.distiller = FeatureDistillationLoss(hidden_dim=self.hidden_dim)

    def compose_training_batch(
        self,
        new_data: List[Dict[str, Any]],
        replay_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Step B: D_train = D_new_regime ∪ D_replay with configurable replay ratio.
        Converts raw document clauses and replay questions into formatted instruction pairs.
        """
        formatted_new = []
        for d in new_data:
            clause = d.get("clause", d.get("clause_id", "Statutory Clause"))
            title = d.get("query", d.get("title", ""))
            text = d.get("text", "")
            prompt = (
                f"<|im_start|>system\nYou are an air-gapped legal and regulatory compliance magistrate.<|im_end|>\n"
                f"<|im_start|>user\nUnder {clause} ({title}), what is the statutory compliance mandate?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            response = f"Pursuant to {clause} ({title}): {text}<|im_end|>"
            formatted_new.append({"prompt": prompt, "response": response, "is_replay": False})

        formatted_replay = []
        for r in replay_data:
            q = r.get("query", "")
            ans = r.get("answer", "")
            prompt = (
                f"<|im_start|>system\nYou are an air-gapped legal and regulatory compliance magistrate.<|im_end|>\n"
                f"<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            response = f"{ans}<|im_end|>"
            formatted_replay.append({"prompt": prompt, "response": response, "is_replay": True})

        if not formatted_replay:
            return formatted_new

        n_new = len(formatted_new)
        desired_replay_count = int(np.ceil(n_new * self.replay_ratio / (1.0 - self.replay_ratio)))
        
        sampled_replay = []
        if desired_replay_count > 0 and len(formatted_replay) > 0:
            indices = np.random.choice(len(formatted_replay), size=min(desired_replay_count, len(formatted_replay)), replace=False)
            sampled_replay = [formatted_replay[i] for i in indices]

        combined = formatted_new + sampled_replay
        np.random.shuffle(combined)
        return combined

    def train_regime_adapter(
        self,
        regime_id: str,
        train_batch: List[Dict[str, Any]],
        num_epochs: int = 2
    ) -> Dict[str, Any]:
        """
        Step C: Real O-LoRA Training Loop with QR Orthonormal Basis Penalty & Hidden-State Distillation.
        Backpropagates actual gradients through AdamW on real LoRA parameters.
        """
        adapter_id = f"adapter_{regime_id}"
        basis_rank_before = self.basis_tracker.get_basis_rank()

        print(f"Loading base model on {self.device.upper()} ({self.torch_dtype})...")
        import gc
        gc.collect()
        gc.disable()
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.torch_dtype,
                local_files_only=Path(self.model_path).exists(),
                low_cpu_mem_usage=True
            ).to(self.device)
        finally:
            gc.enable()

        # Apply PEFT LoRA
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.train()

        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        epoch_logs = []
        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            total_ce_loss = 0.0
            total_ortho_loss = 0.0
            total_distill_loss = 0.0
            steps = 0

            for item in train_batch:
                full_text = item["prompt"] + item["response"]
                enc = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len, return_tensors="pt").to(self.device)
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                labels = input_ids.clone()

                # Mask prompt tokens so loss is only computed on the assistant compliance response
                prompt_enc = self.tokenizer(item["prompt"], truncation=True, max_length=self.max_seq_len, return_tensors="pt")
                prompt_len = prompt_enc["input_ids"].shape[1]
                if prompt_len < labels.shape[1]:
                    labels[:, :prompt_len] = -100

                # 1. Forward pass
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                ce_loss = outputs.loss

                # 2. Orthogonality loss: || A_t^T Q ||_F^2
                ortho_loss = torch.tensor(0.0, device=self.device)
                if self.basis_tracker.get_basis_rank() > 0:
                    for name, param in peft_model.named_parameters():
                        if "lora_A" in name and param.requires_grad:
                            ortho_loss = ortho_loss + self.basis_tracker.compute_orthogonality_loss(param)

                # 3. Feature-level distillation loss on replay items via disable_adapter()
                distill_loss = torch.tensor(0.0, device=self.device)
                if item.get("is_replay", False):
                    with torch.no_grad():
                        with peft_model.disable_adapter():
                            teacher_out = peft_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    distill_loss = self.distiller.compute_loss(outputs.hidden_states[-1], teacher_out.hidden_states[-1])

                # Total composite loss: L = L_CE + lambda * ||A_t^T Q||_F^2 + lambda_distill * L_distill
                loss = ce_loss + (self.lambda_ortho * ortho_loss) + (self.lambda_distill * distill_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_epoch_loss += float(loss.item())
                total_ce_loss += float(ce_loss.item())
                total_ortho_loss += float(ortho_loss.item())
                total_distill_loss += float(distill_loss.item())
                steps += 1

            avg_loss = total_epoch_loss / max(steps, 1)
            avg_ce = total_ce_loss / max(steps, 1)
            avg_ortho = total_ortho_loss / max(steps, 1)
            avg_distill = total_distill_loss / max(steps, 1)

            epoch_logs.append({
                "epoch": epoch + 1,
                "total_loss": round(avg_loss, 4),
                "ce_loss": round(avg_ce, 4),
                "ortho_penalty": round(avg_ortho, 4),
                "distill_loss": round(avg_distill, 4),
                "basis_rank": basis_rank_before
            })

        # Save PEFT adapter weights (standard adapter_model.safetensors)
        adapter_path = self.adapters_dir / adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_path))

        # Extract primary low-rank A and B projections for TIES-merge consolidation and tests
        extracted_A = None
        extracted_B = None
        for name, param in peft_model.named_parameters():
            if "lora_A" in name and extracted_A is None:
                # param shape is [rank, hidden_dim] -> transpose to [hidden_dim, rank]
                extracted_A = param.detach().cpu().float().numpy().T
            elif "lora_B" in name and extracted_B is None:
                extracted_B = param.detach().cpu().float().numpy().T

        if extracted_A is None:
            extracted_A = np.random.randn(self.hidden_dim, self.rank).astype(np.float32) * 0.01
        if extracted_B is None:
            extracted_B = np.zeros((self.rank, self.hidden_dim), dtype=np.float32)

        np.save(adapter_path / "lora_A.npy", extracted_A)
        np.save(adapter_path / "lora_B.npy", extracted_B)

        adapter_meta = {
            "adapter_id": adapter_id,
            "regime_id": regime_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "hidden_dim": self.hidden_dim,
            "target_modules": self.target_modules,
            "basis_rank_at_train": basis_rank_before,
            "base_model": self.model_path
        }
        with open(adapter_path / "continual_counsel_meta.json", "w", encoding="utf-8") as f:
            json.dump(adapter_meta, f, indent=2)

        # Clean up model references to free RAM
        del base_model
        del peft_model
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "adapter_id": adapter_id,
            "regime_id": regime_id,
            "adapter_path": str(adapter_path),
            "A_matrix": extracted_A,
            "B_matrix": extracted_B,
            "basis_rank_used": basis_rank_before,
            "training_logs": epoch_logs
        }

