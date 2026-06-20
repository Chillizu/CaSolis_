"""
Mamba + Multi-Token Prediction (MTP) + JEPA world model

Predicts n future tokens in parallel (Meta ICML 2024)
+ predicts output embedding in latent space (JEPA-style)

Architecture:
  embed → Mamba trunk → norm → [LM head 0: token_{t+1}]
                                [LM head 1: token_{t+2}]
                                [LM head 2: token_{t+3}]
                                [LM head 3: token_{t+4}]
                                [World head: output embedding]
"""

import torch, torch.nn as nn, torch.nn.functional as F
from arch.mamba_model import MambaBlock

V = 2022  # base vocab size
TV = V + 19  # total vocab (with special tokens)
PAD = TV - 1  # padding token

class MambaMTP(nn.Module):
    """Mamba + Multi-Token Prediction + JEPA world model"""

    def __init__(self, d_model=1024, n_pred=4):
        super().__init__()
        self.d_model = d_model
        self.n_pred = n_pred

        self.embed = nn.Embedding(TV, d_model)
        self.mamba = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

        # Multi-token prediction heads (one per future offset)
        self.lm_heads = nn.ModuleList([
            nn.Linear(d_model, V) for _ in range(n_pred)
        ])

        # JEPA-style world model head: predict output embedding in latent space
        self.world_head = nn.Linear(d_model, d_model)

    def forward(self, tokens, output_ids=None):
        """
        Args:
            tokens: (B, S) token IDs
            output_ids: (B, O) output token IDs for world loss (optional)
        Returns:
            lm_loss: cross-entropy over n_pred heads
            world_loss: MSE in latent embedding space
            head_losses: list of individual head losses (for monitoring)
        """
        B, S = tokens.shape
        x = self.embed(tokens)          # (B, S, d)
        y = self.norm(self.mamba(x))    # (B, S, d)

        # ---- Multi-token prediction losses ----
        lm_losses = []
        valid_count = 0

        for i, head in enumerate(self.lm_heads):
            shift = i + 1
            if shift >= S:
                break

            logits = head(y[:, :-shift])         # (B, S-shift, V)
            targets = tokens[:, shift:]           # (B, S-shift)
            mask = targets < V                    # exclude special tokens

            if mask.sum() > 0:
                loss = F.cross_entropy(
                    logits[mask].reshape(-1, V),
                    targets[mask].reshape(-1)
                )
                lm_losses.append(loss)

        if lm_losses:
            lm_loss = sum(lm_losses) / len(lm_losses)
        else:
            lm_loss = torch.tensor(0.0, device=tokens.device)

        # ---- JEPA world loss: predict output embedding ----
        world_loss = torch.tensor(0.0)
        if output_ids is not None and output_ids.numel() > 2:
            h_last = y[:, -1]                     # (B, d)
            pred_emb = self.world_head(h_last)    # (B, d)

            # Target: mean embedding of output tokens (detached)
            out_emb = self.embed(output_ids).mean(dim=1)  # (B, d)
            world_loss = 0.5 * F.mse_loss(pred_emb, out_emb.detach())

        return lm_loss, world_loss, [l.item() for l in lm_losses]

    @torch.no_grad()
    def generate(self, seed_ids, n=60, temp=0.85):
        """Standard autoregressive generation (uses head 0 only)"""
        out = list(seed_ids)
        self.eval()
        for _ in range(n):
            x = torch.tensor([out[-120:]]).long()
            h = self.norm(self.mamba(self.embed(x)))
            logits = self.lm_heads[0](h)  # head_0 for next token
            lp = F.softmax(logits[0, -1] / temp, dim=-1)
            nt = torch.multinomial(lp, 1).item()
            if nt >= V:
                break
            out.append(nt)
        self.train()
        return out

    @torch.no_grad()
    def generate_with_plan(self, seed_ids, n=60, temp=0.85):
        """
        Generate using plan information from MTP heads.
        Uses uncertainty across heads to guide generation:
        - If all heads agree → confident → keep going
        - If heads disagree → explore more (higher temp)
        """
        out = list(seed_ids)
        self.eval()

        for _ in range(n):
            x = torch.tensor([out[-60:]]).long()
            h = self.norm(self.mamba(self.embed(x)))
            h_last = h[0, -1]  # (d,)

            # Get predictions from all heads
            all_logits = []
            for head in self.lm_heads:
                all_logits.append(F.softmax(head(h_last), dim=-1))

            # Average agreement: mean of pairwise KL div
            # High agreement = all heads predict similar tokens
            agreement = 0
            for i in range(len(all_logits)):
                for j in range(i+1, len(all_logits)):
                    agreement += -(all_logits[i] * all_logits[j].log()).sum()
            agreement /= len(all_logits) * (len(all_logits)-1) / 2 + 1e-8

            # Low agreement = surprising = curious → higher temp
            adaptive_temp = temp * (2.0 - agreement.clamp(0, 1))
            lp = all_logits[0]  # use head_0 but with adaptive temperature
            reweighted = lp ** (1.0 / adaptive_temp)
            reweighted /= reweighted.sum()

            nt = torch.multinomial(reweighted, 1).item()
            if nt >= V:
                break
            out.append(nt)

        self.train()
        return out


def test():
    """Quick sanity check: model builds and trains"""
    model = MambaMTP(d_model=1024, n_pred=4)
    print(f"📐 MambaMTP: {sum(p.numel() for p in model.parameters()):,} params")

    B, S = 4, 64
    tokens = torch.randint(0, 100, (B, S))
    out_ids = torch.randint(0, 100, (B, 20))

    lm_loss, w_loss, head_l = model(tokens, out_ids)
    loss = lm_loss + 0.3 * w_loss
    loss.backward()

    print(f"   Forward + backward: ✅")
    print(f"   LM loss: {lm_loss.item():.4f} ({len(head_l)} heads active)")
    print(f"   Head losses: {[f'{x:.4f}' for x in head_l]}")
    print(f"   World loss: {w_loss.item():.4f}")

    # Test generation
    seed = torch.randint(0, 100, (1, 20)).tolist()[0]
    gen = model.generate(seed, n=30)
    print(f"   Gen: {len(gen)} tokens (seed: 20)")
    gen2 = model.generate_with_plan(seed, n=30)
    print(f"   Plan-gen: {len(gen2)} tokens (seed: 20)")
    print("✅ MambaMTP OK")


if __name__ == "__main__":
    test()
