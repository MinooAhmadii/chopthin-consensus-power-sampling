"""Sequential Monte Carlo decoding for power sampling.

N particles decode one token at a time. Each carries an importance weight for the target
p(y | x)^alpha, with alpha ramped from 1 to its final value over the first `alpha_ramp_tokens`
tokens. At every block boundary the effective sample size (ESS) is checked; when it drops
below `ess_threshold * N` the population is resampled with either systematic resampling
(Power-SMC) or Chopthin (CCPS). Chopthin keeps its unequal output weights; "chopthin_reset"
is the ablation that resets them to uniform.

After resampling, only the unique ancestors' KV caches are kept until the next forward pass,
then expanded back to N. This is copy-on-write bookkeeping, not an approximation.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from chopthin import chopthin


@dataclass
class SMCConfig:
    n_particles: int = 32
    alpha: float = 2.0
    temperature: float = 0.5           # proposal temperature; the paper uses 1/alpha
    max_new_tokens: int = 4096
    min_new_tokens: int = 100          # EOS is masked in the proposal before this many tokens
    top_p: float = 0.9
    alpha_ramp_tokens: int = 100
    block_size: int = 64
    ess_threshold: float = 0.5
    stop_on_boxed: bool = True         # a particle stops once its tail holds a non-empty \boxed{}
    boxed_check_window_tokens: int = 256
    resampler: str = "systematic"      # "systematic" | "chopthin" | "chopthin_reset"
    eta: float = 5.8284271247461903    # chopthin weight-ratio bound, 3 + sqrt(8)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def top_p_filter_(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumprobs > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
    logits.masked_fill_(mask, -float("inf"))
    return logits


def effective_sample_size(w, eps=1e-12):
    w = w.clamp_min(eps)
    return float(1.0 / torch.sum(w * w).item())


def _alpha_ramp(t, alpha_final, ramp_T):
    if ramp_T <= 1:
        return float(alpha_final)
    if t < ramp_T:
        frac = float(t + 1) / float(ramp_T)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


_BOX_ONLY_RE = re.compile(r"^[\s{}]*$")


def _has_nonempty_boxed(text):
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            j = idx + len(macro)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                right = None
                for k in range(j, len(text)):
                    c = text[k]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            right = k
                            break
                if right is not None:
                    content = text[j + 1 : right].strip()
                    while (
                        len(content) >= 2 and content[0] == "{" and content[-1] == "}"
                    ):
                        inner = content[1:-1].strip()
                        if inner == content:
                            break
                        content = inner
                    if content and (_BOX_ONLY_RE.fullmatch(content) is None):
                        return True
            start = idx + len(macro)
    return False


# ---------------------------------------------------------------------------
# Resamplers: both return ancestor indices; chopthin also returns new weights
# ---------------------------------------------------------------------------

def systematic_resample(w, generator=None):
    N = w.numel()
    device = w.device
    if generator is None:
        u0 = torch.rand((), device=device)
    else:
        u0 = torch.rand((), device=device, generator=generator)
    positions = (u0 + torch.arange(N, device=device)) / N
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    idx = idx.clamp_max(N - 1).to(torch.long)
    return idx


def chopthin_resample(w, eta, N, generator):
    """Torch wrapper around the pure-Python chopthin(). A seed for chopthin's internal draws
    is taken from `generator`, so the result is tied to the resampling RNG stream."""
    device = w.device
    w_list = w.detach().to("cpu", torch.float64).tolist()
    seed = int(
        torch.randint(0, 2_147_483_647, (1,), generator=generator, device=device).item()
    )
    idx, new_w = chopthin(w_list, float(eta), int(N), random.Random(seed))
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
    new_w_t = torch.tensor(new_w, dtype=torch.float32, device=device)
    return idx_t, new_w_t


# ---------------------------------------------------------------------------
# KV cache: keep one copy per unique ancestor, expand back to N before a forward pass
# ---------------------------------------------------------------------------

def _recursive_select_batch(obj, idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if idx.numel() == 0:
            return obj
        if obj.dim() >= 1 and obj.size(0) >= idx.max().item() + 1:
            return obj.index_select(0, idx.to(obj.device)).contiguous()
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_select_batch(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_select_batch(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_select_batch(v, idx) for k, v in obj.items()}
    return obj


def _recursive_expand_batch(obj, expand_idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if expand_idx.numel() == 0:
            return obj
        U = int(expand_idx.max().item()) + 1
        if obj.dim() >= 1 and obj.size(0) == U:
            return obj.index_select(0, expand_idx.to(obj.device))
        return obj
    if isinstance(obj, tuple):
        return tuple(_recursive_expand_batch(x, expand_idx) for x in obj)
    if isinstance(obj, list):
        return [_recursive_expand_batch(x, expand_idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_expand_batch(v, expand_idx) for k, v in obj.items()}
    return obj


def select_cache_subset(model, past, idx):
    """Shrink the cache from N particles to the unique ancestors `idx` (in place when possible)."""
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, idx)
        except Exception:
            pass
    return _recursive_select_batch(past, idx)


def expand_cache(model, past, expand_idx):
    """Expand the cache from U unique entries back to N particles; expand_idx[i] is particle
    i's slot in the U-sized batch."""
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(expand_idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, expand_idx)
        except Exception:
            pass
    return _recursive_expand_batch(past, expand_idx)


def _build_prompt_cache(model, input_ids_1, N):
    """Run the prompt once at batch size 1, then replicate its cache to N particles."""
    out = model(input_ids=input_ids_1, use_cache=True)
    past_1 = getattr(out, "past_key_values", None)
    logits_1 = out.logits[:, -1, :]
    expand_idx = torch.zeros(N, dtype=torch.long, device=input_ids_1.device)
    past_N = expand_cache(model, past_1, expand_idx)
    logits_N = logits_1.expand(N, -1).contiguous()
    del out
    torch.cuda.empty_cache()
    return past_N, logits_N


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_smc(
    model,
    tokenizer,
    input_ids: torch.Tensor,  # [1, L]
    cfg: SMCConfig,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode N particles for one prompt. Returns the final sequences, their log weights,
    the weight-drawn particle, and per-run statistics."""
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    device = input_ids.device
    # Two RNG streams: token sampling and resampling. Keeping them separate means both
    # resamplers see identical token draws until their populations actually diverge, so
    # the systematic and Chopthin arms are paired per problem.
    g_sample = torch.Generator(device=device)
    g_resample = torch.Generator(device=device)
    if seed is not None:
        g_sample.manual_seed(int(seed))
        g_resample.manual_seed(int(seed) + 2_147_483_647)

    N = int(cfg.n_particles)
    prompt_len = input_ids.size(1)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    past, last_logits = _build_prompt_cache(model, input_ids, N)

    total_len = prompt_len + int(cfg.max_new_tokens)
    seqs = torch.full((N, total_len), pad_id, dtype=torch.long, device=device)
    seqs[:, :prompt_len] = input_ids.expand(N, -1)
    cur = prompt_len

    done = torch.zeros(N, dtype=torch.bool, device=device)
    cum_logp = torch.zeros(N, dtype=torch.float32, device=device)
    log_w = torch.zeros(N, dtype=torch.float32, device=device)

    alpha_final = float(cfg.alpha)
    ramp_T = max(int(cfg.alpha_ramp_tokens), 1)
    prev_alpha = 1.0

    stats = {
        "resample_count": 0,
        "ess_history": [],
        "ess_after_history": [],          # chopthin only: ESS of the carried weights
        "unique_ancestors_history": [],
    }

    Tmax = int(cfg.max_new_tokens)
    min_new = max(int(cfg.min_new_tokens), 0)

    # After resampling, particle i reads its logits from slot logical_to_physical[i] of the
    # shrunken batch until the cache is expanded again.
    logical_to_physical = None

    for t in range(Tmax):
        alpha_t = _alpha_ramp(t=t, alpha_final=alpha_final, ramp_T=ramp_T)

        base_logprobs = F.log_softmax(last_logits.float(), dim=-1)
        if cfg.temperature == 1.0:
            prop_logits = last_logits.float()
        else:
            prop_logits = last_logits.float() / cfg.temperature

        if logical_to_physical is not None:
            base_logprobs = base_logprobs[logical_to_physical]  # [N, V]
            prop_logits = prop_logits[logical_to_physical]      # [N, V]

        if t < min_new:
            prop_logits[:, eos_id] = -float("inf")
        if cfg.top_p is not None and cfg.top_p < 1.0:
            top_p_filter_(prop_logits, cfg.top_p)
        prop_logprobs = F.log_softmax(prop_logits, dim=-1)  # [N, V]

        # ---- Sample one token per active particle; finished particles emit EOS ----
        next_tokens = torch.empty(N, dtype=torch.long, device=device)
        active = ~done
        if active.any():
            probs_active = torch.exp(prop_logprobs[active])
            next_tokens[active] = torch.multinomial(
                probs_active, num_samples=1, generator=g_sample
            ).squeeze(-1)
        next_tokens[done] = eos_id

        idx_tok = next_tokens.view(N, 1)
        token_logp = torch.gather(base_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        token_logq = torch.gather(prop_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        token_logp = torch.where(done, torch.zeros_like(token_logp), token_logp)
        token_logq = torch.where(done, torch.zeros_like(token_logq), token_logq)

        # ---- Weight update (Eq. 3 of the paper, with the alpha ramp) ----
        delta = float(alpha_t - prev_alpha)
        if delta != 0.0:
            log_w = log_w + delta * cum_logp
            prev_alpha = float(alpha_t)
        log_w = log_w + (float(alpha_t) - 1.0) * token_logp
        log_w = log_w + (token_logp - token_logq)
        cum_logp = cum_logp + token_logp

        seqs[:, cur] = next_tokens
        cur += 1

        newly_done = (next_tokens == eos_id) & (~done)
        done = done | newly_done

        if cfg.stop_on_boxed:
            active = ~done
            if active.any():
                start_tok = max(prompt_len, cur - int(cfg.boxed_check_window_tokens))
                active_idx = torch.nonzero(active, as_tuple=False).flatten().tolist()
                for i_part in active_idx:
                    gen_text_tail = tokenizer.decode(
                        seqs[i_part, start_tok:cur].tolist(), skip_special_tokens=True
                    )
                    if _has_nonempty_boxed(gen_text_tail):
                        done[i_part] = True

        if done.all():
            break

        # ---- Expand the cache back to N before the forward pass ----
        if logical_to_physical is not None:
            past = expand_cache(model, past, logical_to_physical)
            logical_to_physical = None

        out = model(
            input_ids=next_tokens.view(N, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]

        # ---- Block boundary: check ESS, resample if needed ----
        is_block_end = ((t + 1) % int(cfg.block_size) == 0) or (t == Tmax - 1)
        if is_block_end:
            lw = log_w - torch.logsumexp(log_w, dim=0)
            w = torch.exp(lw)
            ess = effective_sample_size(w)
            stats["ess_history"].append(ess)

            if ess < float(cfg.ess_threshold) * N:
                if cfg.resampler == "systematic":
                    idx_rs = systematic_resample(w, generator=g_resample)
                    post_log_w = None                      # reset to uniform
                elif cfg.resampler in ("chopthin", "chopthin_reset"):
                    idx_rs, new_w = chopthin_resample(w, float(cfg.eta), N, g_resample)
                    stats["ess_after_history"].append(
                        effective_sample_size(new_w / new_w.sum())
                    )
                    post_log_w = (
                        torch.log(new_w.clamp_min(1e-38))   # carry the unequal weights
                        if cfg.resampler == "chopthin"
                        else None                          # chopthin_reset: uniform
                    )
                else:
                    raise ValueError(f"unknown resampler {cfg.resampler!r}")

                unique_ancestors, inverse = torch.unique(idx_rs, return_inverse=True)
                stats["unique_ancestors_history"].append(unique_ancestors.numel())

                seqs = seqs.index_select(0, idx_rs)
                done = done.index_select(0, idx_rs)
                cum_logp = cum_logp.index_select(0, idx_rs)
                past = select_cache_subset(model, past, unique_ancestors)
                last_logits = last_logits.index_select(0, unique_ancestors)
                logical_to_physical = inverse              # [N] -> [0, U)

                if post_log_w is None:
                    log_w = torch.zeros_like(log_w)
                else:
                    log_w = post_log_w.to(log_w.dtype)
                stats["resample_count"] += 1

    # ---- Finish the alpha ramp if decoding ended early ----
    if prev_alpha != alpha_final:
        log_w = log_w + float(alpha_final - prev_alpha) * cum_logp
        prev_alpha = alpha_final

    seqs_out = seqs[:, :cur]
    lw = log_w - torch.logsumexp(log_w, dim=0)
    w = torch.exp(lw)
    chosen_idx = int(torch.multinomial(w, 1, generator=g_resample).item())   # weight draw

    return {
        "sequences": seqs_out,
        "log_w": log_w,
        "w": w,
        "chosen_idx": chosen_idx,
        "chosen_sequence": seqs_out[chosen_idx],
        "stats": stats,
    }
