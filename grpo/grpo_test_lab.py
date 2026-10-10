import torch
import re
from math_verify import parse, verify, ExprExtractionConfig
import sys

def get_per_token_logps(logits, input_ids):
    """CPU version of get_per_token_logps for testing."""
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def reward_correct(item, answer):
    """Same as in grpo_lab.py."""
    # TODO: Implement reward_correct function Done

    answerPattern = r"^(?!.*<answer>.*?</answer>.*<answer>.*?</answer>).*<answer>(.+?)</answer>.*$"
    answerMatch = re.search(answerPattern, answer)
    if not answerMatch:
        return -1.0

    strAnswer = answerMatch.group(1)

    numericalPattern = r'\d+/\d+|\d+\.\d+|\d+'
    numericalMatches = re.findall(numericalPattern, strAnswer)

    if not numericalMatches:
        return -1.0

    expAnswer = parse(strAnswer)
    expTruth = parse(item["A"])
    return 1.0 if verify(expAnswer, expTruth) else -1.0

def reward_format(item, answer):
    """Same as in grpo_lab.py."""
    # TODO: Implement reward_format function Done

    thinkPattern = r"^(?!.*<think>.*?</think>.*<think>.*?</think>).*<think>.+?</think>.*$"
    answerPattern = r"^(?!.*<answer>.*?</answer>.*<answer>.*?</answer>).*<answer>(.+?)</answer>.*$"

    thinkMatch = re.search(thinkPattern, answer)
    answerMatch = re.search(answerPattern, answer)

    return 1.25 if (thinkMatch and answerMatch) else -1.0

def recompute_grpo_loss(vars_dict):
    """Recompute GRPO loss on CPU using saved variables."""
    # Global Variables
    pad_token_id = vars_dict['pad_token_id']
    clip_param = vars_dict['clip_param']
    beta = vars_dict['beta']
    compute_gen_logps = vars_dict['compute_gen_logps']
    
    # Local Variables
    prompt_length = vars_dict['prompt_length']
    inputs = vars_dict['inputs']
    advantages = vars_dict['advantages']
    logits = vars_dict['logits']
    input_ids = inputs[:, 1:]
    refs_per_token_logps = vars_dict['ref_per_token_logps']
    gen_logps = vars_dict.get('gen_logps', None) if compute_gen_logps else None
    assert gen_logps is not None
    
    # TODO: Implement recomputation of GRPO loss (NOTE: This may change slightly when not CPU).

    print(pad_token_id)
    print(input_ids)
    print(input_ids[:, prompt_length:])
    print(input_ids[:, prompt_length - 1:])

    # 1. Compute per-token log probabilities of the model
    per_token_logps = get_per_token_logps(logits, input_ids)

    # 2. Slice to keep only completion tokens (after prompt) (-1 might not be correct)
    per_token_logps = per_token_logps[:, prompt_length - 1:]
    # refs_per_token_logps = refs_per_token_logps[:, :-1]
    # gen_logps = gen_logps[:, :-1]
    
    # 4. Compute per-token KL divergence approximation for regularization
    kl_ratio = torch.exp(refs_per_token_logps - per_token_logps)
    kl_divergence = kl_ratio - (refs_per_token_logps - per_token_logps) - 1

    # print(input_ids[:, prompt_length - 1:])
    # print(gen_logps)
    # print(refs_per_token_logps)
    # print(per_token_logps)
    # print(kl_ratio)
    # print(torch.log(kl_ratio))
    # print(kl_divergence)

    print(input_ids.shape)
    print(prompt_length)
    print(input_ids[:, prompt_length:].shape)
    # print(per_token_logps.shape)
    print(refs_per_token_logps.shape)
    print(gen_logps.shape)
    # print(kl_divergence.shape)
    # sys.exit(0)

    # 5. Create mask for completion tokens (not padding) (-1 might not be correct)
    padMask = input_ids[:, prompt_length - 1:] != pad_token_id
    per_token_logps = torch.mul(per_token_logps, padMask)

    # 6. Compute importance sampling ratio
    imp_ratio = torch.exp(per_token_logps - gen_logps)

    # 7. Clip ratio for PPO-style loss
    clip_ratio = torch.clamp(imp_ratio, 1 - clip_param, 1 + clip_param)

    # 8. Compute per-token GRPO loss
    clip_min = torch.minimum(torch.mul(imp_ratio, advantages), torch.mul(clip_ratio, advantages))
    token_loss = clip_min - torch.mul(kl_divergence, beta)

    # 9. Average loss over completion tokens and batch
    avg_loss = torch.mean(token_loss)

    # 10. Return final loss 
    return -avg_loss


if __name__ == '__main__':
    print("Running GRPO lab tests...")
    print("=" * 40)
    print("Recomputing GRPO loss from saved variables...")
    # Load saved variables
    vars_dict = torch.load('grpo_step_vars.pth', weights_only=True)
    expected_loss = vars_dict['loss']
    
    # Recompute loss on CPU
    computed_loss = recompute_grpo_loss(vars_dict)
    
    print(f"Expected Loss: {expected_loss.item():.6f}")
    print(f"Computed Loss: {computed_loss.item():.6f}")
    print(f"Match: {torch.allclose(expected_loss, computed_loss, atol=1e-3)}")
    print("=" * 40)
    
    # Tests for reward functions
    test_item = {'Q': 'What is 2+2?', 'A': '4'}
    
    # Test reward_correct
    print(f"Testing reward_correct...")
    assert reward_correct(test_item, '<think>2+2=4</think><answer>4</answer>') == 1
    assert reward_correct(test_item, '<think>2+2=5</think><answer>5</answer>') == -1
    assert reward_correct(test_item, 'No number') == -1
    print("reward_correct tests passed")
    print("=" * 40)
    
    # Test reward_format
    print(f"Testing reward_format...")
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer>') == 1.25
    assert reward_format(test_item, '<think>reasoning</think> <answer>answer</answer>') == 1.25  # with space
    assert reward_format(test_item, '<think>reasoning<answer>answer</answer>') == -1  # missing </think>
    assert reward_format(test_item, '<think>reasoning</think><answer>answer</answer><think>extra</think>') == -1  # extra tags
    print("reward_format tests passed")
    print("=" * 40)
