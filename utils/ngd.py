import torch


def apply_natural_gradient(policy, batch_log_probs, damping):
    standard_grad = torch.cat([p.grad.clone().view(-1) for p in policy.parameters()])
    grads = []

    for log_prob in batch_log_probs:
        policy.zero_grad()
        log_prob.backward(retain_graph=True)
        g = torch.cat([p.grad.view(-1) for p in policy.parameters() if p.grad is not None])
        grads.append(g)

    grads = torch.stack(grads)
    FIM = (grads.T @ grads) / len(batch_log_probs)
    FIM += damping * torch.eye(FIM.size(0), device=FIM.device)

    eigenvalues = torch.linalg.eigvalsh(FIM)
    cond_num = (eigenvalues[-1] / (eigenvalues[0] + 1e-8)).item()

    FIM_inv = torch.linalg.inv(FIM)
    nat_grad = FIM_inv @ standard_grad

    idx = 0
    policy.zero_grad()

    for p in policy.parameters():
        length = p.numel()
        if p.grad is None:
            p.grad = nat_grad[idx:idx + length].view(p.shape).clone()
        else:
            p.grad.copy_(nat_grad[idx:idx + length].view(p.shape))
        idx += length

    return cond_num
