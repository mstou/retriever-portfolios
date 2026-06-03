import torch
import pickle

def select_portfolio(scores_file, portfolio_size = 10, device='cpu'):
    """
    Returns:
        portfolio (List[int]):
            Indices of retrievers selected by greedy submodular maximization.

        portfolio_score (float):
            Average per-question score achieved by the greedy portfolio.
        opt_retriever_per_question_score (float):
            Oracle upper bound: average over questions of the best single retriever per question.

        topk_retrievers (List[Tuple[int, float]]):
            The top-k retrievers by ORIGINAL average score [(idx, avg_score), ...].
            
        topk_portfolio_score (float):
            The portfolio score when using those top-k retrievers together (max across them per question).
    """
    with open(scores_file, 'rb') as f:
        scores = pickle.load(f)
    
    scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
    
    R, Q = scores.shape
    k = portfolio_size
    
    # Keep a copy for reporting
    scores_clone = scores.clone()

    # --- Top-k by average (vectorized) ---
    avg_scores = scores.mean(dim=1)
    
    topk_vals, topk_idx = torch.topk(avg_scores, k=k, largest=True)
    topk_retrievers = [(int(i), float(v)) for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]

    # Portfolio score if we just took top-k-by-avg
    topk_portfolio_score = torch.max(scores_clone[topk_idx], dim=0).values.sum() / Q

    # --- Greedy submodular maximization ---
    portfolio = []

    # current best per question so far
    current_max = torch.zeros(Q, device=device, dtype=scores.dtype)

    for _ in range(k):
        marginal = torch.relu(scores - current_max.unsqueeze(0)).sum(dim=1)# marginal contributions summed up over questions, size: [R]

        best = int(torch.argmax(marginal).item())
        portfolio.append(best)

        # Update current_max with the chosen retriever row (elementwise max)
        current_max = torch.maximum(current_max, scores[best])

    # --- Final metrics ---
    opt_retriever_per_question_score = torch.max(scores_clone, dim=0).values.sum() / Q
    portfolio_score = current_max.sum() / Q

    return portfolio, float(portfolio_score), float(opt_retriever_per_question_score), topk_retrievers, float(topk_portfolio_score)
