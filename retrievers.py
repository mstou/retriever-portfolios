from abc import ABC, abstractmethod
from typing import List
from data_classes import Dataset, TextUnit
from text_processing import Embedder
from vector_db import FaissVectorDB
import torch


class Retriever(ABC):
    def __init__(self, dataset: Dataset, embedder: Embedder, vector_db: FaissVectorDB):
        """
        Abstract Retriever class.

        - dataset: A Dataset object -- we assume that the embeddings of the documents have already been calculated.
        - embedder: An Embedder object
        """
        pass
        

    @abstractmethod
    def query(self, text: str, num_results: int = 5) -> List[TextUnit]:
        """
        Given a query text, return the most relevant text units.
        :param text: The query string.
        :param num_results: The number of relevant text units to return.
        :return: A list of the most relevant text units.
        """
        pass


class NaiveRAG(Retriever):
    """Implements Naine RAG, i.e. fetching the top-k nearest neighbors according to some distance"""

    def __init__(self, embedder: Embedder, vector_db: FaissVectorDB, metric="dot"):
        self.embedder = embedder
        self.vector_db = vector_db #FaissVectorDB(metric=metric,dim=self.embedder.get_embedding_dim())
    
    def query(self, text, num_results):
        results = self.vector_db.search(self.embedder.embed(text, role="query"), k = num_results)

        return [x[0] for x in results] # db_search returns TextUnit, distance
    
    def __repr__(self):
        return 'Naive RAG'

class DiscountedSimilarity(Retriever):
    """Discounts the relevance (dot similarity) of text units by a function of
    their similarities to chosen text units"""

    def __init__(self, embedder: Embedder, vector_db: FaissVectorDB, gamma: float, r: float, metric="dot", device="cpu"):
        self.embedder = embedder
        self.vector_db = vector_db
        self.gamma = gamma
        self.r = r
    
    def query(self, text, num_results = 5, pre_filter = 1000):
        """
        Retrieves the text unit with the highest score and discounts the scores 
        of every other text unit based on their distances to the selected text unit
        """
        results = self.vector_db.search(self.embedder.embed(text, role="query"), k = pre_filter) # fetches the 1000 top neighbors and then runs the algorithm on them
        text_units = [x[0] for x in results] # db_search returns TextUnit, distance
        embeddings = torch.stack([torch.tensor(tu.embedding) for tu in text_units])
        scores = torch.tensor([x[1] for x in results]) # initialize scores to similarity with the query vector
        
        selected_units = []
        for _ in range(num_results):
            
            best_idx = torch.argmax(scores).item()
            selected_units.append(text_units[best_idx])
            best_vector = embeddings[best_idx]
            
            # Apply multiplicative discounting
            similarities = torch.matmul(embeddings, best_vector)
            similarities *= (similarities >= self.r) # set similarity to 0 if it's below r
            discounts = torch.exp(-self.gamma * similarities) # discount score by e^{-\gamma * similarity}
            scores *= discounts
            scores[best_idx] = -float("inf")  # Prevent re-selecting the same unit
        
        return selected_units
    
    def __repr__(self):
        return f'DiscountedSimilarity(gamma={self.gamma}, r={self.r})'

class BatchDiscountedSimilarity():
    """
        Efficiently runs retrievals for many DiscountedSimilarity retrievers in a batch.
        Assumes all retrievers share the same embedder and vector DB (so we can prefilter once).
    """
    def __init__(self, retrievers: List[DiscountedSimilarity], device: str = None):
        self.retrievers = retrievers
        self.embedder = retrievers[0].embedder
        self.vector_db = retrievers[0].vector_db

        # Collect params
        self.gammas = torch.tensor([r.gamma for r in retrievers], dtype=torch.float32)
        self.rs = torch.tensor([r.r for r in retrievers], dtype=torch.float32)

        # Device handling
        self.device = device
        self.gammas = self.gammas.to(self.device)
        self.rs = self.rs.to(self.device)
    
    def num_retrievers(self):
        return len(self.retrievers)

    def query(self, text: str, num_results: int = 5, pre_filter: int = 1000, candidates=None, q_vec = None):
        """
            Returns a 2D list (len == number of retrievers) where each sub-list contains the
            selected TextUnit objects for that retriever.

            The prefiltered candidates are fetched ONCE, and discounting is run in batch.
        """
        # Single prefilter search (shared) -- if we are not given q_vec and candidate text units
        if q_vec is not None: qvec = q_vec
        else: qvec = self.embedder.embed(text, role="query")

        # ensuring qvec is a tensor on device
        qvec = torch.tensor(qvec, device=self.device, dtype=torch.float32)

        if candidates is not None: text_units = candidates
        else:
            results = self.vector_db.search(qvec, k=pre_filter)  # [(TextUnit, score), ...]
            text_units = [tu for (tu, _score) in results]

        # Ensure embeddings are a (N, D) tensor -- D is the embedding dimension
        emb_list = [torch.tensor(tu.embedding, dtype=torch.float32) for tu in text_units]
        embeddings = torch.stack(emb_list, dim=0).to(self.device) # (N, D)
        base_scores = (embeddings @ qvec) # (N,)
        # base_scores = torch.tensor([score for (_tu, score) in results], dtype=torch.float32, device=self.device)  # (N,)

        B = len(self.retrievers)
        N = embeddings.shape[0]

        # Initialize batched scores: each retriever starts from the same initial scores
        scores = base_scores.unsqueeze(0).expand(B, N).contiguous()  # (B, N)

        # Greedy selection with batched discounting
        selected_indices = torch.empty((B, num_results), dtype=torch.long, device=self.device)

        # For efficient in-place masking when setting -inf
        batch_arange = torch.arange(B, device=self.device)

        for k in range(num_results):
            # Pick current best idx per retriever
            best_idx = torch.argmax(scores, dim=1)  # (B,)
            selected_indices[:, k] = best_idx

            # Corresponding best vectors per retriever (B, D)
            best_vecs = embeddings[best_idx]

            # similarities per retriever to ALL candidates:
            # (N, D) @ (D, B) -> (N, B) -> transpose -> (B, N)
            sims = (embeddings @ best_vecs.T).T  # (B, N)

            # Mask similarities below each retriever's r
            # rs: (B,), so rs[:, None]: (B, 1)
            mask = (sims >= self.rs[:, None]).to(sims.dtype)
            sims_masked = sims * mask  # (B, N)

            # Compute multiplicative discounts per retriever: exp(-gamma * similarity)
            # gammas: (B,) -> (B,1) broadcasts over N
            discounts = torch.exp(-self.gammas[:, None] * sims_masked)  # (B, N)

            # Apply discounts to current scores
            scores *= discounts

            # Prevent re-selecting the same item for each retriever
            scores[batch_arange, best_idx] = -float("inf")

        # Map indices back to TextUnit objects for each retriever
        selected_text_units_per_retriever = []
        for b in range(B):
            indices = selected_indices[b].tolist()
            selected_text_units_per_retriever.append([text_units[i] for i in indices])

        return selected_text_units_per_retriever

    def __repr__(self):
        params = ", ".join([f"(gamma={r.gamma}, r={r.r})" for r in self.retrievers])
        return f"BatchDiscountedSimilarity[{params}]"


class VendiRetriever(Retriever):
    """
        Fetches documents according to a convex combination of the Vendi Score and similarity to the question.
        More info: https://arxiv.org/abs/2502.11228v2
    """
    def __init__(self, embedder: Embedder, vector_db: FaissVectorDB, s: float, device="cpu"):
        self.embedder = embedder
        self.vector_db = vector_db
        self.s = s
        self.device = device

    def set_s(self, s: float):
        """
        Update the trade-off parameter s in-place so the same
        retriever instance can be reused with different settings.
        """
        self.s = s
        return self
    
    @staticmethod
    def row_norm(x, eps=1e-12):
        # returns a row-normalized tensor
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    @staticmethod
    def vendi_score_from_kernel(K):
        n = K.shape[0]
        w = torch.linalg.eigvalsh(K / n).clamp_min(0)
        H = -(w * w.clamp_min(1e-12).log()).sum()
        return float(torch.exp(H).item())

    def query(self, text, num_results = 5, pre_filter = 1000):
        device = self.device

        q_vec = self.embedder.embed(text, role="query")
        results = self.vector_db.search(q_vec, k = pre_filter) # fetches the 1000 top neighbors and then runs the algorithm on them
        text_units = [x[0] for x in results] # db_search returns TextUnit, distance
        X = torch.stack([torch.as_tensor(tu.embedding, dtype=torch.float32, device=device) for tu in text_units])

        q_vec = torch.as_tensor(q_vec, device=device, dtype=torch.float32)
        
        qn = self.row_norm(q_vec.unsqueeze(0)).squeeze(0) # normalized question vector
        Xn = self.row_norm(X) # normalized embeddings for text units
        q_sims = Xn @ qn # similarities with query vector (size PRE_FILTER x 1)
        K = Xn @ Xn.T    # pairwise similarities of text units

        picked = torch.zeros(Xn.shape[0], dtype=torch.bool, device=device) # boolean map of picked documents
        selected_units = []

        for _ in range(num_results): # picked the document with the biggest marginal improvement of the score
            best_j, best_score = -1, float("-inf")
            base_idx = torch.where(picked)[0].tolist() # already picked indices
            for j in range(Xn.shape[0]):   # trying to add doc j
                if picked[j]: continue
                T = base_idx + [j]         # new set of docs
                Kj = K[T][:, T]            # similarities between new set
                VS = self.vendi_score_from_kernel(Kj)  # vendi score of new set
                SS = float(q_sims[T].sum().item())    # similarities of new set
                VRS = self.s * VS + (1.0 - self.s) * SS
                if VRS > best_score:
                    best_score, best_j = VRS, j
            picked[best_j] = True
            selected_units.append(text_units[best_j])

        return selected_units

    def __repr__(self):
        return f'VendiRetriever(s={self.s})'


class BatchVendiRetriever():
    """
        Efficiently runs retrievals for many VendiRetrievers in a batch.
        Assumes all retrievers share the same embedder and vector DB (so we can prefilter once).
    """
    def __init__(self, retrievers: List[VendiRetriever], device: str = None):
        self.retrievers = retrievers
        self.embedder = retrievers[0].embedder
        self.vector_db = retrievers[0].vector_db
        self.device = device
        
        self.s = torch.tensor([r.s for r in retrievers], dtype=torch.float32)
        self.s = self.s.to(self.device)
        
    def num_retrievers(self):
        return len(self.retrievers)
    
    @staticmethod
    def row_norm(x, eps=1e-12):
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    @staticmethod
    def vendi_score_batched(K_batch: torch.Tensor) -> torch.Tensor:
        """
            K_batch: (B, m, m) stacked cosine kernels for trial sets.
            Returns VS for each (B,).
        """
        m = K_batch.shape[-1]
        w = torch.linalg.eigvalsh(K_batch / m)      # (B, m)
        H = -(w * (w + 1e-12).log()).sum(dim=-1)    # (B,)
        return torch.exp(H)                         # (B,)
    
    def query(self, text, num_results = 5, pre_filter = 1000, candidates=None, q_vec = None):
        """
            Optimized single-question retrieval for many retrievers and one query:
            - t=0: pick top-sim doc once for all retrievers
            - t=1: compute VS/SS once for base {j*}, broadcast over s to pick per retriever
            - t>=2: fall back to per-retriever greedy with batched eig over remaining candidates
        """
        device = self.device

        if q_vec is None:
            q_vec = self.embedder.embed(text, leave_on_device=True, role="query")
        else:
            q_vec = torch.tensor(q_vec, dtype=torch.float32, device=device)
        
        if candidates is None:
            results = self.vector_db.search(q_vec.detach().cpu().numpy(), k=pre_filter)
            text_units = [tu for (tu, _score) in results]
        else:
            text_units = candidates
        
        # Shared matrices
        X = torch.stack([torch.as_tensor(tu.embedding, dtype=torch.float32, device=device) for tu in text_units]) # (M, d)
        qn = self.row_norm(q_vec.unsqueeze(0)).squeeze(0)                            # (d,)
        Xn = self.row_norm(X)                                                       # (M, d)
        q_sims = Xn @ qn                                                            # (M,)
        K = Xn @ Xn.T                                                               # (M, M)

        R = self.num_retrievers()
        M = Xn.shape[0]
        k = min(num_results, M)

        picked = torch.zeros(R, M, dtype=torch.bool, device=device)
        sum_ss = torch.zeros(R, dtype=torch.float32, device=device)
        outputs = [[] for _ in range(R)]

        # All retrievers pick the same first document
        j_star = int(torch.argmax(q_sims).item())
        picked[:, j_star] = True
        sum_ss += q_sims[j_star]
        for r in range(R):
            outputs[r].append(text_units[j_star])
        
        if k == 1: return outputs

        # All the VS and SS are the same at this point, we calculate them once
        # then we compute the scores based on each different s

        rem_idx = [j for j in range(M) if j != j_star]
        if rem_idx:
            Kj_list = []
            for j in rem_idx:
                idx = [j_star, int(j)]  
                Kj_list.append(K[idx][:, idx])
            K_batch = torch.stack(Kj_list, dim=0)

            VS_all = self.vendi_score_batched(K_batch)
            SS_all_vec = q_sims[j_star] + q_sims[rem_idx]

            # Broadcast over all retrievers' s to pick per-r best j
            S = self.s.view(-1, 1) # (R,1)
            VRS = S * VS_all.view(1, -1) + (1.0 - S) * SS_all_vec.view(1, -1)
            best_locals = torch.argmax(VRS, dim=1).tolist()

            for r in range(R):
                j_best = rem_idx[best_locals[r]]
                picked[r, j_best] = True
                sum_ss[r] += q_sims[j_best]
                outputs[r].append(text_units[j_best])

        if k == 2: return outputs

        # From this point onwards we run Greedy normally
        for _ in range(2, k):
            for r in range(R):
                mask_rem = ~picked[r]
                rem_idx = torch.where(mask_rem)[0].tolist()
                if not rem_idx:
                    continue
                base_idx = torch.where(picked[r])[0].tolist()

                # Stack submatrices in the SAME order as rem_idx
                Kj_list = []
                SS_all_list = []
                for j in rem_idx:
                    idx = base_idx + [int(j)]
                    Kj_list.append(K[idx][:, idx])
                    SS_all_list.append(sum_ss[r] + q_sims[j])

                K_batch = torch.stack(Kj_list, dim=0)                    # (Rj, t+1, t+1)
                VS_all = self.vendi_score_batched(K_batch)               # (Rj,)
                SS_all = torch.tensor(SS_all_list, device=device)        # (Rj,)

                VRS_all = self.s[r] * VS_all + (1.0 - self.s[r]) * SS_all

                best_local = int(torch.argmax(VRS_all).item())
                best_j = rem_idx[best_local]

                picked[r, best_j] = True
                sum_ss[r] += q_sims[best_j]
                outputs[r].append(text_units[best_j])

        return outputs
    
    @torch.no_grad()
    def batch_query(self, texts, num_results = 5, pre_filter = 1000, candidates_per_query = None, batch_size = 10_000):
        """
            Batch retrieval for all texts at the same time.
            Optionally give the pre-filtered candidates via an argument.
        """
        device = self.device
        R = len(self.retrievers)
        Q = len(texts)

        qmat = self.embedder.embed(texts, leave_on_device=True, role="query")
        qvecs = [qmat[i].to(device) for i in range(qmat.shape[0])]

        if candidates_per_query is None:
            candidates_per_query = []
            for qv in qvecs:
                res = self.vector_db.search(qv.detach().cpu().numpy(), k=pre_filter)
                candidates_per_query.append([tu for (tu, _s) in res])
        
        Xn_q, K_q, q_sims_q, tus_q = [], [], [], []
        for q in range(Q):
            tus = candidates_per_query[q]    # prefiltered text units for this query
            tus_q.append(tus)
            X = torch.stack([torch.as_tensor(tu.embedding, dtype=torch.float32, device=device) for tu in tus]) # embeddings of text_units
            qn = self.row_norm(qvecs[q].unsqueeze(0)).squeeze(0)          # normalized question vector   (d,)
            Xn = self.row_norm(X)                                         # normalized embedding vectors (M_q, d)
            K = Xn @ Xn.T                                                 # pairwise similarities in the pre-filtered units (M_q, M_q)
            q_sims = Xn @ qn                                              # similarities with the question vector           (M_q,)
            Xn_q.append(Xn); K_q.append(K); q_sims_q.append(q_sims)       # gather all matrices
        
        # state per (q,r)
        outputs = [[[] for _ in range(R)] for _ in range(Q)]
        picked = [[set() for _ in range(R)] for _ in range(Q)]
        sum_ss = torch.zeros((Q, R), dtype=torch.float32, device=device)

        # greedy steps
        for t in range(num_results):
            # Build one global trial list across all (q,r)
            K_blocks, SS_num, owner_q, owner_r, owner_j = [], [], [], [], []

            for q in range(Q):
                Mq = K_q[q].shape[0]

                for r in range(R):
                    base = sorted(picked[q][r])  # size t for active retrievers
                    # remaining indices for this (q,r)
                    rem_mask = torch.ones(Mq, dtype=torch.bool, device=device)
                    if base:
                        base_idx_t = torch.tensor(base, device=device, dtype=torch.long)
                        rem_mask[base_idx_t] = False
                    rem_idx = torch.where(rem_mask)[0].tolist()
                    
                    # append all trials
                    for j in rem_idx:
                        idx = base + [int(j)]
                        K_blocks.append(K_q[q][idx][:, idx])           # (t+1, t+1)
                        SS_num.append(sum_ss[q, r] + q_sims_q[q][j])
                        owner_q.append(q); owner_r.append(r); owner_j.append(int(j))
            
            # We process the calculations in batches
            owner_q_t = torch.tensor(owner_q, device=device)
            owner_r_t = torch.tensor(owner_r, device=device)
            owner_j_t = torch.tensor(owner_j, device=device)

            VRS_all = torch.empty(len(K_blocks), dtype=torch.float32, device=device)
            start = 0
            while start < len(K_blocks):
                end = min(start + batch_size, len(K_blocks))
                K_batch = torch.stack(K_blocks[start:end], 0)           # (Bch, t+1, t+1)
                VS = self.vendi_score_batched(K_batch)                  # (Bch,)
                SS = torch.stack(SS_num[start:end], 0)                  # (Bch,)
                s_owner = self.s[owner_r_t[start:end]]                  # (Bch,)
                VRS_all[start:end] = s_owner * VS + (1.0 - s_owner) * SS
                start = end
            
            # For each (q,r), pick best j among its trials
            for q in range(Q):
                # mask for this query
                mq = (owner_q_t == q)

                # within this query, do per-retriever selection
                for r in range(R):
                    mqr = mq & (owner_r_t == r)
                    idxs = torch.where(mqr)[0]
                    best_local = idxs[torch.argmax(VRS_all[mqr])]
                    j_best = int(owner_j_t[best_local].item())

                    picked[q][r].add(j_best)
                    sum_ss[q, r] += q_sims_q[q][j_best]
                    outputs[q][r].append(tus_q[q][j_best])
        
        return outputs

        
