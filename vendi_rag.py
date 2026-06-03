import os
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import constants as C
from models import OpenAI_LLM
from retrievers import VendiRetriever
from text_processing import Embedder
from utils import extract_tagged_answer, exact_match_score, f1_support
from vector_db import FaissVectorDB
from tqdm import tqdm
from transformers import AutoTokenizer


def judge_vendi_rag_prompt(query: str, answer: str, passages: List[str]) -> tuple[str, str]:
    """
    Builds the judge_vendi_rag prompt:
      - System: instructions for an LLM judge scoring (C, R, Q).
      - User: concrete query, answer, and retrieved passages.

    The model is expected to output the numeric quality score Q_t
    inside <quality>...</quality></final>, with NOTHING after </final>.
    """
    system_prompt = (
        "IMPORTANT OUTPUT CONTRACT (READ CAREFULLY):\n"
        "- Your reply MUST end with a SINGLE line of the form:\n"
        "    <quality>Q_t</quality></final>\n"
        "- Q_t must be a single floating point number (e.g., 3.67).\n"
        "- You may optionally include explanation text BEFORE this final line.\n"
        "- You MUST NOT output anything after </final>.\n"
        "- If you do NOT include the final <quality>...</quality></final> line, "
        "your answer is INVALID.\n\n"
        "ROLE AND SCORING TASK:\n"
        "You are an expert LLM-based judge tasked with evaluating the quality of answers "
        "in a Retrieval-Augmented Generation (RAG) system. Your evaluation will consider "
        "the following aspects:\n\n"
        "1. Coherence: Assess whether the provided answer is logically consistent and flows "
        "smoothly, without conflicting statements or gaps in reasoning.\n"
        "2. Relevance: Evaluate how well the answer addresses the query based on the information "
        "from the retrieved documents.\n"
        "3. Query Alignment: Determine how closely the answer aligns with the specific query asked, "
        "ensuring that the response is focused and appropriate.\n\n"
        "Your evaluation will be quantified based on the following scoring system:\n"
        "- Coherence Score (C): [1 - 10], where 10 is perfectly coherent.\n"
        "- Relevance Score (R): [1 - 10], where 10 is highly relevant to the query.\n"
        "- Query Alignment Score (Q): [1 - 10], where 10 is perfectly aligned.\n\n"
        "Provide a quality score Q_t as the average of these individual scores:\n\n"
        "Q_t = mean(C, R, Q)\n\n"
        "Remember: whatever reasoning you produce, you MUST finish with a line "
        "exactly of the form <quality>Q_t</quality></final>."
    )

    user_prompt = f"Query:\n{query}\n\n"
    user_prompt += "Retrieved Passages:\n"
    for i, passage in enumerate(passages):
        user_prompt += f"Passage {i}:\n{passage}\n\n"
    user_prompt += "Answer:\n"
    user_prompt += f"{answer}\n\n"
    user_prompt += (
        "======\n"
        "Your task:\n"
        "- Carefully read the query, the retrieved passages, and the answer.\n"
        "- Assign Coherence (C), Relevance (R), and Query Alignment (Q) scores in [1, 10].\n"
        "- Compute Q_t = mean(C, R, Q).\n"
        "- You may optionally provide a brief explanation.\n"
        "- IMPORTANT: At the very end of your reply, you MUST output a single line of the form:\n"
        "  <quality>Q_t</quality></final>\n"
        "- Q_t must be a single floating point number (e.g., 7.5).\n"
        "- Do NOT output anything after </final>.\n"
        "- Even if you run out of space, always include the final <quality>...</quality></final> line.\n"
    )

    return system_prompt, user_prompt


class VendiRAGAdaptive:
    """
    Adaptive Vendi-RAG loop:
        - Start from an initial s value.
        - Repeat for a fixed number of steps:
            * Retrieve with VendiRetriever(s).
            * Answer with an LLM given the retrieved passages.
            * Judge the answer with another LLM to get a score.
            * Use the score to update s.

    All logs are plain dictionaries.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_db: FaissVectorDB,
        answer_llm: OpenAI_LLM,
        judge_llm: OpenAI_LLM,
        answer_prompt_fn: Callable[[str, List[str]], tuple[str, str]],
        num_steps: int = 3,
        k_docs: int = 4,
        prefilter_k: int = 1000,
        device: str = "cuda",
        questions_dataset: Optional[Any] = None,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Args:
            embedder: Shared embedder for the retriever.
            vector_db: Vector DB containing document embeddings.
            answer_llm: LLM used to produce answers.
            judge_llm: LLM used to judge answers.
            answer_prompt_fn: (question, passages) -> (system_prompt, user_prompt).
            num_steps: Number of adaptive iterations per question.
            k_docs: Number of documents to retrieve per step.
            prefilter_k: Number of candidates to prefilter in VendiRetriever.
            device: Device string passed to VendiRetriever.
            questions_dataset: Optional dataset with a .questions list for run_dataset.
            log_callback: Optional hook to stream logs (e.g. to disk) after each step.
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.answer_llm = answer_llm
        self.judge_llm = judge_llm
        self.answer_prompt_fn = answer_prompt_fn
        self.judge_prompt_fn = judge_vendi_rag_prompt
        self.update_s_fn = _default_update_s
        self.num_steps = num_steps
        self.k_docs = k_docs
        self.prefilter_k = prefilter_k
        self.device = device
        self.questions_dataset = questions_dataset
        self.log_callback = log_callback

    def run_one(
        self,
        question: str,
        initial_s: float,
        question_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the adaptive Vendi-RAG loop on a single question.

        Returns a dict with:
            - question_id
            - question
            - final_s
            - final_answer
            - steps: list of per-step dict logs.
        """
        # Single retriever instance, we only update its s parameter across steps.
        # We force the internal Vendi computations to run on CPU for stability.
        retriever = VendiRetriever(self.embedder, self.vector_db, initial_s, device="cpu")
        s = initial_s
        history: List[Dict[str, Any]] = []

        def _snapshot_tokens(llm: OpenAI_LLM) -> Dict[str, float]:
            return {
                "prompt_tokens": float(getattr(llm, "prompt_tokens", 0)),
                "completion_tokens": float(getattr(llm, "completion_tokens", 0)),
                "total_tokens": float(getattr(llm, "total_tokens", 0)),
                "num_calls": float(getattr(llm, "num_calls", 0)),
            }

        answer_tokens_before = _snapshot_tokens(self.answer_llm)
        judge_tokens_before = _snapshot_tokens(self.judge_llm)

        for step in range(self.num_steps):
            step_log: Dict[str, Any] = {
                "step": step,
                "s": s,
            }

            # Retrieval
            t0 = time.perf_counter()
            retriever.set_s(s)
            text_units = retriever.query(
                question,
                num_results=self.k_docs,
                pre_filter=self.prefilter_k,
            )
            step_log["retrieval_time"] = time.perf_counter() - t0
            step_log["retrieved_docs"] = [
                {
                    "doc_id": getattr(tu, "doc_id", None),
                    "text": getattr(tu, "text", None),
                }
                for tu in text_units
            ]

            # Answer with LLM
            passages = [d["text"] for d in step_log["retrieved_docs"] if d["text"] is not None]
            t0 = time.perf_counter()
            sys_prompt, user_prompt = self.answer_prompt_fn(question, passages)
            answer = self.answer_llm.answer(system_prompt=sys_prompt, user_prompt=user_prompt)
            step_log["answer_time"] = time.perf_counter() - t0
            step_log["answer_system_prompt"] = sys_prompt
            step_log["answer_user_prompt"] = user_prompt
            step_log["answer"] = answer

            # Judge with LLM
            t0 = time.perf_counter()
            j_sys, j_user = self.judge_prompt_fn(question, answer, passages)
            judge_raw = self.judge_llm.answer(
                system_prompt=j_sys,
                user_prompt=j_user,
                max_tokens=512,
                temperature=0.5,
            )
            step_log["judge_time"] = time.perf_counter() - t0
            step_log["judge_system_prompt"] = j_sys
            step_log["judge_user_prompt"] = j_user
            step_log["judge_raw"] = judge_raw
            step_log["judge_score"] = _parse_judge_score(judge_raw)

            history.append(step_log)

            if self.log_callback is not None:
                self.log_callback(
                    {
                        "question_id": question_id,
                        "question": question,
                        "current_s": s,
                        "step_log": step_log,
                    }
                )

            # Update s for next step
            s = self.update_s_fn(s, step_log["judge_score"], step, history)

        # Simple final answer policy: pick the step with the best judge_score
        best_step = max(history, key=lambda h: h.get("judge_score", float("-inf")))
        answer_tokens_after = _snapshot_tokens(self.answer_llm)
        judge_tokens_after = _snapshot_tokens(self.judge_llm)

        def _delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
            return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before.keys()}

        result = {
            "question_id": question_id,
            "question": question,
            "final_s": history[-1]["s"],
            "final_answer": best_step.get("answer"),
            "steps": history,
            "token_usage": {
                "answer": _delta(answer_tokens_after, answer_tokens_before),
                "judge": _delta(judge_tokens_after, judge_tokens_before),
            },
        }
        return result

    def run_dataset(
        self,
        initial_s: float,
        max_questions: Optional[int] = None,
        start_idx: int = 0,
        max_workers: int = 16,
        output_file: Optional[str] = None,
        checkpoint_every: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Run the adaptive loop over all questions in the attached questions_dataset.

        This implementation:
          - Keeps retrieval single-threaded (for safety with Faiss / Torch).
          - Runs answer + judge LLM calls concurrently via a ThreadPoolExecutor.
          - Pipelines work so that as soon as a question's judge result is ready,
            we immediately run the next retrieval for that question (if any).

        Assumes questions_dataset.questions is a list of dicts with at least a
        "question" field (and optionally an "id" field).
        """
        if self.questions_dataset is None:
            raise ValueError("questions_dataset is not set")

        questions = getattr(self.questions_dataset, "questions", None)
        if questions is None:
            raise ValueError("questions_dataset does not have a 'questions' attribute")

        end_idx = len(questions)
        if max_questions is not None:
            end_idx = min(end_idx, start_idx + max_questions)

        num_total = max(0, end_idx - start_idx)
        if num_total == 0:
            return []

        # Prepare results buffer (for checkpointing)
        results_buffer: List[Optional[Dict[str, Any]]] = [None] * num_total

        # If an output_file is provided and exists, try to resume from it
        if output_file is not None and os.path.exists(output_file):
            try:
                with open(output_file, "rb") as f:
                    old = pickle.load(f)
                old_results = old.get("results")
                if isinstance(old_results, list) and len(old_results) == num_total:
                    results_buffer = old_results  # type: ignore[assignment]
                    print(
                        f"[VendiRAGAdaptive] Resuming from checkpoint: "
                        f"{sum(r is not None for r in results_buffer)}/{num_total} questions done.",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[VendiRAGAdaptive] Warning: could not load checkpoint from {output_file}: {exc}",
                    flush=True,
                )

        # Per-question state (only for questions not already completed)
        q_states: List[Dict[str, Any]] = []
        for rel_idx, i in enumerate(range(start_idx, end_idx)):
            if results_buffer[rel_idx] is not None:
                continue  # already done
            q = questions[i]
            q_text = q["question"]
            q_id = str(q.get("id", i))
            q_states.append(
                {
                    "rel_idx": rel_idx,
                    "id": q_id,
                    "question": q_text,
                    "s": float(initial_s),
                    "current_step": 0,
                    "steps": [],  # list of step_log dicts
                }
            )

        if not q_states:
            # Everything was already done in the checkpoint
            return [r for r in results_buffer if r is not None]

        # Vendi computations on CPU; embedder may still run on GPU.
        retriever = VendiRetriever(self.embedder, self.vector_db, initial_s, device="cpu")

        # Map futures -> (stage, q_state, step_log, t_start)
        future_to_meta: Dict[Any, Any] = {}
        total_steps = len(q_states) * self.num_steps
        num_completed_questions = sum(r is not None for r in results_buffer)

        def _schedule_retrieval_and_answer(q_state: Dict[str, Any]):
            """
            Runs retrieval synchronously for this question and schedules
            the answer-LLM call in the executor.
            """
            step_idx = q_state["current_step"]
            s_val = q_state["s"]

            step_log: Dict[str, Any] = {
                "step": step_idx,
                "s": s_val,
            }

            # Retrieval (single-threaded)
            t0 = time.perf_counter()
            retriever.set_s(s_val)
            text_units = retriever.query(
                q_state["question"],
                num_results=self.k_docs,
                pre_filter=self.prefilter_k,
            )
            step_log["retrieval_time"] = time.perf_counter() - t0
            step_log["retrieved_docs"] = [
                {
                    "doc_id": getattr(tu, "doc_id", None),
                    "text": getattr(tu, "text", None),
                }
                for tu in text_units
            ]

            passages = [d["text"] for d in step_log["retrieved_docs"] if d["text"] is not None]
            sys_prompt, user_prompt = self.answer_prompt_fn(q_state["question"], passages)
            step_log["answer_system_prompt"] = sys_prompt
            step_log["answer_user_prompt"] = user_prompt
            step_log["answer_time"] = None
            step_log["judge_time"] = None

            q_state["steps"].append(step_log)

            t_ans = time.perf_counter()
            fut = executor.submit(self.answer_llm.answer, sys_prompt, user_prompt)
            future_to_meta[fut] = ("answer", q_state, step_log, t_ans)

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=total_steps,
            desc="VendiRAGAdaptive",
            unit="step",
        ) as pbar:
            # Seed: first retrieval + answer for every question
            for q_state in q_states:
                _schedule_retrieval_and_answer(q_state)

            # Process LLM futures, scheduling subsequent stages as they complete
            while future_to_meta:
                # Snapshot keys so we don't mutate during iteration
                for fut in as_completed(list(future_to_meta.keys())):
                    stage, q_state, step_log, t_start = future_to_meta.pop(fut)
                    if stage == "answer":
                        try:
                            answer = fut.result()
                        except Exception as exc:
                            answer = f"[ERROR in answer LLM: {exc}]"
                        step_log["answer"] = answer
                        step_log["answer_time"] = time.perf_counter() - t_start

                        passages = [
                            d["text"]
                            for d in step_log["retrieved_docs"]
                            if d.get("text") is not None
                        ]
                        j_sys, j_user = self.judge_prompt_fn(
                            q_state["question"],
                            answer,
                            passages,
                        )
                        step_log["judge_system_prompt"] = j_sys
                        step_log["judge_user_prompt"] = j_user

                        t_judge = time.perf_counter()
                        # Increase max_tokens for judge outputs to allow more detailed scoring
                        j_fut = executor.submit(self.judge_llm.answer, j_sys, j_user, 512, 0.5)
                        future_to_meta[j_fut] = ("judge", q_state, step_log, t_judge)

                    else:  # stage == "judge"
                        try:
                            judge_raw = fut.result()
                        except Exception as exc:
                            judge_raw = f"[ERROR in judge LLM: {exc}]"

                        step_log["judge_raw"] = judge_raw
                        step_log["judge_time"] = time.perf_counter() - t_start
                        score = _parse_judge_score(judge_raw)
                        step_log["judge_score"] = score

                        # Optional external logging callback
                        if self.log_callback is not None:
                            self.log_callback(
                                {
                                    "question_id": q_state["id"],
                                    "question": q_state["question"],
                                    "current_s": q_state["s"],
                                    "step_log": step_log,
                                }
                            )

                        # Update s for the next step (even if we might stop afterwards)
                        q_state["s"] = self.update_s_fn(
                            q_state["s"],
                            score,
                            step_log["step"],
                            q_state["steps"],
                        )
                        q_state["current_step"] += 1

                        # We completed one adaptive step for this question
                        pbar.update(1)

                        # Schedule next step if we still have steps left
                        if q_state["current_step"] < self.num_steps:
                            _schedule_retrieval_and_answer(q_state)
                        else:
                            # Question finished: finalize its result and place into buffer
                            steps = q_state["steps"]
                            if steps:
                                best_step = max(
                                    steps, key=lambda h: h.get("judge_score", float("-inf"))
                                )
                                results_buffer[q_state["rel_idx"]] = {
                                    "question_id": q_state["id"],
                                    "question": q_state["question"],
                                    "final_s": steps[-1]["s"],
                                    "final_answer": best_step.get("answer"),
                                    "steps": steps,
                                }
                                num_completed_questions += 1

                                # Periodic checkpoint
                                if (
                                    output_file is not None
                                    and num_completed_questions % checkpoint_every == 0
                                ):
                                    try:
                                        payload = {
                                            "results": results_buffer,
                                            "start_idx": start_idx,
                                            "num_questions": num_total,
                                        }
                                        with open(output_file, "wb") as f:
                                            pickle.dump(payload, f)
                                        print(
                                            f"[VendiRAGAdaptive] Checkpoint saved to {output_file} "
                                            f"({num_completed_questions}/{num_total} questions).",
                                            flush=True,
                                        )
                                    except Exception as exc:
                                        print(
                                            f"[VendiRAGAdaptive] Warning: failed to save checkpoint "
                                            f"to {output_file}: {exc}",
                                            flush=True,
                                        )

        # Final save if requested
        if output_file is not None:
            try:
                payload = {
                    "results": results_buffer,
                    "start_idx": start_idx,
                    "num_questions": num_total,
                }
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "wb") as f:
                    pickle.dump(payload, f)
                print(
                    f"[VendiRAGAdaptive] Final results saved to {output_file} "
                    f"({num_completed_questions}/{num_total} questions).",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[VendiRAGAdaptive] Warning: failed to save final results to {output_file}: {exc}",
                    flush=True,
                )

        # Build final list (skip any None slots defensively)
        return [r for r in results_buffer if r is not None]


# ---------------------------------------------------------------------------
# Parsing helpers for VendiRAGAdaptive experiment traces
# ---------------------------------------------------------------------------

def _vendirag_load_results(dataset: str, num_docs: int, model_key: str):
    """
    Loads raw VendiRAGAdaptive results from disk, handling checkpoint payloads.
    """
    path = C.get_vendirag_results(dataset, num_docs, model_key)
    print(f"[parse_vendirag] Loading VendiRAG results from: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    else:
        results = payload

    # Filter out None slots (partial runs)
    results = [r for r in results if r is not None]
    print(f"[parse_vendirag] Loaded {len(results)} per-question traces.")
    return path, results


def _vendirag_load_questions_test(dataset: str):
    path = C.get_questions_test(dataset)
    print(f"[parse_vendirag] Loading test questions from: {path}")
    with open(path, "rb") as f:
        questions_dataset = pickle.load(f)
    return questions_dataset


def _vendirag_build_tokenizer(model_key: str):
    """
    Use the local LLM directory for token counting.
    This is approximate but consistent across experiments.
    """
    model_path = C.LLM_DIR[model_key]
    print(f"[parse_vendirag] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def _vendirag_count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def parse_vendirag(
    dataset: str,
    num_docs: int,
    model_key: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    For a given dataset / num_docs / model, load the vendirag results and the
    test questions, parse each step, and save a structured 'parsed' object.

    Per question, we store:
      - question_id, question, final_s, final_answer
      - steps: list of dicts, each containing:
        - s
        - answer_time, judge_time
        - answer_raw, judge_raw
        - answer_tokens, judge_tokens
        - answer_prompt_tokens, judge_prompt_tokens
        - quality (parsed judge_score)
        - retrieved_doc_ids
        - recall of retrieved docs (if gold doc ids available)
        - parsed_answer (from <answer>..</answer>)
        - exact_match (EM) against gold answer (if available)
    """
    _, raw_results = _vendirag_load_results(dataset, num_docs, model_key)
    questions_dataset = _vendirag_load_questions_test(dataset)
    tokenizer = _vendirag_build_tokenizer(model_key)

    questions = questions_dataset.questions
    if len(raw_results) > len(questions):
        print(
            f"[parse_vendirag] Warning: more VendiRAG results ({len(raw_results)}) "
            f"than questions ({len(questions)}); truncating.",
            flush=True,
        )
        raw_results = raw_results[: len(questions)]

    parsed_results: List[Dict[str, Any]] = []

    for q_idx, r in enumerate(raw_results):
        q = questions[q_idx]
        gold_docs = q.get("target") or q.get("support_docs") or []
        gold_answer = q.get("answer")

        question_id = r.get("question_id", q_idx)
        question_text = r.get("question", q.get("question"))

        steps_in = r.get("steps", []) or []
        steps_out: List[Dict[str, Any]] = []

        for step in steps_in:
            s_val = step.get("s")
            answer_raw = step.get("answer") or ""
            judge_raw = step.get("judge_raw") or ""

            ans_sys = step.get("answer_system_prompt") or ""
            ans_usr = step.get("answer_user_prompt") or ""
            j_sys = step.get("judge_system_prompt") or ""
            j_usr = step.get("judge_user_prompt") or ""

            answer_prompt = ans_sys + "\n" + ans_usr
            judge_prompt = j_sys + "\n" + j_usr

            # Token counts
            answer_tokens = _vendirag_count_tokens(tokenizer, answer_raw)
            judge_tokens = _vendirag_count_tokens(tokenizer, judge_raw)
            answer_prompt_tokens = _vendirag_count_tokens(tokenizer, answer_prompt)
            judge_prompt_tokens = _vendirag_count_tokens(tokenizer, judge_prompt)

            # Retrieved docs + recall
            retrieved = step.get("retrieved_docs", []) or []
            doc_ids = [d.get("doc_id") for d in retrieved if d.get("doc_id") is not None]

            recall = None
            if gold_docs:
                _, _, rec = f1_support(doc_ids, gold_docs)
                recall = rec

            # Parsed answer from <answer>...</answer>
            parsed_answer = extract_tagged_answer(answer_raw, tag="answer")

            # Exact match (if we have a gold answer)
            if gold_answer is not None and parsed_answer is not None:
                em = exact_match_score(parsed_answer, gold_answer)
            else:
                em = None

            step_out = {
                "step": step.get("step"),
                "s": s_val,
                "retrieval_time": step.get("retrieval_time"),
                "answer_time": step.get("answer_time"),
                "judge_time": step.get("judge_time"),
                "answer_raw": answer_raw,
                "judge_raw": judge_raw,
                "answer_tokens": answer_tokens,
                "judge_tokens": judge_tokens,
                "answer_prompt_tokens": answer_prompt_tokens,
                "judge_prompt_tokens": judge_prompt_tokens,
                "quality": step.get("judge_score"),
                "retrieved_doc_ids": doc_ids,
                "recall": recall,
                "parsed_answer": parsed_answer,
                "exact_match": em,
            }
            steps_out.append(step_out)

        parsed_results.append(
            {
                "question_idx": q_idx,
                "question_id": question_id,
                "question": question_text,
                "final_s": r.get("final_s"),
                "final_answer": r.get("final_answer"),
                "steps": steps_out,
                # carry over token_usage if present on a per-question basis
                "token_usage": r.get("token_usage"),
            }
        )

    out_path = C.get_vendirag_parsed(dataset, num_docs, model_key)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "dataset": dataset,
                "num_docs": num_docs,
                "model": model_key,
                "parsed_results": parsed_results,
            },
            f,
        )

    print(f"[parse_vendirag] Saved parsed Vendi-RAG traces to: {out_path}")
    return out_path, parsed_results


def evaluate_vendirag_early_stopping(
    dataset: str,
    num_docs: int,
    model_key: str,
    max_steps_list: List[int] = None,
    quality_threshold: float = 0.85,
) -> Dict[int, Dict[str, Any]]:
    """
    Given parsed VendiRAG traces, simulate the early-stopping strategy
    described in the Vendi-RAG paper:

      - For a fixed max number of steps M:
          * Run steps sequentially.
          * If at some step t the (normalized) quality >= quality_threshold,
            stop immediately and select that step as the answer.
          * Otherwise, after executing up to M steps (or fewer if not enough),
            select the step with the largest quality among the executed ones.

      - For each question and each M, we log:
          * recall of the selected step
          * EM of the selected step (0/1; None treated as 0)
          * total_tokens = sum over executed steps of
                (answer_tokens + judge_tokens +
                 answer_prompt_tokens + judge_prompt_tokens)

    Returns:
        A dict keyed by max_steps (int), where each value is:
            {
              "avg_recall": float,
              "avg_em": float,
              "avg_tokens": float,
              "per_question": [
                  {
                    "question_idx": int,
                    "selected_step": int,
                    "recall": float,
                    "em": float,
                    "total_tokens": int,
                  },
                  ...
              ],
            }
    """
    if max_steps_list is None:
        max_steps_list = [2, 5, 10, 15, 20]

    parsed_path = C.get_vendirag_parsed(dataset, num_docs, model_key)
    print(f"[evaluate_vendirag_early_stopping] Loading parsed traces from: {parsed_path}")
    with open(parsed_path, "rb") as f:
        payload = pickle.load(f)
    parsed_results: List[Dict[str, Any]] = payload.get("parsed_results", [])
    print(f"[evaluate_vendirag_early_stopping] Parsed questions: {len(parsed_results)}")

    summaries: Dict[int, Dict[str, Any]] = {}

    for M in max_steps_list:
        per_q: List[Dict[str, Any]] = []
        recalls: List[float] = []
        ems: List[float] = []
        tokens_list: List[int] = []

        for q in parsed_results:
            steps = q.get("steps", []) or []
            if not steps:
                continue

            max_available = len(steps)
            max_exec = min(M, max_available)  # number of steps we are allowed to execute

            total_tokens = 0
            selected_step_idx = None
            best_quality = float("-inf")
            best_idx = 0

            for i in range(max_exec):
                step = steps[i]
                # Tokens for this step
                t_step = (
                    int(step.get("answer_tokens") or 0)
                    + int(step.get("judge_tokens") or 0)
                    + int(step.get("answer_prompt_tokens") or 0)
                    + int(step.get("judge_prompt_tokens") or 0)
                )
                total_tokens += t_step

                quality = step.get("quality")
                if quality is None:
                    q_norm = 0.0
                else:
                    # normalize assuming original scale ~ [0,10]
                    q_norm = float(quality) / 10.0

                # Track best quality among executed steps
                if q_norm > best_quality:
                    best_quality = q_norm
                    best_idx = i

                # Early stopping if threshold met
                if q_norm >= quality_threshold:
                    selected_step_idx = i
                    break

            if selected_step_idx is None:
                selected_step_idx = best_idx

            sel_step = steps[selected_step_idx]
            rec = float(sel_step.get("recall") or 0.0)
            em_val = sel_step.get("exact_match")
            em = 1.0 if em_val else 0.0

            recalls.append(rec)
            ems.append(em)
            tokens_list.append(total_tokens)

            per_q.append(
                {
                    "question_idx": q.get("question_idx"),
                    "selected_step": selected_step_idx,
                    "recall": rec,
                    "em": em,
                    "total_tokens": total_tokens,
                }
            )

        if not per_q:
            summaries[M] = {
                "avg_recall": 0.0,
                "avg_em": 0.0,
                "avg_tokens": 0.0,
                "per_question": [],
            }
            continue

        n = len(per_q)
        summaries[M] = {
            "avg_recall": sum(recalls) / n,
            "avg_em": sum(ems) / n,
            "avg_tokens": sum(tokens_list) / n,
            "per_question": per_q,
        }

        print(
            f"[evaluate_vendirag_early_stopping] M={M}: "
            f"avg_recall={summaries[M]['avg_recall']:.4f}, "
            f"avg_em={summaries[M]['avg_em']:.4f}, "
            f"avg_tokens={summaries[M]['avg_tokens']:.1f} "
            f"(n={n})"
        )

    return summaries


def _default_update_s(
    current_s: float,
    judge_score: float,
    step_idx: int,
    history: List[Dict[str, Any]],
) -> float:
    """
    Paper-inspired update rule:
        - judge_score is assumed in [0, 10]
        - normalize: q_norm = clamp(judge_score, 0, 10) / 10
        - new s := 1 - q_norm  (higher quality -> smaller s)
    """
    # Clamp score to [0, 10] for safety
    score = max(0.0, min(10.0, float(judge_score)))
    q_norm = score / 10.0
    s_new = 1.0 - q_norm
    # Ensure s stays in [0, 1]
    return max(0.0, min(1.0, s_new))


def _parse_judge_score(text: str) -> float:
    """
    Parse a numeric score from the judge LLM output.
    Expects the score inside <quality>...</quality> tags, e.g.:
        <quality>7.5</quality></final>
    Uses utils.extract_tagged_answer and falls back to numeric literals
    in the text. Emits warnings and returns 0.0 when parsing fails.
    """
    import re

    if not text:
        print(
            "[VendiRAGAdaptive] Warning: empty judge output; defaulting score to 0.0\n"
            "[VendiRAGAdaptive] judge_raw = ''"
        )
        return 0.0

    # Try to extract from <quality> tag via helper
    tagged = extract_tagged_answer(text, tag="quality")
    if tagged is not None:
        try:
            return float(tagged)
        except ValueError:
            print(
                "[VendiRAGAdaptive] Warning: could not parse numeric value "
                "from <quality> tag; falling back to regex.\n"
                f"[VendiRAGAdaptive] judge_raw = {text}"
            )

    # Fallback: parse numeric values from the text
    print(
        "[VendiRAGAdaptive] Warning: judge output missing <quality> tag; "
        "trying to parse numeric value in the text.\n"
        f"[VendiRAGAdaptive] judge_raw = {text}"
    )
    # Prefer the LAST numeric value (where Q_t is most likely to appear),
    # but keep the FIRST as a backup if needed.
    all_nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    for val in (all_nums[-1], all_nums[0]) if all_nums else []:
        try:
            return float(val)
        except ValueError:
            print(
                "[VendiRAGAdaptive] Warning: regex-based numeric parsing of judge "
                "output failed; defaulting score to 0.0.\n"
                f"[VendiRAGAdaptive] judge_raw = {text}"
            )

    print(
        "[VendiRAGAdaptive] Warning: judge LLM did not conform to expected "
        "<quality>...</quality></final> format and no numeric score could be "
        "parsed; defaulting score to 0.0.\n"
        f"[VendiRAGAdaptive] judge_raw = {text}"
    )
    return 0.0
