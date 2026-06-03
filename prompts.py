def graph_entity_extraction_system_prompt():
    return (
        "You extract named entities from text for a graph-RAG index. "
        "Return only salient entity mentions explicitly present in the text. "
        "Prefer people, organizations, locations, works, events, and other concrete named entities. "
        "Do not infer missing facts. Do not explain. Output only:\n"
        "<entities>\n"
        "...\n"
        "</entities>\n"
        "with one entity per line."
    )


def graph_entity_extraction_user_prompt(text):
    return (
        "Extract the named entities from the following text. "
        "Only include entities that appear explicitly in the text. "
        "If there are no useful entities, return empty tags.\n\n"
        "Text:\n"
        "<text>\n"
        f"{text}\n"
        "</text>"
    )


def scifact_answer_only_repeat_passages(question, answers, passages_list):
    system_prompt = (
        "You are a scientific claim judge. Your job is to choose which answer is the most likely to be correct for the given claim. This includes judging both the relevance of the used passages and the quality of the produced answer"
        "There are several candidate answers produced by different models, each using specific passages from the literature as context. "
        "For each answer, you are given both the answer text and the passages used to create it."
    )
    
    user_prompt = (
        f"\nSetting:\nEach model had to answer a claim with either 'SUPPORT' or 'CONTRADICT'.\n"
        f"Claim: {question}\n\n"
        "Below, several models have provided their answers using context passages.\n\n"
    )
    
    for idx, (ans, passages) in enumerate(zip(answers, passages_list)):
        user_prompt += (
            f"\nAnswer {idx}:\n"
            f"{ans.strip()}\n\n"
            "This answer has been produced using the following passages as context:\n"
        )
        for pidx, passage in enumerate(passages):
            user_prompt += f"  Passage {pidx}: {passage}...\n\n"
        user_prompt += "======\n"
    
    user_prompt += (
        "Your task:\n"
        "  - Read each answer and its supporting passages carefully.\n"
        "  - At the end, choose the answer that is **best supported by the given passages** for the claim. Do not reward answers that are well written but lack actual evidence from the passages.\n"
        "  - Prefer answers that are correct **and** are justified by the context passages provided. If none are well supported, pick the least bad answer.\n"
        "  - Please keep your explanation as brief as possible. ALWAYS include the verdict of the selected answer <judge> tag at the end, like <judge>CONTRADICT</judge> or <judge>SUPPORT</judge>.\n"
        "======\n"
    )
    return system_prompt, user_prompt

def scifact_answer_only_no_repeat(question, answers, passages_list):
    """
        Returns (system_prompt, user_prompt)
        - Each answer shows which passage indices it used.
        - All unique passages are listed at the end.
        - No per-answer explanation requested.
    """
    # Build unioned passages, preserving order
    unioned_passages = []
    passage_to_index = {}
    for passages in passages_list:
        for p in passages:
            key = p.strip()
            if key not in passage_to_index:
                passage_to_index[key] = len(unioned_passages)
                unioned_passages.append(key)

    # For each answer, which passages does it use?
    answer_passage_indices = []
    for passages in passages_list:
        indices = [passage_to_index[p.strip()] for p in passages]
        answer_passage_indices.append(indices)

    system_prompt = (
        "You are a scientific claim judge. Your job is to choose which answer is the most likely to be correct for the given claim. This includes judging both the relevance of the used passages and the quality of the produced answer"
        "Each answer below was generated using a subset of the context passages listed at the end. "
        "For each answer, we indicate which passages were used."
    )
    user_prompt = (
        f"Claim: {question}\n\n"
        "Below, you are given answers from several models. For each answer, the indices of the passages used are given.\n"
        "======\n"
    )
    for idx, (ans, indices) in enumerate(zip(answers, answer_passage_indices)):
        user_prompt += (
            f"\nAnswer {idx} (uses passages {indices}):\n"
            f"{ans.strip()}\n"
            "======\n"
        )

    user_prompt += "\nAll passages (referred by index above):\n"
    for i, p in enumerate(unioned_passages):
        user_prompt += f"Passage {i}: {p}\n\n"
    user_prompt += "======\n"
    user_prompt += (
         "Your task:\n"
        "  - Read each answer and its supporting passages carefully.\n"
        "  - At the end, choose the answer that is **best supported by the given passages** for the claim. Do not reward answers that are well written but lack actual evidence from the passages.\n"
        "  - Prefer answers that are correct **and** are justified by the context passages provided. If none are well supported, pick the least bad answer.\n"
        "  - Please keep your explanation as brief as possible. ALWAYS include the verdict of the selected answer <judge> tag at the end, like <judge>CONTRADICT</judge> or <judge>SUPPORT</judge>.\n"
        "======\n"
    )
    return system_prompt, user_prompt

def scifact_answer_and_explanation_repeat_passages(question, answers, passages_list):
    system_prompt = (
        "You are a scientific claim judge. Your job is to choose which answer is the most likely to be correct for the given claim. This includes judging both the relevance of the used passages and the quality of the produced answer"
        "For each answer, you are given the text and its supporting context passages. "
        "You must first provide a short evaluation (1-2 sentences) for every answer, then choose the best one at the end."
    )
    user_prompt = (
        f"\nSetting:\nYou are given the following claim, which must be answered as either 'SUPPORT' or 'CONTRADICT':\n"
        f"Claim: {question}\n\n"
        "Several models have answered this claim using various context passages (shown below).\n\n"
    )
    
    for idx, (ans, passages) in enumerate(zip(answers, passages_list)):
        user_prompt += (
            f"\nAnswer {idx}:\n"
            f"{ans.strip()}\n\n"
            "This answer has been produced using the following passages as context:\n"
        )
        for pidx, passage in enumerate(passages):
            user_prompt += f"  Passage {pidx}: {passage}...\n\n"
        user_prompt += "======\n"
    
    user_prompt += (
        "Your tasks:\n"
        "1. For each answer, provide a short evaluation (1-2 sentences) covering:\n"
        "   - (a) Whether the passages provide enough evidence to confidently answer the claim\n"
        "   - (b) How well the answer is actually supported by the provided passages\n"
        "2. After evaluating all answers, choose the best answer, i.e. the one most likely to be correct.\n"
        "3. Put the verdict of the best answer inside a <judge> tag at the end of your reply, i.e. <judge>CONTRADICT</judge> if the selected answer says that the claim is contradicted and <judge>SUPPORT</judge> if it says that the claim is supported.\n"
        "   - Do not output anything else after the <judge> tag.\n"
        "   - Your reply must ALWAYS include the <judge> tag, even if you reach the token limit.\n"
        "======\n"
    )
    return system_prompt, user_prompt

def scifact_answer_explanation_no_repeat(question, answers, passages_list):
    """
    Returns (system_prompt, user_prompt)
    - Each answer shows which passage indices it used.
    - All unique passages are listed at the end.
    - LLM must provide a short evaluation for each answer.
    """
    # Build unioned passages, preserving order
    unioned_passages = []
    passage_to_index = {}
    for passages in passages_list:
        for p in passages:
            key = p.strip()
            if key not in passage_to_index:
                passage_to_index[key] = len(unioned_passages)
                unioned_passages.append(key)

    # For each answer, which passages does it use?
    answer_passage_indices = []
    for passages in passages_list:
        indices = [passage_to_index[p.strip()] for p in passages]
        answer_passage_indices.append(indices)

    system_prompt = (
        "You are a scientific claim judge. Your job is to choose which answer is the most likely to be correct for the given claim. This includes judging both the relevance of the used passages and the quality of the produced answer"
        "Each answer below was generated using a subset of the context passages listed at the end. "
        "For each answer, we indicate which passages were used."
        "Your job is to briefly evaluate each answer and then choose the best one."
    )
    user_prompt = (
        f"\nSetting:\nYou are given the following claim, which must be answered as either 'SUPPORT' or 'CONTRADICT':\n"
        f"Claim: {question}\n\n"
        "Several models have answered this claim using various context passages (shown below).\n\n"
        "For each answer, the indices of the passages it used are specified; the full text of all passages is provided at the end of this prompt."

    )

    for idx, (ans, indices) in enumerate(zip(answers, answer_passage_indices)):
        user_prompt += (
            f"\nAnswer {idx} (uses passages {indices}):\n"
            f"{ans.strip()}\n"
            "======\n"
        )

    user_prompt += "\nAll passages (referred by index above):\n"
    for i, p in enumerate(unioned_passages):
        user_prompt += f"Passage {i}: {p}\n\n"
    user_prompt += "======\n"
    user_prompt += (
        "Your tasks:\n"
        "1. For each answer, provide a short evaluation (1-2 sentences) covering:\n"
        "   - (a) Whether the passages provide enough evidence to confidently answer the claim\n"
        "   - (b) How well the answer is actually supported by the provided passages\n"
        "2. After evaluating all answers, choose the best answer, i.e. the one most likely to be correct.\n"
        "3. Put the verdict of the best answer inside a <judge> tag at the end of your reply, i.e. <judge>CONTRADICT</judge> if the selected answer says that the claim is contradicted and <judge>SUPPORT</judge> if it says that the claim is supported.\n"
        "   - Do not output anything else after the <judge> tag.\n"
        "   - Your reply must ALWAYS include the <judge> tag, even if you reach the token limit.\n"
        "======\n"
    )
    return system_prompt, user_prompt

def scifact_super_answerer(question, answers, passages_list):
    """
    Judge is shown all unique passages first, then all hints (answers from smaller models, with passage indices).
    At the end, the LLM must generate its own answer, justified by the passages, and put the final verdict in a <judge> tag.
    Returns (system_prompt, user_prompt)
    """
    # Union all passages (preserve order, dedupe)
    unioned_passages = []
    passage_to_index = {}
    for passages in passages_list:
        for p in passages:
            key = p.strip()
            if key not in passage_to_index:
                passage_to_index[key] = len(unioned_passages)
                unioned_passages.append(key)

    # For each answer, which passage indices did it use?
    answer_passage_indices = []
    for passages in passages_list:
        indices = [passage_to_index[p.strip()] for p in passages]
        answer_passage_indices.append(indices)

    system_prompt = (
        "You are a scientific claim expert. You will be given a scientific claim, a set of context passages from the literature, "
        "and several answers generated by smaller models using different subsets of these passages as hints. "
        "Your job is to generate your own best answer, fully based on all the context passages."
    )
    user_prompt = (
        f"Claim:\n{question}\n\n"
        "Below are context passages gathered from the literature that may support or contradict the claim. "
        "Read them carefully before reviewing the hint answers.\n"
        "======\n"
    )

    # List all unique passages first
    user_prompt += "Context Passages (referenced by index):\n"
    for i, passage in enumerate(unioned_passages):
        user_prompt += f"Passage {i}:\n{passage}\n\n"
    user_prompt += "======\n"

    # List hints (answers from smaller models)
    user_prompt += "Below are several hint answers, each produced by a smaller model using only a subset of the passages above. These are for your reference only.\n"
    for idx, (ans, indices) in enumerate(zip(answers, answer_passage_indices)):
        user_prompt += (
            f"\nHint {idx} (produced by a smaller model using passages {indices}):\n"
            f"{ans.strip()}\n"
            "------\n"
        )

    # Final instructions
    user_prompt += (
        "\nNow, based on the claim and ALL the context passages above, write your own answer as concisely as possible (max 2 sentences of justification). "
        "Your reasoning should cite passages by index when possible. "
        "If evidence is ambiguous, pick the answer best supported by the passages.\n\n"
        "Your reply MUST end with your final decision inside a <judge> tag, using only SUPPORT or CONTRADICT (e.g., <judge>SUPPORT</judge>), and output NOTHING after the tag. "
        "If you run out of space, always include at least the <judge> tag at the end.\n"
        "======"
    )
    return system_prompt, user_prompt

def hotpotqa_answer_only_no_repeat(question, answers, passages_list):
    """
    Judge picks the SINGLE best-supported answer and may give a brief justification.
    The final line MUST be the actual answer text inside <judge>...</judge> (NOT an index).

    If the chosen candidate contains an <answer>...</answer> tag, COPY ITS CONTENT VERBATIM
    into <judge>...</judge>. Otherwise, output the minimal final answer phrase.
    """

    # Build unioned passages, preserving order
    unioned_passages = []
    passage_to_index = {}
    for passages in passages_list:
        for p in passages:
            key = p.strip()
            if key not in passage_to_index:
                passage_to_index[key] = len(unioned_passages)
                unioned_passages.append(key)

    # Map each answer's used passages -> indices (based on provided passages_list)
    answer_passage_indices = []
    for passages in passages_list:
        indices = [passage_to_index[p.strip()] for p in passages]
        answer_passage_indices.append(indices)

    system_prompt = (
        "You are a question-answering judge.\n"
        "- Given a question, several candidate answers, and supporting passages, choose the SINGLE answer best supported by the passages.\n"
        "- Prefer evidence-mode answers with strong direct support from the passages.\n"
        "- If no candidate has clear evidence, you may consider guess-mode answers. In that case, prefer the one with "
        "the most plausible justification, but reflect its lower confidence.\n"
        "- Use confidence scores as a guide: high-confidence evidence-mode answers should usually beat low-confidence guesses.\n"
        "- Never invent new facts beyond what is in the candidates or the passages.\n"
        "- You may write a brief justification (1–3 sentences) comparing the top candidates.\n"
        "- FINAL OUTPUT RULE: End your response with ONE line that contains ONLY the final answer text inside <judge>...</judge></final>.\n"
        "- IMPORTANT: Do NOT output an index. If the chosen candidate includes an <answer>...</answer> tag, COPY ITS CONTENT VERBATIM into <judge>...</judge>. "
        "If it lacks an <answer> tag, output the minimal final answer phrase.\n"
        "- Examples of valid final lines:\n"
        "    <judge>1971</judge></final>\n"
        "    <judge>Eiffel Tower</judge></final>\n"
        "- Do NOT wrap the answer in quotes and do NOT include explanations inside the <judge> tags."
        "Also Do NOT output anything after the </final> tag."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        "Below are candidate answers from several models. For each answer, we list the indices of the passages it used.\n"
        "======\n"
    )

    for idx, (ans, indices) in enumerate(zip(answers, answer_passage_indices)):
        user_prompt += (
            f"\nAnswer {idx} (uses passages {indices}):\n"
            f"{ans.strip()}\n"
            "------\n"
        )

    user_prompt += "\nAll passages (referred by index above):\n"
    for i, p in enumerate(unioned_passages):
        user_prompt += f"Passage {i}: {p}\n\n"

    user_prompt += (
        "======\n"
        "Your task:\n"
        "- Read the answers and passages carefully.\n"
        "- Provide a brief justification (1–3 sentences).\n"
        "- Then END with a single line: <judge>FINAL_ANSWER_TEXT</judge></final>.\n"
    )

    return system_prompt, user_prompt

def hotpotqa_super_answerer(question, answers, passages_list):
    """
    Super-answerer for HotpotQA.

    Returns (system_prompt, user_prompt)

    Flow:
      1) Show ALL unique context passages (indexed).
      2) Show hint answers with indices of passages they used.
      3) Ask the model to generate its OWN final answer (short span),
         briefly justify with citations to passage indices, and end with
         <judge>FINAL_ANSWER</judge></final>.
    """
    # 1) Union & index passages (preserve order)
    unioned_passages = []
    passage_to_index = {}
    for passages in passages_list:
        for p in passages:
            key = p.strip()
            if key not in passage_to_index:
                passage_to_index[key] = len(unioned_passages)
                unioned_passages.append(key)

    # 2) Map each hint's passages to indices
    answer_passage_indices = []
    for passages in passages_list:
        indices = [passage_to_index[p.strip()] for p in passages]
        answer_passage_indices.append(indices)

    # 3) Prompts
    system_prompt = (
        "You are a Q & A expert. You will receive a question, a set of context passages, "
        "and several hint answers (each with a mode and confidence score), produced by smaller models using subsets of the passages.\n "
        "Your job is to carefully read ALL passages and produce your OWN best final answer."
        "The passages are considered to be correct but might be potentially irrelevant to the question."
        "- IMPORTANT: End your response with your final answer inside "
        "<judge>...</judge> immediately followed by </final>, with NOTHING after </final>.\n"
        "- Example of the last line: <judge>1971</judge></final>"
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        "Carefully read the context passages below (referenced by index). Then review the hint answers.\n"
        "======\n"
        "Context Passages:\n"
    )

    for i, passage in enumerate(unioned_passages):
        user_prompt += f"Passage {i}:\n{passage}\n\n"

    user_prompt += "======\n"
    user_prompt += "Hint Answers (from smaller models; each shows the passages it used by index):\n"
    for idx, (ans, indices) in enumerate(zip(answers, answer_passage_indices)):
        user_prompt += (
            f"\nHint {idx} (uses passages {indices}):\n"
            f"{ans.strip()}\n"
            "------\n"
        )

    # Final instructions: concise answer + brief justification + <judge>FINAL_ANSWER</judge>
    user_prompt += (
        "\nNow write your own answer, supported by the passages above.\n"
        "- Keep the final answer as a short word/phrase/name/date/etc. (no extra wording).\n"
        "- Provide a brief justification, citing passage indices like [P3], [P7] when possible.\n"
        "- If evidence is ambiguous, choose the answer best supported by the passages.\n\n"
        "IMPORTANT FORMAT:\n"
        "Explanation: your brief justification here with citations like [P0], [P4]\n"
        "Final: <judge>FINAL_ANSWER</judge></final>\n"
        "Output NOTHING after the </final> tag.\n"
        "======"
    )

    return system_prompt, user_prompt

def selector_prompt(question, answers, passages_list):
    """
        Judge that selects the SINGLE best answer based only on the question
        and the candidate answers (with their explanations and mode).
        No passages are shown.

        Final line MUST be ONLY <judge>ACTUAL_ANSWER</judge></final>.
    """

    system_prompt = (
        "You are a question-answering judge.\n"
        "- You will be given a question and several candidate answers.\n"
        "- Each answer includes an explanation and a mode (evidence or guess).\n"
        "- Your task is to select the SINGLE answer that is most trustworthy.\n"
        "- Prefer answers in evidence mode with clear, well-structured explanations.\n"
        "- If no evidence-mode answers are persuasive, you may choose a guess-mode answer, "
        "but only if the explanation is reasonable and consistent with the question.\n"
        "- Never invent a new answer: always pick one of the candidates.\n"
        "- Provide a short justification (2–5 sentences) comparing candidates, "
        "referencing their mode and the quality of their explanations.\n"
        "\n"
        "STRICT FORMAT:\n"
        "- End with a single line ONLY: <judge>ACTUAL_ANSWER</judge></final>\n"
        "- IMPORTANT: Replace ACTUAL_ANSWER with the real chosen answer.\n"
        "- Do NOT output the word ACTUAL_ANSWER literally.\n"
        "- Examples of correct final lines:\n"
        "    <judge>1998</judge></final>\n"
        "    <judge>Eiffel Tower</judge></final>\n"
    )

    user_prompt = f"Question:\n{question}\n\n"
    user_prompt += "Candidate Answers:\n======\n"
    for i, ans in enumerate(answers):
        user_prompt += f"Answer {i}:\n{ans.strip()}\n------\n"

    user_prompt += (
        "======\n"
        "Your task:\n"
        "- Compare the answers using their explanations and mode.\n"
        "- Write a 2–5 sentence justification.\n"
        "- END with one line ONLY: <judge>ACTUAL_ANSWER</judge></final>\n"
        "- IMPORTANT: Replace ACTUAL_ANSWER with the real chosen answer.\n"
    )

    return system_prompt, user_prompt

def answer_prompt(question, passages):
    """
    Prompt for answering questions
      - Evidence-first answering (context beats prior knowledge when usable).
      - Safe fallback to best-guess ONLY if the passages contain no usable evidence.
      - A brief explanation for judging (mode, used_passages, confidence).
    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are a helpful question-answering assistant.\n"
        "Your objective is to produce the best SHORT answer.\n"
        "\n"
        "Core rules:\n"
        "1) Evidence-first: If the passages contain explicit evidence that entails the answer, use it. "
        "Prefer statements that are specific and unambiguous; favor answers supported by multiple passages.\n"
        "2) No evidence -> best-guess: If the passages are irrelevant, too vague, or do not entail an answer, "
        "give your best-guess from your general knowledge, but mark mode='guess'.\n"
        "3) Never contradict the passages: If any passage clearly contradicts your prior knowledge, trust the passages "
        "unless they are clearly off-topic (irrelevant to the question). Do not invent unsupported details.\n"
        "4) Be concise: The answer must be a single word, name, date, number, or very short phrase.\n"
        "5) Always put the final answer inside <answer>...</answer> tags.\n"
        "\n"
        "Conflict handling:\n"
        "- If passages disagree, pick the answer with the strongest explicit support (more passages, clearer wording). "
        "- If the evidence is ambiguous, output your best guess but mark the mode as 'guess' and explicitly mention this in your explanation.\n"
        "- If multi-hop reasoning is needed, combine facts across passages explicitly.\n"
        "\n"
        "Output format (exactly these two blocks, in this order):\n"
        "<explanation mode=\"evidence|guess\" used_passages=\"[comma-separated indices]\">\n"
        "Brief 1–2 sentence justification referencing passage indices (e.g., 'P2 states X; P4 confirms Y').\n"
        "</explanation>\n"
        "<answer>YOUR SHORT ANSWER</answer></final>"
    )

    user_prompt = ""
    user_prompt += (
        "Here are a few examples of valid outputs:\n"
        "Example question: In what year was Google founded?\n"
        "Example answer (evidence mode):\n"
        "<explanation mode=\"evidence\" used_passages=\"[1,3]\">P1 names the founder; P3 gives the company’s founding year.</explanation>\n"
        "<answer>1998</answer></final>\n\n"
        "Example question: Is Saturn larger than Jupiter?\n"
        "Example answer (guess mode, no usable evidence in passages):\n"
        "<explanation mode=\"guess\" used_passages=\"[]\">No passage compares the sizes; providing best-guess from general knowledge.</explanation>\n"
        "<answer>No</answer></final>\n\n"
        "Example question: Is Kyoto the capital of Japan?\n"
        "Example answer (evidence mode):\n"
        "<explanation mode=\"evidence\" used_passages=\"[0]\">P0 states that Tokyo is Japan’s capital city, so the answer is no.</explanation>\n"
        "<answer>No</answer></final>\n\n"
        "Example question: Who has scored the most points in NBA history?\n"
        "Example answer (evidence mode):\n"
        "<explanation mode=\"evidence\" used_passages=\"[1]\">P1 explicitly states that LeBron James is the NBA's all-time leading scorer.</explanation>\n"
        "<answer>LeBron James</answer></final>\n\n"
        "Example question: Which is larger, Saturn or Jupiter?\n"
        "Example answer (guess mode, no usable evidence in passages):\n"
        "<explanation mode=\"guess\" used_passages=\"[]\">No passage compares sizes; providing best-guess from general knowledge.</explanation>\n"
        "<answer>Jupiter</answer></final>\n"
        "========\n"
        "Now answer the following question.\n"
    )

    user_prompt += f"Question:\n{question}\n\n"
    user_prompt += "Context Passages:\n"
    for i, passage in enumerate(passages):
        user_prompt += f"Passage {i}: {passage}\n\n"

    user_prompt += (
        "========\n"
        "Your task:\n"
        "- Read the passages and determine whether they explicitly support an answer.\n"
        "- If yes, answer in EVIDENCE mode and cite the passage indices you used.\n"
        "- If no, answer in GUESS mode (best-guess from your knowledge).\n"
        "- Do NOT repeat the question. Keep the answer minimal. No extra text outside the required blocks.\n"
        "- ALWAYS conclude your answer with a </final> tag and write nothing after it!\n"
        "======\n\n"
    )

    return system_prompt, user_prompt
