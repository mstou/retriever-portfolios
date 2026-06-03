import os
import json
import random
from abc import ABC, abstractmethod
from data_classes import Document, Dataset, Questions
from constants import DATASET_LOCATION, MUSIQUE, HotpotQA, TRIVIAQA, TWOWIKI

class Loader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_documents(self):
        """Returns a Dataset object."""
        pass

class QuestionsLoader(ABC):
    """Abstract base class for question loaders."""

    @abstractmethod
    def load_questions(self):
        """Returns a Question object."""
        pass


class HotpotQA_DataLoader(Loader):
    def __init__(self, dataset_location: str):
        self.dataset_location = dataset_location
        self.corpus_location = dataset_location + 'corpus.jsonl'
        self.title_to_id = {}

    def load_documents(self):

        dataset = Dataset("HotpotQA")

        with open(self.corpus_location, 'r') as f:
            for line in f:
                article = json.loads(line.strip())
                dataset.add_document(Document(
                    doc_id=article['_id'],
                    text=article['text'],
                    metadata={'title': article['title'], 'url': article['metadata']['url']},
                    no_chunking=False
                ))
                self.title_to_id[article['title']] = article['_id']
        self.dataset = dataset

        return dataset
        
    def get_title_to_id(self):
        return self.title_to_id
    
    
class HotpotQA_QuestionsLoader(QuestionsLoader):
    def __init__(self, dataset_location: str, title_to_id: dict, split: int = 0.8):
        # split = percentage of queries to be used as train
        self.dataset_location = dataset_location
        self.questions_location = dataset_location + 'queries.jsonl'
        self.title_to_id = title_to_id
        self.split = split

    def load_questions(self):
        questions_train = Questions(f"HotpotQA Train")
        questions_test = Questions(f"HotpotQA Test") if self.split < 1 else None

        questions_raw = []

        with open(self.questions_location, 'r') as f:
            for line in f:
                questions_raw.append(json.loads(line.strip()))
        
        
        random.shuffle(questions_raw)

        train_size = int(len(questions_raw)*self.split)
        train_q = questions_raw[:train_size]
        test_q = questions_raw[train_size:]

        for question in train_q:
            questions_train.add_question(
                question=question['text'],
                answer=question['metadata']['answer'],
                target_chunks=[self.title_to_id[s[0]] for s in question['metadata']['supporting_facts']]
            )
        
        for question in test_q:
            questions_test.add_question(
                question=question['text'],
                answer=question['metadata']['answer'],
                target_chunks=[self.title_to_id[s[0]] for s in question['metadata']['supporting_facts']]
            )
        
        return questions_train, questions_test


class MusiQue_DataLoader(Loader):
    """Loads the articles/documents of the MusiQue dataset."""

    def __init__(self, dataset_location: str = DATASET_LOCATION[MUSIQUE]):
        """
        dataset_location : str
            Base path where the MusiQue JSONL files (`train.jsonl` and `test.jsonl`) are stored.
        """
        self.dataset_location = dataset_location
        # Train and test MusiQue files (each line is a QA example with paragraphs)
        self.train_file = os.path.join(self.dataset_location, "train.jsonl")
        self.test_file = os.path.join(self.dataset_location, "test.jsonl")

        # Mapping from (title + paragraph_text) key to a single internal document ID
        # each unique paragraph (by title and text) becomes its own document.
        self.title_to_id = {}

    def load_documents(self):
        """Return a `Dataset` object populated with MusiQue documents.

        All unique paragraphs from both train and test files are collected
        into a single dataset. Paragraphs are considered the same only if
        both their title and paragraph_text match.
        """
        dataset = Dataset("MusiQue")

        # track keys we've already seen to avoid duplicates
        seen_keys = set()

        def _add_from_file(path):
            if not os.path.exists(path):
                return

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)
                    for paragraph in example.get("paragraphs", []):
                        title = paragraph.get("title", "")
                        text = paragraph.get("paragraph_text", "")
                        if not text or not title:
                            continue

                        key = f"{title}||{text}"
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        doc_id = len(dataset.documents)
                        dataset.add_document(
                            Document(
                                doc_id=doc_id,
                                text=text,
                                metadata={"title": title},
                            )
                        )
                        self.title_to_id[key] = doc_id

        _add_from_file(self.train_file)
        _add_from_file(self.test_file)

        self.dataset = dataset
        return dataset

    def get_title_to_id(self):
        """Return the mapping from titles to internal document IDs."""
        return self.title_to_id


class MusiQue_QuestionsLoader(QuestionsLoader):
    """Loads the questions/QA entries of the MusiQue dataset."""

    def __init__(self, dataset_location: str = DATASET_LOCATION[MUSIQUE], title_to_id: dict = None):
        """
        Parameters
        ----------
        dataset_location : str
            Base path where the MusiQue question files (e.g. `train.jsonl`, `test.jsonl`) are stored.
        title_to_id : dict
            Mapping from composite paragraph key (\"title||paragraph_text\") to internal document IDs
            produced by `MusiQue_DataLoader`.
        """
        self.dataset_location = dataset_location
        self.title_to_id = title_to_id or {}
        # Paths to MusiQue question files (train and test)
        self.train_questions_location = os.path.join(self.dataset_location, "train.jsonl")
        self.test_questions_location = os.path.join(self.dataset_location, "test.jsonl")

    def load_questions(self):
        """
        Return a (train, test) tuple of `Questions` objects, using `train.jsonl`
        for the train split and `test.jsonl` for the test split.

        Target document IDs are resolved only for supporting paragraphs
        (`is_supporting == True`) using the same composite key used in
        `MusiQue_DataLoader` (\"title||paragraph_text\").
        """
        questions_train = Questions("MusiQue Train")
        questions_test = Questions("MusiQue Test")

        def _add_from_file(path, questions_obj):
            if not os.path.exists(path):
                return

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)

                    # Only consider answerable questions
                    if not example.get("answerable", True):
                        continue

                    question_text = example.get("question", "")
                    answer = example.get("answer")

                    target_ids = []
                    for paragraph in example.get("paragraphs", []):
                        if not paragraph.get("is_supporting"):
                            continue
                        title = paragraph.get("title", "")
                        text = paragraph.get("paragraph_text", "")
                        if not title or not text:
                            continue

                        key = f"{title}||{text}"
                        doc_id = self.title_to_id.get(key)
                        if doc_id is not None:
                            target_ids.append(doc_id)

                    questions_obj.add_question(
                        question=question_text,
                        answer=answer,
                        target_chunks=target_ids,
                    )

        _add_from_file(self.train_questions_location, questions_train)
        _add_from_file(self.test_questions_location, questions_test)

        return questions_train, questions_test


class TriviaQA_DataLoader(Loader):
    """Loads the articles/documents for the TriviaQA dataset."""

    def __init__(self, dataset_location: str):
        """
        Parameters
        ----------
        dataset_location : str
            Base path where the TriviaQA files (`train.json`/`train.jsonl`, `test.json`/`test.jsonl`) are stored.
        """
        self.dataset_location = dataset_location
        # Support both .json (array) and .jsonl (one object per line)
        def _resolve_split(split_name: str) -> str:
            base = os.path.join(self.dataset_location, split_name)
            for ext in (".json", ".jsonl"):
                candidate = base + ext
                if os.path.exists(candidate):
                    return candidate
            # Default to .json if nothing found yet
            return base + ".json"

        self.train_file = _resolve_split("train")
        self.test_file = _resolve_split("test")

        # Mapping from composite context key ("title||text") to internal document IDs
        self.key_to_id = {}

    def load_documents(self):
        """Return a `Dataset` object populated with TriviaQA documents."""
        dataset = Dataset("TriviaQA")

        # Track seen (title, text) pairs to avoid duplicate documents
        seen_keys = set()

        def _add_contexts_from_file(path):
            if not os.path.exists(path):
                return

            # Load either JSON array or JSONL, following test.py logic
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
            if not txt:
                return
            if txt[0] == "[":
                instances = json.loads(txt)
            else:
                instances = [json.loads(line) for line in txt.splitlines() if line.strip()]

            for instance in instances:
                positive_ctxs = instance.get("positive_ctxs", []) or []
                negative_ctxs = instance.get("negative_ctxs", []) or []
                hard_negative_ctxs = instance.get("hard_negative_ctxs", []) or []

                # Sample at most 5 negatives and 5 hard negatives per question
                if len(negative_ctxs) > 5:
                    negative_ctxs = random.sample(negative_ctxs, 5)
                if len(hard_negative_ctxs) > 5:
                    hard_negative_ctxs = random.sample(hard_negative_ctxs, 5)

                for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
                    title = ctx.get("title", "").strip()
                    text = ctx.get("text", "").strip()
                    if not title or not text:
                        continue

                    key = f"{title}||{text}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    doc_id = len(dataset.documents)
                    dataset.add_document(
                        Document(
                            doc_id=doc_id,
                            text=text,
                            metadata={"title": title},
                        )
                    )
                    self.key_to_id[key] = doc_id

        _add_contexts_from_file(self.train_file)
        _add_contexts_from_file(self.test_file)

        self.dataset = dataset
        return dataset

    def get_key_to_id(self):
        """Return the mapping from composite context keys to internal document IDs."""
        return self.key_to_id


class TriviaQA_QuestionsLoader(QuestionsLoader):
    """Loads the questions/QA entries for the TriviaQA dataset."""

    def __init__(self, dataset_location: str, key_to_id: dict):
        """
        Parameters
        ----------
        dataset_location : str
            Base path where the TriviaQA files (`train.json`/`train.jsonl`, `test.json`/`test.jsonl`) are stored.
        key_to_id : dict
            Mapping from composite context key ("title||text") to internal document IDs
            produced by `TriviaQA_DataLoader`.
        """
        self.dataset_location = dataset_location
        self.key_to_id = key_to_id

        def _resolve_split(split_name: str) -> str:
            base = os.path.join(self.dataset_location, split_name)
            for ext in (".json", ".jsonl"):
                candidate = base + ext
                if os.path.exists(candidate):
                    return candidate
            return base + ".json"

        self.train_questions_location = _resolve_split("train")
        self.test_questions_location = _resolve_split("test")

    def load_questions(self):
        """
        Return a (train, test) tuple of `Questions` objects built from
        the TriviaQA train and test files.
        """
        questions_train = Questions("TriviaQA Train")
        questions_test = Questions("TriviaQA Test")

        def _add_from_file(path, questions_obj, skip_if_no_positive=False):
            if not os.path.exists(path):
                return

            # Load either JSON array or JSONL, same as in the data loader / test.py
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
            if not txt:
                return
            if txt[0] == "[":
                instances = json.loads(txt)
            else:
                instances = [json.loads(line) for line in txt.splitlines() if line.strip()]

            for instance in instances:
                positive_ctxs = instance.get("positive_ctxs", [])
                if skip_if_no_positive and not positive_ctxs:
                    continue

                question_text = instance.get("question", "")
                answers = instance.get("answers", [])

                target_ids = []
                for ctx in positive_ctxs:
                    title = ctx.get("title", "").strip()
                    text = ctx.get("text", "").strip()
                    if not title or not text:
                        continue

                    key = f"{title}||{text}"
                    doc_id = self.key_to_id.get(key)
                    if doc_id is not None and doc_id not in target_ids:
                        target_ids.append(doc_id)

                questions_obj.add_question(
                    question=question_text,
                    answer=answers,
                    target_chunks=target_ids,
                )

        # For train: skip questions with empty positive_ctxs
        _add_from_file(self.train_questions_location, questions_train, skip_if_no_positive=True)
        # For test: include all questions, even if they have no positive contexts
        _add_from_file(self.test_questions_location, questions_test, skip_if_no_positive=False)

        return questions_train, questions_test


class TwoWikiMultiHopQA_DataLoader(Loader):
    """Loads the articles/documents for the 2WikiMultiHopQA dataset."""

    def __init__(self, dataset_location: str):
        """
        Parameters
        ----------
        dataset_location : str
            Base path where the 2WikiMultiHopQA files (`train.json`, `dev.json`) are stored.
        """
        self.dataset_location = dataset_location
        self.train_file = os.path.join(self.dataset_location, "train.json")
        self.dev_file = os.path.join(self.dataset_location, "dev.json")

        # Mapping from article titles to internal document IDs
        self.title_to_id = {}

    def _load_instances(self, path):
        """Load a list of JSON instances from a .json (array) or .jsonl file."""
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        if not txt:
            return []
        if txt[0] == "[":
            return json.loads(txt)
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

    def load_documents(self):
        """Return a `Dataset` object populated with 2WikiMultiHopQA documents.

        Each unique context title becomes a single document whose text is the
        concatenation of its sentences.
        """
        dataset = Dataset("2WikiMultiHopQA")

        for path in [self.train_file, self.dev_file]:
            instances = self._load_instances(path)
            for instance in instances:
                for raw_context in instance.get("context", []):
                    if not isinstance(raw_context, (list, tuple)) or len(raw_context) < 2:
                        continue
                    title = str(raw_context[0]).strip()
                    sentences = raw_context[1]
                    if not isinstance(sentences, (list, tuple)):
                        continue
                    paragraph_text = " ".join(str(s) for s in sentences).strip()
                    if not title or not paragraph_text:
                        continue

                    if title in self.title_to_id:
                        continue

                    doc_id = len(dataset.documents)
                    dataset.add_document(
                        Document(
                            doc_id=doc_id,
                            text=paragraph_text,
                            metadata={"title": title},
                        )
                    )
                    self.title_to_id[title] = doc_id

        self.dataset = dataset
        return dataset

    def get_title_to_id(self):
        """Return the mapping from titles to internal document IDs."""
        return self.title_to_id


class TwoWikiMultiHopQA_QuestionsLoader(QuestionsLoader):
    """Loads the questions/QA entries for the 2WikiMultiHopQA dataset."""

    def __init__(self, dataset_location: str, title_to_id: dict):
        """
        Parameters
        ----------
        dataset_location : str
            Base path where the 2WikiMultiHopQA files (`train.json`, `dev.json`) are stored.
        title_to_id : dict
            Mapping from article titles to internal document IDs produced by
            `TwoWikiMultiHopQA_DataLoader`.
        """
        self.dataset_location = dataset_location
        self.title_to_id = title_to_id
        self.train_file = os.path.join(self.dataset_location, "train.json")
        self.dev_file = os.path.join(self.dataset_location, "dev.json")

    def _load_instances(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        if not txt:
            return []
        if txt[0] == "[":
            return json.loads(txt)
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

    def load_questions(self):
        """
        Return a (train, test) tuple of `Questions` objects built from
        the 2WikiMultiHopQA train (train split) and dev (test split) files.
        """
        questions_train = Questions("2WikiMultiHopQA Train")
        questions_test = Questions("2WikiMultiHopQA Test")

        def _add_from_file(path, questions_obj, skip_if_no_support=False):
            instances = self._load_instances(path)
            for instance in instances:
                question_text = instance.get("question", "")
                answer = instance.get("answer")

                # supporting_facts is a list of [title, sentence_idx] pairs
                supporting_facts = instance.get("supporting_facts", []) or []
                supporting_titles = {str(t[0]).strip() for t in supporting_facts if isinstance(t, (list, tuple)) and len(t) >= 1}

                target_ids = []
                for title in supporting_titles:
                    doc_id = self.title_to_id.get(title)
                    if doc_id is not None and doc_id not in target_ids:
                        target_ids.append(doc_id)

                if skip_if_no_support and not target_ids:
                    continue

                questions_obj.add_question(
                    question=question_text,
                    answer=answer,
                    target_chunks=target_ids,
                )

        # Train split = train only
        _add_from_file(self.train_file, questions_train, skip_if_no_support=True)
        # Test split = dev only (keep even if no supporting titles resolve)
        _add_from_file(self.dev_file, questions_test, skip_if_no_support=False)

        return questions_train, questions_test


def get_document_loader(dataset_name: str) -> Loader:
    if dataset_name == HotpotQA:
        return HotpotQA_DataLoader(DATASET_LOCATION[HotpotQA])
    if dataset_name == MUSIQUE:
        return MusiQue_DataLoader(DATASET_LOCATION[MUSIQUE])
    if dataset_name == TRIVIAQA:
        return TriviaQA_DataLoader(DATASET_LOCATION[TRIVIAQA])
    if dataset_name == TWOWIKI:
        return TwoWikiMultiHopQA_DataLoader(DATASET_LOCATION[TWOWIKI])
    raise ValueError(f"Invalid dataset provided: {dataset_name}")
