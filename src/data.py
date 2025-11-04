import json
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    from src.logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger()

__all__ = ["EvalData", "EvalResult"]


@dataclass
class EvalResult:
    """Single evaluation result."""

    id: str
    query: str
    prompt: str
    answer: str
    prediction: str
    label: Optional[bool] = None

    def __call__(
        self,
        st: Literal["id", "query", "prompt", "answer", "prediction", "label"],
    ):
        return asdict(self).get(st)

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalResult":
        return cls(
            id=data["id"],
            query=data["query"],
            prompt=data["prompt"],
            answer=data["answer"],
            prediction=data["prediction"],
            label=data["label"],
        )

    def set_label(self, label: bool):
        """Adds or updates the label for this result."""
        self.label = label


@dataclass
class EvalData:
    """Unified data evaluation data structure"""

    id: str
    query: str
    golden_doc: List[str]
    reference: List[str]
    ground_truth: str

    def __call__(
        self,
        st: Literal[
            "id",
            "query",
            "golden_doc",
            "reference",
            "ground_truth",
        ],
    ):
        return asdict(self).get(st)

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalData":
        return cls(
            id=data["id"],
            query=data["query"],
            golden_doc=data["golden_doc"],
            reference=data["reference"],
            ground_truth=data["ground_truth"],
        )

    @classmethod
    def to_jsonl(cls, data_list: List["EvalData"], file_path: str) -> None:
        """Save EvalData object list to JSONL file

        Args:
            data_list: EvalData object list
            file_path: JSONL file path to save
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                json_line = json.dumps(asdict(data), ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(
            f"Successfully \033[35mwrite {len(data_list)} benchmark items\033[0m to \033[32m{file_path}\033[0m"
        )

    @classmethod
    def from_jsonl(cls, file_path: str) -> List["EvalData"]:
        """Read data from JSONL file and convert to EvalData object list"""
        data_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines
                    try:
                        data_dict = json.loads(line)
                        data_list.append(cls.from_dict(data_dict))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
                        logger.error(f"Line content: {line.strip()}")
                    except Exception as e:
                        logger.error(
                            f"Error processing line {line_num}: {type(e)}:{str(e)}"
                        )
                        logger.error(f"Line content: {line.strip()}")
        logger.info(
            f"Successfully \033[34mloaded {len(data_list)} benchmark items\033[0m from \033[31m{file_path}\033[0m"
        )
        return data_list

    def get_answer(self) -> str:
        """Get answer"""
        return self.ground_truth


class DataPreprocessBase(ABC):
    """Data preprocessing base class"""

    def __init__(self, prompt_config_path: str = "config/default_prompt_config.json"):
        self.set_prompt_config(prompt_config_path)
        self.name = self._get_name()
        self.doc_pool = self.get_document_pool()

    @abstractmethod
    def _get_name(self) -> str:
        """Abstract method to get the name of the preprocessor, subclasses must implement"""
        pass

    @abstractmethod
    def load_data(self, data_path: str) -> List[EvalData]:
        """Abstract method to load data, subclasses must implement"""
        pass

    @abstractmethod
    def get_corresponding_eval_type(self):
        """Abstract method to get corresponding eval type, subclasses must implement"""
        pass

    @abstractmethod
    def generate_prompt(self, query: str, docs_id: List[str]) -> List[Dict[str, str]]:
        """Abstract method to generate prompt, subclasses must implement"""
        pass

    def get_document_pool(self) -> Dict[str, str]:
        with open("data/documents_pool.json", "r") as f:
            doc_pool = json.load(f)
        return doc_pool

    def set_prompt_config(self, prompt_config_path: str):
        with open(prompt_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.prompt_config = config
            # Initialize random number generators with seed from config
            random_seed = config.get("random_seed", 42)
            self.selection_rng = random.Random(random_seed)
            self.shuffle_rng = random.Random(random_seed)

    def generate_input(
        self, shuffle: bool = True, total_doc_number: int = 10
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate input for the model.
        Args:
            data: Data list
            num_iterations: number of iterations to randomly select from available placeholders.
                           For each query, this parameter determines how many different placeholder
                           versions will be used for evaluation. Each placeholder represents a
                           different version of the same query with different variable substitutions.
            shuffle: whether to shuffle the data
        Returns:
            idxs: list of indices
            queries: list of queries
            prompts: list of prompts
            answers: list of answers
        """
        queries_final = []
        prompts_final = []
        answers_final = []
        idxs_final = []

        for sample in self.data:
            idx = sample.id
            query = sample.query

            answers = sample.get_answer()
            golden_docs_ready = sample.golden_doc
            noise_docs_ready = self.generate_noise_docs(
                sample, total_doc_number - len(golden_docs_ready)
            )
            docs_ready = golden_docs_ready + noise_docs_ready
            if shuffle:
                self.shuffle_rng.shuffle(docs_ready)
            prompt = self.generate_prompt(query, docs_ready)
            prompts_final.append(prompt)
            answers_final.append(answers)
            queries_final.append(query)
            idxs_final.append(idx)

        return idxs_final, queries_final, prompts_final, answers_final

    def generate_noise_docs(self, sample: EvalData, noise_doc_number: int) -> List[str]:
        """
        Generate noise docs for the sample.
        """
        if noise_doc_number > len(sample.reference):
            logger.info(f"Sample {sample.id} has insufficient reference length: {len(sample.reference)}")
        return self.selection_rng.sample(
            sample.reference, min(noise_doc_number, len(sample.reference))
        )
