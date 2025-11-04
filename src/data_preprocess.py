import json
from typing import Dict, List

from .data import DataPreprocessBase, EvalData
from .report import NAME_TO_EVAL_CLASS


class DataPreprocess(DataPreprocessBase):
    """General data preprocessor"""

    def __init__(self, prompt_config_path: str, data_path: str):
        super().__init__(prompt_config_path=prompt_config_path)
        self.data = self.load_data(data_path=data_path)

    def _get_name(self) -> str:
        """Get the name of the general preprocessor"""
        return "general"

    def load_data(self, data_path: str) -> List[EvalData]:
        """Load general data"""
        return EvalData.from_jsonl(data_path)

    def get_corresponding_eval_type(self):
        pass

    def generate_prompt(self, query: str, docs_id: List[str]) -> List[Dict[str, str]]:
        """Format the prompt for the model."""
        # get document str from doc_id
        docs = []
        for _id in docs_id:
            docs.append(self.doc_pool[_id])

        docs_text = "\n".join(
            [f"<doc_{i+1}>" + doc + "</doc>" for i, doc in enumerate(docs)]
        )

        return [
            {
                "role": "system",
                "content": self.prompt_config["system_prompt"],
            },
            {
                "role": "user",
                "content": self.prompt_config["user_prompt"].format(
                    docs=docs_text, query=query
                ),
            },
        ]


class PubmedQAPreprocess(DataPreprocessBase):
    """PubmedQA specialized preprocessor"""

    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
        )
        self.data = self.load_data(data_path="data/pubmed.jsonl")

    def load_data(self, data_path: str) -> List[EvalData]:
        """Load general data"""
        return EvalData.from_jsonl(data_path)

    def _get_name(self) -> str:
        """Load PubmedQA data"""
        return "pubmedqa"

    def get_corresponding_eval_type(self):
        """Get corresponding eval type for PubmedQA"""
        return NAME_TO_EVAL_CLASS["pubmedqa"]

    def generate_prompt(self, query: str, docs_id: List[str]) -> List[Dict[str, str]]:
        """Format the prompt for the model."""
        # get document str from doc_id
        docs = []
        for _id in docs_id:
            docs.append(self.doc_pool[_id])
        docs_text = "\n".join(
            [f"<doc_{i+1}>" + doc + "</doc>" for i, doc in enumerate(docs)]
        )
        return [
            {
                "role": "system",
                "content": self.prompt_config["system_prompt"],
            },
            {
                "role": "user",
                "content": self.prompt_config["user_prompt"].format(
                    docs=docs_text, query=query
                ) + ", only response in one of 'yes', 'no' and 'maybe'",
            },
        ]


class HotpotQAPreprocess(DataPreprocess):
    """HotpotQA specialized preprocessor"""

    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
            data_path="data/hotpot_distractor.jsonl",
        )

    def _get_name(self) -> str:
        """Get the name of the HotpotQA preprocessor"""
        return "hotpotqa"

    def get_corresponding_eval_type(self):
        """Get corresponding eval type for HotpotQA"""
        return NAME_TO_EVAL_CLASS["hotpotqa"]


class PopQAPreprocess(DataPreprocess):
    """PopQA specialized preprocessor"""

    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
            data_path="data/popqa.jsonl",
        )

    def _get_name(self) -> str:
        """Get the name of the PopQA preprocessor"""
        return "popqa"

    def get_corresponding_eval_type(self):
        """Get corresponding eval type for PopQA"""
        return NAME_TO_EVAL_CLASS["popqa"]


class MusiqueQAPreprocess(DataPreprocess):
    """MusiqueQA specialized preprocessor"""

    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
            data_path="data/musique.jsonl",
        )

    def _get_name(self) -> str:
        """Get the name of the MusiqueQA preprocessor"""
        return "musiqueqa"

    def get_corresponding_eval_type(self):
        """Get corresponding eval type for MusiqueQA"""
        return NAME_TO_EVAL_CLASS["musiqueqa"]


class TwoWIKIPreprocess(DataPreprocess):
    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
            data_path="data/2wiki.jsonl",
        )

    def _get_name(self) -> str:
        return "2wiki"

    def get_corresponding_eval_type(self):
        return NAME_TO_EVAL_CLASS["2wiki"]


class TriviaQAPreprocess(DataPreprocess):
    def __init__(self):
        super().__init__(
            prompt_config_path="config/api_prompt_config_en.json",
            data_path="data/triviaqa.jsonl",
        )

    def _get_name(self) -> str:
        return "triviaqa"

    def get_corresponding_eval_type(self):
        return NAME_TO_EVAL_CLASS["triviaqa"]



NAME_TO_CLASS = {
    "hotpotqa": HotpotQAPreprocess,
    "popqa": PopQAPreprocess,
    "musiqueqa": MusiqueQAPreprocess,
    "pubmedqa": PubmedQAPreprocess,
    "2wiki": TwoWIKIPreprocess,
    "triviaqa": TriviaQAPreprocess,
}
