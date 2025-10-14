"""
Test cases for src/data.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import DataPreprocess, EvalData


class TestEvalData:
    """Test EvalData class functionality"""

    def test_evaldata_creation(self):
        """Test creating EvalData instance"""
        data = EvalData(
            id="test_001",
            query="What is the capital of France?",
            golden_doc=["Paris is the capital of France."],
            reference=["Paris is the capital of France."],
            ground_truth="Paris",
        )

        assert data.id == "test_001"
        assert data.query == "What is the capital of France?"
        assert data.golden_doc == ["Paris is the capital of France."]
        assert data.reference == ["Paris is the capital of France."]
        assert data.ground_truth == "Paris"

    def test_evaldata_call_method(self):
        """Test EvalData __call__ method"""
        data = EvalData(
            id="test_002",
            query="What is 2+2?",
            golden_doc=["Basic arithmetic"],
            reference=["Basic arithmetic"],
            ground_truth="4",
        )

        assert data("id") == "test_002"
        assert data("query") == "What is 2+2?"
        assert data("golden_doc") == ["Basic arithmetic"]
        assert data("reference") == ["Basic arithmetic"]
        assert data("ground_truth") == "4"

    def test_evaldata_from_dict(self):
        """Test creating EvalData from dictionary"""
        data_dict = {
            "id": "test_003",
            "query": "What is the color of the sky?",
            "golden_doc": ["The sky is blue."],
            "reference": ["The sky is blue."],
            "ground_truth": "Blue",
        }

        data = EvalData.from_dict(data_dict)
        assert data.id == "test_003"
        assert data.query == "What is the color of the sky?"
        assert data.ground_truth == "Blue"

    def test_evaldata_get_answer(self):
        """Test get_answer method"""
        data = EvalData(
            id="test_004",
            query="What is the largest planet?",
            golden_doc=["Jupiter is the largest planet."],
            reference=["Jupiter is the largest planet."],
            ground_truth="Jupiter",
        )

        assert data.get_answer() == "Jupiter"

    def test_evaldata_jsonl_io(self):
        """Test JSONL save and load functionality"""
        test_data = [
            EvalData(
                id="test_005",
                query="What is the capital of Japan?",
                golden_doc=["Tokyo is the capital of Japan."],
                reference=["Tokyo is the capital of Japan."],
                ground_truth="Tokyo",
            ),
            EvalData(
                id="test_006",
                query="What is the capital of Germany?",
                golden_doc=["Berlin is the capital of Germany."],
                reference=["Berlin is the capital of Germany."],
                ground_truth="Berlin",
            ),
        ]

        # Test saving to JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            EvalData.to_jsonl(test_data, temp_path)
            assert os.path.exists(temp_path)

            # Test loading from JSONL
            loaded_data = EvalData.from_jsonl(temp_path)
            assert len(loaded_data) == 2
            assert loaded_data[0].id == "test_005"
            assert loaded_data[1].id == "test_006"
            assert loaded_data[0].ground_truth == "Tokyo"
            assert loaded_data[1].ground_truth == "Berlin"

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataPreprocess:
    """Test DataPreprocess class functionality"""

    def test_datapreprocess_initialization(self):
        """Test DataPreprocess initialization"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Based on the following documents: {docs}\n\nQuestion: {query}",
                "random_seed": 42,
            }
            json.dump(config_data, f)
            config_path = f.name

        # Create a temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data_content = '{"id": "test", "query": "test", "golden_doc": ["test"], "reference": ["test"], "ground_truth": "test"}'
            f.write(data_content)
            data_path = f.name

        try:
            # Test initialization with custom config and data
            preprocessor = DataPreprocess(
                prompt_config_path=config_path, data_path=data_path
            )

            assert preprocessor.name == "general"
            assert (
                preprocessor.prompt_config["system_prompt"]
                == "You are a helpful assistant."
            )
            assert (
                preprocessor.prompt_config["user_prompt"]
                == "Based on the following documents: {docs}\n\nQuestion: {query}"
            )
            assert preprocessor.prompt_config["random_seed"] == 42

        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(data_path):
                os.unlink(data_path)

    def test_datapreprocess_generate_prompt(self):
        """Test prompt generation"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": "Based on the following documents: {docs}\n\nQuestion: {query}",
                "random_seed": 42,
            }
            json.dump(config_data, f)
            config_path = f.name

        # Create a temporary data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data_content = '{"id": "test", "query": "test", "golden_doc": ["test"], "reference": ["test"], "ground_truth": "test"}'
            f.write(data_content)
            data_path = f.name

        try:
            preprocessor = DataPreprocess(
                prompt_config_path=config_path, data_path=data_path
            )

            query = "What is the capital of France?"
            docs = ["Paris is the capital of France.", "France is a country in Europe."]

            prompt = preprocessor.generate_prompt(query, docs)

            assert len(prompt) == 2
            assert prompt[0]["role"] == "system"
            assert prompt[0]["content"] == "You are a helpful assistant."
            assert prompt[1]["role"] == "user"
            assert "Paris is the capital of France" in prompt[1]["content"]
            assert "What is the capital of France?" in prompt[1]["content"]

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(data_path):
                os.unlink(data_path)


if __name__ == "__main__":
    pytest.main([__file__])
