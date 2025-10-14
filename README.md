# RAGQA-Leaderboard

A standardized and fair evaluation framework and leaderboard for Retrieval-Augmented Generation (RAG) systems.

`RAGQA-Leaderboard` aims to provide researchers and developers with a unified and reproducible benchmark for evaluating the performance of RAG models. We have integrated a suite of popular and high-frequency question-answering datasets and offer a streamlined, one-click evaluation pipeline that generates detailed reports, making model comparison and analysis easier than ever.

## ✨ Key Features

*   **📊 Standardized Evaluation Framework**: Provides a unified and fair evaluation pipeline, ensuring that different models are compared under the same conditions for reproducible results.

*   **📚 Comprehensive Dataset Integration**:
    *   Integrates a wide range of popular QA datasets used in the RAG domain.
    *   Covers diverse question types including **Single-Hop**, **Multi-Hop**, and **Domain-Specific** scenarios.
    *   Includes benchmarks like `HotpotQA`, `PopQA`, `MusiqueQA`, `TriviaQA`, and more.

*   **📈 Multi-Dimensional Metrics**:
    *   Supports core evaluation metrics such as **Accuracy**, **F1 Score**, and **Exact Match** to provide a holistic view of model performance.

*   **📄 One-Click Reporting**:
    *   Generate comprehensive evaluation reports with a single command.
    *   Outputs reports in both **HTML** for easy visualization and analysis, and **JSON** for programmatic use. This makes it effortless to analyze and compare performance across different models.

*   **🧩 Modular RAG Evaluation**: Go beyond end-to-end testing. This framework allows for the isolated evaluation of individual RAG components—such as the **Retriever** and the **Generator**—enabling targeted analysis and debugging.

*   **🚀 Flexible Model Inference**:
    *   **API-based**: Evaluate models served via API endpoints (e.g., OpenAI, Anthropic, or custom-hosted models).
    *   **Local Inference**: Supports high-performance, offline evaluation of local models using libraries like **vLLM** for maximum speed and efficiency.

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/RAGQA-Leaderboard.git
   cd RAGQA-Leaderboard
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download Data

```bash
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download AQ-MedAI/RAG-OmniQA --repo-type=dataset
```

4. (Optional) Install local inference dependencies  
If you want to use local models (e.g., `transformers` or `vllm`):

```bash
pip install transformers vllm
```

5. (Optional) Install API inference dependencies  
If you want to use the OpenAI API for inference:

```bash
pip install openai
```

---

## 📋 Usage

### 1. Run Evaluation

To evaluate a model, use the following `make` command:

```bash
make eval-all
```

This will evaluate the model for all datasets specified in the `Makefile`.

If you want to evaluate a specific dataset, use:

```bash
make eval-single DATASETS="hotpotqa popqa"
```

Alternatively, you can use the Python script directly:

```bash
python eval.py --model-name "Qwen3" --model-path "/path/to/model" --eval-dataset hotpotqa popqa
```

### 2. Customize Configuration

You can modify the configuration files in the `config/` directory (e.g., `api_prompt_config_en.json`) to customize evaluation parameters.

### 3. Generate Reports

After evaluation, HTML reports and JSON results will be saved in the `reports/` directory.

---

## 📂 Project Structure

```
RAGQA-Leaderboard/
├── src/                    # Core source code
│   ├── data.py             # Data processing module
│   ├── report/             # Report generation module
│   ├── models/             # Model interface module
│   ├── eval_main.py        # Evaluation entry point
│   └── logger.py           # Logging module
├── data/                   # Datasets
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── reports/                # Output directory for evaluation reports
├── eval.py                 # Main evaluation script
├── README.md               # Project documentation
└── Makefile                # Automation scripts
```

---

## 🔍 Datasets Collection

## Dataset Construction

We have curated a comprehensive dataset by collecting and processing popular question-answering (QA) datasets. Our processing pipeline ensures that each question is paired with its corresponding **golden document(s)** and a set of **noise documents** (approximately 50). Additionally, we provide a large-scale **retrieval pool** consisting of approximately 1.09 million documents from Wikipedia.

The detailed construction process is as follows:

### 1. Question Collection

We collected a total of **30,135 queries** from three main categories of QA datasets:

*   **Single-Hop:** We adopted the data split from [MIRAGE](https://github.com/nlpai-lab/MIRAGE?tab=readme-ov-file) and selected queries from **NQ**, **TriviaQA**, and **PopQA**.
*   **Multi-Hop:** We included all queries from **HotpotQA**, **MuSiQue-Ans**, and **2WikiMultiHopQA**.
*   **Domain-Specific:** We selected 500 queries from the **PubMedQA** test set.

### 2. Golden Document Collection

*   **Single-Hop:** For these datasets, we directly used the golden documents provided in the MIRAGE project.
*   **Multi-Hop:** The HotpotQA (distractor setting), MuSiQue-Ans, and 2WikiMultiHopQA datasets inherently provide multiple golden documents for each multi-hop question, which we used directly.
*   **Domain-Specific:** For PubMedQA, questions are generated from article abstracts. We treat these source abstracts as the golden documents.

### 3. Noise Document Collection

We employed [Contriever-MS](https://github.com/facebookresearch/contriever) as our retriever. For each question, we retrieved the **top 50 documents** from our Wikipedia corpus. These retrieved documents, after removing any exact matches with the corresponding golden document(s), constitute the set of noise documents.

### 4. Retrieval Pool Construction

The final retrieval pool was constructed by merging all golden and noise documents from the entire collection and then performing a final deduplication to ensure a unique set of documents.


## 📊 Supported Datasets

- **HotpotQA**: Multi-hop question answering dataset.
- **PopQA**: Single-hop question answering dataset.
- **MusiqueQA**: Multi-hop question answering dataset.
- **TriviaQA**: General knowledge question answering dataset.
- **2Wiki**: Multi-hop question answering dataset.
- **PubmedQA**: Biomedical question answering dataset.

---

## ⚙️ Configuration

- **`config/api_prompt_config_en.json`**: Default configuration for English evaluation.
- **`config/api_prompt_config_ch.json`**: Default configuration for Chinese evaluation.
- **`config/default_prompt_config.json`**: General configuration file.

---

## 🧪 Testing

Run the following command to execute the test cases:

```bash
pytest tests/
```

---

## 🙏 Acknowledgements

We gratefully acknowledge the creators and maintainers of the publicly available datasets integrated into RAGQA-Leaderboard. Specifically:


- **HotpotQA** ([Yang et al., EMNLP 2018](https://github.com/hotpotqa/hotpot)):  
  A multi-hop question answering dataset.  
  [Paper Link](https://arxiv.org/abs/1809.09600)

- **PopQA** ([Mallen et al., ACL 2023](https://huggingface.co/datasets/akariasai/PopQA)):  
  A factoid question answering dataset.  
  [Paper Link](https://arxiv.org/abs/2212.10511)

- **MusiqueQA** ([Trivedi et al., TACL 2022](https://github.com/StonyBrookNLP/musique)):  
  Multi-hop compositional QA dataset.  
  [Paper Link](https://arxiv.org/abs/2108.00573)

- **TriviaQA** ([Joshi et al., ACL 2017](https://github.com/mandarjoshi90/triviaqa)):  
  Large-scale QA dataset.  
  [Paper Link](https://arxiv.org/abs/1705.03551)

- **2Wiki** ([Ho et al., NAACL 2021](https://github.com/amazon-science/2wikimultihop)):  
  Multi-hop complex QA dataset.  
  [Paper Link](https://arxiv.org/abs/2104.08207)

- **PubmedQA** ([Jin et al., BioRxiv 2019](https://github.com/pubmedqa/pubmedqa)):  
  Biomedical QA dataset.  
  [Paper Link](https://arxiv.org/abs/1909.06146)

These datasets are copyright of their respective authors and we use them solely for research and non-commercial evaluation purposes. Please cite their works appropriately if you use the leaderboard or these datasets in your own publication.


## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📞 Contact

- **Author**: AQ-Med Team
- **Email**: tanzhehao.tzh@antgroup.com, jiaoyihan.yh@antgroup.com