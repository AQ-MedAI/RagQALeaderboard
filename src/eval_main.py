import argparse
import json
from typing import List, Tuple

from .data import EvalResult
from .data_preprocess import NAME_TO_CLASS
from .logger import get_logger

logger = get_logger()
SUPPORTED_DATASET = list(NAME_TO_CLASS.keys())


def get_eval(args):
    model_name = args.model_name
    model_path = args.model_path
    output_path = args.output_path

    shuffle = args.shuffle
    batch_size = args.batch_size
    temperature = args.temperature
    total_doc_number = args.total_doc_number
    # breakpoint()
    # 采样
    # ragdata.data = ragdata.data[:1]
    if "http" in model_path:
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            from .models import APIModel

            model = APIModel(
                url=model_path,
                model=model_name,
                api_key=args.api_key,
                inference_mode=args.inference_mode,
            )
        else:
            from .models import OpenAIModel

            model = OpenAIModel(
                url=model_path,
                model=model_name,
                api_key=args.api_key,
                inference_mode=args.inference_mode,
            )

    else:
        if not args.inference_mode:
            if "qwen3" in model_name.lower():
                from .models import Qwen3Vllm

                model = Qwen3Vllm(plm=model_path, think_mode=False)
            else:
                from .models import CommonModelVllm

                model = CommonModelVllm(plm=model_path)
        else:
            from .models import InferModelVllm

            model = InferModelVllm(plm=model_path)

    return_result_dict = {}
    for eval_dataset_name in args.eval_dataset:
        ragdata = NAME_TO_CLASS[eval_dataset_name]()

        idxs, queries, prompts, answers = ragdata.generate_input(
            shuffle=shuffle, total_doc_number=total_doc_number
        )
        predictions = model.batch_generate(prompts, temperature, batch_size=batch_size)

        rageval = ragdata.get_corresponding_eval_type()()
        for i in range(len(idxs)):
            result = EvalResult(
                id=idxs[i],
                query=queries[i],
                prompt=prompts[i],
                answer=answers[i],
                prediction=predictions[i],
            )
            rageval.add_result(result)

        rageval.save_results(
            f"{output_path}/{model_name}_eval_result_{ragdata.name}_#{total_doc_number}.jsonl"
        )

        return_result_dict[eval_dataset_name] = rageval.results

    print(
        f"{output_path}/{model_name}_eval_result_{ragdata.name}_#{total_doc_number}.jsonl"
    )
    # print("acc_scores:", eval_results.acc_scores)
    return return_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total_doc_number",
        type=int,
        default=30,
        help="total doc number when infering",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="api key of api models"
    )
    parser.add_argument("--model-name", type=str, default="Qwen", help="model name")
    parser.add_argument(
        "--inference-mode",
        type=bool,
        default=False,
        help="whether inference model or not",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="api key of api models or local model path",
    )
    parser.add_argument(
        "--eval-dataset",
        nargs="+",
        choices=SUPPORTED_DATASET,
        default=SUPPORTED_DATASET,
        help="Specify which datasets to evaluate. Must be one or more of 'a', 'b', 'c'. (e.g., --eval-dataset a c)",
    )
    parser.add_argument("--output_path", type=str, default="./", help="output path")
    parser.add_argument(
        "--custom_config",
        type=str,
        default=None,
        help="custom config path",
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="rate of noisy passages"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="rate of correct passages"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="number of external passages",
    )
    parser.add_argument("--gpu", type=int, default=8, help="number of iterations")
    args = parser.parse_args()
    get_eval(args)
