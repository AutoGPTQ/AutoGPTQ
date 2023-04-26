from argparse import ArgumentParser

from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--quantized_model_dir", type=str, help="Directory that saves quantized model.")
    parser.add_argument("--repo_id", type=str, help="The name of the repository you want to push to.")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Directory that saves tokenizer, defaults to None, will not upload tokenizer if not specified."
    )
    parser.add_argument("--commit_message", type=str, default=None, help="Message to commit while pushing.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Which device to load the model."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether or not the repository created should be private."
    )
    parser.add_argument(
        "--use_temp_dir",
        action="store_true",
        help="Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub."
    )
    args = parser.parse_args()

    push_to_hub_kwargs = {
        "repo_id": args.repo_id,
        "commit_message": args.commit_message,
        "private": args.private,
        "use_temp_dir": args.use_temp_dir
    }

    model = AutoGPTQForCausalLM.from_quantized(args.quantized_model_dir, device=args.device)
    model.push_to_hub(**push_to_hub_kwargs)
    model.config.push_to_hub(**push_to_hub_kwargs)
    model.quantize_config.push_to_hub(**push_to_hub_kwargs)

    if args.tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        tokenizer.push_to_hub(**push_to_hub_kwargs)


if __name__ == "__main__":
    main()
