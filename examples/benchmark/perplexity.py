import os
import argparse
from auto_gptq.utils import Perplexity

if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    python examples/benchmark/perplexity.py \
        --model_name TheBloke/open-llama-7b-open-instruct-GPTQ \
        --model_basename gptq_model-4bit-128g \
        --is_quantized
    
    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='gpt2', help="Model name.")
    parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use.")
    parser.add_argument("--dataset_path", type=str, default='wikitext', help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default='test', help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default='text', help="Column in the dataset containing the text.")
    parser.add_argument("--is_quantized", action=argparse.BooleanOptionalAction, default=False, help="Is the model GPTQ quantized?")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.is_quantized:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name,
            model_basename=args.model_basename,
            use_safetensors=True,
            trust_remote_code=True
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    ppl = Perplexity(model, tokenizer, args.device, args.dataset_path, args.dataset_name, args.split, args.text_column)
    ppl.calculate_perplexity(args.n_ctx, args.n_batch)
