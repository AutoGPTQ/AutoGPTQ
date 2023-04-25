import json
import random
import time
from argparse import ArgumentParser

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer, TextGenerationPipeline


def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--save_and_reload", action="store_true")
    parser.add_argument("--fast_tokenizer", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=True
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size)
    )

    examples = load_data("dataset/alpaca_data_cleaned.json", tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]

    model.quantize(examples_for_quant)

    if not args.quantized_model_dir:
        args.quantized_model_dir = args.pretrained_model_dir

    if args.save_and_reload:
        model.save_quantized(args.quantized_model_dir)
        model = AutoGPTQForCausalLM.from_quantized(args.quantized_model_dir, device="cuda:0")

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda:0")
    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt']}")
        print(f"origin: {example['output']}")
        start = time.time()
        generated_text = pipeline(
            example['prompt'],
            return_full_text=False,
            num_beams=1,
            max_length=len(example["input_ids"]) + 128  # use this instead of max_new_token to disable UserWarning when integrate with logging
        )[0]['generated_text']
        end = time.time()
        print(f"quant: {generated_text}")
        num_new_tokens = len(tokenizer(generated_text)["input_ids"])
        print(f"generate {num_new_tokens} tokens using {end-start: .4f}s")
        print("=" * 42)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
