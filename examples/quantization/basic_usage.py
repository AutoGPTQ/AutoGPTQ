import torch
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM


use_bitblas = True
# pretrained_model_dir = "facebook/opt-125m"
# pretrained_model_dir = "dist/opt-125m"
# quantized_model_dir = "dist/opt-125m-4bit-128g"
# quantized_model_dir = "dist/opt-125m-4bit-128g-BitBLAS-original"
# quantized_model_dir = "dist/opt-125m-4bit-128g-BitBLAS-quantized"
# pretrained_model_dir = "dist/Llama-2-70B-Chat-BitBLAS"
# quantized_model_dir = "dist/Llama-2-70B-Chat-BitBLAS"
# quantized_model_dir = "dist/Wizard-Vicuna-7B-Uncensored-BitBLAS-propagate-all"
# pretrained_model_dir = "dist/Wizard-Vicuna-7B-Uncensored-GPTQ"
# quantized_model_dir = "dist/Wizard-Vicuna-7B-Uncensored-BitBLAS"
# pretrained_model_dir = "dist/Wizard-Vicuna-7B-Uncensored-BitBLAS"
pretrained_model_dir = "dist/Llama-2-70B-Chat-BitBLAS"
quantized_model_dir = "dist/Llama-2-70B-Chat-BitBLAS"

# pretrained_model_dir = "dist/Llama-2-70B-Chat-BitBLAS-Propagate-all"
# quantized_model_dir = "dist/Llama-2-70B-Chat-BitBLAS-Propagate-all"

# pretrained_model_dir = "dist/vicuna-13b-v1.5"
# pretrained_model_dir = "dist/vicuna-13b-v1.5-INT4-BitBLAS"
# quantized_model_dir = "dist/vicuna-13b-v1.5-INT4-BitBLAS"
# pretrained_model_dir = "dist/vicuna-13b-v1.5-INT4-BitBLAS-pa"
# quantized_model_dir = "dist/vicuna-13b-v1.5-INT4-BitBLAS-pa"
# quantized_model_dir = "dist/vicuna-13b-v1.5-INT4-GPTQ"
# pretrained_model_dir = "dist/vicuna-13b-v1.5-INT4-GPTQ"

# if use_bitblas:
#     quantized_model_dir += "-BitBLAS"
# else:
#     quantized_model_dir += "-GPTQ"


def profile(model, input_data):
    import time

    import numpy as np
    model = model.cuda()
    model.eval()

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        # print("Warming up ...")
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    # examples = [
    #     tokenizer(
    #         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    #     )
    # ]

    # quantize_config = BaseQuantizeConfig(
    #     bits=4,  # quantize model to 4-bit
    #     group_size=-1,  # it is recommended to set the value to 128
    #     desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    #     is_bitblas_format=False,
    # )

    # # load un-quantized model, by default, the model will always be loaded into CPU memory
    # model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # model.quantize(examples)

    # # save quantized model
    # model.save_quantized(quantized_model_dir)
    # tokenizer.save_pretrained(quantized_model_dir)

    # push quantized model to Hugging Face Hub.
    # to use use_auth_token=True, Login first via huggingface-cli login.
    # or pass explcit token with: use_auth_token="hf_xxxxxxx"
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

    # alternatively you can save and push at the same time
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

    # save quantized model using safetensors
    # model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_bitblas=use_bitblas)
    # download quantized model from Hugging Face Hub and load to the first GPU
    # model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

    # -- simple token evaluate --
    input_ids = torch.ones((1, 1), dtype=torch.long, device="cuda:0")
    outputs = model(input_ids=input_ids)
    print("Profile ", profile(model, input_ids))
    print(f"output logits {outputs.logits.shape}: \n", outputs.logits)
    # inference with model.generate
    print(
        tokenizer.decode(
            model.generate(
                **tokenizer("auto_gptq is", return_tensors="pt").to(model.device)
            )[0]
        )
    )

    # # or you can also use pipeline
    # pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    # print(pipeline("auto-gptq is")[0]["generated_text"])

    model.save_quantized(quantized_model_dir)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
