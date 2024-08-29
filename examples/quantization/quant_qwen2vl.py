import logging

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import Qwen2VLGPTQForConditionalGeneration


# Set up logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quantize_config = BaseQuantizeConfig(
    bits=8,  # 4 or 8
    group_size=128,
    damp_percent=0.1,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
)

# Load your processor and model with AutoGPTQ
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2VLGPTQForConditionalGeneration.from_pretrained(model_path, quantize_config)


# Then you need to prepare your data for calibaration. What you need to do is just put samples into a list,
# each of which is a typical chat message as shown below. you can specify text and image in `content` field:
# dataset = [
#     # message 0
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me who you are."},
#         {"role": "assistant", "content": "I am a large language model named Qwen..."},
#     ],
#     # message 1
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "file:///path/to/your/image.jpg"},
#                 {"type": "text", "text": "Output all text in the image"},
#             ],
#         },
#         {"role": "assistant", "content": "The text in the image is balabala..."},
#     ],
#     # other messages...
#     ...,
# ]
# here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.
def prepare_dataset(n_sample: int = 20) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "generate a caption for this image"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset()


def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


calib_data = []
batch_size = 1
for batch in batched(dataset, batch_size):
    text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(batch)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    calib_data.append(inputs)


# Then just run the calibration process by one line of code:
model.quantize(calib_data, cache_examples_on_gpu=False)

# Finally, save the quantized model:
model.save_quantized(quant_path, use_safetensors=True)
processor.save_pretrained(quant_path)

# Optional: save the quantized model compatible with transformers
# from optimum.gptq import GPTQQuantizer
# from transformers.utils.quantization_config import QuantizationMethod
# gptq_quantizer = GPTQQuantizer.from_dict(model.quantize_config.to_dict())
# model.model.is_quantized = True
# model.model.quantization_method = QuantizationMethod.GPTQ
# model.model.config.quantization_config = gptq_quantizer.to_dict()
# gptq_quantizer.save(model.model, quant_path, max_shard_size="4GB", safe_serialization=True)
# processor.save_pretrained(quant_path)

# Then you can obtain your own GPTQ quantized model for deployment. Enjoy!
