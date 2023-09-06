from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

parser = argparse.ArgumentParser(description='Simple AutoGPTQ example')
parser.add_argument('model_name_or_path', type=str, help='Model folder or repo')
parser.add_argument('--model_basename', type=str, help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_slow', action="store_true", help='Use slow tokenizer')
parser.add_argument('--use_safetensors', action="store_true", help='Model file basename if model is not named gptq_model-Xb-Ygr')
parser.add_argument('--use_triton', action="store_true", help='Use Triton for inference?')
parser.add_argument('--bits', type=int, default=4, help='Specify GPTQ bits. Only needed if no quantize_config.json is provided')
parser.add_argument('--group_size', type=int, default=128, help='Specify GPTQ group_size. Only needed if no quantize_config.json is provided')
parser.add_argument('--desc_act', action="store_true", help='Specify GPTQ desc_act. Only needed if no quantize_config.json is provided')

args = parser.parse_args()

quantized_model_dir = args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=not args.use_slow)

try:
   quantize_config = BaseQuantizeConfig.from_pretrained(quantized_model_dir)
except:
    quantize_config = BaseQuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            desc_act=args.desc_act
        )

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
        use_safetensors=True,
        model_basename=args.model_basename,
        device="cuda:0",
        use_triton=args.use_triton,
        quantize_config=quantize_config)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

prompt = "Tell me about AI"
prompt_template=f'''### Human: {prompt}
### Assistant:'''

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

