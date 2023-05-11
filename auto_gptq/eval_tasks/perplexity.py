import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from os.path import isfile

class Perplexity:
    def __init__(self, model, tokenizer=None, text=None, dataset='wikitext', device='cuda:0'):
        #TODO: Detect if model is already instantiated, or is path to model_dir
        if type(model) == str:
            pass

        #TODO: Properly handle multiple datasets, type checking, etc.
        if text is None:
            if type(dataset) == str:
                if dataset == 'wikitext':
                    # Special handling for wikitext, so that the output precisely matches
                    # the text people are using with llama.cpp's perplexity
                    # which is a raw file load of wikitext-v2-raw-v1/wiki.test.raw from ZIP at https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
                    wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                    wikilist = [' \n' if s == '' else s for s in wikidata['text']]
                    self._text = ''.join(wikilist)
            else:
                self._text = ''.join(dataset)
        else:
            self._text = text

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def run(self, n_ctx, n_batch):
        #TODO: Tokenising the dataset takes a couple of minutes, so might be nice to persist and reload tokenised result
        #      this should be handled elsewhere, and be optional. Also needs hashing to confirm saved file is right
        #      for the passed dataset.
        saved_tokens = "/workspace/tokens.pth"
        if not isfile(saved_tokens):
            print("Tokenising")
            tokens = self._tokenizer(self._text, truncation=False, return_tensors='pt').input_ids.to(self._device)
            print("Saving tokens for later use")
            torch.save(tokens, saved_tokens)
        else:
            tokens = torch.load(saved_tokens)

        #TODO: could we tokenise in batches, to avoid having to do it all upfront?
        #      but we need to know the length of the data tokens in order to know number of batches?
        len_tokens = len(tokens[0])
        print("Length of data:", len_tokens)
        n_chunk = len_tokens // n_ctx

        n_vocab = self._model.config.vocab_size

        nll = 0.0
        count = 0

        # Algorithm duplicated from llama.cpp's perplexity so that results can be compared to the many ppl figures published already
        # https://github.com/ggerganov/llama.cpp/blob/master/examples/perplexity/perplexity.cpp
        print(f'Calculating perplexity over {n_chunk} chunks, batch_size={n_batch}')

        progress = tqdm(range(n_chunk))
        progress.set_description(f"Perplexity: - ")
        results = ""
        for i in progress:
            start = i * n_ctx
            end = start + n_ctx

            num_batches = (n_ctx + n_batch - 1) // n_batch

            logits = []

            for j in range(num_batches):
                batch_start = start + j * n_batch
                batch_size  = min(end - batch_start, n_batch)

                token_org = tokens[0][batch_start].item()

                if j == 0:
                    tokens[0][batch_start] = self._tokenizer.bos_token_id

                with torch.no_grad():
                    outputs = self._model(tokens[:, batch_start:batch_start+batch_size])
                    batch_logits = outputs.logits

                tokens[0][batch_start] = token_org

                logits.append(batch_logits.detach())

            for j in range(min(512, n_ctx // 2), n_ctx - 1):
                tok_logits = logits[0][0][j].cpu().numpy()
                prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

                nll += -np.log(prob)
                count += 1

            ppl = np.exp(nll / count)
            #TODO: this results string gives the result of each batch, and duplicates llama.cpp's output
            #      but currently nothing is being done with it - optionally return or print it?
            results += f'[{i+1}]{ppl:.4f}, '
            progress.set_description(f"Perplexity: {ppl:.4f}")

        print(f"Perplexity: {ppl:.4f}")
        #print(results)

#TODO: this is test code for the above class, to be cleaned up and removed from this file
model_dir = "/workspace/models/huggyllama_llama-7b"
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
print("Loading model")
mod = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')

ppl = Perplexity(mod, tok)

n_ctx = 512
n_batch = 512
ppl.run(n_ctx, n_batch)
