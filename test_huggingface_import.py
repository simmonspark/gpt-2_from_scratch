"""
Ensure that we can load huggingface/transformer GPTs into minGPT
"""

import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
# -----------------------------------------------------------------------------

class TestHuggingFaceImport(unittest.TestCase):

    def test_gpt2(self):
        model_type = 'gpt2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prompt = "이번 국정감사에서는 "
        
        # create a minGPT and a huggingface/transformers model
        from mingpt.model import GPT
        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257  # openai's model vocabulary
        model_config.block_size = 1024
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # init a HF model too
        model = GPT(model_config)
        model.load_state_dict(torch.load("./model.pt", map_location=device))
        # ship both to device
        model.to(device)
        model_hf.to(device)

        # set both to eval mode
        model.eval()
        model_hf.eval()

        # tokenize input prompt
        # ... with mingpt
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        model_hf.config.pad_token_id = model_hf.config.eos_token_id  # suppress a warning
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)

        x1 = encoded_input['input_ids']
        x2 = encoded_input['input_ids']


        logits1, loss = model(x1)
        logits2 = model_hf(x2).logits


        # now draw the argmax samples from each
        y1 = model.generate(x1, max_new_tokens=1024, do_sample=True)[0]
        y2 = model_hf.generate(x2, max_new_tokens=100, do_sample=True)[0]

        out1 = tokenizer.decode(y1.cpu().squeeze())
        out2 = tokenizer.decode(y2.cpu().squeeze())
        print(out1)


if __name__ == '__main__':
    unittest.main()
