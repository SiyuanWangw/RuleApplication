import sys
import torch
print(torch.__version__)
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class LLamaGenerator:
    def __init__(self, model_name, max_new_tokens=1024) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.half() 
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
           self.model = torch.compile(self.model)
        
        self.generation_config = GenerationConfig(
            num_return_sequences=1,
            pad_token_id=0
        )
        self.max_new_tokens = max_new_tokens

    def model_generate(self, system_input, prompt):
        prompt = system_input + "\n" + prompt
        messages = [
            {"role": "user", "content": prompt}
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = encodeds.to(device)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s).split("<|start_header_id|>assistant<|end_header_id|>")[1].split('<|eot_id|>')[0].strip()
        return output