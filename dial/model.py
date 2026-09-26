import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dial.heads import FiLMHead, make_head

DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
HEAD_ARMS = ('dial', 'film', 'film_tanh', 'film_clamp')
ARMS = HEAD_ARMS + ('temp', 'lora', 'caa')


def chat(prompt):
    return f'Human: {prompt}\n\nAssistant:'


def load_base(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map='auto')
    return tok, model


class SteeredLM:
    def __init__(self, model, tok, arm, head=None, T=5.0, caa_direction=None, caa_layer=16):
        self.tok, self.arm, self.head, self.T = tok, arm, head, T
        inner = model.get_base_model() if isinstance(model, PeftModel) else model
        self.body, self.lm_head = inner.model, inner.lm_head
        self.device = self.lm_head.weight.device

        if arm == 'caa':
            self._c = 0.0
            self.direction = caa_direction.float().to(self.device)
            self.body.layers[caa_layer].register_forward_hook(self._caa_hook)

    def _caa_hook(self, module, inputs, output):
        h = output[0]
        return (h + self._c * self.direction.to(h.dtype),) + tuple(output[1:])

    def logits(self, input_ids, c, attention_mask=None, past=None, use_cache=False):
        if self.arm == 'caa':
            self._c = float(c)
        out = self.body(input_ids=input_ids, attention_mask=attention_mask,
                        past_key_values=past, use_cache=use_cache, return_dict=True)
        h = out.last_hidden_state
        if self.head is not None:
            y = self.head(h.float(), c.float()).to(h.dtype)
            h = y if isinstance(self.head, FiLMHead) else h + y
        logits = self.lm_head(h)
        if self.arm == 'temp':
            logits = logits * (1 + c.view(-1, 1, 1) / self.T).to(logits.dtype)
        return logits, out.past_key_values

    @torch.no_grad()
    def generate(self, prompt, c, max_new):
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(self.device)
        c = torch.tensor([float(c)], device=self.device)
        past, cur, new = None, ids, []
        for _ in range(max_new):
            logits, past = self.logits(cur, c, past=past, use_cache=True)
            cur = logits[:, -1].argmax(-1, keepdim=True)
            if cur.item() == self.tok.eos_token_id:
                break
            new.append(cur.item())
        return self.tok.decode(new, skip_special_tokens=True)

    @torch.no_grad()
    def final_state(self, prompt):
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(self.device)
        return self.body(input_ids=ids, return_dict=True).last_hidden_state.float()


def load_steered(arm, model_name=DEFAULT_MODEL, adapter=None, head_path=None, T=5.0, cap=0.5,
                 caa_path=None, caa_layer=16):
    tok, base = load_base(model_name)
    device = base.lm_head.weight.device
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()

    head = None
    if arm in HEAD_ARMS:
        if head_path is None:
            raise ValueError(f'arm {arm} needs a head checkpoint')
        head = make_head(arm, base.config.hidden_size, T, cap)
        head.load_state_dict(torch.load(head_path, map_location='cpu'))
        head.to(device).eval()

    direction = None
    if arm == 'caa':
        direction = torch.load(caa_path, map_location='cpu')['direction_unit']

    return SteeredLM(model, tok, arm, head, T, direction, caa_layer)
