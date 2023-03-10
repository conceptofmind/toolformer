# Adapted from https://github.com/CarperAI/trlx/blob/93c90cbdc3c6b463f565b09340ca1f74271285c5/examples/hh/triton_config.pbtxt

import argparse
import os
from string import Template

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Path to HF checkpoint with the base model"
)

parser.add_argument(
    "--max-batch-size", type=int, default=4, help="Maximum batch size for inference"
)

parser.add_argument(
    "--revision",
    type=str,
    required=False,
    help="Optional branch/commit of the HF checkpoint",
)

parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

device = torch.device(args.device)


class ModelLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids).logits


class InferModel(nn.Module):
    def __init__(self, traced_model):
        super().__init__()
        self.traced_model = traced_model

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        tensor_of_seq_len: torch.Tensor,
        temperature: torch.Tensor,
    ):
        for _ in range(tensor_of_seq_len.shape[1] - 1):
            logits = self.traced_model(input_ids)
            next_token = torch.multinomial(
                torch.softmax(logits[:, -1, :] / temperature, dim=-1), 1
            ).squeeze(1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        # in TorchScript, the above logits var lifetime doesn't escape the loop's scope
        logits = self.traced_model(input_ids).float()
        next_token = torch.multinomial(
            torch.softmax(logits[:, -1, :] / temperature, dim=-1), 1
        ).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return input_ids.int(), logits


print(f"Converting {args.model} to TorchScript...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = ModelLogits(AutoModelForCausalLM.from_pretrained(args.model))
model.eval()
model.requires_grad_(False)
model = model.half().to(device)

input = tokenizer("annotator model's hash is 0x", return_tensors="pt").to(device)
print(f"{model(input.input_ids)=}")

traced_script_module = torch.jit.trace(model, input.input_ids)

print(f"{traced_script_module(input.input_ids)=}")

print("Scripting generation wrapper...")

scripted_generator_model = torch.jit.script(InferModel(traced_script_module))

print(f"{input.input_ids=}")
x = input.input_ids, torch.empty(1, 5).cuda(), torch.full([1, 1], 1.0).cuda()
print(f"{(scripted_generator_model(*x))=}")
print(f"{tokenizer.decode(scripted_generator_model(*x)[0][0])=}")

sanitized_name = args.model.replace("/", "--")
print("Model renamed to ", sanitized_name)

print("Saving TorchScript model...")

os.makedirs(f"model_store/{sanitized_name}/1", exist_ok=True)
scripted_generator_model.save(f"model_store/{sanitized_name}/1/traced-model.pt")

config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "triton_config.pbtxt"
)
with open(config_path) as f:
    template = Template(f.read())
config = template.substitute(
    {"model_name": sanitized_name, "max_batch_size": args.max_batch_size}
)
with open(f"model_store/{sanitized_name}/config.pbtxt", "w") as f:
    f.write(config)
