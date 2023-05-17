from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler

from safetensors.torch import load_file
import lora
import torch

model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/ec2-user/github/stable-diffusion-v1-5"
model_id = "/home/ec2-user/github/sd_trt/sd_model/my_down"
# pipeline = DiffusionPipeline.from_pretrained(model_id)
model_id = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

unet_prefix = "lora_unet"
text_encoder_prefix = "lora_te"
lora_path = "/home/ec2-user/github/sd_trt/lora_model/blindbox.safetensors"
alpha = 1.0

# pipeline = lora.load_lora(pipeline, lora_path, LORA_PREFIX_UNET=unet_prefix, LORA_PREFIX_TEXT_ENCODER=text_encoder_prefix, alpha=alpha)

pipeline.to("cuda")

text = "a beautiful girl"
image = pipeline(prompt=text, num_inference_steps=20, generator=[torch.Generator(device="cuda").manual_seed(1)]).images[0]

image.save("2.png")