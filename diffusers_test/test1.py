from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler

from safetensors.torch import load_file
import lora
import torch
import time
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/ec2-user/github/stable-diffusion-v1-5"
model_id = "/home/ec2-user/github/sd_trt/sd_model/my_down"
# pipeline = DiffusionPipeline.from_pretrained(model_id)
model_id = "/home/ec2-user/github/sd_trt/vivid_paina"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# unet_prefix = "lora_unet"
# text_encoder_prefix = "lora_te"
# lora_path = "/home/ec2-user/github/sd_trt/lora_model/blindbox.safetensors"
# alpha = 1.0

# pipeline = lora.load_lora(pipeline, lora_path, LORA_PREFIX_UNET=unet_prefix, LORA_PREFIX_TEXT_ENCODER=text_encoder_prefix, alpha=alpha)

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

print(pipeline.scheduler)

pipeline.to("cuda")

text = "masterpiece, best quality, 1girl, vivid_paina, blue hair, brown eyes, short hair"
text = "a nude girl"
text = "a good girl"
text = "a sleep girl"
neg = "ugly, bad face, fused hand, fused feet, worst quality, low quality, bad hands, missing fingers, weapon, sword, holding, text, signature"

warmup_count = 10
for i in range(10):
    image = pipeline(prompt=text, negative_prompt=neg, guidance_scale=7.0, num_inference_steps=20, height=768, width=512, generator=torch.Generator(device="cuda").manual_seed(91652449)).images[0]

torch.cuda.synchronize()
start = time.time()

image = pipeline(prompt=text, negative_prompt=neg, guidance_scale=7.0, num_inference_steps=20, height=768, width=512, generator=torch.Generator(device="cuda").manual_seed(91652449)).images[0]

torch.cuda.synchronize()
end = time.time()
print(f"cost: {end - start}")

import random
name = random.randint(1000, 9999)
name = str(name) + ".png"

image.save(name)
print("[torch] image save: ", name)