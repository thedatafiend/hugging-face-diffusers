from tkinter import image_names
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-3", 
    scheduler=lms,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=True
).to("cuda")

prompt = "the ottboss owns everyone in diablo"
with autocast("cuda"):
    images = pipe(prompt)["sample"]
    
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

image_grid(images, 2, 2)
# image.save("ottbossv2.png")