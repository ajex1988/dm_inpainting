import PIL
import requests
import torch
from io import BytesIO
import numpy as np
import cv2
import os
import sys
from diffusers import StableDiffusionInpaintPipeline


def run_inpainting_generator(img, mask, prompt, generator=None):
    """
    Run LDM based inpainting with fixed generator
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    height = int(np.floor(img.height / 8) * 8)
    width = int(np.floor(img.width / 8) * 8)
    inpainted = pipe(prompt=prompt, image=img, mask_image=mask, height=height, width=width, strength=1, generator=generator).images[0]
    return inpainted


def inpaint_video_generator(img_dir, mask_dir, prompt, out_dir, generator=None):
    """
    LDM based video inpainting using generator
    """
    img_name_list = os.listdir(img_dir)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img_id, ext = os.path.splitext(img_name)
        mask_path = os.path.join(mask_dir, img_id + '.png')
        img = PIL.Image.open(img_path)
        img = img.convert('RGB')
        mask = PIL.Image.open(mask_path)
        mask = mask.convert('RGB')
        inpainted = run_inpainting_generator(img=img,
                                             mask=mask,
                                             prompt=prompt,
                                             generator=generator)
        out_path = os.path.join(out_dir, img_id + '.png')
        inpainted.save(out_path)


def task_inpaint_video_generator_cmd():
    """
    Do the video inpainting using a generator
    """
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]
    prompt = sys.argv[4]
    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(10)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inpaint_video_generator(img_dir=img_dir,
                            mask_dir=mask_dir,
                            prompt=prompt,
                            out_dir=out_dir,
                            generator=generator)
