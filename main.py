import PIL
import requests
import torch
from io import BytesIO
import numpy as np
import cv2
import os
import sys
from diffusers import StableDiffusionInpaintPipeline


def run_inpainting(img, mask, prompt):
    """
    LDM based inpainting
    img, mask: PIL Image
    return inpainted img
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    height = int(np.floor(img.height/8)*8)
    width = int(np.floor(img.width/8)*8)
    inpainted = pipe(prompt=prompt, image=img, mask_image=mask, height=height, width=width, strength=1).images[0]
    return inpainted





def inpaint_video(img_dir, mask_dir, prompt, out_dir):
    """
    Run LDM based inpainting is a frame-wise fashion
    We assume that mask images are png
    """
    img_name_list = os.listdir(img_dir)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img_id, ext = os.path.splitext(img_name)
        mask_path = os.path.join(mask_dir, img_id+'.png')
        img = PIL.Image.open(img_path)
        img = img.convert('RGB')
        mask = PIL.Image.open(mask_path)
        mask = mask.convert('RGB')
        inpainted = run_inpainting(img=img,
                                   mask=mask,
                                   prompt=prompt)
        out_path = os.path.join(out_dir, img_id+'.png')
        inpainted.save(out_path)



def task_inpaint_video_cmd():
    """
    Do the video inpainting using the command line args
    """
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]
    prompt = sys.argv[4]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inpaint_video(img_dir=img_dir,
                  mask_dir=mask_dir,
                  prompt=prompt,
                  out_dir=out_dir)




def task_inpaint_group_photo():
    img_path = "/workspace/shared-dir/zzhu/tmp/20240102/group.png"
    mask_path = "/workspace/shared-dir/zzhu/tmp/20240102/group_mask.png"
    out_path = "/workspace/shared-dir/zzhu/tmp/20240102/group_inpainted.png"
    prompt = "A nice restaurant, high resolution, at the corner of a strip mall"
    img = PIL.Image.open(img_path)
    img = img.convert('RGB')
    mask = PIL.Image.open(mask_path)
    mask = mask.convert('RGB')
    inpainted = run_inpainting(img=img,
                               mask=mask,
                               prompt=prompt)
    inpainted.save(out_path)


def run_outpainting_1024_512(img, prompt):
    """
    Outpaint an 512x512 image to 1024x512
    """
    img_left = np.zeros((512, 512, 3), dtype=np.uint8)
    img_left[:, 256:, :] = img[:, :256, :]
    img_left = PIL.Image.fromarray(img_left)

    mask_left = np.zeros((512, 512), dtype=np.uint8)
    mask_left[:, :256] = 255
    mask_left = PIL.Image.fromarray(mask_left)

    img_right = np.zeros((512, 512, 3), dtype=np.uint8)
    img_right[:, :256, :] = img[:, 256:, :]
    img_right = PIL.Image.fromarray(img_right)

    mask_right = np.zeros((512, 512), dtype=np.uint8)
    mask_right[:, 256:] = 255
    mask_right = PIL.Image.fromarray(mask_right)

    outpainted_left = run_inpainting(img=img_left,
                                     mask=mask_left,
                                     prompt=prompt)
    outpainted_right = run_inpainting(img=img_right,
                                      mask=mask_right,
                                      prompt=prompt)

    outpainted = PIL.Image.new("RGB", (1024, 512), "white")
    outpainted.paste(outpainted_left, (0, 0))
    outpainted.paste(outpainted_right, (512, 0))
    return outpainted_left, outpainted_right, outpainted


def multiband_blending(img_list, mask_list):
    pass


def task_outpaint_1():
    img_path = "/workspace/shared-dir/zzhu/tmp/20240102/group.png"
    out_path = "/workspace/shared-dir/zzhu/tmp/20240102/group_outpainted.png"
    prompt = "A nice restaurant, high resolution, at the corner of a strip mall"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_l, out_r, out = run_outpainting_1024_512(img=img,
                                   prompt=prompt)
    out_l.save("/workspace/shared-dir/zzhu/tmp/20240102/group_outpainted_l.png")
    out_r.save("/workspace/shared-dir/zzhu/tmp/20240102/group_outpainted_r.png")
    out.save(out_path)


def main():
    #task_inpaint_group_photo()
    #task_outpaint_1()
    task_inpaint_video_cmd()

if __name__ == "__main__":
    main()
