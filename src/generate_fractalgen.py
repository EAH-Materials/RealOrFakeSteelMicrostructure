import os

import torch
import numpy as np
from models import fractalgen
from torchvision.utils import save_image
from tqdm import tqdm
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

model_type = "fractalmar_base_in256"
num_conds = 5
model = fractalgen.__dict__[model_type](
    guiding_pixel=True,
    num_conds=num_conds,
    class_num=30
).to(device)

model_path = "/home/models/fractalgen/"
state_dict = torch.load(os.path.join(model_path, "checkpoint-last.pth"))["model"]
model.load_state_dict(state_dict)
model.eval() # important!

seed = 0 
torch.manual_seed(seed)
np.random.seed(seed)
class_labels = np.arange(30) 
num_iter_list = 64, 16, 16
cfg_scale = 1 
cfg_schedule = "constant" 
temperature = 1.1
filter_threshold = 1e-3
samples_per_row = 3 

for i in tqdm(class_labels):
  for n in range(10):
    print(i,n)
    label_gen = torch.Tensor([i]).long().cuda()
    class_embedding = model.class_emb(label_gen)
    if not cfg_scale == 1.0:
      class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(label_gen.size(0), 1)], dim=0)

    with torch.no_grad():
      with torch.cuda.amp.autocast():
        sampled_images = model.sample(
          cond_list=[class_embedding for _ in range(num_conds)],
          num_iter_list=num_iter_list,
          cfg=cfg_scale, cfg_schedule=cfg_schedule,
          temperature=temperature,
          filter_threshold=filter_threshold,
          fractal_level=0,
          visualize=False)

    # Denormalize images.
    pix_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
    pix_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
    sampled_images = sampled_images * pix_std + pix_mean
    sampled_images = sampled_images.detach().cpu()

    # Save & display images
    save_image(sampled_images, os.path.join(model_path, f"samples_{i}_{n}.png"), nrow=int(samples_per_row), normalize=True, value_range=(0, 1))