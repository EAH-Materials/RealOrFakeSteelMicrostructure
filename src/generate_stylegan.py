import pickle
import torch
import datetime
import os

from PIL import Image

def generate_fakes(G, save_path, count):
    torch.manual_seed(0)
    for i in range(count):
        generate_fake(G, save_path, i)

def generate_fake(G, save_path, i):
    z = torch.randn([1, G.z_dim]).cuda()

    c = None # class labels not used

    img = G(z, c)
    img = img.squeeze()
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    im = Image.fromarray(img)
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{save_path}/generated_{name}_{i}.png"
    im.save(save_path)

def main():
    model_name = "network-snapshot-005000.pkl"
    base_path = f"model"
    model_path = f"{base_path}/{model_name}"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    generator = model['G_ema'].cuda()
    generator.eval()

    save_path = os.path.join(base_path, "generated_fakes")
    os.makedirs(save_path, exist_ok=True)

    count = 10
    with torch.no_grad():
        generate_fakes(generator, save_path, count) 

if __name__ == "__main__":
    main()