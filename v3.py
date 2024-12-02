import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class YUVImageDataset(Dataset):
    def __init__(self, yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1):
        self.yuv_dir = yuv_dir
        self.diff_map_dir = diff_map_dir
        self.image_size = image_size  # (height, width)
        self.mask_threshold = mask_threshold
        self.image_pairs = self._load_image_pairs()
        print(f"Loaded {len(self.image_pairs)} image pairs.")

    def _load_image_pairs(self):
        image_pairs = []
        for yuv_filename in os.listdir(self.yuv_dir):
            if yuv_filename.startswith("denoised_") and yuv_filename.endswith(".raw"):
                base_name = yuv_filename[len("denoised_"):-len(".raw")]
                diff_map_filename = f"difference_{base_name}.png"
                diff_map_path = os.path.join(self.diff_map_dir, diff_map_filename)
                if os.path.exists(diff_map_path):
                    yuv_path = os.path.join(self.yuv_dir, yuv_filename)
                    image_pairs.append((yuv_path, diff_map_path))
                else:
                    print(f"Warning: Difference map {diff_map_filename} not found for {yuv_filename}.")
        return image_pairs

    def load_yuv_frame(self, yuv_path):
        height, width = self.image_size
        y_size = width * height
        uv_width = width // 2
        uv_height = height // 2
        uv_size = uv_width * uv_height
    
        frame_size = int(y_size + 2 * uv_size)
    
        with open(yuv_path, 'rb') as f:
            yuv_bytes = f.read(frame_size)
            y = np.frombuffer(yuv_bytes[0:y_size], dtype=np.uint8).reshape((height, width))
            u = np.frombuffer(yuv_bytes[y_size:y_size+uv_size], dtype=np.uint8).reshape((uv_height, uv_width))
            v = np.frombuffer(yuv_bytes[y_size+uv_size:y_size+2*uv_size], dtype=np.uint8).reshape((uv_height, uv_width))
    
            # Ensure arrays are writable
            y = np.array(y, copy=True)
            u = np.array(u, copy=True)
            v = np.array(v, copy=True)
    
            # Upsample U and V to match Y dimensions using nearest neighbor
            u = u.repeat(2, axis=0).repeat(2, axis=1)
            v = v.repeat(2, axis=0).repeat(2, axis=1)
    
        return y, u, v

    def load_diff_map(self, diff_map_path):
        diff_map = Image.open(diff_map_path).convert('L')
        diff_map = diff_map.resize(self.image_size[::-1])  # PIL uses (width, height)
        return np.array(diff_map)

    def generate_mask_from_diff_map(self, diff_map):
        diff_map = diff_map / 255.0
        mask = np.where(diff_map > self.mask_threshold, 0, 1).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        yuv_path, diff_map_path = self.image_pairs[idx]
        y_channel, u_channel, v_channel = self.load_yuv_frame(yuv_path)
        diff_map = self.load_diff_map(diff_map_path)
        mask = self.generate_mask_from_diff_map(diff_map)

        # Apply mask to all channels
        masked_y_channel = y_channel.copy()
        masked_u_channel = u_channel.copy()
        masked_v_channel = v_channel.copy()

        masked_y_channel[mask == 0] = 0
        masked_u_channel[mask == 0] = 0
        masked_v_channel[mask == 0] = 0

        # Create tensors and normalize to [0,1]
        y_tensor = torch.from_numpy(y_channel).unsqueeze(0).float() / 255.0
        u_tensor = torch.from_numpy(u_channel).unsqueeze(0).float() / 255.0
        v_tensor = torch.from_numpy(v_channel).unsqueeze(0).float() / 255.0

        masked_y_tensor = torch.from_numpy(masked_y_channel).unsqueeze(0).float() / 255.0
        masked_u_tensor = torch.from_numpy(masked_u_channel).unsqueeze(0).float() / 255.0
        masked_v_tensor = torch.from_numpy(masked_v_channel).unsqueeze(0).float() / 255.0

        # Stack channels to create 3-channel images
        yuv_tensor = torch.cat([y_tensor, u_tensor, v_tensor], dim=0)
        masked_yuv_tensor = torch.cat([masked_y_tensor, masked_u_tensor, masked_v_tensor], dim=0)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat mask for each channel

        return masked_yuv_tensor, yuv_tensor, mask_tensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels changed to 3
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output channels changed to 3
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        output = self.decoder(enc)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels changed to 3
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def adversarial_loss(pred, target):
    return nn.BCELoss()(pred, target)

def reconstruction_loss(pred, target):
    return nn.L1Loss()(pred, target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
generator = nn.DataParallel(generator)

discriminator = Discriminator().to(device)
discriminator = nn.DataParallel(discriminator)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train_gan(dataloader, num_epochs=100):
    for epoch in range(num_epochs):
        for i, (masked_yuv, yuv, mask) in enumerate(dataloader):
            masked_yuv = masked_yuv.to(device)
            yuv = yuv.to(device)
            mask = mask.to(device)

            # Train Discriminator
            real_output = discriminator(yuv)
            real_labels = torch.ones_like(real_output).to(device)
            fake_labels = torch.zeros_like(real_output).to(device)

            optimizer_D.zero_grad()
            d_loss_real = adversarial_loss(real_output, real_labels)

            generated = generator(masked_yuv)
            fake_output = discriminator(generated.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(generated)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            g_rec_loss = reconstruction_loss(generated * mask, yuv * mask)
            g_loss = g_adv_loss + g_rec_loss
            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


yuv_dir = '/home/msh7377/dataset_full' 
diff_map_dir = '/home/msh7377/detection_map_full'

dataset = YUVImageDataset(yuv_dir, diff_map_dir, image_size=(720, 640), mask_threshold=0.1)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced batch size due to increased data size
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

train_gan(train_loader, num_epochs=10)

def yuv_to_rgb(y, u, v):
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

# def visualize_results(generator, test_loader, num_images=5):
#     generator.eval()
#     with torch.no_grad():
#         count = 0
#         for masked_yuv, yuv, mask in test_loader:
#             masked_yuv = masked_yuv.to(device)
#             generated_yuv = generator(masked_yuv).cpu().numpy()

#             masked_yuv = masked_yuv.cpu().numpy()
#             yuv = yuv.cpu().numpy()

#             for i in range(generated_yuv.shape[0]):
#                 if count >= num_images:
#                     return

#                 gen_yuv = generated_yuv[i] * 255.0
#                 mask_img = masked_yuv[i] * 255.0
#                 original_yuv = yuv[i] * 255.0

#                 # Convert to uint8
#                 gen_yuv = gen_yuv.astype(np.uint8)
#                 mask_img = mask_img.astype(np.uint8)
#                 original_yuv = original_yuv.astype(np.uint8)

#                 # Convert YUV to RGB
#                 gen_rgb = yuv_to_rgb(gen_yuv[0], gen_yuv[1], gen_yuv[2])
#                 mask_rgb = yuv_to_rgb(mask_img[0], mask_img[1], mask_img[2])
#                 original_rgb = yuv_to_rgb(original_yuv[0], original_yuv[1], original_yuv[2])

#                 fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#                 axs[0].imshow(mask_rgb)
#                 axs[0].set_title("Masked Image")
#                 axs[0].axis('off')

#                 axs[1].imshow(gen_rgb)
#                 axs[1].set_title("Generated Image")
#                 axs[1].axis('off')

#                 axs[2].imshow(original_rgb)
#                 axs[2].set_title("Original Image")
#                 axs[2].axis('off')

#                 plt.show()
#                 count += 1

# visualize_results(generator, test_loader, num_images=5)

save_path = '/home/msh7377/Muneeb/Models'
os.makedirs(save_path, exist_ok=True)
torch.save(generator.state_dict(), os.path.join(save_path, 'Generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(save_path, 'Discriminator.pth'))
print("Models saved successfully.")
