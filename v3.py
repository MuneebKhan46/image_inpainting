
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


device_ids = [0, 1]  # Use both GPUs
device = torch.device(f"cuda:{device_ids[0]}")


class YUVImageDataset(Dataset):
    def __init__(self, yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1, mode='train', triplets_list=None):
        self.yuv_dir = yuv_dir
        self.diff_map_dir = diff_map_dir
        self.image_size = image_size  # (height, width)
        self.mask_threshold = mask_threshold
        self.mode = mode

        if triplets_list is None:
            self.image_triplets = self._load_image_triplets()
        else:
            self.image_triplets = triplets_list
        print(f"Loaded {len(self.image_triplets)} image triplets for mode {self.mode}.")

    def _load_image_triplets(self):
        image_triplets = []
        for filename in os.listdir(self.yuv_dir):
            if filename.startswith('original_') and filename.endswith('.raw'):
                base_name = filename[len('original_'):-len('.raw')]
                original_path = os.path.join(self.yuv_dir, filename)
                denoised_filename = f'denoised_{base_name}.raw'
                denoised_path = os.path.join(self.yuv_dir, denoised_filename)
                diff_map_filename = f'difference_{base_name}.png'
                diff_map_path = os.path.join(self.diff_map_dir, diff_map_filename)

                if os.path.exists(denoised_path) and os.path.exists(diff_map_path):
                    image_triplets.append((original_path, denoised_path, diff_map_path))
                else:
                    print(f"Missing files for base name {base_name}")
        return image_triplets

    def load_y_channel(self, yuv_path):
        height, width = self.image_size  # Note the order: height, width
        y_size = width * height
        with open(yuv_path, 'rb') as f:
            y_channel = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((height, width))
        return y_channel

    def _generate_mask(self, diff_map_path=None):
        height, width = self.image_size  # Note the order: height, width
        
        if diff_map_path is None:
            # Generate random mask
            mask = np.ones((height, width), dtype=np.uint8)
            num_shapes = np.random.randint(5, 15)
            for _ in range(num_shapes):
                x = np.random.randint(0, width - 200)
                y = np.random.randint(0, height - 200)
                w = np.random.randint(10, 200)
                h = np.random.randint(10, 200)
                mask[y:y+h, x:x+w] = 0
            return mask
        else:
            diff_map = Image.open(diff_map_path).convert('L')
            diff_map = diff_map.resize((width, height))  # Note the order: width, height for PIL
            mask = np.array(diff_map) / 255.0
            return np.where(mask > self.mask_threshold, 0, 1).astype(np.uint8)

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        original_path, denoised_path, diff_map_path = self.image_triplets[idx]

        if self.mode == 'train':
            y_channel = self.load_y_channel(original_path)
            ground_truth_y_channel = y_channel.copy()
            mask = self._generate_mask()
        elif self.mode == 'test':
            y_channel = self.load_y_channel(denoised_path)
            ground_truth_y_channel = self.load_y_channel(original_path)
            mask = self._generate_mask(diff_map_path)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        assert y_channel.shape == mask.shape, f"Shape mismatch: y_channel {y_channel.shape} vs mask {mask.shape}"
        
        masked_y_channel = y_channel.copy()
        masked_y_channel[mask == 0] = 0

        y_tensor = torch.from_numpy(ground_truth_y_channel).unsqueeze(0).float() / 255.0
        masked_y_tensor = torch.from_numpy(masked_y_channel).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return masked_y_tensor, y_tensor, mask_tensor, y_channel

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
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
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        output = self.decoder(enc)
        return output




class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # Up to relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # Up to relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])  # Up to relu3_3
        for param in self.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3]




def perceptual_loss(vgg, predicted, ground_truth):
    predicted_rgb = predicted.repeat(1, 3, 1, 1)
    ground_truth_rgb = ground_truth.repeat(1, 3, 1, 1)
    features_pred = vgg(predicted_rgb)
    features_gt = vgg(ground_truth_rgb)
    loss = 0
    for f_pred, f_gt in zip(features_pred, features_gt):
        loss += nn.functional.l1_loss(f_pred, f_gt)
    return loss


def gram_matrix(features):
    (b, ch, h, w) = features.size()
    features = features.view(b, ch, w * h)
    G = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    return G / (ch * h * w)


def style_loss(vgg, predicted, ground_truth):
    predicted_rgb = predicted.repeat(1, 3, 1, 1)
    ground_truth_rgb = ground_truth.repeat(1, 3, 1, 1)
    features_pred = vgg(predicted_rgb)
    features_gt = vgg(ground_truth_rgb)
    loss = 0
    for f_pred, f_gt in zip(features_pred, features_gt):
        G_pred = gram_matrix(f_pred)
        G_gt = gram_matrix(f_gt)
        loss += nn.functional.l1_loss(G_pred, G_gt)
    return loss


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def combined_loss(vgg, alpha=1.0, beta=0.1, gamma=0.01):
    def loss_fn(predicted, ground_truth):
        pixel_loss = nn.functional.l1_loss(predicted, ground_truth)
        p_loss = perceptual_loss(vgg, predicted, ground_truth)
        s_loss = style_loss(vgg, predicted, ground_truth)
        tv_loss = total_variation_loss(predicted)
        return alpha * pixel_loss + beta * p_loss + gamma * (s_loss + tv_loss)
    return loss_fn




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generator = Generator().to(device)
# generator = nn.DataParallel(generator)

generator = Generator().to(device)
generator = nn.DataParallel(generator, device_ids=device_ids)


optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# vgg = VGGFeatures().to(device)
# vgg = nn.DataParallel(vgg)

vgg = VGGFeatures().to(device)
vgg = nn.DataParallel(vgg, device_ids=device_ids)


loss_function = combined_loss(vgg, alpha=1.0, beta=0.1, gamma=0.01)

yuv_dir = '/home/msh7377/dataset_full' 
diff_map_dir = '/home/msh7377/detection_map_full'

full_dataset = YUVImageDataset(yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1)

dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.25 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]


train_triplets = [full_dataset.image_triplets[i] for i in train_indices]
test_triplets = [full_dataset.image_triplets[i] for i in test_indices]

train_dataset = YUVImageDataset(yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1, mode='train', triplets_list=train_triplets)

test_dataset = YUVImageDataset(yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1, mode='test', triplets_list=test_triplets)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)



def train_model(dataloader, num_epochs=1):
    for epoch in range(num_epochs):
        generator.train()
        for i, (masked_y, y_channel, mask, _) in enumerate(dataloader):
            masked_y = masked_y.to(device)
            y_channel = y_channel.to(device)
            mask = mask.to(device)

            optimizer_G.zero_grad()
            generated = generator(masked_y)
            
            loss = loss_function(generated * mask, y_channel * mask)
            loss.backward()
            optimizer_G.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        validate_model(generator, test_loader)




def validate_model(generator, test_loader):
    generator.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for masked_y, y_channel, mask, _ in test_loader:
            masked_y = masked_y.to(device)
            y_channel = y_channel.to(device)
            generated = generator(masked_y)
            generated_np = generated.cpu().squeeze(1).numpy()
            y_channel_np = y_channel.cpu().squeeze(1).numpy()
            for i in range(generated_np.shape[0]):
                psnr = compare_psnr(y_channel_np[i], generated_np[i], data_range=1.0)
                ssim = compare_ssim(y_channel_np[i], generated_np[i], data_range=1.0)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Validation - Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")




def save_images(generator, test_loader, save_base_dir, num_images=5):
    gt_dir = os.path.join(save_base_dir, "GT")
    generated_dir = os.path.join(save_base_dir, "Generated")
    mask_dir = os.path.join(save_base_dir, "Mask")
    masked_dir = os.path.join(save_base_dir, "Masked")

    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for i, (masked_y, y_channel, mask, _) in enumerate(test_loader):
            masked_y = masked_y.to(device)
            y_channel = y_channel.to(device)
            mask = mask.to(device)
            generated = generator(masked_y).cpu().squeeze(1).numpy()
            masked_y = masked_y.cpu().squeeze(1).numpy()
            y_channel = y_channel.cpu().squeeze(1).numpy()
            mask = mask.cpu().squeeze(1).numpy()

            for j in range(len(generated)):
                masked_input_path = os.path.join(masked_dir, f"masked_input_{i}_{j}.png")
                cv2.imwrite(masked_input_path, (masked_y[j] * 255).astype(np.uint8))

                ground_truth_path = os.path.join(gt_dir, f"ground_truth_{i}_{j}.png")
                cv2.imwrite(ground_truth_path, (y_channel[j] * 255).astype(np.uint8))

                generated_path = os.path.join(generated_dir, f"generated_{i}_{j}.png")
                cv2.imwrite(generated_path, (generated[j] * 255).astype(np.uint8))

                mask_path = os.path.join(mask_dir, f"mask_{i}_{j}.png")
                cv2.imwrite(mask_path, (mask[j] * 255).astype(np.uint8))

                num_images -= 1
                if num_images <= 0:
                    return




train_model(train_loader, num_epochs=1)

save_images(generator, test_loader, save_base_dir='/home/msh7377/Muneeb/Output', num_images=5)
