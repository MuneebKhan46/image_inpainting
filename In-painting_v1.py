import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim



class YUVImageDataset(Dataset):
    def __init__(self, yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1, mode='train', triplets_list=None):
        self.yuv_dir = yuv_dir
        self.diff_map_dir = diff_map_dir
        self.image_size = image_size
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
        width, height = self.image_size
        y_size = width * height
        with open(yuv_path, 'rb') as f:
            y_channel = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((height, width))
        return y_channel

    def load_diff_map(self, diff_map_path):
        diff_map = Image.open(diff_map_path).convert('L')
        diff_map = diff_map.resize(self.image_size)
        return np.array(diff_map)

    def generate_mask_from_diff_map(self, diff_map):
        diff_map = diff_map / 255.0
        mask = np.where(diff_map > self.mask_threshold, 0, 1).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.image_triplets)
    
    def __getitem__(self, idx):
        original_path, denoised_path, diff_map_path = self.image_triplets[idx]


        if self.mode == 'train':
            input_image_path = original_path
        elif self.mode == 'test':
            input_image_path = denoised_path
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        y_channel = self.load_y_channel(input_image_path)
        ground_truth_y_channel = self.load_y_channel(original_path)

        diff_map = self.load_diff_map(diff_map_path)
        mask = self.generate_mask_from_diff_map(diff_map)

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



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
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

test_dataset = YUVImageDataset( yuv_dir, diff_map_dir, image_size=(720, 1280), mask_threshold=0.1, mode='test', triplets_list=test_triplets)



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




def train_gan(dataloader, num_epochs=100, model_save_dir="/home/msh7377/Muneeb/Models"):
    os.makedirs(model_save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        for i, (masked_y, y_channel, mask, _) in enumerate(dataloader):
            masked_y = masked_y.to(device)
            y_channel = y_channel.to(device)
            mask = mask.to(device)
                        
            optimizer_D.zero_grad()
            real_output = discriminator(y_channel)
            real_labels = torch.ones_like(real_output).to(device)
            fake_labels = torch.zeros_like(real_output).to(device)
            d_loss_real = adversarial_loss(real_output, real_labels)
            
            generated = generator(masked_y)
            fake_output = discriminator(generated.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            fake_output = discriminator(generated)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            g_rec_loss = reconstruction_loss(generated * mask, y_channel * mask)
            g_loss = g_adv_loss + g_rec_loss
            g_loss.backward()
            optimizer_G.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
    torch.save(generator.state_dict(), os.path.join(model_save_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_save_dir, "discriminator_final.pth"))
    print("Training completed. Final models saved.")





def test_model(generator, test_loader, output_dir="/home/msh7377/Muneeb/Output"):
    # Create subdirectories for test results
    gt_dir = os.path.join(output_dir, "GT")
    generated_dir = os.path.join(output_dir, "Generated")
    masked_dir = os.path.join(output_dir, "Masked")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    generator.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (masked_y, y_channel, mask, _) in enumerate(test_loader):
            masked_y = masked_y.to(device)
            y_channel = y_channel.to(device)
            generated = generator(masked_y)
            
            # Convert to numpy arrays for metrics calculation and saving images
            generated_np = generated.cpu().squeeze(1).numpy()
            y_channel_np = y_channel.cpu().squeeze(1).numpy()
            masked_y_np = masked_y.cpu().squeeze(1).numpy()
            
            for i in range(generated_np.shape[0]):
                psnr = compare_psnr(y_channel_np[i], generated_np[i], data_range=1.0)
                ssim = compare_ssim(y_channel_np[i], generated_np[i], data_range=1.0)
                total_psnr += psnr
                total_ssim += ssim
                count += 1
                
                # Save images to appropriate subdirectories
                Image.fromarray((masked_y_np[i] * 255).astype(np.uint8)).save(
                    os.path.join(masked_dir, f"batch{batch_idx}_image{i}_masked.png"))
                Image.fromarray((y_channel_np[i] * 255).astype(np.uint8)).save(
                    os.path.join(gt_dir, f"batch{batch_idx}_image{i}_ground_truth.png"))
                Image.fromarray((generated_np[i] * 255).astype(np.uint8)).save(
                    os.path.join(generated_dir, f"batch{batch_idx}_image{i}_generated.png"))
    
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Validation - Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
    generator.train()






train_gan(train_loader, num_epochs=100, model_save_dir="/home/msh7377/Muneeb/Models")



test_model(generator, test_loader, output_dir="/home/msh7377/Muneeb/Output")
