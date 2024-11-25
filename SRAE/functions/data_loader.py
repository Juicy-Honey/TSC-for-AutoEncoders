import torch
import os
import zipfile
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

# zip_name, bs, scale, im_size=None
def un_zip (name, guide_name=None, path="../datasets"):

    if guide_name == None:
      zip_path  = f"{path}/{name}.zip"
    else:
      zip_path = f"{path}/{name}_{guide_name}.zip"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/dataset')

    if guide_name==None:
      hr_path = f'/content/dataset/{name}' # dataset folder
    else:
      hr_path = f'/content/dataset/{name}_{guide_name}'
    return hr_path

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

class Dataset_SR(torch.utils.data.Dataset):
    def __init__(self, filenames, data_path, guide_path, transform=None, scale=4, lr_size=16, num_patches=4, testMode=False):
        self.filenames = filenames  
        self.path = data_path
        self.g_path = guide_path
        self.transform = transform
        self.scale = scale
        self.img_size = lr_size
        self.num_patches = num_patches
        self.testMode = testMode

    def __len__(self):
        return len(self.filenames)

    def load_image(self, filepath):
        return Image.open(filepath).convert('RGB')

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # Folder paths
        hr_path = os.path.join(self.path, "HR")
        lr_path = os.path.join(self.path, "LR")
        g2_path = os.path.join(self.g_path, "g2")
        g4_path = os.path.join(self.g_path, "g4")

        # Load images
        hr_image = self.load_image(os.path.join(hr_path, filename))
        lr_image = self.load_image(os.path.join(lr_path, filename))
        guide2x_image = self.load_image(os.path.join(g2_path, filename))
        guide4x_image = None
        if self.scale == 4:
            guide4x_image = self.load_image(os.path.join(g4_path, filename))

        if self.testMode:
            # Use full-size patches
            hr_patch = hr_image
            lr_patch = lr_image

            guide2x_patch = guide2x_image
            guide_patches = [guide2x_patch]
            if self.scale == 4:
                guide_patches.append(guide4x_image)            

            # Apply transforms (if any)
            if self.transform:
                hr_patch = self.transform(hr_patch)
                lr_patch = self.transform(lr_patch)
                guide_patches = [self.transform(patch) for patch in guide_patches]

            data = [hr_patch, lr_patch]
            guides = guide_patches
            
            patches = (data, guides, filename)

            return patches

        else:
            patches = []
            lr_width, lr_height = lr_image.size
            for _ in range(self.num_patches):
                # Random starting coordinates for LR patch
                lr_x = random.randint(0, lr_width - self.img_size)
                lr_y = random.randint(0, lr_height - self.img_size)
                
                # Corresponding HR coordinates
                hr_x = lr_x * self.scale
                hr_y = lr_y * self.scale
                hr_patch_size = self.img_size * self.scale

                # Crop patches
                lr_patch = lr_image.crop((lr_x, lr_y, lr_x + self.img_size, lr_y + self.img_size))
                hr_patch = hr_image.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))
                guide2x_patch = guide2x_image.crop(
                    (lr_x * 2, lr_y * 2, (lr_x + self.img_size) * 2, (lr_y + self.img_size) * 2)
                )
                guide_patches = [guide2x_patch]
                if self.scale == 4:
                    guide4x_patch = guide4x_image.crop(
                        (lr_x * 4, lr_y * 4, (lr_x + self.img_size) * 4, (lr_y + self.img_size) * 4)
                    )
                    guide_patches.append(guide4x_patch)

                # Apply the same augmentation to all patches
                if self.transform:
                    seed = random.randint(0, 9999)  # Set a random seed for reproducibility
                    torch.manual_seed(seed)
                    hr_patch = self.transform(hr_patch)
                    torch.manual_seed(seed)
                    lr_patch = self.transform(lr_patch)
                    torch.manual_seed(seed)
                    guide_patches = [self.transform(patch) for patch in guide_patches]

                # Prepare data and guides
                data = (hr_patch, lr_patch)
                guides = tuple(guide_patches)

                # Append to patches
                patches.append((data, guides, filename))
            
            return patches

###########main
def load_ds(zip_name, guide_name, bs, scale, lr_size=None, num_patches=1):
  data_path  = un_zip(zip_name)      # unzip dataset at "/content/dataset/{zipname}"
  guide_path = un_zip(zip_name, guide_name) # unzip guide dataset

  transform = transforms.Compose([
    transforms.ToTensor()
  ])
  
  hr_path = os.path.join(data_path, "HR")
  filenames = [x for x in os.listdir(hr_path)]

  if lr_size==None: testMode = True 
  else: testMode = False

  dataset_SR = Dataset_SR(filenames, data_path, guide_path, transform, scale,lr_size=lr_size, num_patches=num_patches, testMode=testMode)
  print(f'{len(dataset_SR)} images loaded')
  
  if testMode:
    data_loader = torch.utils.data.DataLoader(dataset_SR, batch_size=bs, shuffle=True, collate_fn=collate_fn)
  else:
    data_loader = torch.utils.data.DataLoader(dataset_SR, batch_size=bs, shuffle=True)

  return data_loader

def collate_fn(batch):
    return batch







