import net_tester
import data_loader

import torch.nn.functional as F
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import math
from pathlib import Path


def trainer(device, model, optimizer, num_epochs, dataloader, testLoaders, modelName, st_epoch=0):
  model = model.to(device)

  criterion = nn.MSELoss()
  # criterion = nn.L1Loss()

  for epoch in range(num_epochs):
      total_loss = 0
      total_mse = 0

      batch_count = 0
      num_steps = int(len(dataloader))

      tqdm_dataloader = tqdm(dataloader, desc=f'Epoch {epoch+1+st_epoch}/{num_epochs}', leave=False)

      step_loss = 0
      step_mse = 0

      for step, patches in enumerate(tqdm_dataloader):
        for p_step, ((hr_images, lr_images), gs, filenames) in enumerate(patches):
          model.train()

          batch_count += 1

########### GT , LR
          hr_images = hr_images.to(device)
          lr_images = lr_images.to(device)
  ######### guides
          guides = [gs[0].to(device)]
          if model.scale == 4:
            guides.append(gs[1].to(device))

  ######### predict!
          sr_images, out = model(lr_images, guides, trainMode=True) # Y Channel img

  ######### losses
          channels = [3, 64, 128, 256, 512, 512]
          a = []
          count = 1

          a.append(count) #init
          for i in range(model.scale//2): 
            a.append(10)
          for i in range(model.depth-1, -1, -1):
            a.append(10) # decoder

          a[len(a)-1] *= 10

          loss = 0
          gt = model.get_gt(hr_images)
          for i in range(len(gt)):
            loss += criterion(out[i], gt[i])*a[i]
            if i == (len(gt)-1): mse = criterion(out[i], gt[i])

  ######### step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

  ######### Update Status
          total_loss += loss.item()
          total_mse  += mse.item()

          tqdm_dataloader.set_postfix({'Loss': total_loss/batch_count, 'MSE': total_mse/batch_count })
      print(f'Epoch [{epoch+1+st_epoch}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, MSE: {total_mse/len(dataloader):.4f}')

      # Save Model
      model_path = Path(f"../models/temp/{modelName}")
      model_path.mkdir(parents=True, exist_ok=True)
      save_path = os.path.join(model_path, f'{modelName}_e{epoch+1+st_epoch}.pth')
      torch.save(model.state_dict(), save_path)

      model_path = Path(f"../models/save")
      model_path.mkdir(parents=True, exist_ok=True)
      save_path = os.path.join(model_path, f'{modelName}.pth')
      torch.save(model.state_dict(), save_path)

      model_path = Path(f"../models/optim/{modelName}")
      model_path.mkdir(parents=True, exist_ok=True)
      save_path = os.path.join(model_path, f'optim_{modelName}_e{epoch+1+st_epoch}.pth')
      torch.save(optimizer.state_dict(), save_path)

      # Validate
      model.eval()
      for (tl, tn) in testLoaders:
        print(f"[{tn}]")
        save_path = f"../outputs/{modelName}/{tn}"
        net_tester.test_set(tl, device, model, save_path=save_path)
      print("\n\n")
      
