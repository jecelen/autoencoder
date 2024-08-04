import os
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
import scipy.ndimage

class AutoencoderDataset(Dataset):
  def __init__(self, img_dir, train = None):
        self.img_dir = img_dir
        self.data = os.listdir(self.img_dir)
        self.train = train

  def __len__(self):
    return len(self.data)
  
  @staticmethod
  def interpolate_nan(image):
        nan_mask = np.isnan(image)
        image[nan_mask] = 0
        interpolated_image = scipy.ndimage.morphology.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        return image[tuple(interpolated_image)]

  def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data[idx])
        img = tifffile.imread(img_name)
        if img.ndim == 2:  # Checa se a imagem tem duas dimensões (escala de cinza)
            img = img[:, :, np.newaxis]
        img = img.transpose((2, 0, 1))  # Transposta para (canais, altura, largura)
        if np.isnan(img).any(): #remove os NaN com vizinhos próximos
              img = self.interpolate_nan(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.nn.functional.adaptive_avg_pool2d(img, (224, 224))

        if img.shape[0] < 23: #ajustando numero de canais
            padding = torch.zeros((23 - img.shape[0], 224, 224))
            img = torch.cat((img, padding), 0)
        return img

    
