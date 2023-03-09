"""Import necessary packages"""
import torch
from torchvision import transforms

"""All task can tuning"""

DATASET = '500_birds_data/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKER = 2
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-5
NUM_CLASS = 500
WEIGHT_DECAY = 1e-2
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "model_checkpoint.pt"
TRAIN_DIR = DATASET + 'train/'
TEST_DIR = DATASET + 'test/'
VAL_DIR = DATASET + 'valid/'

"""Transform for train dataset"""
train_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                     transforms.RandomCrop(IMAGE_SIZE),
                                     transforms.RandomVerticalFlip(0.5),
                                     transforms.RandomHorizontalFlip(0.01),
                                     transforms.RandomGrayscale(0.05),
                                     transforms.CenterCrop(IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],)

"""Transform for val/test dataset"""
test_val_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                         transforms.CenterCrop(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],)
