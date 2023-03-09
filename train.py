"""Import all necessary package"""
import config
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from dataset import Dataset_Load
from utils import train_step, val_step, test_step, load_checkpoint, save_checkpoint, Confusion_Matrix
from model import Yolov8_cls


def main():

    # Create tensorboard file to keep track process training
    writer = SummaryWriter(f'runs_drop/')
    step = 0

    # Setup data
    train_loader, val_loader, test_loader, class_labels, test_targets = Dataset_Load(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        train_transform=config.train_transform,
        test_transform=config.test_val_transform,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKER)

    # Setup all necessary of model
    model_yolo = Yolov8_cls(3, num_classes=config.NUM_CLASS).to(config.DEVICE)
    optimizer = Adam(model_yolo.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    # Load check point
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model_yolo, optimizer, config.LR, config.DEVICE)

    # Add model graph to tensorboard
    writer.add_graph(model_yolo, torch.randn((1, 3, 224, 224)).to(config.DEVICE))


    for epoch in range(config.EPOCHS):
    
        # Training process
        train_loss, train_acc = train_step(model_yolo, train_loader, loss_fn, optimizer, config.DEVICE)
    
        # Validation process
        val_loss, val_acc = val_step(model_yolo, val_loader, loss_fn, config.DEVICE)
    
        # Add information to tensorboard
        writer.add_scalar(f'Training Loss', train_loss, global_step=step)
        writer.add_scalar(f'Validation Loss', val_loss, global_step=step)
        writer.add_scalar(f'Training Accuracy', train_acc, global_step=step)
        writer.add_scalar(f'Validation Accuracy', val_acc, global_step=step)
        step += 1
    
        # Print information
        print(f'Epoch: {epoch+1}')
        print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.2f}%')
        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc * 100:.2f}%')
    
        # Sve model each 5 epochs
        if (epoch + 1) % 5 == 0 & config.SAVE_MODEL:
            save_checkpoint(model_yolo, optimizer, "model_checkpoint_drop.pt")
        print('--------------------------------------------------------------------')

    # Final testing process
    test_loss, test_acc, y_preds_tensor = test_step(model_yolo, test_loader, loss_fn, config.DEVICE)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.2f}%')
    Confusion_Matrix(class_labels, y_preds_tensor, test_targets)

if __name__ == '__main__':
    main()


