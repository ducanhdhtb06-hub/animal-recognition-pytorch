import os.path
import shutil

from torch.utils.checkpoint import checkpoint

from AnimaDataset import AnimaDataset
from simplenetwork import simpleCNN
from torch.utils.data import DataLoader , Dataset
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
import torch.optim as optim
import torch
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser("CNN")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--r", type=str, default="animals_v2/animals", help="root path of dataset")
    parser.add_argument("--checkpoint", type=str, default="train_models/last_model.pt", help="root path of dataset")
    args = parser.parse_args()
    return args
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    num_epochs = 100

    args = get_args()
    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=5
        ),
        Resize((args.image_size, args.image_size)),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.4,
            hue=0.1
        ),
        ToTensor(),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = AnimaDataset(root=args.r, train=True, transform=train_transform)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)
    test_dataset = AnimaDataset(root=args.r, train=False, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    model = simpleCNN(num_classes=10)
    train_models = "train_models"


    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")
    if not os.path.isdir(train_models):
        os.mkdir(train_models)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    writer = SummaryWriter("tensorboard")
    if torch.cuda.is_available():
        model.cuda()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint , weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_accuracy = checkpoint["best_accuracy"]
    else :
        start_epoch = 0
        best_accuracy = 0

    for epoch in range(  start_epoch , args.epochs):
        model.train()
        progress_bar = tqdm(train_loader , colour= "green")
        for iters , (images , labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar('train/loss', loss, epoch * len(train_loader) + iters)
            progress_bar.set_description("Epoc {}/{}. Iteration {}/{}. loss{}".format(epoch+1, args.epochs, iters + 1, len(train_loader), loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        all_predictions = []
        all_labels = []

        for iters, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                indices = torch.argmax(outputs.cpu(), 1)
                all_predictions.extend(indices)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [indices.item() for indices in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions),
                              class_names=test_dataset.categories, epoch=epoch)
        print("Epoch {}: Accurancy: {}".format(epoch+1, accuracy_score(all_labels, all_predictions)))
        writer.add_scalar('test/accuracy', accuracy_score(all_labels, all_predictions), epoch+1)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': accuracy_score(all_labels, all_predictions)
        }
        torch.save(checkpoint, "{}/last_model.pt".format(train_models))
        if best_accuracy < accuracy_score(all_labels, all_predictions):
            best_accuracy = accuracy_score(all_labels, all_predictions)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }
            torch.save(checkpoint, "{}/best_model.pt".format(train_models))
