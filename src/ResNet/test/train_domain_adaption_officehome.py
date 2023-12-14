import argparse
import torch
import random
from torch import optim
from torch import nn
import numpy as np
from torchvision import datasets, transforms
import models.resnet
import models.alexnet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, Subset

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", choices=["resnet","alexnet"], help="Network", default='resnet')
    parser.add_argument("--source", choices=["Art","Clipart","Product","Real World"], help="Source", nargs='+')
    parser.add_argument("--target", choices=["Art","Clipart","Product","Real World"], help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=65, help="Number of classes")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")

    return parser.parse_args()

def get_model(model_name, pretrained=True, **kwargs):
    if model_name == 'resnet':
        model_class = getattr(models.resnet, 'resnet18')
    elif model_name == 'alexnet':
        model_class = getattr(models.alexnet, 'alex')
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model_class(pretrained=pretrained, **kwargs)

#Accelerate the training
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)

args = get_args() 
args.source = ["Art","Clipart","Product"]
args.target = "Real World"
print("Target domain: {}".format(args.target))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
args.n_classes = 65
model = get_model(args.model_name, pretrained=True, classes=args.n_classes)
model = model.to(device)

# Data loaders
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.RandomGrayscale(0.1),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224,224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_dataset = datasets.ImageFolder(
    f'/home/Domain_Generalization/OfficeHomeDataset_10072016/{args.target}', 
    transform=val_transform)
target_loader = torch.utils.data.DataLoader(target_dataset , batch_size=args.batch_size, shuffle=True)

concatenated_datasets = []
total_size = 0

for name in args.source:
    dataset = datasets.ImageFolder(
        f'/home/Domain_Generalization/OfficeHomeDataset_10072016/{name}', 
        transform=train_transform
    )
    total_size += len(dataset)
    concatenated_datasets.append(dataset)
    
combined_dataset = ConcatDataset(concatenated_datasets)
indices = list(range(total_size))
random.shuffle(indices)
random_subset = Subset(combined_dataset, indices)
source_loader = DataLoader(random_subset, batch_size=args.batch_size, shuffle=True)



# Optimizer and scheduler
if args.train_all:
    params = model.parameters()
else:
    params = model.get_params(args.learning_rate)
    
optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=False, lr=args.learning_rate)
#optimizer = optim.Adam(params, lr=lr)
step_size = int(args.epochs * .8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

print("Step size: %d" % step_size)

# Training and testing loop
results = {"test": torch.zeros(args.epochs)}
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.folder_name)
global_step = 0

for now_epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    writer.add_scalar('Learning Rate', lr, now_epoch)

    # Training
    model.train()


    for it, batch in enumerate(source_loader):
        data, labels = batch
        
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        data_fliped = torch.flip(data, (3,)).detach().clone()
        data = torch.cat((data, data_fliped))
        labels_labels_flip = torch.cat((labels, labels))

        class_logit = model(data, labels_labels_flip, True, now_epoch)
        class_loss = criterion(class_logit, labels_labels_flip)
        _, cls_pred = class_logit.max(dim=1)
        loss = class_loss

        loss.backward()
        optimizer.step()

        class_loss_train = class_loss.item()
        accuracy = torch.sum(cls_pred == labels_labels_flip.data).item() / data.shape[0]
        writer.add_scalar('Loss/Train', class_loss_train, global_step)
        writer.add_scalar('Accuracy/Train', accuracy, global_step)
        global_step += 1
        
        print(f"Training - running batch {it}/{len(source_loader)} of epoch {now_epoch}/{args.epochs} - "
                f"loss: {class_loss_train} - acc: {accuracy * 100 : 2f}%")
        del loss, class_loss, class_logit

    # Testing
    model.eval()
    with torch.no_grad():
        class_correct = 0
        total = len(target_loader.dataset)
        for it, batch in enumerate(target_loader):
            data, labels = batch
            
            data, labels = data.to(device), labels.to(device)

            class_logit = model(data, labels, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == labels.data)

        class_acc = float(class_correct) / total
        writer.add_scalar('Test/Accuracy', class_acc, now_epoch)
        print(f"Testing - acc: {class_acc : 2f}%")
        results['test'][now_epoch] = class_acc

# Output results
test_res = results["test"]
idx_best = test_res.argmax()
print(f"Best test {test_res.max()}, corresponding test {test_res[idx_best]} - best test: {test_res.max()}, best epoch: {idx_best}")

# Save the model
model_path = './model_domain_adaption_last.pth'  
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")