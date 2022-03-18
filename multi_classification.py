import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

from PIL import *
from PIL import ImageFile
from PIL import Image
from efficientnet_pytorch import EfficientNet

import wandb

#System settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['WANDB_CONSOLE'] = 'off'
#Coloring for print outputs
class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MultilabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.resnet = models.resnet34(pretrained=True)
        #self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.efficient_net = EfficientNet.from_pretrained(model_name="efficientnet-b2", num_classes=2)
        #self.model_wo_fc = nn.Sequential(*(list(self.efficient_net.children())[:-1]))
        inch = self.efficient_net._fc.in_features
        self.hair_dense = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.hair_short = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.hair_medium = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.black_frame = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.ruler_mark = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.other = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=inch, out_features=2)
        )
        self.efficient_net._fc = Identity()

    def forward(self, x):
        x = self.efficient_net(x)

        return {
            "hair_dense": self.hair_dense(x),
            "hair_short": self.hair_short(x),
            "hair_medium": self.hair_medium(x),
            "black_frame": self.black_frame(x),
            "ruler_mark": self.ruler_mark(x),
            "other": self.other(x)
        }

class BiasDataset(Dataset):
  def __init__(self, root_path: str, annotationfile_path: str, transform=None, train=True):
    self.path = root_path
    self.train = train
    self.transform = transform
    if self.train:
      self.annotationfile_path = annotationfile_path
      self.folder = [
              x.strip().split()[0] for x in open(self.annotationfile_path)
          ]
    else:
      included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
      self.folder = sorted([fn for fn in os.listdir(self.path)
              if any(fn.endswith(ext) for ext in included_extensions)])

  def __len__(self):
    if self.train:
      return len(self.folder)
    else:
      return len(os.listdir(self.path))

  def __getitem__(self,idx):
    if self.train:
      img_loc = os.path.join(self.path, self.folder[idx].split(',')[0])
      translation_dict = [int(label) for label in self.folder[idx].split(',')[1:]]

      label1 = translation_dict[0]
      label2 = translation_dict[1]
      label3 = translation_dict[2]
      label4 = translation_dict[3]
      label5 = translation_dict[4]
      label6 = translation_dict[5]
    else:
      img_loc = os.path.join(self.path, self.folder[idx])
    image = Image.open(img_loc).convert('RGB')
    single_img = self.transform(image)
    
    if self.train:
      return {'image':single_img, 'labels': {"label_hair_dense": label1,
                                             "label_hair_short": label2,
                                             "label_hair_medium": label3,
                                             "label_black_frame": label4,
                                             "label_ruler_mark": label5,
                                             "label_other": label6
                                            }
             }
    else:
      return {'image':single_img, 'name': self.folder[idx]}

def criterion(loss_func,outputs,pictures):
  losses = 0
  for _, key in enumerate(outputs):
    losses += loss_func(outputs[key], pictures['labels'][f'label_{key}'].to(device))
  return losses

def training(model, device, lr_rate,epochs, train_loader, wandb_flag=True):
  num_epochs = epochs
  losses = []
  checkpoint_losses = []

  optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
  n_total_steps = len(train_loader)

  loss_func = nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
     for i, pictures in enumerate(train_loader):
        images = pictures['image'].to(device)
        pictures = pictures

        outputs = model(images)

        loss = criterion(loss_func,outputs, pictures)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % (int(n_total_steps/1)) == 0:
            checkpoint_loss = torch.tensor(losses).mean().item()
            checkpoint_losses.append(checkpoint_loss)
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {checkpoint_loss:.4f}')
            if wandb_flag:
              wandb.log({f'train/training_loss': checkpoint_loss, 'epoch':epoch+1})
        if (i+1) % (int(n_total_steps/1)) == 0:
            n_correct,n_samples,n_class_correct,n_class_samples = validation(model, test_loader, len(images),
                                                                             classes_hair_dense, classes_hair_short, classes_hair_medium,
                                                                             classes_black_frame, classes_ruler_mark, classes_other)
            class_acc(n_correct, n_samples, n_class_correct, n_class_samples, class_list, wandb_flag)
        
  return checkpoint_losses, optimizer

def validation(model, dataloader, batch_size, *args):

  with torch.no_grad():
    n_correct = []
    n_class_correct = []
    n_class_samples = []
    n_samples = 0

    for arg in args:
      n_correct.append(len(arg))
      n_class_correct.append([0 for _ in range(len(arg))])
      n_class_samples.append([0 for _ in range(len(arg))])

    for pictures in dataloader:
      images = pictures['image'].to(device)
      outputs = model(images)
      labels = [pictures['labels'][picture].to(device) for picture in pictures['labels']]

      for i,out in enumerate(outputs):
        _, predicted = torch.max(outputs[out],1)
        n_correct[i] += (predicted == labels[i]).sum().item()

        if i == 0:
          n_samples += labels[i].size(0)
        for k in range(batch_size):
          label = labels[i][k]
          pred = predicted[k]
          if (label == pred):
              n_class_correct[i][label] += 1
          n_class_samples[i][label] += 1
          
  return n_correct,n_samples,n_class_correct,n_class_samples

def class_acc(n_correct,n_samples,n_class_correct,n_class_samples,class_list, wandb_flag=True):
    for i in range(len(class_list)):
      print("-------------------------------------------------")
      acc = 100.0 * n_correct[i] / n_samples if n_samples != 0 else 0
      print(color.BOLD + color.RED + f'Overall class performance: {round(acc,1)} %' + color.END)
      for k in range(len(class_list[i])):
          acc = 100.0 * n_class_correct[i][k] / n_class_samples[i][k]  if n_class_samples[i][k] != 0 else 0
          print(f'Accuracy of {class_list[i][k]}: {round(acc,1)} %')
          if wandb_flag:
            wandb.log({'val/Acc_'+class_list[i][k]: round(acc,1)})
    print("-------------------------------------------------")

def test(model, dataloader, save_path):
  file = open(save_path,"w")
  with torch.no_grad():
    for pictures in dataloader:
      images = pictures['image'].to(device)
      outputs = model(images)
      img_labels = [pictures['name']]
      for out in outputs:
        _, predicted = torch.max(outputs[out],1)
        img_labels.append([str(j) for j in predicted.cpu().tolist()])
      file.writelines([','.join(line)+'\n' for line in list(zip(*img_labels))])
  file.close()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", help="path to all images")
    parser.add_argument(
        "--ann_path",
        type=str,
        default=None,
        help="path to annotations (default: None)",
    )
    parser.add_argument(
        "--mode",
        default="val",
        choices=["train", "test", "val"],
        help="mode for proces which will be done",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="train/test ratio (default: 0.8)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        metavar="EPOCHS",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers (default: 4)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch size (default: 16)"
    )
    parser.add_argument("--model_path", help="path to save or read model",
                        default="multiclasificator_efficientnet-b2_uGAN.pth")
    parser.add_argument("--save_path", help="path to save pseudoannotations",
                        default="annotations.csv")
    parser.add_argument(
        "--seed", type=int, default=2022, help="random seed (default: 2022)"
    )

    # wandb settings
    parser.add_argument(
        "--wandb_flag",
        action="store_true",
        default=False,
        help="Launch experiment and log metrics with wandb",
    )
    return parser


if __name__ == "__main__":
  parser = get_args_parser()
  args = parser.parse_args()

  # set the seed for reproducibility
  seed = args.seed
  torch.manual_seed(seed)
  np.random.seed(seed)

  #Getting the data
  DATA_DIR = args.img_path

  # model path to load trained model or to save model after training
  PATH = args.model_path

  # choose mode beetwen val, train, test
  mode = args.mode
  label_flag = False if args.mode == 'test' else True

  # source annotation path
  annotationfile_path = args.ann_path

  #Pre-processing transformations
  data_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((256,256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ])

  classes_hair_dense = ['None_hair_dense', 'Yes_hair_dense']
  classes_hair_short = ['None_hair_short', 'Yes_hair_short']
  classes_hair_medium = ['None_hair_medium', 'Yes_hair_medium']
  classes_black_frame = ['None_black_frame', 'Yes_black_frame']
  classes_ruler_mark = ['None_ruler_mark', 'Yes_ruler_mark']
  classes_other = ['None_other', 'Yes_other']
  header = [
      "hair_dense",
      "hair_short",
      "hair_medium",
      "black_frame",
      "ruler_mark",
      "other"
  ]
  class_list = [classes_hair_dense,classes_hair_short,classes_hair_medium,classes_black_frame,classes_ruler_mark,classes_other]

  dataset = BiasDataset(root_path=DATA_DIR,
                        annotationfile_path=annotationfile_path,
                        transform=data_transforms,
                        train=label_flag)

  if mode == 'train':
    #Split the data in training and testing
    train_val_ratio = args.ratio
    train_len = round(len(dataset) * train_val_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    #Create the dataloader for each dataset
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, 
                              num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, 
                             num_workers=args.num_workers, drop_last=True)
    if args.wandb_flag:
      wandb.init(project="dai-healthcare", entity='eyeforai', group='cls_biases',
                 config={"model": "efficientnet-b2"})

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultilabelClassifier().to(device)
    checkpoint_losses, optimizer = training(model, device, args.lr, args.epochs, train_loader,
                                            wandb_flag=args.wandb_flag)

    torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': checkpoint_losses[-1],
                }, PATH)
    n_correct,n_samples,n_class_correct,n_class_samples = validation(model, test_loader, args.batch,
                                                                     classes_hair_dense, classes_hair_short, classes_hair_medium,
                                                                     classes_black_frame, classes_ruler_mark, classes_other)
    class_acc(n_correct, n_samples, n_class_correct, n_class_samples, class_list, wandb_flag=args.wandb_flag)
  else:
    test_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, 
                             num_workers=args.num_workers, drop_last=True)
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultilabelClassifier().to(device)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if mode == 'test':
      # path to save annotations
      SAVE_PATH = args.save_path
      test(model, test_loader, SAVE_PATH)
    elif mode == 'val':
      n_correct,n_samples,n_class_correct,n_class_samples = validation(model,test_loader, args.batch,
                                                                       classes_hair_dense, classes_hair_short, classes_hair_medium,
                                                                       classes_black_frame, classes_ruler_mark, classes_other)
      class_acc(n_correct, n_samples, n_class_correct, n_class_samples, class_list, wandb_flag=False)
    else:
      print("Wrong mode!")
