# TODO: your imports
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms, models, datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_data_loader(batch_size):
      # convert to tensor, normalize
  transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  #load sets
  full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  #split sets
  train_subset = Subset(full_train_set, range(0, 40000))
  val_subset = Subset(full_train_set, range(40000, 50000))

  #dataloading
  train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

class Brick(nn.Module):
    oneexp = 1
    def __init__(self, inp, outp, stride=1):
        super().__init__()
        
        #conv and norm layer implementation
        self.conv1 = nn.Conv2d(inp, outp, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outp)

        self.conv2 = nn.Conv2d(outp, outp, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outp)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or inp != self.oneexp * outp:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inp,
                    self.oneexp * outp,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.oneexp * outp),
            )

    #forward pass that goes thr the layers
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        #resnet layer implementation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.l1 = self.create(block, 64, num_blocks[0], stride=1)
        self.l2 = self.create(block, 128, num_blocks[1], stride=2)
        self.l3 = self.create(block, 256, num_blocks[2], stride=2)
        self.l4 = self.create(block, 512, num_blocks[3], stride=2)
        self.avgpl = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.oneexp, num_classes)

    #layer builder function
    def create(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        ls = []
        for stride in strides:
            ls.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.oneexp
        return nn.Sequential(*ls)

    #forward pass thru the entire NN resnet 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.l1(out)  
        out = self.l2(out)  
        out = self.l3(out)  
        out = self.l4(out)  
        out = self.avgpl(out)  
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out

def init_train_var(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
    return criterion, optimizer

def train(train_loader, val_loader, batch_size):
    #prerequisite variables 
    model = ResNet(Brick, [2,2,2,2])
    model.to(device)
    criterion, optimizer = init_train_var(model)
    best_val_acc = 0.0
    best_model_state = None

    #run for 50 epochs 
    for epoch in range(50):
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        total_train = 0

        #train for each input an label
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            #calculate loss
            epoch_train_loss += loss.item() * x.size(0)  # Multiply by batch size
            _, predicted = torch.max(pred.data, 1)
            train_correct += (predicted == y).sum().item()
            total_train += y.size(0)

        #train loss 
        train_loss = epoch_train_loss / total_train
        train_acc = train_correct / total_train

        #now validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)

                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(pred.data, 1)
                val_correct += (predicted == y).sum().item()
                total_val += y.size(0)

        #val loss 
        val_loss = val_loss / total_val
        val_acc = val_correct / total_val

        #save new best instance 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_model.pth')

        #print metrics
        print(f"[INFO] EPOCH: {epoch + 1}/{50}")
        print(f"Train loss: {train_loss:.6f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.6f}, Val accuracy: {val_acc:.4f}\n")

    # Load best instance
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test(model_path, test_loader):
    model_path.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_path(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def main():
    torch.multiprocessing.freeze_support()   # optional
    train_loader, val_loader, test_loader = create_data_loader(batch_size=64)
    model = train(train_loader, val_loader, batch_size=64)
    test_acc = test(model, test_loader)
    print(f"\nBest Model Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
