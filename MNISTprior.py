import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import RandAugment

num_atoms = 3
epochs = 100
batch_size = 8

transform = transforms.Compose([
    transforms.ToTensor(),
      # Use a single mean and a single std dev
])

randomAugment = transforms.Compose([
    RandAugment(),  # Add RandAugment
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=randomAugment)
if __name__ == "__main__":
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    trainiter = iter(trainloader)

samples_per_label = 3
label_indices = {i: [] for i in range(10)}
for idx, (image,label) in enumerate(trainset):
  if len(label_indices[label]) < samples_per_label:
    label_indices[label].append(idx)
  if all(len(indices) == samples_per_label for indices in label_indices.values()):
    break
selected_indices = [idx for indices in label_indices.values() for idx in indices]
subset = Subset(trainset, selected_indices)

labeled_batch_size = 16
labeled_loader = DataLoader(subset, batch_size = batch_size, shuffle=True)
labeled_iter = iter(labeled_loader)



testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class PriorNet(nn.Module):
  def __init__(self, num_atoms):
    super(PriorNet, self).__init__()
    self.atoms = nn.ModuleList([Net() for i in range(num_atoms)])
  def forward(self,x):
    return [atom(x) for atom in self.atoms]

def H_y(p): #p is 0th dimension atom, 1st dimension input image, 2nd dimension softmax probability for labels
  p = p.sum(0)
  p_y_x = p / (num_atoms)
  log_p_y_x = torch.log(p_y_x)

  return (-1) * (p_y_x * log_p_y_x).sum(1).mean(0)

def H_y_w(p):
  log_p = torch.log(p)
  entropy = torch.sum(p * log_p)
  return (-1) * entropy / batch_size / num_atoms
  '''
  labels = labels.unsqueeze(0).unsqueeze(2).expand(num_atoms,batch_size,1)
  p_y_w = p.gather(2, labels).squeeze(2)
  log_p_y_w = torch.log(p_y_w)
  p_correct = (p_y_w * log_p_y_w).sum()
  return (-1) * p_correct / num_atoms / batch_size
  '''


#calculate Loss
def calculate_loss(p,outputs,labels):
  l_u = H_y(p) - H_y_w(p)
  l_x = sum([criterion(x,labels) for x in outputs]) / num_atoms
  #print("H_y: ",H_y(p).item())
  #print("H_y_w: ",H_y_w(p).item())
  #print("L_x: ",l_x.item())
  #print("L_u: ",l_u.item())
  return l_x #- l_u



model = PriorNet(num_atoms)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(epochs):
  optimizer.zero_grad()
  unlabeled_images, _ = next(iter(trainloader))
  inputs, labels = next(iter(labeled_loader))
  p = torch.stack([F.softmax(logits,dim=1) for logits in model(unlabeled_images)]) #p is 0th dimension atom, 1st dimension input image, 2nd dimension softmax probability for labels

  loss = calculate_loss(p,model(inputs),labels)
  loss.backward()
  optimizer.step()
  print("Loss: " ,loss.item())

model.eval()
accuracies = []
correct = 0
total = 0
with torch.no_grad():
  for images,labels in testloader:
    outputs = model(images)
    aggregated_output = torch.mean(F.softmax(torch.stack(outputs),dim=1),dim = 0)
    _, predicted = torch.max(aggregated_output,1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)


for atom in model.atoms:
  print("Atom Tested")
  total_accuracy = 0
  with torch.no_grad():
    for images, labels in testloader:
      outputs = atom(images)
      _, predictions = torch.max(outputs,1)
      accuracy = (predictions == labels).float().mean().item()
      total_accuracy += accuracy
  average_accuracy = total_accuracy / len(testloader)
  accuracies.append(average_accuracy)


print(correct/total)
print(accuracies)
