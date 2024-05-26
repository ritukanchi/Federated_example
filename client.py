import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp 
from flwr_datasets import FederatedDataset #changing line according to how we want to import our dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor #not required as its image processing related, our dataset is values 
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #convolution layer 1 , main building block of CNN 
        self.pool = nn.MaxPool2d(2, 2) #max pooling layer , divides the inputs and applies functions to it 
        self.conv2 = nn.Conv2d(6, 16, 5) #convolution lyer 2
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #fully connected layer based on product of input vector and weights matrix giving a output vector ( fully interconnected layer w the previous layer)
        self.fc2 = nn.Linear(120, 84) #sub layer to flatten the output for better network handling capabilities 
        self.fc3 = nn.Linear(84, 10)#  same as above 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#training function 
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss() #loss function i.e cross entropy loss 
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimization algorithm (stochastic gradient descent main part of FL )
    for _ in range(epochs): #iterates over dataset for specified epochs or number of trials 
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"] #change 
            labels = batch["label"] #change
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward() 
            optimizer.step()


def test(net, testloader): #testing of the mode l
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss() #loss unction as before 
    correct, loss = 0, 0.0
    with torch.no_grad(): #disables the gradient computation for efficiency , gradient computation finds rate of change for loss function by how much its changing and is used as input for backprogation on basis of training hence remove it 
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE) #change 
            labels = batch["label"].to(DEVICE) #change
            outputs = net(images) 
            loss += criterion(outputs, labels).item() #loss calculation 
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset) #accuracy calculation 
    return loss, accuracy


def load_data(partition_id):
    """Load partition CIFAR10 data.""" #change to health data 
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3}) 
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42) #splits data into training and testing sets 80% train 20% test 
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch): # change but dont think its required as its not images 
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2],
    default=0,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_known_args()[0].partition_id #recognition of partition ids when given as commands 
#also allows selecting hwhich partition to use 

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data(partition_id=partition_id)
#initializes the model and moves to appropriate device i.e phone from server in this case 
# load data based on selected partition id given earlier 


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config): #model paramaters as a list of numpy arrays 
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters): #sets model paramaters from list of numpy arrays 
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): #training the model for one epoch and returns updated parameters  
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config): #evaluates model and returns loss and accuracy 
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client() #creae and return insatnce of client 


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
#start client flower connecting to loopback address i.e local server 