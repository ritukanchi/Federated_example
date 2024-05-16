# Flower Example using PyTorch

This introductory example to Flower uses PyTorch with dataset [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

### Clone the project 

### Installing Dependencies

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

______________________________________________________________________

## Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Start the Flower clients which will participate in the learning. We need to specify the partition id to
use different partitions of the data on different nodes.  To do so open two more terminal windows and run the
following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1
```

You will see that PyTorch is starting a federated training. 

