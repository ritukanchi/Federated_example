from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig #flower to configure and run FL server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics #accuracy 


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] #muliply each clients accuracy by number of examples they provided
    examples = [num_examples for num_examples, _ in metrics] #extracts number of examples from each client 

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


# Define config
config = ServerConfig(num_rounds=3) #number of FL rounds 


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy, #fedavg strategy with custom metric aggregation function 
)


# Legacy mode
if __name__ == "__main__": #runs only if script executred directly
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
