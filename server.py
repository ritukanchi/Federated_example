import sys #variables used by interpreter to interact
import numpy as np #support for matrices and arrays 
import flwr as fl #frameowkr for FL 

class SaveModelStrategy(fl.server.strategy.FedAvg): #strategy for fedaverae
    def aggregate_fit( #
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

def main():
    if len(sys.argv) != 2:
        print("abey bhidu run: python(python3) pythonfile.py <port no>")
        sys.exit(1)

    try:
        port = int(sys.argv[1])
    except ValueError:
        print("Error: Port must be an integer")
        sys.exit(1)

    # Create strategy and run server
    strategy_MINE = SaveModelStrategy()
    server_address = f"localhost:{port}"
    config = fl.server.ServerConfig(num_rounds=3)
    fl.server.start_server(
        server_address=server_address,
        config=config,
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy_MINE
    )

if __name__ == "_main_":
    main()