import flwr as fl
from flwr.server.strategy import FedAvg
from utils import create_keras_model, load_model_weights
import tensorflow as tf

def start_server(num_rounds=3):
    model = create_keras_model()
    load_model_weights(model, file_path='model.weights.h5')
    
    def evaluate_metrics_aggregation_fn(evaluations):
        accuracies = [metrics["accuracy"] * metrics["num_examples"] for _, metrics in evaluations]
        total_examples = sum(metrics["num_examples"] for _, metrics in evaluations)
        return {"accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0.0}
    
    strategy = FedAvg(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    # Start Flower server
    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server(num_rounds=1)