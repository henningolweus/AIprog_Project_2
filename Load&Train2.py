import numpy as np
import json
from ANET import ANet  # Make sure your ANet class has appropriate methods
from MCTS import ReplayBuffer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def train_network(anet, states, target_probs, epochs, batch_size):
    """ Train the network and return the evaluation metrics. """
    history = anet.train(states, target_probs, epochs=epochs, batch_size=batch_size)
    # Depending on how `anet.train` is implemented, make sure it returns some form of history object
    # that includes loss and accuracy for each epoch.
    return history

def main():
    config = load_config('config.json')

    board_size = 7
    total_games = config['hex_game']['total_games']

    iteration_limit = config['mcts']['iteration_limit']



    # Load network if exists or initialize a new one
    #try:
        #anet.load_net("trained_anet.h5")
    #except FileNotFoundError:
        #print("No saved model found, initializing a new model.")

    replay_buffer = ReplayBuffer()
    replay_buffer.load_from_file('Replay_buffer_seven_NEW200.json')

    
    # Hyperparameters to tweak
    batch_sizes = [32, 64, 128, 256]  # Example batch sizes
    learning_rates = [0.1,0.01, 0.001, 0.0001]  # Example learning rates

    best_accuracy = 0
    best_params = {}

    for batch_size in batch_sizes:
        for lr in learning_rates:
            save_interval = 100
            batch_size = batch_size
            num_epochs = 5 # Number of epochs to train ANET for each batch.
            anet = ANet(
                board_size=7, 
                learning_rate=lr,
                hidden_layers=config['anet']['hidden_layers'],
                activation=config['anet']['activation'],
                optimizer_name=config['anet']['optimizer']
)
            
            print(f"Training with batch size {batch_size} and learning rate {lr}")
            states, target_probs_dicts = replay_buffer.sample(batch_size)
            target_probs = np.array([replay_buffer.convert_probs_to_array(prob_dict, board_size) for prob_dict in target_probs_dicts])

            #history = 
            train_network(anet, states, target_probs, num_epochs, batch_size)
            #final_accuracy = history.history['accuracy'][-1]  # Modify according to how history object stores accuracy

            #if final_accuracy > best_accuracy:
                #best_accuracy = final_accuracy
                #best_params = {'batch_size': batch_size, 'learning_rate': lr}
                #anet.save_net(f"anet_optimized_bs{batch_size}_lr{lr}.h5")
                #print(f"New best model saved with accuracy {best_accuracy}")

    print(f"Best parameters: {best_params} with accuracy {best_accuracy}")


if __name__ == "__main__":
    main()