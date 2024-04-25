import numpy as np
import json
from ANET import ANet  # Make sure your ANet class has appropriate methods
from MCTS import ReplayBuffer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def main():
    config = load_config('config.json')

    board_size = config['hex_game']['board_size']
    total_games = config['hex_game']['total_games']
    iteration_limit = config['mcts']['iteration_limit']

    save_interval = config['training']['save_interval']
    sample_size = config['training']['sample_size']
    num_epochs = config['training']['num_epochs'] # Number of epochs to train ANET for each batch.
    anet = ANet(
        board_size=config['hex_game']['board_size'], 
        learning_rate=config['anet']['learning_rate'],
        hidden_layers=config['anet']['hidden_layers'],
        activation=config['anet']['activation'],
        optimizer_name=config['anet']['optimizer']
    )

    # Load network if exists or initialize a new one
    #try:
        #anet.load_net("trained_anet.h5")
    #except FileNotFoundError:
        #print("No saved model found, initializing a new model.")

    replay_buffer = ReplayBuffer()
    replay_buffer.load_from_file('Replay_buffer_seven_NEW200.json')
    anet.save_net(f"anet_saved_data_seven0.h5")
    for i in range(1,total_games+1):
        
        # Ensure there's enough data in the buffer to sample
        batch_size = config["training"]["sample_size"]

        if len(replay_buffer.buffer) >= batch_size:
            states, target_probs_dicts = replay_buffer.sample(min(sample_size, len(replay_buffer)))
            target_probs = np.array([replay_buffer.convert_probs_to_array(prob_dict, board_size = board_size) for prob_dict in target_probs_dicts])
            anet.train(states, target_probs, epochs=num_epochs)
            anet.save_net("Loaded_anet_seven200.h5")
            print("Training complete and model saved.")
        else:
            print("Not enough data in replay buffer to perform training.")
        if i % save_interval == 0:
            anet.save_net(f"anet_saved_data_seven{i}.h5")
            print(f"Saved ANET parameters after game {i}.")

if __name__ == "__main__":
    main()
