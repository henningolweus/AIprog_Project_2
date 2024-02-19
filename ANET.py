import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class ANet:
    def __init__(self, board_size, learning_rate=0.001, hidden_layers=[64, 64], activation='relu', optimizer_name='adam', num_cached_nets=10):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.num_cached_nets = num_cached_nets

        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        
        # Assuming your input is a flat array including the game state and player indicator
        input_shape = (self.board_size * self.board_size * 2 + 1,)  # Total inputs including player indicator
        
        # Start building your model from the Dense layers directly
        model.add(layers.InputLayer(input_shape=input_shape))  # Define the input shape of the model
        
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation))
        
        # Output layer
        model.add(layers.Dense(self.board_size**2, activation='softmax'))
        
        optimizer = self._get_optimizer()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def _get_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            return optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'adagrad':
            return optimizers.Adagrad(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, state):
        # Ensure state is reshaped correctly to match the expected input shape of the model
        if state.ndim == 1:  # If state is a flat array, reshape it to include batch dimension
            state = state.reshape(1, -1)
        return self.model.predict(state)

    def save_net(self, filename):
        self.model.save(filename)

    def load_net(self, filename):
        self.model = models.load_model(filename)



# Example usage
anet = ANet(board_size=4, learning_rate=0.001, hidden_layers=[64, 64], activation='relu', optimizer_name='adam', num_cached_nets=10)

game_state = np.zeros((anet.board_size*anet.board_size*2 + 1,))  # Include the player indicator in the game state array

# Predict the move distribution for the given game state and current player
move_distribution = anet.predict(game_state)

# Create the ANet instance

# Define a test game state
game_state = np.array([[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,-1]])  # Example input reshaped for prediction

# Predict the move distribution for the given game state
move_distribution = anet.predict(game_state)
print("Predicted move distribution:", move_distribution)
