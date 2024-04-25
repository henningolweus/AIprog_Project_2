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
        
        input_shape = (self.board_size*self.board_size + 1,)  # Board size squared + player indicator
        model.add(layers.InputLayer(input_shape=input_shape))  # Input layer
        
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation)) # Hidden layers
        model.add(layers.Dense(self.board_size**2, activation="softmax"))  # Output layer
        
        optimizer = self._get_optimizer() # Optimizer
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # Compile model

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
        if state.ndim == 1: 
            state = state.reshape(1, -1)
        return self.model.predict(state, verbose=0)

    def save_net(self, filename):
        self.model.save(filename)

    def load_net(self, filename):
        self.model = models.load_model(filename)