import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras import layers





# AE--------------------------------------------------------------------------------------------------#

# Define the Autoencoder Model
def create_AE(input_dim=1, latent_dim=100, activation='relu', loss='mae', optimizer='adam'):
    autoencoder = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        layers.Flatten(),
        layers.Dense(latent_dim, activation=activation),
        layers.Dense(input_dim, activation='sigmoid')  # Decoder to reconstruct the input
    ])
    
    autoencoder.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
    
    return autoencoder

# Manual GridSearchCV function
def run_AE(X_train_scaled, X_test_scaled, param_grid=None):
    if param_grid is None:
        param_grid = {
            'input_dim': [X_train_scaled.shape[1]], 
            'latent_dim': [10, 25, 50, 100],
            'activation': ['relu', 'sigmoid', 'tanh'],
            'loss': ['mae'],#, 'binary_crossentropy'],#
            'optimizer': ['sgd', 'adam'],
            'epochs': [10],
            'batch_size': [32]
        }

    # Manually define a function for model training
    def fit_model(input_dim, latent_dim, activation, loss, optimizer, epochs, batch_size):
        autoencoder = create_AE(input_dim=input_dim, latent_dim=latent_dim, activation=activation, 
                                 loss=loss, optimizer=optimizer)
        
        # Fit the model with validation data
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test_scaled, X_test_scaled), verbose=0)
        return autoencoder

    # Custom GridSearchCV logic
    best_model = None
    best_params = None
    best_score = float('inf')

    for latent_dim in param_grid['latent_dim']:
        for activation in param_grid['activation']:
            for loss in param_grid['loss']:
                for optimizer in param_grid['optimizer']:
                    for epochs in param_grid['epochs']:
                        for batch_size in param_grid['batch_size']:
                            
                            # Fit the model with a specific combination of hyperparameters
                            autoencoder = fit_model(X_train_scaled.shape[1], latent_dim, activation, loss, optimizer, epochs, batch_size)
                            
                            # Get the validation loss (first item in the returned list)
                            val_loss = autoencoder.evaluate(X_test_scaled, X_test_scaled, verbose=0)[0]
                            # Track the best parameters based on validation loss
                            if val_loss < best_score:
                                best_score = val_loss
                                best_params = {
                                    'latent_dim': latent_dim,
                                    'activation': activation,
                                    'loss': loss,
                                    'optimizer': optimizer,
                                    'epochs': epochs,
                                    'batch_size': batch_size
                                }
                                best_model = autoencoder
    print(f"latent dimension: {latent_dim}, activation:{activation}, opt:{optimizer},MAE:{best_score}")               
    # Ensure the best model is built
    best_model.predict(X_train_scaled[:1])  # Call the model on one sample to build it

    # **New Fix: Call the encoder with proper outputs**
    encoder_layer = tf.keras.Sequential(best_model.layers[:2])  # Extract only the encoder layers
    
    # Ensure the encoder is also built
    encoder_layer.predict(X_train_scaled[:1])  # Trigger the encoder to initialize

    # Generate latent features from encoder representation
    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train = AE_train.add_prefix('feature_')


    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test = AE_test.add_prefix('feature_')
    
    return AE_train, AE_test











