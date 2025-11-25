import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

class BiLSTMSentimentModel:
    """
    Bidirectional LSTM model for sentiment analysis
    """
    
    def __init__(self, vocab_size=50000, embedding_dim=128, lstm_units=128, 
                 max_length=100, num_classes=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """
        Build Bi-LSTM architecture
        """
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding'
            ),
            
            # Spatial dropout for regularization
            SpatialDropout1D(0.2),
            
            # Bidirectional LSTM
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=False),
                name='bilstm'
            ),
            
            # Dropout layer
            Dropout(0.5),
            
            # Dense layer
            Dense(64, activation='relu', name='dense_1'),
            Dropout(0.3),
            
            # Output layer
            Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss function
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_callbacks(self, checkpoint_path='model/checkpoints/bilstm_best.h5'):
        """
        Get training callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """
        Print model summary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.summary()
    
    def save_model(self, path='model/checkpoints/bilstm_sentiment.h5'):
        """
        Save model to file
        """
        if self.model is None:
            raise ValueError("Model not built.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='model/checkpoints/bilstm_sentiment.h5'):
        """
        Load model from file
        """
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return self.model

def create_model(vocab_size=50000, embedding_dim=128, lstm_units=128, 
                 max_length=100, num_classes=3):
    """
    Factory function to create and compile model
    """
    model_builder = BiLSTMSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        num_classes=num_classes
    )
    
    model_builder.build_model()
    model_builder.compile_model()
    
    print("Model architecture:")
    model_builder.summary()
    
    return model_builder

if __name__ == '__main__':
    # Test model creation
    model_builder = create_model()
