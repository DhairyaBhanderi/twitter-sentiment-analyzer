import pandas as pd
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
import sys
sys.path.append('..')

from preprocessing.cleaner import TextSequencer
from model.bilstm_model import BiLSTMSentimentModel

class ModelTrainer:
    """
    Train Bi-LSTM sentiment model
    """
    
    def __init__(self, vocab_size=50000, max_length=100, embedding_dim=128, lstm_units=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.sequencer = TextSequencer(vocab_size, max_length)
        self.model_builder = None
        self.history = None
    
    def load_data(self, train_path='data/train.csv', val_path='data/val.csv'):
        """
        Load training and validation data
        """
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Use processed_text if available, else text
        text_col = 'processed_text' if 'processed_text' in train_df.columns else 'text'
        
        X_train = train_df[text_col].values
        y_train = train_df['sentiment'].values
        
        X_val = val_df[text_col].values
        y_val = val_df['sentiment'].values
        
        return X_train, y_train, X_val, y_val
    
    def prepare_sequences(self, X_train, X_val):
        """
        Convert texts to sequences
        """
        # Fit tokenizer on training data
        self.sequencer.fit_tokenizer(X_train)
        
        # Transform to sequences
        X_train_seq = self.sequencer.texts_to_sequences(X_train)
        X_val_seq = self.sequencer.texts_to_sequences(X_val)
        
        # Save tokenizer
        self.sequencer.save_tokenizer('model/tokenizer.pkl')
        
        return X_train_seq, X_val_seq
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights for imbalanced datasets
        """
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print("\nClass weights:")
        for cls, weight in class_weights.items():
            print(f"Class {cls}: {weight:.4f}")
        
        return class_weights
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
        """
        Train the model
        """
        # Build model
        self.model_builder = BiLSTMSentimentModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            max_length=self.max_length,
            num_classes=3
        )
        
        self.model_builder.build_model()
        self.model_builder.compile_model()
        
        print("Model Summary:")
        self.model_builder.summary()
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks('model/checkpoints/bilstm_best.h5')
        
        # Train model
        print("\nTraining model...")
        self.history = self.model_builder.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model_builder.save_model('model/checkpoints/bilstm_final.h5')
        
        return self.history
    
    def save_training_history(self, path='model/training_history.json'):
        """
        Save training history to JSON
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
        }
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"\nTraining history saved to {path}")

def train_sentiment_model():
    """
    Main training pipeline
    """
    trainer = ModelTrainer(
        vocab_size=50000,
        max_length=100,
        embedding_dim=128,
        lstm_units=128
    )
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val = trainer.load_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train_seq, X_val_seq = trainer.prepare_sequences(X_train, X_val)
    
    # Train model
    history = trainer.train(X_train_seq, y_train, X_val_seq, y_val, epochs=20, batch_size=64)
    
    # Save history
    trainer.save_training_history()
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

if __name__ == '__main__':
    train_sentiment_model()
