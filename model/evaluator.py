import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')

from tensorflow import keras
from preprocessing.cleaner import TextSequencer

class ModelEvaluator:
    """
    Evaluate trained sentiment model
    """
    
    def __init__(self, model_path='model/checkpoints/bilstm_best.h5', 
                 tokenizer_path='model/tokenizer.pkl'):
        self.model = keras.models.load_model(model_path)
        self.sequencer = TextSequencer()
        self.sequencer.load_tokenizer(tokenizer_path)
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def load_test_data(self, test_path='data/test.csv'):
        """
        Load test data
        """
        df = pd.read_csv(test_path)
        
        text_col = 'processed_text' if 'processed_text' in df.columns else 'text'
        
        X_test = df[text_col].values
        y_test = df['sentiment'].values
        
        return X_test, y_test
    
    def predict(self, texts):
        """
        Predict sentiment for texts
        """
        sequences = self.sequencer.texts_to_sequences(texts)
        predictions = self.model.predict(sequences, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_labels, confidences
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        """
        print("Evaluating model on test set...")
        
        # Predict
        y_pred, confidences = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall),
            'mean_confidence': float(np.mean(confidences))
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"Mean Confidence: {np.mean(confidences):.4f}")
        print("="*50 + "\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_map.values()))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, cm, y_pred
    
    def plot_confusion_matrix(self, cm, save_path='model/confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_map.values(),
                    yticklabels=self.label_map.values())
        plt.title('Confusion Matrix - Sentiment Classification', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def save_metrics(self, metrics, path='model/metrics.json'):
        """
        Save metrics to JSON
        """
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {path}")
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text
        """
        pred, conf = self.predict([text])
        sentiment = self.label_map[pred[0]]
        confidence = conf[0]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'label': int(pred[0])
        }

def evaluate_model():
    """
    Main evaluation pipeline
    """
    evaluator = ModelEvaluator(
        model_path='model/checkpoints/bilstm_best.h5',
        tokenizer_path='model/tokenizer.pkl'
    )
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = evaluator.load_test_data()
    print(f"Test samples: {len(X_test)}")
    
    # Evaluate
    metrics, cm, y_pred = evaluator.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(cm)
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    # Test single prediction
    print("\nTesting single prediction:")
    test_text = "This product is amazing! I love it so much!"
    result = evaluator.predict_single(test_text)
    print(f"Text: {result['text']}")
    print(f"Predicted Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")

if __name__ == '__main__':
    evaluate_model()
