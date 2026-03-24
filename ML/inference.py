import joblib

from ML.config import MODEL_PATH, VECTORIZER_PATH
from ML.preprocess import TextPreprocessor

class SpamInferenceService:
    def __init__(self)->None:
        print("[1/3] Initializing inference service...")
        self.preprocessor = TextPreprocessor()
        print(f"Loading vectorizer from: {VECTORIZER_PATH}")
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        print(f"Loading model from: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        print("Inference service is ready.")

    def predict (self,text:str) -> dict:
        print("[2/3] Preprocessing input text...")
        cleaned_text = self.preprocessor.clean_text(text)
        print(f"Cleaned text: {cleaned_text}")

        print("[3/3] Transforming text and running prediction...")
        vectorized_text = self.vectorizer.transform([cleaned_text])
        print(f"Vectorized text shape: {vectorized_text.shape}")

        predicted_label = self.model.predict(vectorized_text)[0]
        prediction_probabilities = self.model.predict_proba(vectorized_text)[0]
        confidence = float(max(prediction_probabilities))
        print(f"Predicted label: {predicted_label}")
        print(f"Prediction probabilities: {prediction_probabilities}")
        print(f"Confidence: {confidence:.4f}")


        return{
            "input_text" : text,
            "cleaned_text" : cleaned_text,
            "predicted_label" : int(predicted_label),
            "confidence" : confidence,
        }

if __name__ == "__main__":
    service = SpamInferenceService()

    sample_text = "Congratulations ! You have won an iphone. click this link now  !!!"
    result = service.predict(sample_text)

    print("Inference result : ")
    print(result)

