import gradio as gr
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_sentiment(review):
    prediction = model.predict([review])[0]
    probability = model.predict_proba([review])[0]
    label = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = round(max(probability) * 100, 2)
    return f"{label} (Confidence: {confidence}%)"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter Movie Review", lines=5),
    outputs=gr.Textbox(label="Sentiment"),
    title="IMDB Sentiment Analyzer",
    description="Enter a movie review to analyze its sentiment"
)

iface.launch()