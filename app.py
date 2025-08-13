from flask import Flask, request, render_template
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

# Define model architecture BEFORE loading
class CalorieRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.last_hidden_state[:, 0]).squeeze()

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("model/tokenizer")

# Load quantized model (with weights_only=False since it's a pickled object)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model/model_full_quantized.pt", map_location=device, weights_only=False)
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("food_text", "")

        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            predicted_log_cal = output.item()
            predicted_cal = torch.expm1(torch.tensor(predicted_log_cal)).item()
            prediction = round(predicted_cal, 2)

    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
