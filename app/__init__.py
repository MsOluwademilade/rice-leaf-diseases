import os
from flask import Flask
from .config import Config

app = Flask(__name__)
app.config.from_object(Config)
from app import views

if __name__ == "__main__":
    app.run(debug=True, port=os.environ.get("PORT", 5000))
