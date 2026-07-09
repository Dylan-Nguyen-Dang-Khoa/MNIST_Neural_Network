from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
from pathlib import Path

from neural_network import Network, IO

app = Flask(__name__, template_folder="../templates")

WEIGHTS_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "0.97814" / "model_kaggle_parameters.npz"
)

nn = Network()
IO.load_parameters(nn.l1, nn.l2, nn.l3, filepath=WEIGHTS_PATH)
