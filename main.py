from src.models import AutoEncoder, RecurrentReversiblePredictor

if __name__ == "__main__":
    ae = AutoEncoder([1], [1], [1,1,1])
    rpm = RecurrentReversiblePredictor(32,32, 32, 6, 16)