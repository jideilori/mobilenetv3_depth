# Load and evalute on Eigen's test data
from evaluate import load_test_data, evaluate
from model import MobileNetv3_model

rgb, depth, crop = load_test_data()
evaluate(MobileNetv3_model, rgb, depth, crop)