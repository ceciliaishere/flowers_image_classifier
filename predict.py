import argparse
import json
from utils import load_model, process_image, predict

parser = argparse.ArgumentParser(
    description='Flower Classification Deep Neural Network',
)

parser.add_argument('image_path', action="store", type=str)
parser.add_argument('model', action="store", type=str)
parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int)
parser.add_argument('--category_names', action="store",
                    dest="category_names_json", type=str)

args = parser.parse_args()


### Processing Image and Loading Model
processed_image = process_image(args.image_path)
model = load_model(args.model)

### Running Prediction
probs, classes = predict(processed_image, model, args.top_k)

### Printing Results
print("Class: \t\t\t Probability:")
print("----- \t\t\t -----")
if args.category_names_json == None:
    for i in range(len(probs)):
        print(str(classes[i]+1), "\t\t\t", probs[i])
else:
    with open(args.category_names_json, 'r') as f:
        class_names = json.load(f)
    for i in range(len(probs)):
        print(class_names[str(classes[i]+1)], "\t\t", probs[i])
        
        
#Test Cases
#$ python predict.py test_images/cautleya_spicata.jpg model.h5
#$ python predict.py test_images/cautleya_spicata.jpg model.h5 --top_k 5
#$ python predict.py test_images/cautleya_spicata.jpg model.h5 --category_names label_map.json
#$ python predict.py test_images/cautleya_spicata.jpg model.h5 --top_k 8 --category_names label_map.json


