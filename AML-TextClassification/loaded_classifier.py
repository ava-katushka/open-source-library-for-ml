from textclassifier import TextClassifier
import pickle

filename_names = "./names.pkl"
with open(filename_names, "rb") as f: 
    names_loaded = pickle.load(f)

textClassifier = TextClassifier()
textClassifier.load("./class.pkl")

with open("./ex.txt", "r") as f:
    X = [f.read()]

predicted = textClassifier.predict(X)
offset = 0
mask = textClassifier.get_support()
print "results:"
for i in range(len(predicted[0])):
    while (mask[offset] != True):
        offset += 1
    if (predicted[0][i] != 0):
        print names_loaded[offset]
    offset +=1