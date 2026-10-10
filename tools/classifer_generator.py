import json

# Load the annotation JSON from a file
with open(
    "/home/osama/PCB_Inspection/App/tools/output_jsons/PW881-CD.json", "r"
) as file:
    annotations = json.load(file)

# Default classification template
classification_template = [
    "Not Missing",
    "Not Shifted",
    "Not an IC",
    "Ignore",
    "YELLOW",
]

# Generate classification JSON
classifier_json = {}
counter = 1  # For naming C1, C2, C3...

for component in annotations.keys():
    classifier_json[component] = [0, classification_template]
    counter += 1

# Save the classification JSON to a file
with open("classifier_config.json", "w") as file:
    json.dump(classifier_json, file, indent=4)

print("Classifier JSON generated successfully as 'classifier.json'.")
