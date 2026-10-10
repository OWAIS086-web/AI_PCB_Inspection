import json
import re

template = json.load(
    open("/home/osama/PCB_Inspection/App/configs/template_2/annotations.json", "r")
)
options_template = json.load(
    open(
        "/home/osama/PCB_Inspection/App/configs/template_2/classifier_config.json", "r"
    )
)
groups = {}
for group_name, group_labels in template.items():
    label_options = set(options_template[group_name][-1])
    orientation_options = set([options_template[group_name][0]])
    match = re.search(r"[A-Za-z]+", group_name)
    group_label = match.group() if match else ""
    if group_label not in groups:
        groups[group_label] = {
            "to_label": list(group_labels.keys()),
            "group_components": [group_name],
            "orientation_options": orientation_options,
            "label_options": label_options,
        }
    else:
        for option in label_options:
            groups[group_label]["label_options"].add(option)

        for option in orientation_options:
            groups[group_label]["orientation_options"].add(option)

        groups[group_label]["group_components"].append(group_name)

for group_label, group in groups.items():
    groups[group_label]["label_options"] = list(groups[group_label]["label_options"])
    groups[group_label]["orientation_options"] = list(
        groups[group_label]["orientation_options"]
    )

json.dump(groups, open("tools/label_groups.json", "w"), indent=4)
print("Created label groups")

import pdb

pdb.set_trace()
