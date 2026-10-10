import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from PIL import Image, ImageTk
import os


class PCBLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Component Label Editor")
        self.root.geometry("1100x700")  # Slightly larger window
        self.json_file = "/home/osama/PCB_Inspection/App/tools/classifier_config.json"  # Default JSON file
        self.image_folder = "/home/osama/PCB_Inspection/App/tools/PW881-CD_components"  # Folder containing component images
        self.data = self.load_json()
        self.selected_component = None  # Store the selected component

        # UI Elements
        self.component_list = tk.Listbox(root, width=30, height=15)
        self.component_list.grid(row=0, column=0, rowspan=7, padx=10, pady=10)
        self.component_list.bind("<<ListboxSelect>>", self.load_component)

        self.labels = ["Missing", "Shifted", "IC", "Ignore", "Color", "Rotation"]
        self.options = {
            "Missing": ["Not Missing", "Missing"],
            "Shifted": ["Not Shifted", "Shifted"],
            "IC": ["Not an IC", "IC is downwards", "IC is upwards"],
            "Ignore": ["Ignore", "Short", "Not Short"],
            "Color": ["Not_Capacitor", "YELLOW", "BROWN", "WHITE"],
            "Rotation": [
                str(angle) for angle in range(0, 361, 90)
            ],  # Rotation angles in 90-degree increments
        }

        self.entries = {}
        for i, label in enumerate(self.labels):
            tk.Label(root, text=label).grid(row=i, column=1, sticky="w", padx=10)
            self.entries[label] = ttk.Combobox(root, values=self.options[label])
            self.entries[label].grid(row=i, column=2, padx=10)

        self.save_button = tk.Button(root, text="Save", command=self.save_changes)
        self.save_button.grid(row=6, column=2, pady=10)

        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=3, rowspan=7, padx=10, pady=10)

        self.load_components()

    def load_json(self):
        try:
            with open(self.json_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "JSON file not found!")
            return {}
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format!")
            return {}

    def load_components(self):
        """Load component names into the listbox"""
        self.component_list.delete(0, tk.END)
        for component in self.data.keys():
            self.component_list.insert(tk.END, component)

    def load_component(self, event):
        """Load selected component's data into the fields"""
        selected = self.component_list.curselection()
        if not selected:
            return

        self.selected_component = self.component_list.get(
            selected[0]
        )  # Store selection

        if (
            self.selected_component not in self.data
            or not isinstance(self.data[self.selected_component], list)
            or len(self.data[self.selected_component]) < 2
        ):
            messagebox.showerror(
                "Error", f"Invalid data for component: {self.selected_component}"
            )
            return

        rotation = self.data[self.selected_component][0]  # Rotation value
        values = self.data[self.selected_component][1]

        if not isinstance(values, list) or len(values) != 5:
            messagebox.showerror(
                "Error", f"Invalid attributes for component: {self.selected_component}"
            )
            return

        for i, key in enumerate(self.labels[:-1]):  # Excluding rotation
            self.entries[key].set(values[i])
        self.entries["Rotation"].set(str(rotation))

        self.display_image(self.selected_component)

    def display_image(self, component):
        """Display the image of the selected component"""
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(self.image_folder, f"{component}{ext}")
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((300, 300), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(img)
                self.image_label.config(image=img, text="")
                self.image_label.image = img
                return
        self.image_label.config(image="", text="No Image Found")

    def save_changes(self):
        """Save updated values to the JSON file"""
        if not self.selected_component:
            messagebox.showwarning("Warning", "Select a component to update")
            return

        new_rotation = int(self.entries["Rotation"].get())
        new_values = [
            self.entries[key].get() for key in self.labels[:-1]
        ]  # Excluding rotation

        self.data[self.selected_component] = [new_rotation, new_values]

        with open(self.json_file, "w") as f:
            json.dump(self.data, f, indent=4)

        # messagebox.showinfo("Success", f"Updated {self.selected_component}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PCBLabelApp(root)
    root.mainloop()
