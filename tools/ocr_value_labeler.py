import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from PIL import Image, ImageTk  # Required for image display
import cv2


class ResistorLabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resistor Component Editor")
        self.root.geometry("900x600")

        self.data = {}
        self.json_file = ""
        self.image_folder = ""
        self.selected_component = None  # Store selected component name

        # Select JSON File Button
        self.select_json_button = tk.Button(
            root, text="Select JSON File", command=self.select_json
        )
        self.select_json_button.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        # Select Image Folder Button
        self.select_image_button = tk.Button(
            root, text="Select Image Folder", command=self.select_image_folder
        )
        self.select_image_button.grid(row=0, column=2, padx=10, pady=10, columnspan=2)

        # UI Elements
        self.component_list = tk.Listbox(root, width=30, height=20)
        self.component_list.grid(row=1, column=0, rowspan=6, padx=10, pady=10)
        self.component_list.bind("<<ListboxSelect>>", self.load_component)

        tk.Label(root, text="Value:").grid(row=1, column=1, padx=10, sticky="w")
        self.value_entry = tk.Entry(root)
        self.value_entry.grid(row=1, column=2, padx=10)

        tk.Label(root, text="Rotation:").grid(row=2, column=1, padx=10, sticky="w")
        self.rotation_entry = ttk.Combobox(root, values=[0, 90, 180, 270])
        self.rotation_entry.grid(row=2, column=2, padx=10)

        self.save_button = tk.Button(
            root, text="Save Changes", command=self.save_changes
        )
        self.save_button.grid(row=3, column=2, pady=10)

        # Create a frame for the image display
        self.image_frame = tk.Frame(root, width=350, height=400, bg="gray")
        self.image_frame.grid(
            row=0, column=4, rowspan=10, padx=10, pady=10, sticky="nsew"
        )

        # Image Label inside the frame
        self.image_label = tk.Label(
            self.image_frame, text="No Image Selected", bg="gray"
        )
        self.image_label.pack(expand=True, fill="both")

    def select_json(self):
        """Allow user to select the JSON file"""
        self.json_file = filedialog.askopenfilename(
            title="Select JSON File", filetypes=[("JSON Files", "*.json")]
        )
        if not self.json_file:
            return

        self.load_json()
        self.load_components()

    def select_image_folder(self):
        """Allow user to select the folder containing images"""
        self.image_folder = filedialog.askdirectory(title="Select Image Folder")

    def load_json(self):
        """Load the JSON file"""
        if not os.path.exists(self.json_file):
            messagebox.showerror("Error", "JSON file not found!")
            return

        try:
            with open(self.json_file, "r") as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format!")

    def load_components(self):
        """Load components into the Listbox"""
        self.component_list.delete(0, tk.END)
        for component in self.data.keys():
            self.component_list.insert(tk.END, component)

    def load_component(self, event):
        """Load the selected component and display its image"""
        selected = self.component_list.curselection()
        if not selected:
            return

        self.selected_component = self.component_list.get(
            selected[0]
        )  # Store selection
        value, rotation = self.data.get(self.selected_component, ["0", 0])

        self.value_entry.delete(0, tk.END)
        self.value_entry.insert(0, value)
        self.rotation_entry.set(rotation)

        # Display image if folder is selected
        if self.image_folder:
            image_path = os.path.join(
                self.image_folder, f"{self.selected_component}.jpg"
            )

            if os.path.exists(image_path):
                self.display_image(image_path)
            else:
                self.image_label.config(text="Image Not Found", image="", bg="gray")

    def display_image(self, image_path):
        """Display the selected image in the dedicated frame."""
        try:
            print(f"Loading image from: {image_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError("Image file does not exist!")

            # Open and resize image while maintaining aspect ratio
            image = Image.open(image_path)
            image.thumbnail((350, 400), Image.Resampling.LANCZOS)

            # Convert to Tkinter-compatible format
            self.photo = ImageTk.PhotoImage(image)

            # Update label inside the new frame
            self.image_label.config(image=self.photo, text="", bg="white")
            self.image_label.image = self.photo  # Prevent garbage collection
            self.root.update_idletasks()

        except Exception as e:
            print(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Could not open image: {str(e)}")

    def save_changes(self):
        """Save updated values and rotations back to JSON"""
        if not self.selected_component:
            messagebox.showwarning("Warning", "No component selected!")
            return

        new_value = self.value_entry.get()

        try:
            new_rotation = int(self.rotation_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Rotation must be a number!")
            return

        # Update the JSON data dictionary
        self.data[self.selected_component] = [new_value, new_rotation]

        # Save the updated data back to the JSON file
        if self.json_file:
            try:
                with open(self.json_file, "w") as f:
                    json.dump(self.data, f, indent=4)
                # messagebox.showinfo("Success", f"Updated {self.selected_component}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save JSON file: {str(e)}")
        else:
            messagebox.showerror("Error", "No JSON file selected!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ResistorLabelApp(root)
    root.mainloop()
