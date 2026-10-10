import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = '/home/osama/anaconda3/envs/pcb/bin/tesseract'


img= "/home/osama/PCB_Inspection/App/old_data/21.png"
# text = pytesseract.image_to_string(Image.open(img))

text = pytesseract.image_to_boxes(img) #pytesseract.image_to_data(Image.open(img), config=r'--oem 3', output_type=pytesseract.Output.DICT)

char_list = []
for line in pytesseract.image_to_boxes(img).splitlines():
    parts = line.split()
    if len(parts) == 6:
        char, x1, y1, x2, y2, _ = parts
        char_list.append(char)

print(char_list)
