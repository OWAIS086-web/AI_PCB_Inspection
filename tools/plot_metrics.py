import matplotlib.pyplot as plt
import re
import numpy as np
import cv2
import os 

def get_metrics(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    brightness = np.mean(image_rgb)
    
    contrast = np.max(image_rgb) - np.min(image_rgb)
    
    avg_red = np.mean(image_rgb[:,:,0])
    avg_green = np.mean(image_rgb[:,:,1])
    avg_blue = np.mean(image_rgb[:,:,2])
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[:,:,1])
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = laplacian.var()
    
    hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    height, width, _ = image.shape
    
    noise_level = np.std(image_rgb)
    
    
    return brightness, contrast, avg_red, avg_green, avg_blue, saturation, sharpness, noise_level


def parse_log(log_data):
    log_pattern = re.compile(r"Brightness: (\d+\.\d+)\nContrast: (\d+\.\d+)\nAvg Red Level: (\d+\.\d+)\nAvg Green Level: (\d+\.\d+)\nAvg Blue Level: (\d+\.\d+)\nSaturation: (\d+\.\d+)\nSharpness \(Laplacian Variance\): (\d+\.\d+)\nResolution: \d+x\d+\nNoise Level \(std dev\): (\d+\.\d+)")
    
    brightness = []
    contrast = []
    avg_red = []
    avg_green = []
    avg_blue = []
    saturation = []
    sharpness = []
    noise_level = []
    
    for match in re.finditer(log_pattern, log_data):
        brightness.append(float(match.group(1)))
        contrast.append(float(match.group(2)))
        avg_red.append(float(match.group(3)))
        avg_green.append(float(match.group(4)))
        avg_blue.append(float(match.group(5)))
        saturation.append(float(match.group(6)))
        sharpness.append(float(match.group(7)))
        noise_level.append(float(match.group(8)))
    
    return brightness, contrast, avg_red, avg_green, avg_blue, saturation, sharpness, noise_level

def plot_metrics_with_highlight(metrics, metric_names, highlight_index, highlight_color='green'):
    os.makedirs('metrics', exist_ok=True)
    for i, metric in enumerate(metrics):
        plt.figure()
        plt.plot(metric, label=metric_names[i])
        
        plt.scatter(highlight_index, metric[highlight_index], color=highlight_color, zorder=5, label=f'Reference Image')
        
        plt.xlabel('Time')
        plt.ylabel(metric_names[i])
        plt.title(f'{metric_names[i]} over Time')
        plt.legend()
        plt.savefig(f"metrics/{metric_names[i]}_highlighted.png")  

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
    return log_data

def get_ideal_case_metrics(test_image):
    return get_metrics(test_image)

log_file_path = "logs/inputs.log"
test_image_path = "Test_Images/ref.bmp"  

log_data = read_log_file(log_file_path)

brightness, contrast, avg_red, avg_green, avg_blue, saturation, sharpness, noise_level = parse_log(log_data)
test_image = cv2.imread(test_image_path)
ideal_brightness, ideal_contrast, ideal_avg_red, ideal_avg_green, ideal_avg_blue, ideal_saturation, ideal_sharpness, ideal_noise_level = get_ideal_case_metrics(test_image)

metric_names = [
    'Brightness', 'Contrast', 'Avg Red Level', 'Avg Green Level', 'Avg Blue Level',
    'Saturation', 'Sharpness', 'Noise Level'
]

metrics = [brightness, contrast, avg_red, avg_green, avg_blue, saturation, sharpness, noise_level]

highlight_index = len(brightness) - 1  

plot_metrics_with_highlight(metrics, metric_names, highlight_index, highlight_color='green')
