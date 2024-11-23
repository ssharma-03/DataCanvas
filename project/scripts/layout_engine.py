from PIL import Image, ImageDraw, ImageFont
import os 

def create_infographic(chart_path, insights, output_path):
    img = Image.open(chart_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 10), insights, fill="black", font=font)
    img.save(output_path)
