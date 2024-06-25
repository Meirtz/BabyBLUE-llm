#!/bin/bash

# Create directories
mkdir -p images
mkdir -p download

# Generate random images using Python
python3 << END
import os
import random
from PIL import Image

# Set directory paths
image_dir = "images"

# Generate random images
for i in range(5):  # Change the range to generate more or fewer images
    image_path = os.path.join(image_dir, f"image_{i+1}.png")
    img = Image.new('RGB', (100, 100), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img.save(image_path)
END

echo "Directories and images prepared."
