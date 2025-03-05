# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** Sai Vishal D  
- **Register Number:** 212223230180

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Eagle_in_Flight.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
```

#### 2. Print the image width, height & Channel.
```python
img_gray.shape
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img_gray)
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite("output.png", img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(img_rgb)
plt.title("Color Image")
plt.show()
img_rgb.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped_eagle = image_rgb[y:y+h, x:x+w]
```

#### 8. Resize the image up by a factor of 2x.
```python
scale_factor = 2
new_width = int(cropped_eagle.shape[1] * scale_factor)
new_height = int(cropped_eagle.shape[0] * scale_factor)

resized_image = cv2.resize(cropped_eagle, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
```

#### 9. Flip the cropped/resized image horizontally.
```python
flipped_image = cv2.flip(resized_image, 1)
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img = cv2.imread("Apollo-11-launch.jpg")
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
font_scale = 2
font_thickness = 2
text_color = (255, 255, 255)
padding = 10

image_height, image_width, _ = img.shape
(text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)

x = (image_width - text_width) // 2
y = image_height - padding

cv2.putText(img, text, (x, y), font_face, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rectangle_color = (255, 0, 255)
rectangle_thickness = 3
cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, rectangle_thickness)
```

#### 13. Display the final annotated image.
```python
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Apollo 11 Launch - Annotated")
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img = cv2.imread("boy.jpg")
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
import numpy as np
matrix_ones = np.ones(image.shape, dtype="uint8") * 50
bright_image = cv2.add(image, matrix_ones)
bright_image_rgb = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
# YOUR CODE HERE
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_brighter_rgb = cv2.cvtColor(img_brighter, cv2.COLOR_BGR2RGB)
img_darker_rgb = cv2.cvtColor(img_darker, cv2.COLOR_BGR2RGB)
```

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
img_float = img.astype(np.float32)

matrix1 = img_float * 1.1  # Increase contrast by 10%
matrix2 = img_float * 1.2  # Increase contrast by 20%

img_higher1 = np.clip(matrix1, 0, 255).astype(np.uint8)
img_higher2 = np.clip(matrix2, 0, 255).astype(np.uint8)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_higher1_rgb = cv2.cvtColor(img_higher1, cv2.COLOR_BGR2RGB)
img_higher2_rgb = cv2.cvtColor(img_higher2, cv2.COLOR_BGR2RGB)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(img_lower_rgb)
plt.title("Lower Contrast (0.9x)")

plt.subplot(1, 3, 3)
plt.imshow(img_higher_rgb)
plt.title("Higher Contrast (1.1x)")

plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
B, G, R = cv2.split(img)
zeros = np.zeros_like(B)

B_colored = cv2.merge([B, zeros, zeros])
G_colored = cv2.merge([zeros, G, zeros])
R_colored = cv2.merge([zeros, zeros, R])

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
B_rgb = cv2.cvtColor(B_colored, cv2.COLOR_BGR2RGB)
G_rgb = cv2.cvtColor(G_colored, cv2.COLOR_BGR2RGB)
R_rgb = cv2.cvtColor(R_colored, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(B_rgb)
plt.title("Blue Channel")

plt.subplot(1, 4, 3)
plt.imshow(G_rgb)
plt.title("Green Channel")

plt.subplot(1, 4, 4)
plt.imshow(R_rgb)
plt.title("Red Channel")

plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged_img = cv2.merge([B, G, R])
merged_rgb = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(hsv_img)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(H, cmap="gray")
plt.title("Hue Channel (H)")

plt.subplot(1, 4, 3)
plt.imshow(S, cmap="gray")
plt.title("Saturation Channel (S)")

plt.subplot(1, 4, 4)
plt.imshow(V, cmap="gray")
plt.title("Value Channel (V)")

plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_img = cv2.merge([B, G, R])
merged_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(merged_rgb)
plt.title("Merged HSV Image")

plt.subplot(1, 3, 3)
plt.imshow(H, cmap="gray")
plt.title("Hue (H) Channel")

plt.show()
```

## Output:
- **i)** Read and Display an Image.
- ![Eagle_in_Flight](https://github.com/user-attachments/assets/88044571-7d56-4338-a926-512caa808e8c)
  
- **ii)** Adjust Image Brightness.
- ![image](https://github.com/user-attachments/assets/9d4746c1-4f39-44e1-b747-7df88603bb4f)
 
- **iii)** Modify Image Contrast.
- ![image](https://github.com/user-attachments/assets/a0d73628-808f-449b-92a9-d9727c27f95c)
 
- **iv)** Generate Third Image Using Bitwise Operations.
- ![image](https://github.com/user-attachments/assets/414b03d5-50ec-4c81-a815-25c03a69b061)


## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

