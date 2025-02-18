import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def numpy_arrays():
    # global arr_1d, arr_2d

    print("1D Array: \n", arr_1d)
    


print("2D Array:\n", arr_2d)
print("")
print("Sum of 1D Array: ", np.sum(arr_1d))
print("Mean of 2D Array: ", np.mean(arr_2d))
print("")
print("Transpose of 2D Array: \n", arr_2d.T)
print("")

numpy_arrays()


# Image Processing with NumPy (Indexing & Slicing in Action)

# Creating a grayscale image using a 2D NumPy array

def Image_Processor():
    image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    print("Original Image:\n", image)

    # Slicing a portion of the image
    cropped = image[1:4, 1:4]
    print("Cropped Section:\n", cropped)

    # Inverting colors
    inverted_image = 255 - image
    print("Inverted Image:\n", inverted_image)

Image_Processor()

# Load an image (update path as needed)
image_path = r"C:\Users\Aquib\Downloads\Class-4\image.jpg"

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
else:
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load the image. Check the file format.")
    else:
        print("Image loaded successfully!")

        # Convert image to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Scaling Transformation
        def scale_image(image, scale_factor):
            new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
            scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            return scaled_image

        # Rotation Transformation
        def rotate_image(image, angle):
            rows, cols = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            return rotated_image

        # Translation Transformation
        def translate_image(image, tx, ty):
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
            return translated_image

        # Example transformations
        scaled_image = scale_image(image, 1.5)  # Scale by 1.5
        rotated_image = rotate_image(image, 45)  # Rotate by 45 degrees
        translated_image = translate_image(image, 50, 30)  # Translate

        # Display images using Matplotlib
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Scaled Image")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Rotated Image")
        axes[2].axis("off")

        axes[3].imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Translated Image")
        axes[3].axis("off")

        plt.show()

# Face Detection Section
image_path = r"C:\Users\Aquib\Downloads\Class-4\image.jpg"

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
else:
    image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Face detection running...")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detect faces
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the detected faces using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Faces")
    plt.axis("off")
    plt.show()
