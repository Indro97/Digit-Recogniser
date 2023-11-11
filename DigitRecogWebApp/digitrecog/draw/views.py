from django.shortcuts import render, redirect, HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import base64
import os
from PIL import Image
from .models import PredictionStats

model_path = 'draw/models/DigitRecogniser.model'

model = tf.keras.models.load_model(model_path)


def draw_view(request):
  return render(request, 'draw.html')




@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            # Get the base64-encoded image data from the frontend
            digit_image_data = request.POST.get('digit_image').split(',')[1]

            # Decode base64 data and create an InMemoryUploadedFile
            image_data = base64.b64decode(digit_image_data)
            image_file = InMemoryUploadedFile(
                ContentFile(image_data),
                None,  # Name of the file
                'uploaded_image.png',  # Actual name of the file
                'image/png',  # Content type
                len(image_data),
                None
            )

            # Save the uploaded image to the 'Image' folder
            image_directory = 'Image'
            os.makedirs(image_directory, exist_ok=True)
            image_path = os.path.join(image_directory, 'uploaded_image.png')
            with open(image_path, 'wb') as f:
                f.write(image_file.read())

            # Process the uploaded image with a white background
            background_color = (255, 255, 255)  # White background color
            image_with_white_background = add_white_background(image_path, background_color)

            # Convert the processed image to grayscale
            gray_image = cv2.cvtColor(image_with_white_background, cv2.COLOR_BGR2GRAY)

            # Invert the grayscale image
            inverted_image = np.invert(gray_image)

            # Normalize pixel values
            normalized_image = inverted_image / 255.0

            # Reshape to match the model's input shape
            image = np.reshape(normalized_image, (1, 28, 28, 1))

            # Make a prediction
            prediction_uploaded = model.predict(image)
            predicted_digit_uploaded = int(np.argmax(prediction_uploaded))
        
            context = {
                'predicted_digit': predicted_digit_uploaded,
                'image_path': image_path,
            }

            return render(request, 'prediction.html', context)

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})

def add_white_background(image_path, background_color):
    # Open the image using Pillow
    img = Image.open(image_path)

    # Create a new image with white background
    new_img = Image.new("RGBA", img.size, background_color)
    new_img.paste(img, (0, 0), img)

    # Save the new image
    new_img_path = os.path.join('Image', 'uploaded_image.png')
    new_img.save(new_img_path)

    return cv2.imread(new_img_path)[:, :, :3]  # Return BGR image without alpha channel


@csrf_exempt
def update_model(request):
    if request.method == 'POST':
        
        feedback = request.POST.get('feedback')

        stats, created = PredictionStats.objects.get_or_create(id=1)

        # Update the correct or incorrect count
        if feedback == 'correct':
            stats.correct_count += 1
        else:
            stats.incorrect_count += 1

        stats.save()

        if feedback == 'correct':
            print("do nothing")
            # return HttpResponse('correct ans')
        else:
            predicted_digit = int(request.POST.get('predicted_digit'))
            image = cv2.imread("Image/uploaded_image.png")[:, :, 0]
            image = np.invert(np.array([image]))
            update_model_internal(model, image, np.array([predicted_digit]))
            # return HttpResponse('Incorrent model updated')

        return HttpResponse(f'correct: {stats.correct_count} and incorrect: {stats.incorrect_count}')
    # Handle other cases or display an error page
    return redirect('error_view')


def update_model_internal(model, new_data, labels):
    new_data = tf.keras.utils.normalize(new_data, axis=1)
    model.fit(new_data, labels, epochs=4)