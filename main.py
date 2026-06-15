import cv2
import numpy as np
import tensorflow as tf

# Load labels
with open("converted_tflite_quantized/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Load model
interpreter = tf.lite.Interpreter(
    model_path="converted_tflite_quantized/model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

print("Model Loaded Successfully")
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to access camera")
        break

    # Resize image
    image = cv2.resize(frame, (width, height))

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # UINT8 model
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.uint8)

    # Prediction
    interpreter.set_tensor(
        input_details[0]['index'],
        image
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )[0]

    index = np.argmax(prediction)

    confidence = prediction[index] / 255.0 * 100

    label = labels[index]

    # Color based on class
    if label == "Healthy Plants":
        color = (0, 255, 0)

    elif label == "Dry Plants":
        color = (0, 255, 255)

    else:
        color = (0, 0, 255)

    text = f"{label} ({confidence:.1f}%)"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow(
        "AI Plant Health Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()