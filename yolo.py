import cv2
import smtplib
import os
from email.message import EmailMessage
from ultralytics import YOLO

# Load YOLO model
model = YOLO("model.pt")  # Replace with your trained model file

# Email Credentials
EMAIL_SENDER = "sender mail"  # Your Gmail address
EMAIL_PASSWORD = "api password"   # Your 16-character App Password
EMAIL_RECEIVER = "receiver mail"  # Receiver's Email Address
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL Port

# Function to send email alert with an image
def send_email_alert(detected_object, image_path):
    subject = "⚠️ Alert: Dangerous Object Detected!"
    body = f"A {detected_object} has been detected on the camera! See the attached image."

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the image
    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="detected_object.jpg")

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email alert sent! ({detected_object} detected)")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Initialize Webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Unable to access webcam")
    exit()

while True:
    success, frame = webcam.read()
    
    if not success:
        continue  # Skip if frame is not captured properly
    
    # Run YOLO detection
    results = model(frame, conf=0.5)

    detected_objects = []
    
    # Check if knife or scissors are detected
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID of detected object
            class_name = model.names[class_id]  # Get object name
            
            if class_name in ["person", "scissors"]:  # Modify class names as per your model
                detected_objects.append(class_name)

    # Send email if knife or scissors are detected
    if detected_objects:
        image_path = "detected_object.jpg"
        cv2.imwrite(image_path, frame)  # Save the detected frame

        for obj in detected_objects:
            send_email_alert(obj, image_path)

        os.remove(image_path)  # Delete the image after sending

    # Display detections
    cv2.imshow("Live Camera", results[0].plot())  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
