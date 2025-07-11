import cv2
from deepface import DeepFace


capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    
    result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

    if result:
        result = result[0]   

        age = result['age']
        gender = result['dominant_gender']
        emotion = result['dominant_emotion']
        region = result['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        label = f'{gender}, {int(age)}, {emotion}'
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
    cv2.imshow("Age, Gender, Emotion Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()