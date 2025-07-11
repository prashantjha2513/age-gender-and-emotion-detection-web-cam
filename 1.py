from deepface import DeepFace


result = DeepFace.analyze(img_path = "bnm.jpg", actions = ['emotion'])
print("Dominant Emotion:",result[0]["dominant_emotion"])
result = DeepFace.analyze(img_path = "bnm.jpg", actions = ['age'])
print("Predicted age:",result[0]["age"])
result = DeepFace.analyze(img_path = "bnm.jpg", actions = ['gender'])
print("Dominant Gender:",result[0]["dominant_gender"])

