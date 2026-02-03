from keras.models import load_model

model = load_model("emotion_model.h5")

model.summary()