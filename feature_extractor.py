from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
import os

# Load all image paths from saved pickle
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load VGGFace model using ResNet50 backend
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Function to extract feature from image using VGGFace
def feature_extractor(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result
    except Exception as e:
        print(f"[Error] Failed on {img_path}: {e}")
        return None

# Extract features for each image
features = []
for file in tqdm(filenames):
    if os.path.exists(file):
        feature = feature_extractor(file, model)
        if feature is not None:
            features.append(feature)
    else:
        print(f"[Warning] File not found: {file}")

# Save extracted features to pickle
pickle.dump(features, open('embedding.pkl', 'wb'))

print("✅ Feature extraction completed and saved to 'embedding.pkl'")
