# ✨ Which Bollywood Celebrity Are You? 🎬

Ever wondered which Bollywood star you resemble the most?  
Upload your photo and let our **AI-powered face recognition app** find your celebrity twin from Bollywood! 🧠📸
## 🔍 How It Works

🧠 **AI Models Used**:
- **Face Detection** → [MTCNN](https://github.com/ipazc/mtcnn)
- **Feature Extraction** → [VGGFace (ResNet50)](https://github.com/rcmalli/keras-vggface)
- **Matching** → Cosine Similarity via `scikit-learn`

📸 **Workflow**:
1. Upload your image
2. Face is detected and cropped
3. Features are extracted using ResNet50
4. Compared with known Bollywood faces
5. Your closest celebrity match is revealed!
