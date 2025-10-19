import os
import gdown

def download_model():
    """Download model from Google Drive if not present"""
    model_path = 'models/best_model.pth'
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Only download if model doesn't exist
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        
        # Your Google Drive File ID
        file_id = "1jnoXxzaPObvShpwr39qOmnAneW1iqZVz"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            print("✅ Model downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print("✅ Model already exists locally")
    
    return True

if __name__ == "__main__":
    download_model()
