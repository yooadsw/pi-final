import streamlit as st
st.set_page_config(
    page_title="SIBI Translator", 
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)
import cv2
import traceback
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import threading
import time
import os
from pathlib import Path
from tensorflow import keras
import gc     
import psutil

# Fungsi untuk monitoring memory
def check_memory():
    memory = psutil.virtual_memory()
    return f"Memory: {memory.percent}% used"

# Tambahkan di main() function
if st.checkbox("Show System Info"):
    st.write(check_memory())

try:
    from keras.models import load_model
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow tidak tersedia. Model dinamis tidak dapat dimuat.")

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.error("streamlit-webrtc tidak tersedia. Fitur live video tidak dapat digunakan.")

# --- Inisialisasi MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Ganti bagian load_all_models() dengan:
@st.cache_resource
def load_all_models():
    """Load all models with error handling"""
    models = {}
    
    # Check if model files exist - cek di folder models/ dan root
    model_paths = {
        'model_statis': ['models/model_statis.pkl', 'model_statis.pkl'],
        'scaler_statis': ['models/scaler_statis.pkl', 'scaler_statis.pkl'],
        'le_statis': ['models/label_encoder_statis.pkl', 'label_encoder_statis.pkl'],
        'model_dinamis': ['models/model_dinamis_jz.keras', 'model_dinamis_jz.keras']
    }
    
    # Function to find existing file
    def find_model_file(possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    # Find model files
    model_files = {}
    missing_files = []
    
    for name, paths in model_paths.items():
        found_path = find_model_file(paths)
        if found_path:
            model_files[name] = found_path
        else:
            missing_files.append(f"{name} (checked: {', '.join(paths)})")
    
    if missing_files:
        st.error(f"File model tidak ditemukan: {', '.join(missing_files)}")
        st.info("Pastikan file model telah diupload ke repository GitHub Anda")
        
        # Show current directory contents for debugging
        st.write("Current directory contents:")
        for root, dirs, files in os.walk('.'):
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 2 * level
            st.write(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                st.write(f"{subindent}{file}")
        
        return None, None, None, None

    try:
        # Load static models
        models['model_statis'] = joblib.load(model_files['model_statis'])
        models['scaler_statis'] = joblib.load(model_files['scaler_statis'])
        models['le_statis'] = joblib.load(model_files['le_statis'])
        
        # Load dynamic model only if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            models['model_dinamis'] = load_model(model_files['model_dinamis'])
        else:
            models['model_dinamis'] = None
            st.warning("Model dinamis tidak dapat dimuat karena TensorFlow tidak tersedia")
        
        return models['model_statis'], models['scaler_statis'], models['le_statis'], models['model_dinamis']
        
    except Exception as e:
        st.error(f"Error memuat model")
        traceback.print_exc()
        return None, None, None, None


def mediapipe_detection(image, model):
    """MediaPipe detection with error handling"""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        st.error(f"Error dalam deteksi MediaPipe: {str(e)}")
        return image, None

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    if results is None:
        return np.zeros(126)  # 21*3*2 for both hands
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    try:
        a = np.array(a); b = np.array(b); c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: 
            angle = 360 - angle
        return angle
    except:
        return 0.0

def extract_geometric_features(landmarks):
    """Extract geometric features from hand landmarks"""
    try:
        if len(landmarks) < 63:  # 21 landmarks * 3 coordinates
            return np.zeros(15)  # Return default features
        
        wrist_coords = np.array(landmarks[0:3])
        normalized_landmarks = []
        
        for i in range(0, len(landmarks), 3):
            normalized_landmarks.extend(np.array(landmarks[i:i+3]) - wrist_coords)
        
        features = []
        
        # Distance features
        for i in [4, 8, 12, 16, 20]:
            if i*3+2 < len(normalized_landmarks):
                point = np.array(normalized_landmarks[i*3 : i*3+3])
                features.append(np.linalg.norm(point))
            else:
                features.append(0.0)
        
        # Angle features
        for i in range(5):
            try:
                base_idx = (i*4+1)*3
                pip_idx = (i*4+2)*3
                dip_idx = (i*4+3)*3
                tip_idx = (i*4+4)*3
                
                if tip_idx+1 < len(normalized_landmarks):
                    base_joint = np.array(normalized_landmarks[base_idx : base_idx+2])
                    pip_joint = np.array(normalized_landmarks[pip_idx : pip_idx+2])
                    dip_joint = np.array(normalized_landmarks[dip_idx : dip_idx+2])
                    tip_joint = np.array(normalized_landmarks[tip_idx : tip_idx+2])
                    
                    angle1 = calculate_angle(base_joint, pip_joint, dip_joint)
                    angle2 = calculate_angle(pip_joint, dip_joint, tip_joint)
                    
                    features.extend([angle1, angle2])
                else:
                    features.extend([0.0, 0.0])
            except:
                features.extend([0.0, 0.0])
        
        return features[:15]  # Ensure exactly 15 features
        
    except Exception as e:
        return np.zeros(15)

# --- Global Variables untuk Threading ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = "Menunggu..."
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = 0.0
if 'hand_detected' not in st.session_state:
    st.session_state.hand_detected = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Standby"

# --- WebRTC Video Processor ---
class SIBIVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Load models
        try:
            self.model_statis, self.scaler_statis, self.le_statis, self.model_dinamis = load_all_models()
            self.models_loaded = all([
                self.model_statis is not None,
                self.scaler_statis is not None, 
                self.le_statis is not None
            ])
        except Exception as e:
            self.models_loaded = False
            st.error(f"Error loading models in processor: {str(e)}")
        
        # Configuration
        self.actions_dinamis = np.array(['J', 'Z', 'Lainnya'])
        self.sequence_length = 30
        self.confidence_threshold = 0.98
        self.prediction_interval = 5
        self.frame_counter = 0
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # Thread lock
        self.lock = threading.Lock()
    
    def transform(self, frame):
        """Main video processing function"""
        try:
            # Convert av.VideoFrame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            if not self.models_loaded:
                cv2.putText(img, "Error: Model tidak dapat dimuat", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return img
            
            # Flip frame horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # MediaPipe detection
            image, results = mediapipe_detection(img, self.holistic)
            
            if results is None:
                cv2.putText(image, "Error: MediaPipe detection failed", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return image
            
            # Initialize variables
            hand_detected = False
            
            # Draw landmarks
            if results.right_hand_landmarks:
                hand_detected = True
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
            elif results.left_hand_landmarks:
                hand_detected = True
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
            
            # Update session state
            with self.lock:
                st.session_state.hand_detected = hand_detected
            
            # Extract keypoints and add to buffer
            keypoints = extract_keypoints(results)
            self.sequence_buffer.append(keypoints)
            
            # Prediction logic
            self.frame_counter += 1
            if self.frame_counter % self.prediction_interval == 0:
                self._perform_prediction()
            
            # Draw predictions on frame
            self._draw_predictions(image)
            
            return image
            
        except Exception as e:
            # Create error image
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: {str(e)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img
    
    def _draw_predictions(self, image):
        """Draw predictions on the image"""
        try:
            prediction_text = st.session_state.prediction_result
            confidence = st.session_state.confidence_score
            model_type = st.session_state.model_type
            hand_detected = st.session_state.hand_detected
            
            # Main prediction display
            cv2.putText(image, f"Prediksi: {prediction_text}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            
            # Confidence and model type
            cv2.putText(image, f"Confidence: {confidence:.1f}%", (15, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Model: {model_type}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hand detection status
            hand_status = "Terdeteksi" if hand_detected else "Tidak Terdeteksi"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(image, f"Tangan: {hand_status}", (15, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            cv2.putText(image, f"Display Error: {str(e)}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def _perform_prediction(self):
        """Perform prediction using both dynamic and static models"""
        try:
            if len(self.sequence_buffer) == self.sequence_length:
                # Try dynamic model first if available
                if self.model_dinamis is not None and TENSORFLOW_AVAILABLE:
                    try:
                        input_data = np.expand_dims(list(self.sequence_buffer), axis=0)
                        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                        res_dinamis = self.model_dinamis.predict(input_tensor, verbose=0)[0]
                        
                        pred_dinamis = self.actions_dinamis[np.argmax(res_dinamis)]
                        confidence = res_dinamis[np.argmax(res_dinamis)]
                        
                        # Update session state with thread lock
                        with self.lock:
                            if (pred_dinamis in ['J', 'Z']) and confidence > self.confidence_threshold:
                                st.session_state.prediction_result = pred_dinamis
                                st.session_state.confidence_score = confidence * 100
                                st.session_state.model_type = "Dinamis"
                                return
                    except Exception as e:
                        pass  # Fall back to static model
                
                # Static model prediction
                self._static_prediction()
        except Exception as e:
            with self.lock:
                st.session_state.prediction_result = "Error"
                st.session_state.confidence_score = 0.0
                st.session_state.model_type = "Error"
    
    def _static_prediction(self):
        """Perform static model prediction"""
        try:
            if not self.models_loaded:
                return
            
            last_frame_keypoints = self.sequence_buffer[-1]
            
            # Try right hand first, then left hand
            if np.any(last_frame_keypoints[63:]):
                single_hand_keypoints = last_frame_keypoints[63:]
            elif np.any(last_frame_keypoints[:63]):
                single_hand_keypoints = last_frame_keypoints[:63]
            else:
                single_hand_keypoints = None
            
            if single_hand_keypoints is not None:
                features_statis = extract_geometric_features(single_hand_keypoints)
                
                if len(features_statis) > 0:
                    scaled_features = self.scaler_statis.transform([features_statis])
                    pred_statis_encoded = self.model_statis.predict(scaled_features)
                    prediction = self.le_statis.inverse_transform(pred_statis_encoded)[0]
                    
                    with self.lock:
                        st.session_state.prediction_result = prediction
                        st.session_state.confidence_score = 85.0  # Placeholder confidence
                        st.session_state.model_type = "Statis"
                else:
                    with self.lock:
                        st.session_state.prediction_result = "..."
                        st.session_state.confidence_score = 0.0
                        st.session_state.model_type = "Menunggu"
            else:
                with self.lock:
                    st.session_state.prediction_result = "..."
                    st.session_state.confidence_score = 0.0
                    st.session_state.model_type = "Menunggu"
                    
        except Exception as e:
            with self.lock:
                st.session_state.prediction_result = "Error"
                st.session_state.confidence_score = 0.0
                st.session_state.model_type = "Error"

# --- Main Application ---
def debug_camera_access():
    """Debug camera access"""
    st.subheader("🔍 Camera Debug Info")
    
    # Check browser compatibility
    st.write("**Browser Info:**")
    st.write("- Pastikan menggunakan Chrome/Firefox/Edge")
    st.write("- Hindari Safari (limited WebRTC support)")
    
    # Check HTTPS
    st.write("**HTTPS Status:**")
    if st.query_params.get('https') == 'true':
        st.success("✅ Running on HTTPS")
    else:
        st.warning("⚠️ Running on HTTP - Camera may not work properly")
        st.info("Try: `streamlit run app.py --server.enableCORS=false --server.enableWebsocketCompression=false`")
    
    # Check permissions
    st.write("**Camera Permissions:**")
    st.info("Pastikan browser memberikan izin akses kamera")
    
    # Alternative camera test
    st.write("**Alternative Solutions:**")
    st.code("""
# 1. Run with HTTPS locally:
streamlit run app.py --server.enableCORS=false --server.enableWebsocketCompression=false

# 2. Use ngrok for HTTPS tunnel:
pip install pyngrok
ngrok http 8501

# 3. Try different browser (Chrome recommended)
    """)

def main():
    with st.sidebar:
        st.title("🤟 SIBI Translator")
        st.markdown("**Sistem Penerjemah Isyarat Bahasa Indonesia**")
        st.markdown("---")
        
        # Camera debug info
        debug_camera_access()
        
        if WEBRTC_AVAILABLE:
            try:
                model_statis, scaler_statis, le_statis, model_dinamis = load_all_models()
                
                if all([model_statis is not None, scaler_statis is not None, le_statis is not None]):
                    st.success("✅ Model Statis Berhasil Dimuat")
                    
                    if model_dinamis is not None:
                        st.success("✅ Model Dinamis Berhasil Dimuat")
                    else:
                        st.warning("⚠️ Model Dinamis Tidak Tersedia")
                    
                    # Model Information
                    with st.expander("📊 Informasi Model"):
                        st.markdown("**🔥 Model Statis**")
                        st.write("- Random Forest Classifier")
                        st.write("- 24 huruf diam (A-I, K-Y)")
                        st.write("- Fitur geometris tangan")
                        
                        if model_dinamis is not None:
                            st.markdown("**⚡ Model Dinamis**")
                            st.write("- LSTM Neural Network")
                            st.write("- 2 huruf bergerak (J, Z)")
                            st.write("- Sequence keypoints")
                        else:
                            st.markdown("**⚡ Model Dinamis**")
                            st.write("- Tidak tersedia (TensorFlow required)")
                        
                else:
                    st.error("❌ Error memuat model")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Error memuat model: {str(e)}")
                st.stop()
        else:
            st.error("❌ WebRTC tidak tersedia")
            st.info("Aplikasi memerlukan streamlit-webrtc untuk berfungsi")
            st.stop()
        
        st.markdown("---")
        
        # Usage Instructions
        with st.expander("📝 Cara Penggunaan"):
            st.markdown("""
            1. **Klik Play** pada video stream
            2. **Izinkan** akses kamera browser
            3. **Posisikan** tangan di depan kamera
            4. **Buat** isyarat alfabet SIBI
            5. **Lihat** hasil prediksi real-time
            """)
        
        # Tips
        with st.expander("💡 Tips Optimal"):
            st.markdown("""
            - Gunakan pencahayaan yang cukup
            - Pastikan background kontras
            - Posisikan tangan jelas di tengah
            - Jaga jarak optimal dari kamera
            - Buat gerakan yang jelas dan stabil
            """)
        
        # Statistics
        with st.expander("📈 Statistik Model"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Huruf Statis", "24")
            with col2:
                st.metric("Huruf Dinamis", "2" if model_dinamis is not None else "0")

    # --- Main Content Area ---
    st.header("🎯 Penerjemahan Real-time")

    if not WEBRTC_AVAILABLE:
        st.error("WebRTC tidak tersedia. Aplikasi tidak dapat berfungsi.")
        return

    # Layout utama
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.subheader("📊 Status")
        
        # Real-time status updates
        status_placeholder = st.empty()
        hand_status_placeholder = st.empty()
        model_status_placeholder = st.empty()

    with col2:
        st.subheader("🎥 Live Camera Feed")
        
        # # Simplified WebRTC Configuration for localhost
        # RTC_CONFIGURATION = RTCConfiguration({
        #     "iceServers": [
        #         {"urls": ["stun:stun.l.google.com:19302"]},
        #     ]
        # })
        
        # WebRTC Streamer dengan konfigurasi yang diperbaiki
        try:
            webrtc_ctx = webrtc_streamer(
                    key="sibi-translator-main",
                    video_processor_factory=SIBIVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False}
                )

            
            if webrtc_ctx.state.playing:
                st.success("🎥 Camera is active!")
            else:
                st.info("📷 Click 'START' to begin camera")
                
        except Exception as e:
            st.error(f"Error initializing WebRTC: {str(e)}")
            st.info("Trying alternative camera configuration...")
            
            # Fallback konfigurasi sederhana
            try:
                webrtc_ctx = webrtc_streamer(
                    key="sibi-translator-fallback",
                    video_processor_factory=SIBIVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=False
                )
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")
                st.markdown("""
                **Manual Setup Required:**
                1. Install required packages: `pip install streamlit-webrtc`
                2. Run with: `streamlit run app.py --server.enableCORS=false`
                3. Use Chrome/Firefox browser
                4. Allow camera permissions
                """)
                return

    with col3:
        st.subheader("📋 Info")
        info_placeholder = st.empty()

    # --- Prediction Results Area ---
    st.subheader("🔮 Hasil Prediksi")
    col_pred1, col_pred2, col_pred3 = st.columns(3)

    with col_pred1:
        prediction_placeholder = st.empty()

    with col_pred2:
        confidence_placeholder = st.empty()

    with col_pred3:
        model_type_placeholder = st.empty()

    # --- Status Updates (Real-time) ---
    def update_status():
        """Update status displays"""
        try:
            if 'webrtc_ctx' in locals() and webrtc_ctx.state.playing:
                # Camera status
                status_placeholder.success("🟢 Kamera Aktif")
                
                # Hand detection status
                if st.session_state.hand_detected:
                    hand_status_placeholder.success("✅ Tangan Terdeteksi")
                else:
                    hand_status_placeholder.warning("❌ Tangan Tidak Terdeteksi")
                
                # Model status
                model_status_placeholder.info(f"🧠 Model: {st.session_state.model_type}")
                
                # Info
                info_placeholder.info("🔄 Sistem aktif")
                
                # Prediction results
                prediction_placeholder.markdown(f"## **{st.session_state.prediction_result}**")
                confidence_placeholder.metric("Confidence", f"{st.session_state.confidence_score:.1f}%")
                model_type_placeholder.metric("Model Type", st.session_state.model_type)
                
            else:
                status_placeholder.info("📷 Kamera Siap")
                hand_status_placeholder.info("⏸️ Standby")
                model_status_placeholder.info("⏸️ Model Standby")
                info_placeholder.info("💤 Klik START untuk memulai")
                prediction_placeholder.markdown("## **Menunggu...**")
                confidence_placeholder.metric("Confidence", "0.0%")
                model_type_placeholder.metric("Model Type", "Standby")
        except Exception as e:
            st.error(f"Error updating status: {str(e)}")

    # Auto-refresh status every second
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

    current_time = time.time()
    if current_time - st.session_state.last_update > 1:  # Update every second
        update_status()
        st.session_state.last_update = current_time

    # Initial status update
    update_status()

    # --- Enhanced Troubleshooting Section ---
    with st.expander("🔧 Troubleshooting Camera Issues"):
        st.markdown("""
        **🎯 Solusi Umum:**
        
        **1. Browser Settings:**
        - Gunakan Chrome/Firefox (recommended)
        - Pastikan camera permission: `chrome://settings/content/camera`
        - Clear browser cache dan cookies
        
        **2. Streamlit Configuration:**
        ```bash
        streamlit run app.py --server.enableCORS=false --server.enableWebsocketCompression=false
        ```
        
        **3. HTTPS Solutions:**
        ```bash
        # Option 1: Use ngrok
        pip install pyngrok
        ngrok http 8501
        
        # Option 2: Use localtunnel
        npm install -g localtunnel
        lt --port 8501
        ```
        
        **4. Alternative Test:**
        - Test camera dengan aplikasi lain (Zoom, Teams)
        - Restart browser setelah memberikan permission
        - Coba incognito/private browsing mode
        
        **5. System Check:**
        - Windows: Device Manager → Cameras
        - macOS: System Preferences → Security & Privacy → Camera
        - Linux: `lsusb` untuk check USB camera
        """)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🎓 Alfabet SIBI | 💻 Teknologi: WebRTC + MediaPipe + TensorFlow + Scikit-learn + Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

# Tambahkan try-catch wrapper untuk main function
def safe_main():
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator")
        
        # Debug info
        if st.checkbox("Show Debug Info"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    safe_main()