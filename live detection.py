import cv2
import numpy as np

class AntiSpoofingDetector:
    def __init__(self):
        self.last_frame = None
        self.frame_count = 0
        
    def detect_spoofing(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        
        # 1. Blur detection
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        results.append(("Blur", blur_value >= 80, blur_value))  # True if real
        
        # 2. Color diversity check
        color_std = np.std(frame)
        results.append(("Color", color_std >= 40, color_std))
        
        # 3. Edge density check
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])
        results.append(("Edges", edge_density >= 7, edge_density))
        
        # 4. Reflection check
        bright_pixels = np.sum(frame > 220) / frame.size
        results.append(("Reflection", bright_pixels <= 0.25, bright_pixels*100))
        
        # 5. Motion detection (every 5 frames)
        if self.frame_count % 5 == 0 and self.last_frame is not None:
            diff = cv2.absdiff(gray, cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY))
            motion = np.mean(diff)
            results.append(("Motion", motion >= 3, motion))
        
        self.last_frame = frame.copy()
        
        # Calculate final decision (weighted)
        weights = {
            "Blur": 0.25,
            "Color": 0.2,
            "Edges": 0.25,
            "Reflection": 0.2,
            "Motion": 0.1
        }
        
        score = 0
        details = []
        for name, is_real, value in results:
            weight = weights.get(name, 0)
            if is_real:
                score += weight
            details.append(f"{name}: {value:.1f}")
        
        is_real = score >= 0.65  # Threshold for final decision
        
        return is_real, score, ", ".join(details)

def main():
    detector = AntiSpoofingDetector()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect spoofing
        is_real, score, details = detector.detect_spoofing(frame)
        
        # Prepare display
        status = "REAL" if is_real else "FAKE"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        
        # Draw results
        cv2.putText(frame, f"Status: {status} ({score:.2f})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, details, (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Anti-Spoofing Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
