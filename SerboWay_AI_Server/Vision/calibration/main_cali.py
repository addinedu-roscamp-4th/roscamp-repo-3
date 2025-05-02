import os
import pickle
from calibration import calibrate_camera, live_video_correction

if __name__ == "__main__":
    pkl_path = r'/home/addinedu/roscamp-repo-3/SerboWay_AI_Server/Vision/calibration/camera_calibration.pkl'
    if os.path.exists(pkl_path):
        print("Loading existing calibration data...")
        with open(pkl_path, 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print('There is no any pkl file.')
    
    print("Starting live video correction...")
    live_video_correction(calibration_data)