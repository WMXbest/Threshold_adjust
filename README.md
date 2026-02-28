# Threshold_calibration
## 1. Requirements
Python >= 3.8
Recommended environment: Anaconda (Spyder IDE)
2. File Structure
- `GUI_predict1.py`  
  Main graphical interface for prediction and threshold calibration.
  - `custom_function1.py`  
  Supporting utility functions.
- `Find_opthre_simplified1.py`  
  An additional easy example.
 ## 3. How to Run the Program (Spyder Recommended)
1. Open Anaconda Navigator.
2. Launch Spyder.
3. Open this project folder.
4. Open the file:
   GUI_predict1.py
5. Click "Run" (green arrow).
The graphical user interface will start automatically.
## 4. Reproducing Core Results
To reproduce the threshold calibration procedure:
- Load prediction scores from a source model.
- Provide calibration set scores and labels.
- The program computes:
  - Target specificity
  - Calibrated threshold
  - ## 5. License
This code is provided for academic research purposes.
