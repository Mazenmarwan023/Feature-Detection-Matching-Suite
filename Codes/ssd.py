from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QSlider, QGroupBox, 
                             QStackedWidget, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import time
from sift import Sift


class MatchFeatures(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Feature Matching (SSD/NCC)")
        self.setGeometry(200, 200, 1500, 1200)
        self.main_window = main_window
        self.image1 = None  # First image
        self.image2 = None  # Second image
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.initUI()

    def initUI(self):
        self.main_widget = QWidget()
        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()

        # Image display group
        group_box = QGroupBox("Image Matching")
        box_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()

        # Image 1 display
        self.image1_layout = QVBoxLayout()
        self.image1_label = QLabel("Image 1")
        self.image1_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setFixedSize(400, 400)
        
        # Image 2 display
        self.image2_label = QLabel("Image 2")
        self.image2_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setFixedSize(400, 400)
        
        # Result display
        self.result_label = QLabel("Matching Result")
        self.result_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(800, 400)

        # Add image displays to layout
        images_layout.addWidget(self.image1_label)
        images_layout.addWidget(self.image2_label)
        images_layout.addWidget(self.result_label)

        # Control buttons
        self.upload1_button = QPushButton("Upload Image 1")
        self.upload2_button = QPushButton("Upload Image 2")
        self.reset_button = QPushButton("Reset")
        self.save_button = QPushButton("Save Result")
        
        buttons_layout.addWidget(self.upload1_button)
        buttons_layout.addWidget(self.upload2_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.save_button)

        # Navigation buttons
        self.harris_button = QPushButton("Harris Corner")
        self.sift_button = QPushButton("SIFT Features")
        buttons_layout.addWidget(self.harris_button)
        buttons_layout.addWidget(self.sift_button)

        # Add to main box layout
        box_layout.addLayout(images_layout)
        box_layout.addLayout(buttons_layout)
        group_box.setLayout(box_layout)

        # Matching controls
        controls_group = QGroupBox("Matching Parameters")
        controls_layout = QVBoxLayout()
        
        # Matching method selection
        self.method_label = QLabel("Matching Method:")
        self.ssd_button = QPushButton("SSD Matching")
        self.ncc_button = QPushButton("NCC Matching")
        
        # Threshold controls
        self.threshold_label = QLabel("Match Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(70)
        self.threshold_value = QLabel("0.7")
        
        # Max matches control
        self.max_matches_label = QLabel("Max Matches to Show:")
        self.max_matches_input = QLineEdit("50")
        
        # Add controls to layout
        controls_layout.addWidget(self.method_label)
        controls_layout.addWidget(self.ssd_button)
        controls_layout.addWidget(self.ncc_button)
        controls_layout.addWidget(self.threshold_label)
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.threshold_value)
        controls_layout.addWidget(self.max_matches_label)
        controls_layout.addWidget(self.max_matches_input)
        controls_group.setLayout(controls_layout)

        # Connect signals
        self.upload1_button.clicked.connect(lambda: self.load_image(1))
        self.upload2_button.clicked.connect(lambda: self.load_image(2))
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.harris_button.clicked.connect(self.switch_to_harris)
        self.ssd_button.clicked.connect(self.apply_ssd)
        self.ncc_button.clicked.connect(self.apply_ncc)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        # Main layout
        main_layout.addWidget(controls_group, 0, 0)
        main_layout.addWidget(group_box, 0, 1)
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

        # Style sheet
        self.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: black;
            }
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
            }
            QPushButton {
                font-size: 16px;
                padding: 8px;
                border: 1px solid #8f8f91;
                border-radius: 6px;
                background-color: #f0f0f0;
                min-width: 120px;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def load_image(self, image_num):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )

        if file_path:
            image = cv2.imread(file_path)
            if image_num == 1:
                self.image1 = image
                self.display_image(image, self.image1_label)
                # Extract features when both images are loaded
                if self.image2 is not None:
                    self.extract_features()
            else:
                self.image2 = image
                self.display_image(image, self.image2_label)
                # Extract features when both images are loaded
                if self.image1 is not None:
                    self.extract_features()

    def display_image(self, img, label):
        if len(img.shape) == 2:  # Grayscale
            q_img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        else:  # Color
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], 
                          rgb_img.shape[1] * 3, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def extract_features(self):
        """Extract SIFT features from both images"""
        sift = cv2.SIFT_create()
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        self.keypoints1, self.descriptors1 = sift.detectAndCompute(gray1, None)
        self.keypoints2, self.descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Show keypoints on images
        img1_kp = cv2.drawKeypoints(self.image1, self.keypoints1, None)
        img2_kp = cv2.drawKeypoints(self.image2, self.keypoints2, None)
        self.display_image(img1_kp, self.image1_label)
        self.display_image(img2_kp, self.image2_label)

    def match_features_ssd(self, descriptors1, descriptors2):
        """Sum of Squared Differences matching"""
        matches = []
        for i, d1 in enumerate(descriptors1):
            ssd = np.sum((d1 - descriptors2) ** 2, axis=1)
            best_match_idx = np.argmin(ssd)
            matches.append(cv2.DMatch(i, best_match_idx, ssd[best_match_idx]))

        return matches

    def match_features_ncc(self, descriptors1, descriptors2):
        """Normalized Cross Correlation matching (corrected)"""
        matches = []
        
        # Mean-center and normalize descriptors (as per the formula)
        desc1_mean = np.mean(descriptors1, axis=1, keepdims=True)
        desc2_mean = np.mean(descriptors2, axis=1, keepdims=True)
        desc1_norm = (descriptors1 - desc1_mean) / (np.std(descriptors1, axis=1, keepdims=True) + 1e-6)
        desc2_norm = (descriptors2 - desc2_mean) / (np.std(descriptors2, axis=1, keepdims=True) + 1e-6)
        
        for i, d1 in enumerate(desc1_norm):
            ncc = np.dot(d1, desc2_norm.T)  # Dot product = correlation
            best_match_idx = np.argmax(ncc)
            matches.append(cv2.DMatch(i, best_match_idx, ncc[best_match_idx]))
        
        return matches
    def apply_ssd(self):
        if not self.check_features():
            return
            
        start_time = time.time()
        matches = self.match_features_ssd(self.descriptors1, self.descriptors2)
        end_time = time.time()
        
        self.show_matches(matches, f"SSD Matching - Time: {end_time-start_time:.3f}s")

    def apply_ncc(self):
        if not self.check_features():
            return
            
        start_time = time.time()
        matches = self.match_features_ncc(self.descriptors1, self.descriptors2)
        end_time = time.time()
        
        self.show_matches(matches, f"NCC Matching - Time: {end_time-start_time:.3f}s")

    def show_matches(self, matches, title):
        try:
            max_matches = int(self.max_matches_input.text())
        except:
            max_matches = 50
            
        # Sort matches by distance and select top N
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:max_matches]
        
        # Draw matches
        match_img = cv2.drawMatches(
            self.image1, self.keypoints1,
            self.image2, self.keypoints2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add title to image
        cv2.putText(match_img, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        self.display_image(match_img, self.result_label)
        self.processed_image = match_img

    def check_features(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Error", "Please load both images first!")
            return False
        if self.descriptors1 is None or self.descriptors2 is None:
            QMessageBox.warning(self, "Error", "No features detected in one or both images!")
            return False
        return True

    def update_threshold(self, value):
        threshold = value / 100.0
        self.threshold_value.setText(f"{threshold:.2f}")

    def reset_images(self):
        self.image1_label.clear()
        self.image2_label.clear()
        self.result_label.clear()
        self.image1 = None
        self.image2 = None
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.processed_image = None

    def save_output_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Error", "No result to save!")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)

    def switch_to_harris(self):
        self.main_window.stacked_widget.setCurrentIndex(0)

