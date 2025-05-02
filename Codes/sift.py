from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QSlider, QGroupBox, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from math import floor
from numpy.linalg import norm
from cv2 import subtract


class Sift(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Sift")
        self.setGeometry(200, 200, 1500, 1200)
        self.main_window = main_window
        self.image = None  # To store the loaded image
        self.processed_image = None  # To store the processed image
        self.initUI()

    def initUI(self):  
        self.main_widget = QWidget()
        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()

        group_box = QGroupBox()
        box_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()

        # Labels for images
        input_image_layout = QVBoxLayout()
        self.input_label = QLabel("Original Image")
        self.input_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFixedSize(500, 500)
        self.color_mode = QRadioButton("Color")
        self.gray_mode = QRadioButton("Grayscale")
        self.color_mode.setChecked(True)  # Default mode is Color
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.color_mode)
        mode_layout.addWidget(self.gray_mode)
        input_image_layout.addWidget(self.input_label)
        input_image_layout.addLayout(mode_layout)

        self.output_label = QLabel("Processed Image")
        self.output_label.setStyleSheet("background-color: black; border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFixedSize(500, 500)

        images_layout.addLayout(input_image_layout)
        images_layout.addWidget(self.output_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setFixedWidth(150)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button = QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # SIFT parameters controls
        self.octaves_label = QLabel("Number of Octaves:")
        self.octaves_input = QLineEdit("4")
        self.scales_label = QLabel("Number of Scales:")
        self.scales_input = QLineEdit("5")
        self.contrast_label = QLabel("Contrast Threshold:")
        self.contrast_input = QLineEdit("0.01")
        
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.octaves_label)
        params_layout.addWidget(self.octaves_input)
        params_layout.addWidget(self.scales_label)
        params_layout.addWidget(self.scales_input)
        params_layout.addWidget(self.contrast_label)
        params_layout.addWidget(self.contrast_input)

        self.harris_button = QPushButton("Harris")
        self.harris_button.setFixedWidth(150)

        box_layout.addWidget(self.harris_button)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)

        # SIFT menu 
        sift_menu_label = QLabel("Sift menu")
        sift_menu_label.setObjectName("menu")

        self.apply_button = QPushButton("Apply Sift")
        self.apply_button.clicked.connect(self.compute_sift_features)
        
        controls_layout.addWidget(sift_menu_label)
        controls_layout.addLayout(params_layout)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.apply_button)

        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.harris_button.clicked.connect(self.switch_to_homepage)

        main_layout.addLayout(controls_layout, 0, 0)
        main_layout.addWidget(group_box, 0, 1)
        main_layout.setColumnStretch(1,2)

        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

        self.setStyleSheet("""
             QLabel{
                font-size:20px;
                color:white;     
                    }
            QLabel#menu{
                font-size:29px;
                color:white;
                           }
            QPushButton{
                    font-size:18px;
                    padding:10px;
                    border:white 1px solid;
                    border-radius:15px;
                    background-color:white;
                    color:black;         
                        }
        """)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)", options=options
        )

        if file_path:
            # Check which mode is selected
            if self.gray_mode.isChecked():
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            else:
                self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load as color (default)

            self.display_image(self.image, self.input_label)  # Display in input label

    def display_image(self, img, label):
        if len(img.shape) == 2:  # Grayscale image
            height, width = img.shape
            bytes_per_line = width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def reset_images(self):
        self.input_label.clear()  # Clear input image label
        self.output_label.clear()  # Clear output image label
        self.image = None  # Remove stored image
        self.processed_image = None  # Remove stored output

    def save_output_image(self):
        if self.processed_image is None:
            return  # No image to save
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)  # Save using OpenCV

    def switch_to_homepage(self):
        self.main_window.stacked_widget.setCurrentIndex(0)

    # SIFT Implementation from previous file (adapted for PyQt5)
    def create_scale_space(self, image, num_octaves, num_scales, k=np.sqrt(2)):
        scale_space_images = []
        for octave in range(num_octaves):
            octave_images = []
            sigma = 1.6
            for scale in range(num_scales):
                sigma_current = sigma * (k ** scale)
                blurred_image = cv2.GaussianBlur(
                    image, (0, 0), sigmaX=sigma_current, sigmaY=sigma_current)
                octave_images.append(blurred_image)
            scale_space_images.append(octave_images)
            image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), 
                            interpolation=cv2.INTER_NEAREST)
        return scale_space_images

    def diff_of_gaussian(self, scale_space_images):
        diff_of_gaussian_result = []
        for octave in scale_space_images:
            octave_images = []
            for first_image, second_image in zip(octave, octave[1:]):
                octave_images.append(subtract(second_image, first_image))
            diff_of_gaussian_result.append(octave_images)
        return diff_of_gaussian_result

    # Scale Space Extrema Detection
    def keypoint_localization(self, dog_res, contrast_threshold=0.04, num_intervals=3):
        keypoints = []
        threshold = floor(0.5 * contrast_threshold / num_intervals * 255)
        for octave_idx, octave in enumerate(dog_res):
            for scale_idx in range(1, len(octave) - 1):
                for i in range(1, octave[scale_idx].shape[0] - 1):
                    for j in range(1, octave[scale_idx].shape[1] - 1):
                        pixel_value = octave[scale_idx][i, j]
                        neighbors = [octave[scale_idx - 1][i - 1:i + 2, j - 1:j + 2],
                                     octave[scale_idx][i - 1:i + 2, j - 1:j + 2],
                                     octave[scale_idx + 1][i - 1:i + 2, j - 1:j + 2]]
                        if self.is_local_extremum(neighbors, threshold):
                            keypoints.append(
                                (i, j, octave[scale_idx], octave_idx, scale_idx, pixel_value))
        return keypoints


    def is_local_extremum(self, neighbors, threshold):
        center_pixel = neighbors[1][1, 1]
        if abs(center_pixel) <= threshold:
            return False

        all_neighbors = np.concatenate([n.flatten() for n in neighbors])
        all_neighbors = np.delete(all_neighbors, 13)  # remove center

        if center_pixel > 0:
            return center_pixel >= np.max(all_neighbors)
        else:
            return center_pixel <= np.min(all_neighbors)

    
    # Orientation Assignment
    def generate_keypoints_with_orientations(self, keypoints, gaussian_images, contrast_threshold=0.03,
                                         eigenvalue_ratio=10, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        selected = []
        for keypoint in keypoints:
            i, j, image, octave_idx, scale_idx, pixel_value = keypoint
            dx = np.gradient(image, axis=1)
            dy = np.gradient(image, axis=0)
            dxx = np.gradient(dx, axis=1)
            dyy = np.gradient(dy, axis=0)
            dxy = np.gradient(dx, axis=0)

            hessian = np.array([[dxx[i, j], dxy[i, j]], [dxy[i, j], dyy[i, j]]])
            det = np.linalg.det(hessian)
            trace = np.trace(hessian)

            contrast = abs(pixel_value / 255)
            if det <= 0 or (trace ** 2 >= (eigenvalue_ratio + 1) ** 2 * det):
                continue
            if contrast < contrast_threshold:
                continue

            oriented_kps = self.compute_keypoints_with_orientations(keypoint, octave_idx,
                                                                    gaussian_images[octave_idx][scale_idx],
                                                                    radius_factor, num_bins, peak_ratio, scale_factor)
            selected.extend(oriented_kps)

        return selected

    def compute_keypoints_with_orientations(self, keypoint, octave_index, gaussian_image,
                                        radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypoints_oriented = []
        i, j = keypoint[0], keypoint[1]
        scale_idx = keypoint[4]

        scale = scale_factor * scale_idx / (2 ** octave_index)
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        histogram = np.zeros(num_bins)

        height, width = gaussian_image.shape
        center_y = int(round(j / (2 ** octave_index)))
        center_x = int(round(i / (2 ** octave_index)))

        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                ny, nx = center_y + y, center_x + x
                if 1 <= ny < height - 1 and 1 <= nx < width - 1:
                    dx = gaussian_image[ny, nx + 1] - gaussian_image[ny, nx - 1]
                    dy = gaussian_image[ny - 1, nx] - gaussian_image[ny + 1, nx]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    angle = np.rad2deg(np.arctan2(dy, dx)) % 360

                    weight = np.exp(weight_factor * (x**2 + y**2))
                    bin_idx = int(round(angle * num_bins / 360)) % num_bins
                    histogram[bin_idx] += weight * magnitude

        smoothed_hist = np.convolve(histogram, np.array([1, 4, 6, 4, 1]) / 16, mode='same')
        max_val = np.max(smoothed_hist)
        peaks = np.where((smoothed_hist > np.roll(smoothed_hist, 1)) &
                        (smoothed_hist > np.roll(smoothed_hist, -1)))[0]

        for peak in peaks:
            if smoothed_hist[peak] >= peak_ratio * max_val:
                left = smoothed_hist[(peak - 1) % num_bins]
                right = smoothed_hist[(peak + 1) % num_bins]
                interpolated = (peak + 0.5 * (left - right) / (left - 2 * smoothed_hist[peak] + right)) % num_bins
                orientation = 360 - interpolated * 360 / num_bins
                if abs(orientation - 360) < 1e-5:
                    orientation = 0
                keypoints_oriented.append((i, j, keypoint[2], octave_index, scale_idx, keypoint[5], orientation))

        return keypoints_oriented


    def generate_descriptors(self, keypoints, gaussian_images, window_width=4, num_bins=8, 
                           scale_multiplier=3, descriptor_max_value=0.2):
        descriptors = []
        for keypoint in keypoints:
            x, y, _, octave_idx, scale_idx, _, orientation = keypoint
            scale = 1.5 * scale_idx / (2 ** (octave_idx))
            gaussian_image = gaussian_images[octave_idx][scale_idx]
            num_rows, num_cols = gaussian_image.shape
            point = np.array([x, y], dtype=int)
            bins_per_degree = num_bins / 360.
            angle = 360. - orientation
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))
            hist_width = scale_multiplier * 0.5 * scale
            half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))
            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if 0 <= row_bin < window_width and 0 <= col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = int(floor(row_bin)), int(floor(col_bin)), int(floor(orientation_bin))
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                orientation_bin_floor %= num_bins
                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111
            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), 1e-6)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')

    def generate_keypoints_cv_format(self, keypoints_info, size_multiplier=10):
        keypoints = []
        for info in keypoints_info:
            i, j, _, octave_idx, scale_idx, _, orientation = info
            scale = size_multiplier * (1.5 * scale_idx / (2 ** octave_idx))
            pt = (j, i)
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], size=scale, angle=orientation)
            keypoints.append(kp)
        return keypoints

    def compute_sift_features(self):
        if self.image is None:
            return

        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image

        # Get parameters from UI
        try:
            num_octaves = int(self.octaves_input.text())
            num_scales = int(self.scales_input.text())
            contrast_threshold = float(self.contrast_input.text())
        except ValueError:
            num_octaves = 4
            num_scales = 5
            contrast_threshold = 0.04

        # Convert to float32 for processing
        image_float = gray_image.astype('float32')

        # 1. Create scale space
        scale_space = self.create_scale_space(image_float, num_octaves, num_scales)

        # 2. Compute Difference of Gaussians
        dog = self.diff_of_gaussian(scale_space)

        # 3. Detect keypoints
        keypoints = self.keypoint_localization(dog, contrast_threshold)

        # 4. Refine keypoints and assign orientations
        keypoints_with_orientations = self.generate_keypoints_with_orientations(
            keypoints, scale_space, contrast_threshold)

        # 5. Generate descriptors
        descriptors = self.generate_descriptors(keypoints_with_orientations, scale_space)

        # 6. Convert keypoints to OpenCV format for visualization
        keypoints_cv = self.generate_keypoints_cv_format(keypoints_with_orientations)

        # Draw keypoints on the original image
        if len(self.image.shape) == 3:
            output_image = cv2.drawKeypoints(self.image, keypoints_cv, None, 
                                           color=(255, 0, 0), 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            output_image = cv2.drawKeypoints(color_image, keypoints_cv, None, 
                                           color=(255, 0, 0), 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display processing time
        processing_time = time.time() - start_time
        print(f"SIFT processing time: {processing_time:.2f} seconds")

        # Store and display the result
        self.processed_image = output_image
        self.display_image(output_image, self.output_label)