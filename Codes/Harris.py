from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QVBoxLayout, QWidget,QLabel,QFileDialog,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider,QGroupBox, 
                             QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


from sift import Sift
from ssd import MatchFeatures


class Harris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Task_03")
        self.setGeometry(200,200,1500,1200)
        self.image = None  # To store the loaded image
        self.processed_image = None  # To store the loaded image

        self.stacked_widget=QStackedWidget()
        self.sift_page=Sift(self)
        self.ssd_page=MatchFeatures(self)
      

        self.initUI()
    
    def initUI(self):
        
        self.main_widget = QWidget()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.sift_page)
        self.stacked_widget.addWidget(self.ssd_page)
        self.stacked_widget.setCurrentWidget(self.main_widget)


        main_layout = QGridLayout()
        controls_layout = QVBoxLayout()


        group_box = QGroupBox()
        box_layout=QVBoxLayout()
        images_layout=QHBoxLayout()
        buttons_layout=QHBoxLayout()

        # Labels for images
        input_image_layout=QVBoxLayout()
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
        self.reset_button=QPushButton("Reset")
        self.reset_button.setFixedWidth(150)
        self.save_button=QPushButton("Save")
        self.save_button.setFixedWidth(150)
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_button)

        # Next pages buttons
        next_pages_buttons_layout=QHBoxLayout()
        self.sift_button=QPushButton("SIFT")
        self.sift_button.setFixedWidth(150)
        self.ssd_button=QPushButton("SSD & NCC")
        self.ssd_button.setFixedWidth(150)

        next_pages_buttons_layout.addWidget(self.sift_button)
        next_pages_buttons_layout.addWidget(self.ssd_button)
  

        box_layout.addLayout(next_pages_buttons_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(images_layout)
        box_layout.addStretch(1)
        box_layout.addLayout(buttons_layout)
        box_layout.addStretch(1)
        group_box.setLayout(box_layout)




        harris_menu_label=QLabel("Harris menu")
        harris_menu_label.setObjectName("menu")

        # Alpha textbox
        self.alpha_textbox=QLineEdit("0.04")
    

        
        alpha_layout = QVBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha:"))
        alpha_layout.addWidget(self.alpha_textbox)
        # alpha_layout.addWidget(alpha_label)

        # Threshold parameter
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold (%):"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 20)
        self.threshold_slider.setValue(1)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_label = QLabel("1.0")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)

        self.apply_harris_button = QPushButton("Apply Harris")
        self.apply_lambda_button=QPushButton("Apply lambda")
        apply_layout=QHBoxLayout()
        apply_layout.addWidget(self.apply_harris_button)
        apply_layout.addWidget(self.apply_lambda_button)
       


        controls_layout.addWidget(harris_menu_label)
        controls_layout.addStretch(1)
        controls_layout.addLayout(alpha_layout)
        controls_layout.addStretch(1)
        controls_layout.addLayout(threshold_layout)
        controls_layout.addStretch(1)
        controls_layout.addLayout(apply_layout)
     

      
        # Connect buttons
        self.upload_button.clicked.connect(self.load_image)
        self.reset_button.clicked.connect(self.reset_images)
        self.save_button.clicked.connect(self.save_output_image)
        self.apply_harris_button.clicked.connect(self.apply_harris)
        self.apply_lambda_button.clicked.connect(self.apply_lambda_minus)
        self.sift_button.clicked.connect(self.switch_to_sift)
        self.ssd_button.clicked.connect(self.switch_to_ssd)


        main_layout.addLayout(controls_layout,0,0)
        main_layout.addWidget(group_box,0,1)
        main_layout.setColumnStretch(1,2)


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
        
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.stacked_widget)

            
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
    
    def update_threshold(self, value):
        self.threshold_percent = value
        self.threshold_label.setText(f"{self.threshold_percent:.1f}")

    def apply_harris(self):
        if self.image is None:
            return
            
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
            
        # Convert to float32 for calculations
        gray = np.float32(gray)
        
        #Compute gradients Ix and Iy using Sobel operator
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Step 2: Compute products of derivatives
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        # Step 3: Compute the sums of the products of derivatives at each pixel
        # using a Gaussian window (here we use a simple box filter as approximation)
        window_size = 3
        offset = window_size // 2
        height, width = gray.shape
        corner_response = np.zeros_like(gray)
        
        # Pad the images to handle borders
        Ix2_padded = np.pad(Ix2, ((offset, offset), (offset, offset)), mode='constant')
        Iy2_padded = np.pad(Iy2, ((offset, offset), (offset, offset)), mode='constant')
        Ixy_padded = np.pad(Ixy, ((offset, offset), (offset, offset)), mode='constant')

        alpha=float(self.alpha_textbox.text())
        
        for y in range(height):
            for x in range(width):
                # Sum over the window
                Sx2 = np.sum(Ix2_padded[y:y+window_size, x:x+window_size])
                Sy2 = np.sum(Iy2_padded[y:y+window_size, x:x+window_size])
                Sxy = np.sum(Ixy_padded[y:y+window_size, x:x+window_size])
                
                # Compute the determinant and trace
                det = Sx2 * Sy2 - Sxy * Sxy
                trace = Sx2 + Sy2
                
                # Compute the corner response
                corner_response[y, x] = det - alpha * (trace ** 2)
        
        threshold = 0.01 * corner_response.max()
        
        # Create output image (convert back to color if original was color)
        if len(self.image.shape) == 3:
            output = self.image.copy()
        else:
            output = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # Mark corners on the output image
        for y in range(height):
            for x in range(width):
                if corner_response[y, x] > threshold:
                    # Check if this is a local maximum in 3x3 neighborhood
                    neighborhood = corner_response[max(0, y-1):min(height, y+2), 
                                                max(0, x-1):min(width, x+2)]
                    if corner_response[y, x] == neighborhood.max():
                        cv2.circle(output, (x, y), 3, (0, 255, 0), -1)  # Green circle

        self.processed_image=output
        
        self.display_image(output, self.output_label)

    def compute_structure_tensor(self, gray, window_size=3):
        """Compute structure tensor components"""
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        
        # Apply Gaussian blur to the products of derivatives
        Ix2 = cv2.GaussianBlur(Ix2, (window_size, window_size), 1)
        Iy2 = cv2.GaussianBlur(Iy2, (window_size, window_size), 1)
        Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1)
        
        return Ix2, Iy2, Ixy
    
    def apply_lambda_minus(self):
        if self.image is None:
            return
            
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
            
        gray = np.float32(gray)
        Ix2, Iy2, Ixy = self.compute_structure_tensor(gray)
        
        height, width = gray.shape
        lambda_min = np.zeros_like(gray)
        
        # Compute minimum eigenvalue for each pixel
        for y in range(height):
            for x in range(width):
                M = np.array([[Ix2[y,x], Ixy[y,x]],
                             [Ixy[y,x], Iy2[y,x]]])
                eigenvalues = np.linalg.eigvalsh(M)  
                lambda_min[y,x] = np.min(np.abs(eigenvalues))
        
        # Normalize response
        lambda_min = cv2.normalize(lambda_min, None, 0, 255, cv2.NORM_MINMAX)
        
        # Thresholding
        threshold = self.threshold_percent / 100.0 * lambda_min.max()
        corners = np.zeros_like(gray, dtype=np.uint8)
        corners[lambda_min > threshold] = 255
        
        # Create output image
        if len(self.image.shape) == 3:
            output = self.image.copy()
        else:
            output = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # Mark corners
        output[corners == 255] = [0, 255, 0]  # Green color
        
        self.display_image(output, self.output_label)


    def switch_to_sift(self):
        self.stacked_widget.setCurrentIndex(1)
    def switch_to_ssd(self):
        self.stacked_widget.setCurrentIndex(2)
