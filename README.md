# 🔑 Feature Extraction and Matching Suite

## 📌 Overview

This project provides a ground-up implementation of classic **Feature Extraction** and **Feature Matching** algorithms. All core components, including the **Harris Corner Detector**, the **Scale-Invariant Feature Transform (SIFT)** descriptor, and matching metrics like **Sum of Squared Differences (SSD)** and **Normalized Cross Correlation (NCC)**, were built **from scratch** to ensure a thorough understanding of their underlying mathematics and structure.

---

## 🚀 Features

- ✅ **Corner Detection:** Custom implementation of the **Harris Corner Detector** for robust keypoint identification.
- ✅ **Scale-Invariant Descriptors:** Full implementation of the **SIFT** algorithm for creating scale and rotation-invariant feature vectors.
- ✅ **Matching Metrics:** Implemented **SSD** (Euclidean Distance) and **NCC** (Normalized Dot Product) for robust feature correspondence.
- ✅ **Optimization:** Non-maximum suppression applied for efficient keypoint selection.

---

## 🧠 Methodology (Algorithm Implementation)

### 1. Harris Corner Detector
This method identifies corners by measuring how the local image structure changes when moved in any direction.

1.  **Gradient Computation:** Manual calculation of image gradients ($I_x$, $I_y$).
2.  **Structure Tensor (M):** Construction of the $2 \times 2$ structure tensor matrix for each pixel, which summarizes the gradient information in a local neighborhood.
3.  **Corner Response (R):** Calculation of the corner response function: $R = \det(\mathbf{M}) - k \cdot \text{trace}(\mathbf{M})^2$.
4.  **Non-Maximum Suppression:** Filtering out weaker responses to yield distinct, strong corner points.



### 2. SIFT Feature Descriptors
SIFT ensures keypoints are reliable under changes in scale and rotation, a critical requirement for image matching.

1.  **Scale-Space Detection:** Keypoints are found at the extrema of the **Difference of Gaussians (DoG)** pyramid.
2.  **Keypoint Refinement:** Low-contrast and poorly localized points (like those along edges) are filtered out.
3.  **Orientation Assignment:** A dominant orientation is assigned to the keypoint based on local gradient histograms, enabling rotation invariance.
4.  **Descriptor Generation:** A **128-dimensional vector** (descriptor) is built from the weighted gradient orientations in the neighborhood of the keypoint.

### 3. Feature Matching (Correspondence)
Once keypoints and descriptors are extracted, the corresponding features in two images must be found using a similarity metric.

* **Sum of Squared Differences (SSD):** Measures the squared Euclidean distance between two descriptor vectors. Matches are identified by the **minimum** SSD score.
* **Normalized Cross Correlation (NCC):** Measures the linear relationship (similarity) between two image patches/descriptors. Matches are identified by the **maximum** NCC score, providing robustness to linear illumination changes.

---

🖼️ **Screenshots**:

1. Harris Corner Detection

<img width="1792" height="1120" alt="Harris" src="https://github.com/user-attachments/assets/13ccb449-5f03-4518-86d1-896ecdf12c5f" />


2. SIFT Keypoint and Descriptor Visualization
   
<img width="1792" height="1120" alt="SIFT" src="https://github.com/user-attachments/assets/86b3ac6d-a22c-44c9-b4b8-b783e5b2f9f7" />


4. Feature Matching (SSD)
   
<img width="1792" height="1120" alt="SSD" src="https://github.com/user-attachments/assets/ff5fa382-78f7-496f-ae9e-39dca3481490" />

5.Feature Matching (NCC)

<img width="1792" height="1120" alt="NCC" src="https://github.com/user-attachments/assets/aea096b1-1978-44c2-8e23-04dd9bf68620" />

---

## 🛠️ Technologies

- **Python 3.x**
- **NumPy:** For high-speed matrix arithmetic, crucial for gradient and structure tensor calculations.
- **OpenCV (cv2):** Used primarily for image file I/O and visualization.
- **Matplotlib:** Used for plotting keypoints and matching lines.

---

## 📈 Future Work

- Implement **FAST** (Features from Accelerated Segment Test) for faster corner detection.
- Explore the **BRIEF** or **ORB** descriptors for a comparison with SIFT's performance.
- Implement the **RANSAC** algorithm to filter outlier matches and compute a robust homography.

---

## Contributor

<div>
<table align="center">
  <tr>
        <td align="center">
      <a href="https://github.com/Mazenmarwan023" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
        <br />
        <sub><b>Mazen Marwan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mohamedddyasserr" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126451832?v=4" width="150px;" alt="Mohamed yasser"/>
        <br />
        <sub><b>Mohamed yasser</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Seiftaha" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127027353?v=4" width="150px;" alt="Saif Mohamed"/>
        <br />
        <sub><b>Saif Mohamed</b></sub>
      </a>
    </td> 
  </tr>
</table>
</div>


---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
