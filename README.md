# 🦷 Dental X-ray Abnormal Tooth Detection using YOLOv8

This project presents a deep learning-based solution for **automated abnormal tooth detection and diagnosis** from panoramic dental X-ray images using the YOLOv8 model. The system supports **FDI-compliant enumeration and classification** of conditions such as caries, deep caries, apical lesions, and impacted teeth.

> 💡 Although this was a group project submitted for a university course, **the majority of the work—including data processing, model development, training, and documentation—was completed independently by ZhaoYi Yang (z5452741).**

---

## 📌 Project Objective

Manual interpretation of panoramic dental radiographs is labor-intensive, subjective, and error-prone. This project aims to:

- Automate the detection and diagnosis of abnormal teeth
- Improve diagnostic accuracy and consistency
- Explore YOLOv8’s effectiveness on dental X-ray datasets

---

## 🗂 Dataset

The model is trained on the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX), released as part of the **Dental Enumeration and Diagnosis on Panoramic X-rays Challenge**.

- 🔗 Dataset URL: https://huggingface.co/datasets/ibrahimhamamci/DENTEX
- 1005 fully annotated panoramic dental X-ray images
- Multi-level labels: quadrant, tooth number, diagnosis
- Data split: 750 training / 50 validation / 250 test images

> ⚠️ **Note:** Due to licensing restrictions and GitHub's file size limits, this repository includes only **a small subset of training, validation, and test images and annotation files** as examples.  
> Please refer to the dataset link for full reproduction.

---

## 🧠 Model Architecture

YOLOv8n (nano) from the [Ultralytics](https://github.com/ultralytics/ultralytics) library is used for real-time object detection.

- Multi-label classification of dental anomalies
- YOLO-compatible `.yaml` and annotation format
- Data augmentation via `Albumentations`

---

## 🧪 Evaluation Metrics

- **Precision / Recall / F1-score**
- **mAP@0.5**, **mAP@0.5:0.95**

While performance is limited due to compute and training constraints, the model demonstrates the feasibility of automated dental diagnosis from X-rays.

---

## 🔧 Project Structure

```
.
├── dental_xray_detection.ipynb           # Full end-to-end training & inference pipeline
├── dental_xray_detection.pdf             # PDF export of the notebook
├── dental_xray_detection_report.pdf      # Final report
├── disease/                              # Sample input images (subset only)
├── quadrant-enumeration-disease/         # Sample quadrant labels (subset only)
├── validation_data/                      # Sample validation images (subset only)
├── README.md
├── requirements.txt
```

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/dental-xray-detection.git
cd dental-xray-detection

pip install -r requirements.txt

jupyter notebook dental_xray_detection.ipynb
```

---

## 📎 Dependencies

- `ultralytics`
- `torch`, `torchvision`
- `albumentations`
- `opencv-python`
- `matplotlib`, `pandas`, `numpy`

> See `requirements.txt` for full list.

---

## 📌 Limitations & Future Work

- YOLOv8n is lightweight; more powerful variants may improve accuracy
- Overfitting due to limited data
- Future improvements:
  - Better augmentation strategies
  - Model ensembling
  - Deployment in a dental diagnosis interface

---

## 📄 License

This project is intended for educational and academic purposes. The dataset used is publicly released by the DENTEX challenge. Please refer to the DENTEX license for usage details.
