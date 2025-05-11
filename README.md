# 🦺 PPE Detection with YOLOv5

This project focuses on the automatic detection of Personal Protective Equipment (PPE) using YOLOv5, with the goal of improving Health, Safety, and Environment (HSE) monitoring in industrial environments. A web-based demo using Streamlit is also provided.

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🧠 Model Training](#-model-training)
- [🔁 Fine-Tuning Process](#-fine-tuning-process)
- [📊 Evaluation](#-evaluation)
- [🖥️ Web Demo (Streamlit)](#️-web-demo-streamlit)
- [📂 Repository Structure](#-repository-structure)
- [🛠️ Requirements](#️-requirements)
- [🚀 Getting Started](#-getting-started)
- [📸 Sample Results](#-sample-results)
- [📄 License](#-license)

---

## 🎯 Project Overview

Ensuring that workers comply with PPE regulations is critical to workplace safety. Manual monitoring is time-consuming and error-prone. This project leverages **YOLOv5** for real-time detection of the following classes:

- ✅ `HELMET`
- ✅ `GLOVE`
- ✅ `SHOE`
- ❌ `NO-HELMET`
- ❌ `NO-GLOVE`
- ❌ `NO-SHOES`

> The integration of AI in HSE systems provides a powerful, scalable way to automate safety compliance monitoring.

---

## 🧠 Model Training

- Model: YOLOv5s
- Framework: PyTorch
- Initial training on a custom PPE dataset from Roboflow
- Fine-tuning with new samples to improve minority class recognition

Best performing model weights: [`model/best2.pt`](model/)

---

## 🔁 Fine-Tuning Process

Two stages of fine-tuning were conducted:

1. **Initial Training:** On the original dataset
2. **First Fine-Tuning:** On a more balanced dataset to improve detection of `NO-HELMET`,..etc.
3. **Second Fine-Tuning:** Additional targeted examples for underperforming classes

Weights from each phase are saved in the `model/` directory.

---

## 📊 Evaluation

The model was tested on a set of **86 images** with **387 total instances**. Results:

| Class       | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------------|-----------|-----------|--------|---------|--------------|
| GLOVE       | 98        | 0.957     | 0.900  | 0.910   | 0.470        |
| HELMET      | 120       | 0.899     | 0.891  | 0.901   | 0.590        |
| NO-GLOVE    | 36        | 0.968     | 0.848  | 0.917   | 0.600        |
| NO-HELMET   | 16        | 0.477     | 0.438  | 0.384   | 0.092        |
| NO-SHOES    | 10        | 0.822     | 0.700  | 0.839   | 0.410        |
| SHOE        | 107       | 0.776     | 0.748  | 0.794   | 0.448        |
| **All**     | **387**   | **0.816** | **0.754** | **0.791** | **0.435** |


---

## 🖥️ Web Demo (Streamlit)

A lightweight web app is included to test detection in real time.

```bash
streamlit run app.py

    Upload an image to see detection results

    Uses the latest best2.pt model stored in model/

📂 Repository Structure

├── app.py                # Streamlit app
├── data/                 # Original + fine-tuned datasets
├── demoapp/              # App assets
├── evaluation/           # Precision, Recall, mAP data
├── model/                # YOLOv5 best.pt weights
├── notebook/             # Training & evaluation notebooks (Colab)
├── README.md             # This file
├── .gitignore
├── LICENSE

🛠️ Requirements

    Python 3.8+

    PyTorch

    YOLOv5

    Streamlit

    OpenCV

    matplotlib, seaborn


O
🚀 Getting Started

    Clone this repository:

git clone https://github.com/yourusername/ppe-detection.git
cd ppe-detection

Run the Streamlit app:

    streamlit run app.py




This project is licensed under the MIT License.


Developed with passion to enhance safety using smart AI systems. 🤖🦺