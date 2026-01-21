# 🚗 Automatic Number Plate Recognition

A high-performance **Automatic Number Plate Recognition (ANPR)** system.  
This application detects vehicle license plates using a custom-trained **YOLOv8** model and extracts text using the **Groq Cloud API** for lightning-fast inference.

---

## 🌟 Features

- **Vehicle Detection**  
  High-accuracy license plate localization using a custom-trained `best.pt` YOLO model.

- **Groq-Powered OCR**  
  Ultra-fast text extraction from detected plates using **Llama-3-70B-Vision** via Groq.

- **Interactive UI**  
  Built with **Streamlit** for easy image/video uploads and real-time visualization.

- **Data Insights**  
  Trained on high-quality **Kaggle license plate datasets**.

---

## 🛠️ Tech Stack

- **Object Detection:** YOLOv8 (Ultralytics)  
- **OCR Engine:** Groq API (LLM-based Vision)  
- **Frontend:** Streamlit  
- **Language:** Python  
- **Dataset Source:** Kaggle (License Plate Dataset)

---

## 📂 Project Structure

```text
Root Directory
├── app.py                 # Main Streamlit application
├── models/
│   └── best.pt             # Custom trained YOLO model
├── utils/
│   ├── detection.py        # YOLO inference logic
│   └── ocr_engine.py       # Groq API integration
├── requirements.txt        # List of dependencies
└── .env                    # Environment variables (API Keys)
```

## ⚙️ Installation & Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/anpr-groq-project.git
   ```
   ```bash
   cd anpr-groq-project
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**
   ```text
   Create a .env file in the root directory and add your Groq API key:
   GROQ_API_KEY=your_actual_api_key_here
   ```
4. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```

---
   
## 🚀 How It Works

1. Detection Phase

* The user uploads an image through the Streamlit interface.
* The system passes the image through the best.pt YOLOv8 model to detect and localize the license plate.

2. Image Processing

* The detected license plate region is cropped and enhanced to improve OCR accuracy.

3. OCR Phase

* The cropped image is encoded and sent to the Groq API.
* Using the vision capabilities of Llama 3, the license plate text is extracted in milliseconds.

4. Output

* The recognized plate number is displayed on the Streamlit dashboard along with the original image and bounding box.

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. **Fork the project** Create a feature branch
   ```bash
   git checkout -b feature/NewAlgorithm
   ```

2. **Commit your changes**
   ```bash
   git commit -m "Added some feature"
   ```

3. **Push to the branch**
   ```bash
   git push origin feature/NewAlgorithm
   ```

4. **Open a Pull Request**

---

# 👨‍💻 Author
Sathvik Palivela AI / ML & Computer Vision Enthusiast

# 📄 License

This project is licensed under the MIT License.
