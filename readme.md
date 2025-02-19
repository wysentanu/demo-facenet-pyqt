# Face Recognition using FaceNet

## Installation

This project requires **Python 3.12 or higher**.

### Setting Up a Virtual Environment

To manage dependencies efficiently, it is recommended to use a virtual environment.

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
   Or explicitly use Python 3:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the virtual environment:**
   - **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```powershell
     .venv\Scripts\activate
     ```

### Installing Dependencies

Install the required packages using `pip`.

- **MacOS:**
  ```bash
  pip install -r requirements-mac.txt
  ```
  Or explicitly use Python 3:
  ```bash
  pip3 install -r requirements-mac.txt
  ```

- **Windows:**
  ```bash
  pip install -r requirements.txt
  ```
  Or explicitly use Python 3:
  ```bash
  pip3 install -r requirements.txt
  ```
  
### Run the Project

To run the project, run this following command:

  ```bash
  python main.py
  ```
  Or explicitly use Python 3:
  ```bash
  python3 main.py
  ```

## Training the Face Recognizer

### Steps to Create the Dataset Directory

1. **Create the main dataset directory named `faces`**  
   Run the following command in your terminal:  
   ```sh
   mkdir -p faces
   ```

2. **Inside the `faces` directory, create folders for each person**  
   Each person's folder should contain face images in `.jpg`, `.jpeg`, or `.png` format.  


3. **Example Directory Structure:**  
   ```
   faces/
   ├── person1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── person2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```

4. **Ensure each folder contains multiple images of the same person**  
   The model requires multiple images for better recognition.


5. **Proceed with the face recognition process using the dataset.**

### Training the Face Recognizer

1. **Ensure the `faces` folder is properly structured as described above.**

2. **Run the training script:**
   ```sh
   python train.py
   ```
   Or specify Python 3 explicitly:
   ```sh
   python3 train.py
   ```

3. **Training Output:**
   - The script will generate an SQLite database named `faces.db`.
   - It uses the SQLite-vec extension to support vector data types for efficient similarity searches.