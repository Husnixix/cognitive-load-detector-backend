# Cognitive Load Detection

**Cognitive Load Detection** is a Human-Computer Interaction (HCI) prototype that captures **facial cues** (via OpenCV) and **keyboard interaction patterns** (via pynput) to detect cognitive load.  

> ⚠️ Note: Some packages may not work properly on Python 3.10 or higher. Recommended version: **Python 3.9**.

---

## Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/husni-haniffa/cognitive-load-detector-backend
   ```

2. **Navigate to the project directory**
   ```bash
   cd cognitive-load-detector-backend
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

5. **Configure MongoDB**
   - Create a MongoDB account and a database named **`cognitive_load_detection`**.
   - Create a `.env` file in the project root with the following variables:
     ```env
     MONGODB_URI=<your_connection_string>
     MONGODB_DB=cognitive_load_detection
     ```

6. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Start the development server**
   ```bash
   py controller/api.py
   ```

8. **Run the main file in a separate terminal**
   ```bash
   py controller/main.py
   ```

9. **Access the backend server**
   ```
   http://localhost:8000
   ```

---

## Frontend Setup

1. **Clone the frontend repository**
   ```bash
   git clone https://github.com/husni-haniffa/cognitive-load-detection-frontend
   ```

2. **Navigate to the project directory**
   ```bash
   cd cognitive-load-detection-frontend
   ```

3. **Follow the setup instructions** in the frontend repository to complete the full system.
