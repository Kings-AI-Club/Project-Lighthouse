# Setup Instructions

Follow these steps to set up and run the Streamlit application.

**IMPORTANT:** This application requires Python 3.11 or lower due to a TensorFlow compatibility issue with Python 3.12+ on macOS.

## 1. Install Python 3.11 or lower (if needed)

Check if you have Python 3.11 or lower installed:

```bash
python3 --version
```

If you have Python 3.12 or higher, install an older version using Homebrew:

```bash
brew install python@3.11
```

Or if you prefer Python 3.10:

```bash
brew install python@3.10
```

## 2. Create and activate a virtual environment with Python 3.11 or lower

Using Python 3.11 (if installed via Homebrew):
```bash
python3.11 -m venv venv
```

Or using Python 3.10:
```bash
python3.10 -m venv venv
```

Or using the Homebrew path directly:
```bash
/opt/homebrew/bin/python3.10 -m venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

## 3. Install all required dependencies

**CRITICAL:** You must install dependencies INSIDE the activated venv, not globally.

Install all required packages:

```bash
pip install streamlit tensorflow numpy pandas flask focal-loss h5py
```

### Required Packages:
- `streamlit` - For the Streamlit web app (streamlit_app.py)
- `tensorflow` - ML framework for model loading and predictions
- `keras` - High-level neural networks API (included with TensorFlow)
- `numpy` - Numerical computing
- `pandas` - Data manipulation (used in Streamlit app)
- `flask` - Web framework for Flask app (app.py)
- `focal-loss` - Custom loss function for model training
- `h5py` - For loading .h5 model files
- `logging`, `multiprocessing`, `typing`, `os`, `json` - Standard library (no install needed)

## 4. Run the Streamlit application

**IMPORTANT:** Make sure your venv is activated (you should see `(venv)` in your terminal prompt).

If not activated, run:
```bash
source venv/bin/activate
```

Then run the application:
```bash
streamlit run streamlit_app.py
```

**Verify you're using Python 3.11:**
```bash
python --version
```
Should show `Python 3.11.x` NOT `Python 3.13.x`

## Deactivating the virtual environment

When you're done, you can deactivate the virtual environment:

```bash
deactivate
```

---

## Troubleshooting

### Mutex Lock Error on macOS

If you see this error:
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
```

**Cause:** This is a known incompatibility between TensorFlow and Python 3.12+ on macOS. The issue occurs because:
- Python 3.12+ changed how it handles multithreading and fork behavior
- TensorFlow (as of current versions) hasn't fully adapted to these changes on macOS
- The Objective-C runtime's fork safety checks conflict with TensorFlow's threading model

**Solution:** You **must** use Python 3.11 or lower AND install packages inside the venv. Setting environment variables like `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` does NOT fix this issue.

**To fix:**
1. Delete your current venv: `rm -rf venv`
2. Install Python 3.11 or 3.10 if needed: `brew install python@3.11` or `brew install python@3.10`
3. Recreate venv with Python 3.11 or lower:
   - `python3.11 -m venv venv` (if python3.11 command exists)
   - Or `python3.10 -m venv venv` (if python3.10 command exists)
   - Or `/opt/homebrew/bin/python3.10 -m venv venv` (using full Homebrew path)
4. Activate: `source venv/bin/activate`
5. **CRITICAL:** Verify you're in the venv: `which python` should show the venv path
6. Reinstall ALL dependencies INSIDE the venv: `pip install streamlit tensorflow numpy pandas flask focal-loss h5py`
7. Verify streamlit is in venv: `which streamlit` should show the venv path, NOT miniconda or system path
8. Run: `streamlit run streamlit_app.py`

**Common mistake:** Running streamlit from a global installation (e.g., miniconda) while the venv is activated. This causes streamlit to use the wrong Python version. Always verify with `which streamlit` after installing.

### Checking Your Python Version

To verify you're using Python 3.11 in your venv:

```bash
python --version
```

Should output: `Python 3.11.x`
