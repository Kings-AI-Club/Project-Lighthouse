#!/bin/bash

# Setup script for Streamlit application
# This script automates the setup process and launches the app

set -e  # Exit on error

echo "🚀 Starting setup for Streamlit application..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_CMD=""

# Try to find Python 3.11 or 3.10
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✓ Found Python 3.11${NC}"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✓ Found Python 3.10${NC}"
elif command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
    echo -e "${GREEN}✓ Found Python 3.11 (Homebrew)${NC}"
elif command -v /opt/homebrew/bin/python3.10 &> /dev/null; then
    PYTHON_CMD="/opt/homebrew/bin/python3.10"
    echo -e "${GREEN}✓ Found Python 3.10 (Homebrew)${NC}"
else
    # Check default python3 version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -le 11 ]; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
    else
        echo -e "${RED}✗ Python 3.11 or lower is required!${NC}"
        echo -e "${YELLOW}Current version: Python $PYTHON_VERSION${NC}"
        echo -e "${YELLOW}Please install Python 3.11 or 3.10:${NC}"
        echo "  brew install python@3.11"
        echo "  or"
        echo "  brew install python@3.10"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "\n${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Verify we're in the venv
VENV_PYTHON=$(which python)
if [[ "$VENV_PYTHON" == *"venv"* ]]; then
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo -e "${GREEN}  Using: $VENV_PYTHON${NC}"
else
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
fi

# Check Python version in venv
VENV_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}  Python version: $VENV_VERSION${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1
pip install streamlit tensorflow numpy pandas flask focal-loss h5py

# Verify installations
echo -e "\n${YELLOW}Verifying installations...${NC}"
STREAMLIT_PATH=$(which streamlit)
if [[ "$STREAMLIT_PATH" == *"venv"* ]]; then
    echo -e "${GREEN}✓ Streamlit installed in venv${NC}"
    echo -e "${GREEN}  Location: $STREAMLIT_PATH${NC}"
else
    echo -e "${RED}✗ Warning: Streamlit not in venv!${NC}"
    echo -e "${YELLOW}  Location: $STREAMLIT_PATH${NC}"
fi

# Check if streamlit_app.py exists
if [ ! -f "streamlit_app.py" ]; then
    echo -e "${RED}✗ streamlit_app.py not found in current directory!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Setup complete!${NC}"
echo -e "\n${YELLOW}Launching Streamlit application...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}\n"

# Launch the app
streamlit run streamlit_app.py
