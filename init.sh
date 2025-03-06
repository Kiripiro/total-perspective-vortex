#!/bin/bash

ENV_NAME="conda-env"
PYTHON_VERSION="3.9"

install_conda() {
    echo "Downloading Miniconda..."
    if [ "$(uname -s)" == "Darwin" ]; then
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda
    else
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    fi
    export PATH="$HOME/miniconda/bin:$PATH"
}

if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Do you want to install it? (y/n)"
    read -r response
    if [ "$response" == "y" ]; then
        install_conda
    else
        echo "Installation canceled. Conda is required to continue."
        exit 1
    fi
fi

echo "Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment: $ENV_NAME"
source $HOME/miniconda/bin/activate $ENV_NAME

echo "Installing dependencies via Conda..."
conda install numpy pandas matplotlib scikit-learn flake8 -y

echo "Installing dependencies via pip..."
pip install mne awscli
if [ "$(uname -s)" == "Darwin" ]; then
    pip install pyqt5
fi

if [ $? -eq 0 ]; then
    echo "All dependencies have been successfully installed."
else
    echo "An error occurred during dependency installation."
    exit 1
fi

conda activate $ENV_NAME

echo "To run the script, ensure the environment is activated ("conda activate $ENV_NAME"), then execute:"
echo "  python script.py"