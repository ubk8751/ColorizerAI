@echo off
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Virtual environment setup complete!
echo Run "venv\Scripts\activate" to activate the environment.
