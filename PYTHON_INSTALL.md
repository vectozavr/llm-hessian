# 🐍 Installing Python 3.12.4 and Setting Up the Project Environment

This guide walks you through installing **Python 3.12.4** from source and preparing your environment to run this project.

---

## 📦 Step 1: Download and Extract Python 3.12.4
Open a terminal and run the following commands in your **home directory**:

```bash
wget https://www.python.org/ftp/python/3.12.4/Python-3.12.4.tgz

# Extract the archive
 tar -xzf Python-3.12.4.tgz
cd Python-3.12.4
```

---

## ⚙️ Step 2: Configure and Build Python
```bash
./configure --prefix=$HOME/python3.12 --enable-optimizations
make -j$(nproc)
make install
```

This installs Python 3.12.4 into `$HOME/python3.12`.

---

## 🛠 Step 3: Update PATH and Verify Python Version
Add the new Python binary to your environment:

```bash
export PATH=$HOME/python3.12/bin:$PATH
source ~/.bashrc
```

Verify the installation:
```bash
python3.12 --version
```
Expected output:
```
Python 3.12.4
```

---

## 🧪 Step 4: Create and Activate a Virtual Environment
Navigate to your **project directory** and set up a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

---

## 📚 Step 5: Install Project Dependencies
With the virtual environment active, install the required libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ✅ Done
You're all set! You can now start running experiments and exploring the Hessian of Perplexity computations.

Need help? Feel free to open an [issue](https://github.com/vectozavr/llm-hessian/issues) on GitHub.

