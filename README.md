# 🔍 POWL Miner
**Process Mining with the Partially Ordered Workflow Language**

The POWL Miner allows you to perform **process discovery from event logs**, leveraging the **Partially Ordered Workflow Language (POWL) 2.0**. The generated POWL 2.0 models can be viewed and exported as BPMNs or Petri nets (PNML). For more details on POWL 2.0, please refer to the paper: [**Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs**](https://arxiv.org/abs/2505.07052).


## 🚀 Launching as a Streamlit App

You have two options for running the POWL Miner as a Streamlit App:

### ☁️ On the Cloud
Access the hosted version directly:
[**https://powl-miner.streamlit.app/**](https://powl-miner.streamlit.app/)

### 💻 Locally
To run the Streamlit application on your own machine:

  1. Clone this repository.
  2. Install the required dependencies ('requirements.txt') and packages ('packages.txt').
  3. Run:
     ```bash
     streamlit run app.py
     ```
#### <img src="https://raw.githubusercontent.com/mustay/dashboard-icons/master/png/docker.png" alt="Docker icon" width="20" height="20"> Docker:

Alternatively, you can install it using the provided Docker image:

1. Pull the Docker image:
     ```bash
     docker pull ghcr.io/humam-kourani/powl:latest
     ```
2. Run the app:
     ```bash
     docker run -p 8501:8501 powl
     ```




## 🐍 Installing as a Python Library

You can also install the POWL Miner as a Python library to integrate its functionalities into your own scripts.

1. Install the required packages ('packages.txt').
2. Install the library via pip:
    ```bash
    pip install powl
    ```

**👉 Usage Example:**
     Check the `examples/` directory of this repository.
