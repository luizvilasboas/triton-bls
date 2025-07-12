# triton-bls

This repository contains a project for testing the Backend-as-a-Service (BLS) of the Triton Inference Server. The project involves running inference on several models to evaluate the capabilities and performance of Triton's BLS.

## Introduction

The Triton Inference Server, developed by NVIDIA, is an open-source platform that simplifies the deployment of AI models at scale in production. This project aims to test the Backend-as-a-Service (BLS) functionality of the Triton Inference Server by performing inference on a selection of models. Through this testing, we aim to evaluate the performance, scalability, and ease of use of Triton's BLS.

## Requirements

Before running this project, ensure you have the following requirements installed:

- Docker
- NVIDIA Docker Toolkit
- Triton Inference Server (latest version)
- Python 3.8 or higher
- Required Python packages (specified in `requirements.txt`)

## Usage

To run the tests, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/luizvilasboas/triton-bls.git
    cd triton-bls
    ```

2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

3. Build Docker image and run the container:
    ```
    docker build -t triton-inference-server-bls .
    bash server.sh
    ```

4. Start the inference tests:
    ```
    python main.py
    ```

## Contributing

If you're interested in contributing to this project, feel free to open a merge request. We welcome all forms of collaboration!

## License

This project is available under the [The Unlicense](https://github.com/luizvilasboas/triton-bls/blob/main/LICENSE). For more information, please see the LICENSE file.
