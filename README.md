# triton-bls

> A project for testing the Business Logic Scripting (BLS) feature of the NVIDIA Triton Inference Server.

## About the Project

This project explores and evaluates the capabilities of Triton's Business Logic Scripting (BLS). BLS allows for the creation of custom backends that can execute arbitrary Python code, enabling complex pre-processing, post-processing, and control flow logic to be orchestrated by Triton itself. This repository contains a test setup to run inference on several models and evaluate the performance and flexibility of BLS.

## Tech Stack

*   [Python](https://www.python.org/)
*   [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
*   [Docker](https://www.docker.com/)
*   [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Usage

Below are the instructions for you to set up and run the project locally.

### Prerequisites

You need to have the following software installed:

*   [Python](https://www.python.org/downloads/) (3.8 or higher)
*   [Docker](https://docs.docker.com/get-docker/)
*   [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

### Installation and Setup

Follow the steps below:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/luizvilasboas/triton-bls.git
    ```

2.  **Navigate to the project directory**
    ```bash
    cd triton-bls
    ```

3.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Build the Triton Docker image**
    ```bash
    docker build -t triton-inference-server-bls .
    ```

### Workflow

1.  **Start the Triton Server**
    Run the provided script to launch the Triton server in a Docker container with the necessary model repository volume.
    ```bash
    bash server.sh
    ```

2.  **Run the client**
    Execute the client script to start sending inference requests to the server.
    ```bash
    python main.py
    ```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

## License

This project is licensed under The Unlicense. See the `LICENSE` file for details.
