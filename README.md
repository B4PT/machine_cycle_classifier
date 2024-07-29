# Cycle Prediction FastAPI Application

This project sets up a FastAPI application to predict the condition of a cycle based on provided data and pre-trained models. It can be run locally or within a Docker container.

## Features

- Web interface to input a cycle number and receive a prediction.
- Uses the best-performing model based on F1 score for predictions.

## Requirements

- Docker (optional for local development)
- Python 3.9+ (for development outside Docker)

## Setup and Installation

1. **Clone the Repository**

   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>

   ```

2. **Prepare Data**

Ensure you have the following data files in the data_subset directory:

- profile.txt
- FS1.txt
- PS2.txt

3. **Configure**

Edit config.py to set the TRAIN_SIZE and TARGET_COLUMN values.

## Running the Application with Docker

1. **Build the Docker Image**

   ```sh
   docker build -t classify .
   ```

2. **Run the Docker Container**
   ```sh
   docker run -p 80:80 classify
   ```

The application will be available at http://localhost:80.

## Access the Web Interface

Open your web browser and go to http://localhost:80. You should see a form to enter a cycle number and get predictions.

## Troubleshooting

- Error loading ASGI app: Ensure that main.py is correctly specified and that your Dockerfile is in the correct directory.
- Empty Predictions: Verify the data files and ensure they are correctly formatted and located in the data_subset directory.
