import argparse
import requests


def main(cycle_number):
    url = "http://localhost/predict/"
    payload = {"cycle_number": cycle_number, "model_name": "RandomForestClassifier"}

    response = requests.post(url, json=payload)

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send a POST request to the FastAPI endpoint."
    )
    parser.add_argument(
        "cycle_number", type=int, help="The cycle number to send in the request."
    )

    args = parser.parse_args()
    main(args.cycle_number)
