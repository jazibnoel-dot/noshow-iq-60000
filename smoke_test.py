import sys
import requests

url = sys.argv[1].rstrip("/")
sample = {
    "age": 30, "scholarship": 0, "hipertension": 0,
    "diabetes": 0, "alcoholism": 0, "handcap": 0,
    "sms_received": 1, "days_in_advance": 5, "hour_of_booking": 10,
}
tests = [
    ("GET", "/health", None),
    ("POST", "/predict", sample),
    ("GET", "/stats", None),
]
for method, endpoint, body in tests:
    r = requests.request(method, url + endpoint, json=body)
    status = "PASS" if r.status_code == 200 else "FAIL"
    print(f"{status}  {method:4s}  {endpoint}  ->  {r.status_code}")