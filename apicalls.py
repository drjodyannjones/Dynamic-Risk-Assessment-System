import requests
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response1 = requests.post(URL + "/prediction")
response2 = requests.get(URL + "/scoring")
response3 = requests.get(URL + "/summarystats")
response4 = requests.get(URL + "/diagnostics")

# Combine all API responses
responses = {
    'predictions': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json()
}

# Write the responses to a JSON file
with open('api_responses.json', 'w') as f:
    json.dump(responses, f, indent=4)

# Combine the responses as a string with proper formatting
combined_responses = "\n".join(
    [f"{key}:\n{json.dumps(value, indent=4)}" for key, value in responses.items()]
)

# Write the combined responses to a file called apireturns.txt
with open('apireturns.txt', 'w') as f:
    f.write(combined_responses)
