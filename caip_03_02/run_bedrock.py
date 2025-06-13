from call_bedrock import call_bedrock
import boto3

# Create a Bedrock runtime client
bedrock = boto3.client('bedrock-runtime')

# read from file
with open('prompt.txt', 'r') as file:
    prompt = file.read()

print(prompt)

# Call Bedrock and print responses
responses = call_bedrock(bedrock, prompt)

for response in responses:
    print(response)