import boto3
import json

def construct_body(prompt, max_tokens=2000):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": f"""Human: {prompt}"""
            }
        ]
    }

    return body

def call_bedrock(bedrock_client, prompt, max_tokens=2000, modelId='anthropic.claude-3-sonnet-20240229-v1:0'):
    body = construct_body(prompt, max_tokens=max_tokens)

    response = bedrock_client.invoke_model(
        body=json.dumps(body),
        modelId=modelId,
    )

    result = json.loads(response["body"].read())

    contents = result["content"]

    responses = []

    for content in contents:
        responses.append(content["text"])

    return responses
        



if __name__=="__main__":
    bedrock = boto3.client('bedrock-runtime')

    prompt = "I need a few sentences on why gold is the best color."

    responses = call_bedrock(bedrock, prompt)

    for response in responses:
        print(response)

    