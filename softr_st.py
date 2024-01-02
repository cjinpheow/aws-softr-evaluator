import os
import boto3
import json
import streamlit as st
from string import Template
from termcolor import colored, cprint

SOFTR_CONTROLS_PATH = "controls" + os.sep

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9)
maxtokens = st.sidebar.slider("Max Tokens", 1, 2048, 1024)

# Service Offering FTR requirements
requirements = {
    "DEF-001  - Clearly define the core of service offering (what, who, and how)": "DEF-001",
    "PROJ-001 - Define a customer engagement project plan template": "PROJ-001",
    "TECH-001 - Identify relevant AWS technical expertise for the service offering": "TECH-001",
    "RISK-001 - Identify and Mitigate risks associated with the service offering": "RISK-001",
    "SEC-001  - Secure AWS accounts governance": "SEC-001",
    "SEC-002  - Secure access to customer-owned AWS accounts": "SEC-002",
    "SAAS-001 - SaaS components pass a software AWS Foundational Technical Review": "SAAS-001",
    "CUS-001  - Implement a process to collect customer feedback": "CUS-001"
}

requirement = 'DEF-001  - Clearly define the core of service offering (what, who, and how)'
requirement = st.sidebar.radio(
    label='Select a requirement:',
    index=0,
    options=requirements
)
requirement = requirements[requirement]

print("Evaluating control: ", end="")
cprint(requirement, "black", "on_green")

# AWS Bedrock Regions (as of Dec 2023)
regions = {
    "US East (N. Virginia)": "us-east-1",
    "US West (N. California)": "us-west-2",
    "Asia Pacific (Singapore)": "ap-southeast-1",
    "Asia Pacific (Tokyo)": "ap-northeast-1",
    "Europe (Frankfurt)": "eu-central-1",
}

region = 'US East (N. Virginia)'
region = st.sidebar.radio(
    label='Select a region:',
    index=0,
    options=regions
)
region = regions[region]

print("In region: ", end="")
cprint(region, "black", "on_green")

try:
    client = boto3.client('bedrock-runtime', region_name=region)
except Exception as e:
    print("Error: ", e)
    exit(1)

models = {}
streaming_models = []

def get_models(region):
    model_client = boto3.client('bedrock', region_name=region)
    model_data = model_client.list_foundation_models()
    for model in model_data['modelSummaries']:
        if ":" in model['modelId']:
            continue
        if "TEXT" in model['inputModalities'] and "TEXT" in model['outputModalities']:
            models[model['modelName']] = model['modelId']
            if model['responseStreamingSupported']:
                streaming_models.append(model['modelId'])
    
    return models, streaming_models

models, streaming_models = get_models(region)

modelId = 'Titan Text Large'

modelId = st.sidebar.radio(
    label="Select a model:", 
    index=0, 
    options = models
)

modelId = models[modelId]

print("Using model: ", end="")
cprint(modelId, "black", "on_green")

st.title("SOFTR Evaluator 🧑‍💻")

prompt_input = ''

accept = 'application/json'
contentType = 'application/json'

def read_file(file_path):
    if os.path.isfile(file_path):
        text_file = open(file_path, "r")
        data = text_file.read()
        text_file.close()
        return data

prompt_template = read_file(SOFTR_CONTROLS_PATH+requirement+'.prompt')

def construct_prompt(body):
    t = Template(prompt_template)
    return t.substitute(partner=body)

def construct_payload(prompt_input):
    if "cohere.command" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input),
            "max_tokens": maxtokens,
            "temperature": temperature,
            "p": 0.9
        })

    if "meta.llama2" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input),
            "max_gen_len": maxtokens,
            "temperature": temperature,
            "top_p": 0.9
        })

    if "amazon.titan" in modelId:
        body = json.dumps({
            "inputText": construct_prompt(prompt_input),
            "textGenerationConfig": {
                "maxTokenCount": maxtokens,
                "stopSequences": [],
                "temperature": temperature,
                "topP": 0.9
            }
        })

    if "anthropic.claude" in modelId:
        body = json.dumps({
            "prompt": '\n\nHuman: {}\n\nAssistant:'.format(construct_prompt(prompt_input)),
            "max_tokens_to_sample": maxtokens,
            "temperature": temperature,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [ "Human:" ]
        })

    if "ai21.j2" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input),
            "maxTokens": maxtokens,
            "temperature": temperature,
            "topP": 1,
            "stopSequences": [],
            "countPenalty": {
                "scale": 0
            },
            "presencePenalty": {
                "scale": 0
            },
            "frequencyPenalty": {
                "scale": 0
            }
        })
    
    return body

def ai_request(body):
    print("Evaluating partner input:")
    cprint(prompt_input, "black", "on_green")

    print("="*78)
    print(f"Invoking model ({modelId}) with this payload:")
    print("-"*78)
    cprint(json.dumps(json.loads(body), indent=2), "black", "on_cyan")

    try:
        if modelId in streaming_models:
            response = client.invoke_model_with_response_stream(
                body=body, modelId=modelId, accept=accept,  contentType=contentType)
        else:
            response = client.invoke_model(
                body=body, modelId=modelId, accept=accept,  contentType=contentType)
    except Exception as err:
        print(err)
        exit(-1)

    print("="*78)
    return response

def ai_response(response):
    c = st.empty()

    streamed = ""
    if modelId in streaming_models:
        streamed = "streamed "

    print("Here's the {}result: ".format(streamed))

    print("-"*78)
    if modelId in streaming_models:
        chunks = []
        for event in response['body']:
            chunk = ''
            response_chunk = json.loads(event['chunk'].get('bytes'))
            if "amazon.titan" in modelId:
                chunk = response_chunk['outputText']

            if "anthropic.claude" in modelId:
                chunk = response_chunk['completion']

            if "cohere.command" in modelId:
                chunk = response_chunk['generations'][0]['text']

            if "meta.llama2" in modelId:
                chunk = response_chunk['generation']

            if not chunks:
                chunk = chunk.lstrip()

            cprint(chunk, "black", "on_yellow", end="", flush=True)
            chunks.append(chunk)
            c.write(''.join(chunks) + "▌")

        print()
        print("="*78)
        c.write(''.join(chunks))
        return ''.join(chunks)

    else:
        response_body = json.loads(response.get('body').read())

        result = ''
        if "amazon.titan" in modelId:
            result = response_body['results'][0]['outputText']

        if "anthropic.claude" in modelId:
            result = response_body['completion']

        if "ai21.j2" in modelId:
            result = response_body['completions'][0]['data']['text']
            
        if "cohere.command" in modelId:
            result = response_body['generations'][0]['text']
            
        if "meta.llama2" in modelId:
            result = response_body['generation']

        cprint(result.lstrip(), "black", "on_yellow")
        print("="*78)

        c.write(result.lstrip())
        return result.lstrip()

with st.form('my_form'):
    text = st.text_area(label='Enter partner input here:', label_visibility='collapsed',placeholder='Enter partner input here')
    submitted = st.form_submit_button('Submit')
    if submitted:
        payload = construct_payload(text)
        response = ai_request(payload)
        answer = ai_response(response)
