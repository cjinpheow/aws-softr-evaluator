import os
import sys
import boto3
import json
import argparse
import gnureadline # this requirement is needed to lift the input character limit
from string import Template
from simple_term_menu import TerminalMenu
from termcolor import colored, cprint

PROMPT_TEMPLATE = "Improve upon the following text in a critical but helpful way:\n$input"
MAX_TOKENS_COUNT = 2048

# AWS Bedrock Regions (as of Dec 2023)
regions = {
    "US East (N. Virginia)": "us-east-1",
    "US West (N. California)": "us-west-2",
    "Asia Pacific (Singapore)": "ap-southeast-1",
    "Asia Pacific (Tokyo)": "ap-northeast-1",
    "Europe (Frankfurt)": "eu-central-1",
}

parser = argparse.ArgumentParser()

parser.add_argument(
    '--region', 
    type=str,
    required=False,
    help='AWS Region')

parser.add_argument(
    '--model-id', 
    type=str,
    required=False,
    help='The foundation model identifier')

parser.add_argument(
    '--body', 
    type=str,
    required=False,
    help='(blob) Text to improve')

args = parser.parse_args()

def select_region():
    terminal_menu = TerminalMenu(regions, 
                                 menu_highlight_style=("bg_green", "fg_black"),
                                 title="Select a region:")
    terminal_menu.show()
    return regions[terminal_menu.chosen_menu_entry]

region = args.region
if region is None or region not in regions.values():
    region = select_region()

print("In region: ", end="")
cprint(region, "black", "on_green")

try:
    client = boto3.client('bedrock-runtime', region_name=region)
except Exception as e:
    print("Error: ", e)
    exit(1)

streaming_models = []
model_client = boto3.client('bedrock', region_name=region)
model_data = model_client.list_foundation_models()
models = {}
for model in model_data['modelSummaries']:
    if ":" in model['modelId']:
        continue
    if "TEXT" in model['inputModalities'] and "TEXT" in model['outputModalities']:
        models[model['modelName']] = model['modelId']
        if model['responseStreamingSupported']:
            streaming_models.append(model['modelId'])

def select_model(models):
    terminal_menu = TerminalMenu(models, 
                                 menu_highlight_style=("bg_green", "fg_black"),
                                 title="Select a model:")
    terminal_menu.show()
    return models[terminal_menu.chosen_menu_entry]

modelId = args.model_id
if modelId is None or modelId not in models.values():
    modelId = select_model(models)

print("Using model: ", end="")
cprint(modelId, "black", "on_green")

prompt_input = args.body
if prompt_input is None:
    print("Enter text to improve: (Hit Ctrl-D on a blank new line to end) ", flush=True)
    prompt_input = sys.stdin.read()

accept = 'application/json'
contentType = 'application/json'

def construct_prompt(body):
    t = Template(PROMPT_TEMPLATE)
    return t.substitute(input=body)

if "cohere.command" in modelId:
    body = json.dumps({
        "prompt": construct_prompt(prompt_input),
        "max_tokens": MAX_TOKENS_COUNT,
        "temperature": 0.9,
        "p": 0.9
    })

if "meta.llama2" in modelId:
    body = json.dumps({
        "prompt": construct_prompt(prompt_input),
        "max_gen_len": MAX_TOKENS_COUNT,
        "temperature": 0.9,
        "top_p": 0.9
    })

if "amazon.titan" in modelId:
    body = json.dumps({
        "inputText": construct_prompt(prompt_input),
        "textGenerationConfig": {
            "maxTokenCount": MAX_TOKENS_COUNT,
            "stopSequences": [],
            "temperature": 0.9,
            "topP": 0.9
        }
    })

if "anthropic.claude" in modelId:
    body = json.dumps({
        "prompt": '\n\nHuman: {}\n\nAssistant:'.format(construct_prompt(prompt_input)),
        "max_tokens_to_sample": MAX_TOKENS_COUNT,
        "temperature": 0.75,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": [ "Human:" ]
    })

if "ai21.j2" in modelId:
    body = json.dumps({
        "prompt": construct_prompt(prompt_input),
        "maxTokens": MAX_TOKENS_COUNT,
        "temperature": 0.7,
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

print("Evaluating input:")
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

    print()

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
