import boto3
import json
import streamlit as st
from string import Template
from termcolor import colored, cprint
import random
import time

PAGE_TITLE = "I Remember Everything"
PAGE_ICON = ":mechanic:"
APP_TITLE = ":blue[{}] {}".format(PAGE_TITLE, PAGE_ICON)
PROMPT_TEMPLATE = "You are a helpful and succinct assistant. Give a one-sentence reply to the user's question.\n$INPUT"
ASSISTANT_PROMPT = "Hello there! How can I assist you today?"
NEW_CONVERSATION_KEYWORD = "/new"
USER_PROMPT = "What's up? To begin a new conversation, please type \"{}\" without the quotation marks.".format(NEW_CONVERSATION_KEYWORD)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.header(APP_TITLE, divider=False)
st.chat_message("Assistant").markdown(ASSISTANT_PROMPT)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_size" not in st.session_state:
    st.session_state.chat_size = 0

memory_size = st.sidebar.empty()
prompt_template = st.sidebar.text_area(label="Prompt Template", value=PROMPT_TEMPLATE)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
maxtokens = st.sidebar.slider("Max Tokens", 1, 2048, 1024)

def refresh_memory():
    if st.session_state.chat_size > maxtokens:
        token_size = ":red[{}]".format(st.session_state.chat_size)
    elif st.session_state.chat_size > maxtokens*0.8:
        token_size = ":orange[{}]".format(st.session_state.chat_size)
    else:
        token_size = ":green[{}]".format(st.session_state.chat_size)

    memory_size.markdown(body="<small>Est. Tokens in Conversation: {}</small>".format(token_size), unsafe_allow_html=True)

def initialize_session_state():
    st.session_state.messages = []
    st.session_state.chat_size = 0
    refresh_memory()

refresh_memory()

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
if "model" not in st.session_state:
    st.session_state["model"] = modelId
else:
    if modelId != st.session_state["model"]:
        initialize_session_state()
        st.session_state["model"] = modelId

prompt_input = ''

accept = 'application/json'
contentType = 'application/json'

def construct_prompt(body, prompt_template):
    t = Template(prompt_template)
    return t.substitute(INPUT=body)

def construct_payload(prompt_input, prompt_template):
    if "cohere.command" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input, prompt_template),
            "max_tokens": maxtokens,
            "temperature": temperature,
            "p": 0.9
        })

    if "meta.llama2" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input, prompt_template),
            "max_gen_len": maxtokens,
            "temperature": temperature,
            "top_p": 0.9
        })

    if "amazon.titan" in modelId:
        body = json.dumps({
            "inputText": construct_prompt(prompt_input, prompt_template),
            "textGenerationConfig": {
                "maxTokenCount": maxtokens,
                "stopSequences": [ "User:"],
                "temperature": temperature,
                "topP": 0.9
            }
        })

    if "anthropic.claude" in modelId:
        body = json.dumps({
            "prompt": '\n\nHuman: {}\n\nAssistant:'.format(construct_prompt(prompt_input, prompt_template)),
            "max_tokens_to_sample": maxtokens,
            "temperature": temperature,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [ "Human:", "User:" ]
        })

    if "ai21.j2" in modelId:
        body = json.dumps({
            "prompt": construct_prompt(prompt_input, prompt_template),
            "maxTokens": maxtokens,
            "temperature": temperature,
            "topP": 1,
            "stopSequences": [ "User:" ],
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
        cprint(err, "black", "on_red")
        st.error(err)
        exit(-1)

    print("="*78)

    return response

def strip_prefix(result):
    prefix = "Assistant:"
    if result.startswith(prefix):
        result = result[len(prefix):]
    return result.lstrip()

def ai_response(response, c):
    # c = st.empty()

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
                chunk = strip_prefix(chunk)

            cprint(chunk, "black", "on_yellow", end="", flush=True)
            chunks.append(chunk)
            c.markdown(''.join(chunks) + "▌")

        print()
        print("="*78)
        c.markdown(''.join(chunks))
        return strip_prefix(''.join(chunks))

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

        result = strip_prefix(result)
        cprint(result.lstrip(), "black", "on_yellow")
        print("="*78)

        c.markdown(result.lstrip())
        return result.lstrip()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(USER_PROMPT):
    if NEW_CONVERSATION_KEYWORD == prompt:
        initialize_session_state()
        st.rerun()
        exit(0)

    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": prompt})

    # Display user message in chat message container
    st.chat_message("User").markdown(prompt)

    # Display assistant response in chat message container
    full_request = ""
    with st.chat_message("Assistant"):
        message_placeholder = st.empty()

        for m in st.session_state.messages:
            full_request += m["role"] + ": " + m["content"] + "\n"
                             
        payload = construct_payload(full_request, prompt_template)
        response = ai_request(payload)
        answer = strip_prefix(ai_response(response, message_placeholder))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "Assistant", "content": answer})
    full_request += "Assistant: " + answer + "\n"
    st.session_state.chat_size = int(len(full_request)/4)
    refresh_memory()
