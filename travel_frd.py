import os
import io
from google.cloud import vision
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from tempfile import NamedTemporaryFile
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'travel-397705-33bade8e16e4.json'
import langchain 
langchain.debug=True

client = vision.ImageAnnotatorClient()

def detect_landmark(file_path):
    try:
        with io.open(file_path, 'rb') as image_file:
            content = image_file.read()

        image = vision_v1.Image(content=content)
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations

        landmark_list ="" # Create an empty list to store landmark descriptions
        for landmark in landmarks:
            # landmark_list.append({'description': landmark.description})
            landmark_list=landmark.description

        # df = pd.DataFrame(landmark_list)  # Convert the list into a DataFrame
        return landmark_list
    except Exception as e:
        print(e)

import streamlit as st
import time
import langchain
import json
from langchain.chat_models import ChatVertexAI
st.markdown(f"<h1 align=left> TravelDosth: AI Travel Buddy🫂❤♾️</h1>",unsafe_allow_html=True)
def llama():
        # Import the required modules
    from langchain.llms import Clarifai
    from langchain import PromptTemplate, LLMChain
    from pathlib import Path
    from pprint import pprint
    import time
    file_path="chat_history.json"
    try:
        chat_history=json.loads(Path(file_path).read_text())
    except:
        chat_history=[]
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.markdown("<p align=center> OR </p>",unsafe_allow_html=True)
    place=st.text_input("Enter Place Manually:")
    p1=""
    if uploaded_image is not None:
        # Check if the uploaded_image is a valid image file
        if uploaded_image.type.startswith('image'):
            # Create a temporary file to store the uploaded image
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                temp_image.write(uploaded_image.read())

            # Get the path to the temporary image file
            image_path = temp_image.name

            # Call the detect_landmark function with the image path
            ans = detect_landmark(image_path)
            if ans:
                place=str(ans)
            else:
                p1="None"
    s1=st.empty()
    if p1=="None":
        s1=st.warning("Can't detect Landmark please enter location manually:")
        time.sleep(4)
        s1=st.write("")
    if place:
        st.markdown(f"<h1 align=center>{place}'s Travel Frnd</h1>",unsafe_allow_html=True)
        # template="""I want to act as My Close Buddy.I Want to know chat about this place {place}.i will ask the question regarding that place. your task is to give the response as my close buddy and nothing else.don't give complete information at once. just give the response what i asked and don't start with i will help give direct response.my question:{question}
    
        # Previous Conversations ;\n {chat_history}
        # """
        # template = """
        #     Hey AI, I'm a traveler currently in {place}, and I'd like you to be my close buddy during my journey. I'll be asking questions about this place, and I want you to respond as if you're my friend. Keep your responses realistic, and feel free to scold me if needed. Remember, don't give away all the information at once. Just respond to my questions, and don't start with "I will help." 

        #     My question: {question}.

        #     Previous Conversations:
        #     {chat_history}
        # """
        
        # template = """
        #     Hey AI, I'm a traveler currently in {place}, and I'd like you to be my close buddy during my journey. Let's keep it casual and friendly. I'll be asking questions about this place, and I want you to respond as if you're my friend. Feel free to use helping words like "hmm," "haha," and more to make it sound natural and friendly. If needed, you can even scold me gently.

        #     My question: {question}.

        #     Previous Conversations:
        #     {chat_history}
        # """
    #    template = """
    #     Hey AI, I'm a traveler currently in {place}, and I'd like you to be my close buddy during my journey. Let's keep it casual and friendly. I'll be asking questions about this place, and I want you to respond as if you're my friend. Feel free to use helping words like "hmm," "haha," and more to make it sound natural and friendly. If needed, you can even scold me gently.

    #     My question: {question}

    #     If I ask for the location, please provide me with the link to that place on Google Maps: [Google Maps Link](https://www.google.com/maps?q={place}).
        
    #     Previous Conversations:
    #     {chat_history}
    #     """ 
        # template = """
        #     Hey AI, I'm a traveler currently in {place}, and I'd like you to be my close buddy during my journey. Let's keep it casual and friendly. I'll be asking questions about this place, and I want you to respond as if you're my friend. Feel free to use helping words like "hmm," "haha," as well as expressions like "well," "oh, by the way," "so, here's the thing," and more to make it sound natural and friendly. If needed, you can even scold me gently.

        #     My question: {question}

        #     If I ask for the location, please provide me with the link to that place on Google Maps: [Google Maps Link](https://www.google.com/maps?q={place}).

        #     And, remember, when you respond, feel free to call me "my friend" or "my buddy" to make it more personal.

        #     Previous Conversations:
        #     {chat_history}
        # """
        # template = """
        # Hey AI, I'm a traveler currently in {place}, and I'd like you to be my close buddy during my journey. Let's keep it casual and friendly. I'll be asking questions about this place, and I want you to respond as if you're my friend. Feel free to use helping words like "hmm," "haha," as well as expressions like "well," "oh, by the way," "so, here's the thing," and more to make it sound natural and friendly. If needed, you can even scold me gently. 😊

        # My question: {question}

        # Please remember to provide location information only when I explicitly ask for it. Until then, keep it a surprise! 🤐

        # And, remember, when you respond, feel free to call me "my friend" or "my buddy" to make it more personal. 👫 don't forget to include emojis to keep the conversation lively and fun! 😄👍

        # Previous Conversations:
        # {chat_history}
        # """
        template = """
        Hey AI, I'm currently exploring {place}, and I'd love for you to be my virtual travel companion on this exciting journey. Let's keep our conversation relaxed and friendly, just like two pals chatting. I'll be throwing questions your way about this place, and I'd appreciate your responses as if we're close friends. Don't hesitate to sprinkle in some friendly words like "hmm," "haha," and use expressions like "well," "oh, by the way," "so, here's the scoop," and more to make our dialogue feel natural and enjoyable. If necessary, you can even give me some friendly advice. 😊

        Here's my latest question: {question}

        Please keep the location details a secret until I ask for them explicitly; that way, we'll keep the element of surprise intact! 🤐

        And, when you respond, feel free to refer to me as "my friend" or "my buddy" to add that personal touch. 👫 Don't forget to sprinkle in some emojis to keep our chat lively and fun! 😄👍

        A Glimpse of Our Previous Conversations:
        {chat_history}
        """



        prompt = PromptTemplate(template=template, input_variables=["place","chat_history","question"])
        # clarifai_llm = Clarifai(
        #     pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
        # )
        llm=ChatVertexAI(model="chat-bison-32k",temperature=0.9,max_output_tokens=2048)
        # Create LLM chain
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        
       # Streamlit chat interface
        spinner_html = """
        <div class="spinner"></div>

        <style>
        .typing-animation {
            display: inline-block;
            white-space: nowrap; /* Prevents line break */
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """
            
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # message_placeholder.markdown("<p>Typing...</p>",unsafe_allow_html=True)
                message_placeholder.markdown(spinner_html,unsafe_allow_html=True)
                full_response = ""
                ai_response=llm_chain.predict(place=place,chat_history=chat_history,question=question)
                for chunk in ai_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.write(full_response + "▌")
                message_placeholder.write(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            conversation = {
                "User": question,
                "Buddy": ai_response
            }
            chat_history.append(conversation)
            with open("chat_history.json","w") as file:
                json.dump(chat_history,file)
                
llama()