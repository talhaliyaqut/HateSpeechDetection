import streamlit as st
from streamlit import session_state
import json
import os
import whisper
import difflib
import speech_recognition as sr
from deep_translator import GoogleTranslator
from st_audiorec import st_audiorec 
import moviepy.editor as mp
import torch
from transformers import  AutoTokenizer ,AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from pytube import YouTube
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
@st.cache_resource()
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer
def convert_to_prompt(text):
    return f'[INST] In this task, you will be performing a classification exercise aimed at identifying whether the given text contains hate speech or not. Consider the text: {text} </prompt>'   
def transcribe_audio_from_data(file_data):
    with open("temp.mp3", "wb") as f:
        f.write(file_data)
    model = whisper.load_model("base")
    result = model.transcribe("temp.mp3")
    os.remove("temp.mp3")
    return result["text"]

def split_text_into_chunks(text, max_chunk_size=250):
    chunks = []
    start = 0
    prompt_start = "[INST] In this task, you will be performing a classification exercise aimed at identifying whether the given text contains hate speech or not. Consider the text: "
    current_chunk = prompt_start
    while start < len(text):
        end = start + max_chunk_size - len(current_chunk)
        if end > len(text):
            end = len(text)
        chunk = text[start:end]
        current_chunk += chunk
        chunks.append(current_chunk)
        start = end
        current_chunk = prompt_start
    return chunks
def transcribe_video():
    try:
        model = whisper.load_model("base")
        result = model.transcribe("temp2.mp3")
        os.remove("temp2.mp3")
        return result["text"]
    except Exception as e:
        return f"Error in fetching transcript {e}"

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    device = next(model.parameters()).device
    inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        
        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(
                    name,
                    email,
                    age,
                    sex,
                    password,
                    json_file_path,
                )
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")

# def extract_audio(video_file_path, output_dir):
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Extract audio from the video file
#     output_audio_path = os.path.join(output_dir, "audio.wav")
#     subprocess.run(["ffmpeg", "-y", "-i", video_file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_audio_path], check=True)
#     return output_audio_path
def create_account(
    name,
    email,
    age,
    sex,
    password,
    json_file_path="data.json",
):
    try:

        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None
def get_transcript_from_url(url):
    try:
        url_data = urlparse(url)
        id = url_data.query[2::]
        script = YouTubeTranscriptApi.get_transcript(id)
        transcript = ""
        for text in script:
            t = text["text"]
            if t != "[Music]":
                transcript += t + " "
        return transcript
    except:
        try:
            try:
                yt = YouTube(url)
            except:
                return "Connection Error"

            stream = yt.streams.get_by_itag(251)
            stream.download("", "temp.webm")
            model = whisper.load_model("base")
            result = model.transcribe("temp.webm")
            if os.path.exists("temp.webm"):
                os.remove("temp.webm")
            return result["text"]
        except Exception as e:
            return f"Error in fetching transcript {e}"

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")


def main(json_file_path="data.json"):
    st.sidebar.title("Hate Speech Detection")
    page = st.sidebar.radio(
        "Go to", ("Signup/Login", "Dashboard", "Hate Speech Detection"), key="Hate Speech Detection",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio("Select an option", ("Login", "Signup"), key="login_signup")
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)
    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")
    elif page == "Hate Speech Detection":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            model_folder= "hatespeech"
            model, tokenizer = load_model_and_tokenizer(model_folder)

            st.title("Hate Speech Detection")
            
            media_format = st.radio("Choose media format", ("Audio", "Video", "URL"))

            if media_format == "Audio":
                options = ["Record", "Upload"]
                choice = st.radio("Choose an option", options)
                if choice == "Record":
                    recognizer = sr.Recognizer()
                    microphone = sr.Microphone()
                    if st.button("START RECORDING"):
                        with microphone as source:
                            st.info("Listening...")
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            voice_command = recognizer.recognize_google(audio, language="en")
                            print(voice_command)
                        except sr.UnknownValueError:
                            st.write("Could not understand the audio. Please try again.")
                            return
                        except sr.RequestError as e:
                            st.write("Could not understand the audio. Please try again.")
                            return
                        if voice_command:
                            chunks = split_text_into_chunks(transcript)
                            is_hate_speech=False
                            total_chunks = len(chunks)
                            is_hate_speech=False
                            for chunk in chunks:
                                prompt = convert_to_prompt(chunk)
                                predictions = generate_predictions(model, tokenizer, [prompt])
                                if predictions[0] == "Hate Speech":
                                    is_hate_speech = True
                                    break

                            if is_hate_speech:
                                overall_prediction = "Hate Speech"
                                st.markdown(f'<p style="color:red; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)
                            else:
                                overall_prediction = "Not Hate Speech"
                                st.markdown(f'<p style="color:green; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)

                elif choice == "Upload":
                    st.write("Upload an audio file:")
                    audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
                    if audio is not None:
                        st.audio(audio, format="audio/wav")
                        transcription = transcribe_audio_from_data(audio.read()).upper()
                        if transcription:
                            chunks = split_text_into_chunks(transcription)
                            hate_speech_count = 0
                            total_chunks = len(chunks)
                            is_hate_speech=False
                            for chunk in chunks:
                                
                                prompt = convert_to_prompt(chunk)
                                predictions = generate_predictions(model, tokenizer, [prompt])
                                if predictions[0] == "Hate Speech":
                                    is_hate_speech = True
                                    break

                            if is_hate_speech:
                                overall_prediction = "Hate Speech"
                                st.markdown(f'<p style="color:red; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)
                            else:
                                overall_prediction = "Not Hate Speech"
                                st.markdown(f'<p style="color:green; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)

            elif media_format == "Video":
                video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "wmv"])
                if video is not None:
                    with open("temp.mp4", "wb") as f:
                        f.write(video.read())
                    video1 = mp.VideoFileClip('temp.mp4')
                    with st.spinner("Transcribing video..."):
                        audio_file = video1.audio
                        audio = audio_file.write_audiofile('temp2.mp3')
                        
                        transcription = transcribe_video()
                        if transcription:
                            chunks = split_text_into_chunks(transcription)
                            hate_speech_count = 0
                            total_chunks = len(chunks)
                            is_hate_speech=False
                            for chunk in chunks:
                                
                                prompt = convert_to_prompt(chunk)
                                predictions = generate_predictions(model, tokenizer, [prompt])
                                if predictions[0] == "Hate Speech":
                                    is_hate_speech = True
                                    break

                            if is_hate_speech:
                                overall_prediction = "Hate Speech"
                                st.markdown(f'<p style="color:red; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)
                            else:
                                overall_prediction = "Not Hate Speech"
                                st.markdown(f'<p style="color:green; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)

                    
                        
            elif media_format == "URL":
     
                url = st.text_input("Enter the video URL:")
                if url:
                    transcript = get_transcript_from_url(url)
                    if transcript:
                        chunks = split_text_into_chunks(transcript)
                        hate_speech_count = 0
                        total_chunks = len(chunks)
                        is_hate_speech=False
                        for chunk in chunks:
                            prompt = convert_to_prompt(chunk)
                            predictions = generate_predictions(model, tokenizer, [prompt])
                            if predictions[0] == "Hate Speech":
                                is_hate_speech = True
                                break

                        if is_hate_speech:
                            overall_prediction = "Hate Speech"
                            st.markdown(f'<p style="color:red; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)
                        else:
                            overall_prediction = "Not Hate Speech"
                            st.markdown(f'<p style="color:green; font-size:20px;">{overall_prediction}</p>', unsafe_allow_html=True)
                    else:
                        st.error("Error fetching transcript from the URL.")
            
            else:
                st.warning("Please login/signup to app!!.")


if __name__ == "__main__":
    initialize_database()
    main()