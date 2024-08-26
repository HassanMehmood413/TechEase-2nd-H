import streamlit as st
import requests
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from googletrans import Translator

# Initialize Translator
translator = Translator()

# Custom CSS for headings
st.markdown(
    """
    <style>
    .main-title {
        color: #3498db; /* Blue color */
    }
    .header-title {
        color: #e74c3c; /* Red color */
    }
    .sidebar-header {
        color: #2ecc71; /* Green color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to translate text
def translate_text(text, dest_language):
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return None

# IBM Watson credentials
credentials = Credentials(
    url=st.secrets["IBM_WATSON_URL"],  # Fetch from Streamlit secrets
    api_key=st.secrets["IBM_WATSON_API_KEY"]  # Fetch from Streamlit secrets
)

# Initialize API client
client = APIClient(credentials)

# Set default project
project_id = st.secrets["IBM_WATSON_PROJECT_ID"]  # Fetch from Streamlit secrets
client.set.default_project(project_id)

# Initialize model inference
model_id = ModelTypes.GRANITE_13B_CHAT_V2
model = ModelInference(
    model_id=model_id,
    params={
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 500,
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.STOP_SEQUENCES: ["\n"]
    },
    credentials=credentials,
    project_id=project_id
)

# Google Custom Search API details
API_KEY = st.secrets["GOOGLE_API_KEY"], # Fetch from Streamlit secrets
SEARCH_ENGINE_ID = st.secrets["GOOGLE_SEARCH_ENGINE_ID"] # Fetch from Streamlit secrets

# Function to summarize text (dummy implementation, replace with actual summarization)
def summarize_text(text):
    return text[:min(1000, len(text))]  # Simplistic approach for example

# Function to search for articles using Google Custom Search API
def search_articles(query):
    search_url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json()
        articles = results.get('items', [])
        return [(article['title'], article['link']) for article in articles]
    else:
        return []

# Streamlit UI

st.markdown('<h1 class="main-title">Tech Ease</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="header-title">Get Solutions for Your Device Issues</h2>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Options")

# User input section
st.sidebar.subheader("Describe Your Issue")
input_text = st.sidebar.text_area("Enter the issue with your electronic device or mobile phone:", "")

# Language selection
language = st.sidebar.selectbox(
    "Select Output Language",
    ["English", "Spanish", "French", "German"]  # Add other languages as needed
)

# Spacer
st.sidebar.write("\n")  # Adds an empty line

# Heading for Past Searches
st.sidebar.subheader("Past Searches")
if 'history' not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    for i, (text, result) in enumerate(st.session_state.history):
        if st.sidebar.button(f"View {i+1}: {text[:30]}...", key=f"history_{i}"):
            st.session_state.current_response = result  # Load the corresponding result
            st.write(f"Past Solution for: {text}")
            st.success(result)
else:
    st.sidebar.write("No past searches.")

# Spacer
st.sidebar.write("\n")  # Adds an empty line

# Heading for Other Actions
st.sidebar.subheader("Actions")

# Button to get solution
if st.sidebar.button("Get Solution"):
    if input_text:
        prompt = f"Provide a detailed solution to fix the following issue with a mobile power IC. Describe the possible causes of the issue and the steps to troubleshoot and repair it. Issue: {input_text}"
        
        try:
            response = model.generate_text(prompt=prompt)
            st.write("Generated Solution:")
            st.success(response)  # Display the response
            
            # Save the response to session state
            st.session_state.current_response = response
            
            # Save to history
            st.session_state.history.append((input_text, response))
        except Exception as e:
            st.error(f"Failed to generate solution: {e}")
    else:
        st.warning("Please describe the issue before generating a solution.")

# Button to translate text
if 'current_response' in st.session_state:
    if st.sidebar.button("Translate"):
        if language != "English":
            translated_response = translate_text(st.session_state.current_response, language.lower())
            if translated_response:
                st.write(f"Translated Solution ({language}):")
                st.success(translated_response)
        else:
            st.warning("No translation needed; the response is already in English.")

# Button to summarize text
if 'current_response' in st.session_state:
    if st.sidebar.button("Summarize"):
        try:
            summarized_text = summarize_text(st.session_state.current_response)
            st.write("Summarized Solution:")
            st.success(summarized_text)
        except Exception as e:
            st.error(f"Failed to summarize: {e}")

# Button to search for related articles
if st.sidebar.button("Find Articles"):
    if input_text:
        try:
            articles = search_articles(input_text)
            if articles:
                st.write("Related Articles:")
                for title, link in articles:
                    st.markdown(f"[{title}]({link})")
            else:
                st.write("No articles found.")
        except Exception as e:
            st.error(f"Failed to fetch articles: {e}")
    else:
        st.warning("Please describe the issue before searching for articles.")
