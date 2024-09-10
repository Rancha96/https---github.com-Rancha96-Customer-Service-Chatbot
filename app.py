from datetime import datetime
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv 
from pathlib import Path
import io
import csv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from htmlTemplates import css, user_template, bot_template, footer
from langchain.chat_models.openai import ChatOpenAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Directory containing PDF files
PDF_DIRECTORY = "E:/Work/University"

# Get list of PDF files in the directory
pdf_files = [os.path.join(PDF_DIRECTORY, file) for file in os.listdir(PDF_DIRECTORY) if file.endswith(".pdf")]

def get_pdf_text(pdf_files):
    text = ""
    for file in pdf_files:
        pathlib_path = Path(file)
        filename = pathlib_path.name
        st.text(f"displaying {filename}")
        with open(file, 'rb') as f:
            pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = Ollama(model="llama3", temperature = 0.3)
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm = Ollama(model = "gemma2")
    # llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    # Measure the start time
    start_time = time.time()

    if detect_reservation_intent(user_question):
        st.session_state.chat_history.append({'content': user_question, 'role': 'user'})
        st.session_state.reservation_mode = True
        
    # Ensure chat_history is initialized
    else:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []  # Initialize as an empty list
        
        # Define the system prompt
        system_prompt = "You are a helpful and professional customer service agent. Always address user inquiries in a polite and informative manner. If the inquire is not related to restaurant, apologise tell the customer that you only can answer related to restaurant’s menu, food, policies and reservations. If you don't have the information ask them to contact the restaurant "
        full_prompt = f"{system_prompt}\n {user_question}"

        # Get the bot's response
        response = st.session_state.conversation({'question': full_prompt})

        # Append user question and assistant's response to the chat history
        st.session_state.chat_history.append({'content': user_question, 'role': 'user'})
        st.session_state.chat_history.append({'content': response['answer'], 'role': 'assistant'})
        st.session_state.chat_history.append({'role': 'feedback', 'response_feedback': None})  # Add feedback placeholder

        # Measure the end time after generating response
        end_time = time.time()

        # Calculate response time
        response_time = end_time - start_time

         # Save the conversation to the CSV file with response time
        append_to_csv(user_question, response['answer'], None, response_time)



def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            # Ensure the assistant message (regular or confirmation) is displayed correctly
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        elif message['role'] == 'feedback':
            if message['response_feedback'] is None:
                # Display feedback buttons only if feedback is not yet given
                col1, col2, col3 = st.columns([0.1, 1, 12])  # Adjust the proportions for tighter spacing
                with col1:
                    st.write("")  # Empty column for alignment
                with col2:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        message['response_feedback'] = 'positive'
                        capture_feedback(i // 3, 'positive')  # Immediately record feedback in CSV
                with col3:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        message['response_feedback'] = 'negative'
                        capture_feedback(i // 3, 'negative')  # Immediately record feedback in CSV
            else:
                st.write(f"Feedback received: {message['response_feedback']}")

# Function to append chat data to CSV
def append_to_csv(user_question, bot_response, feedback, response_time):
    file_exists = os.path.isfile("chat_history.csv")
    
    with open("chat_history.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header only if the file does not exist
        if not file_exists:
            writer.writerow(["timestamp", "user_question", "bot_response", "feedback","response_time (seconds)"])
        
        # Append the current chat data without feedback initially
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), user_question, bot_response, feedback, response_time])


def update_csv_feedback(row_number, feedback):
    with open("chat_history.csv", mode='r+', newline='', encoding='utf-8') as file:
        reader = list(csv.reader(file))

    # Ensure we're not updating the header row and the correct row exists
    if 1 <= row_number < len(reader):
        reader[row_number][3] = feedback  # Update the feedback column

    # Write the updated content back to the file
    with open("chat_history.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(reader)

def capture_feedback(row_number, feedback):
    # Update feedback in CSV for the correct row
    update_csv_feedback(row_number + 1, feedback)  # row_number + 1 since the first row is the header
    
    # Update the in-memory chat history with feedback
    st.session_state.chat_history[(row_number + 1) * 3 - 1]['response_feedback'] = feedback

# Function to detect if the user wants to make a reservation
def detect_reservation_intent(user_question):
    reservation_intents = ["make a reservation", "book a table", "reserve a table", "table booking","book me","book us"]
    for intent in reservation_intents:
        if intent in user_question.lower():
            st.session_state.reservation_mode = True
            return True
    return False
    

# Function to handle reservations
def reservation_form():
    st.subheader("Table Reservation")
    with st.form(key='reservation_form'):
        name = st.text_input("Name")
        date = st.date_input("Date", datetime.today())
        res_time = st.time_input("Time")
        party_size = st.number_input("Number of people", min_value=1, max_value=20)
        
        submit_button = st.form_submit_button(label='Submit Reservation')

        if submit_button:
            # Format time to show only hours and minutes
            formatted_time = res_time.strftime('%H:%M')

            # Save the reservation to CSV file
            save_reservation_to_csv(name, date, formatted_time, party_size)
            
            
            
            st.session_state.reservation_mode = False 
            confirmation_msg = (
                f"Reservation saved!<br>"
                f"Name: {name}<br>"
                f"Date: {date}<br>"
                f"Time: {formatted_time}<br>"
                f"Party Size: {party_size}"
            )
            st.session_state.chat_history.append({'content': confirmation_msg, 'role': 'assistant'})

            # Add a feedback placeholder for the confirmation message
            st.session_state.chat_history.append({'role': 'feedback', 'response_feedback': None})            

            # Save the confirmation message to CSV
            append_to_csv("Reservation request", f"Reservation saved! Name: {name}, Date: {date}, Time: {formatted_time}, Party Size: {party_size}", None,None)


CSV_FILE_PATH = 'reservations.csv'

# Function to save reservation to CSV
def save_reservation_to_csv(name, date, res_time, party_size):
    file_exists = os.path.isfile(CSV_FILE_PATH)

    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if file does not exist
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time', 'Party Size'])

        writer.writerow([name, date, res_time, party_size])
    st.success(f"Reservation saved! Name: {name}, Date: {date}, Time: {res_time}, Party Size: {party_size}")
    # st.write(bot_template.replace("{{MSG}}", f"Reservation saved! Name: {name}, Date: {date}, Time: {time}, Party Size: {party_size}"), unsafe_allow_html=True)
    

def main():
    # Load .env
    load_dotenv()

    #load genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Set page title
    st.set_page_config(page_title="Customer Service Chatbot", page_icon='🤖')

    # Apply custom CSS to maintain the original UI style
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "reservation_mode" not in st.session_state:
        st.session_state.reservation_mode = None  # Track if the user is making a reservation
    # if "just_completed_reservation" not in st.session_state:
        st.session_state.just_completed_reservation = None  # Track when reservation is done

    # Set page header
    st.header("The Thameside Table Customer Service 🍝", divider="rainbow")

    # Display chat history
    if st.session_state.chat_history:
        display_chat_history()

    if st.session_state.reservation_mode:
    # Show the reservation form if the user wants to make a reservation
        reservation_form()
        
    # User input
    with st.form(key="user_input_form", clear_on_submit=True):
        user_question = st.text_input("How may I assist you?", key="user_input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_question:
        handle_userinput(user_question)
        st.experimental_rerun()  # Refresh the app to display the new message
    
    with st.sidebar:
        st.subheader("My documents")
        with st.spinner("Processing"):
            # Get PDF text
            raw_text = get_pdf_text(pdf_files)

            # Get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # Create vectorstore
            vectorstore = get_vectorstore(text_chunks)

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()