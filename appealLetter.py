import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Function to extract text from uploaded PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Initialize GPT-4 Chat Model with LangChain
def initialize_agent(api_key):
    try:
        # Use the correct ChatOpenAI model
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=api_key
        )
        memory = ConversationBufferMemory()
        return ConversationChain(llm=llm, memory=memory)
    except Exception as e:
        st.error(f"Error initializing OpenAI agent: {e}")
        return None

# Streamlit app
st.title("GPT-4 Medical Claim Appeal Generator")
st.write("Generate medical claim appeal letters, summaries, and suggestions using GPT-4. Upload your documents to get started.")

# Sidebar for OpenAI API Key input
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="Your API key is required to use GPT-4 for generating appeal letters."
)

# Drag-and-drop support for uploading files
st.header("Upload Documents")
eob_file = st.file_uploader("Drag and drop or upload the Explanation of Benefits (EOB)", type=["pdf"])
medical_file = st.file_uploader("Drag and drop or upload the Medical Records", type=["pdf"])
denial_file = st.file_uploader("Drag and drop or upload the Denial Letter", type=["pdf"])

# Preview the content of uploaded documents
if eob_file:
    st.subheader("EOB Preview:")
    eob_text = extract_text_from_pdf(eob_file)
    st.text_area("EOB Content", eob_text, height=200)

if medical_file:
    st.subheader("Medical Records Preview:")
    medical_text = extract_text_from_pdf(medical_file)
    st.text_area("Medical Records Content", medical_text, height=200)

if denial_file:
    st.subheader("Denial Letter Preview:")
    denial_text = extract_text_from_pdf(denial_file)
    st.text_area("Denial Letter Content", denial_text, height=200)

# Process and generate outputs
if st.button("Generate Appeal Letter"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif eob_file and medical_file and denial_file:
        # Initialize the AI agent with the user's API key
        st.write("Initializing AI agent with GPT-4...")
        agent = initialize_agent(api_key)

        if agent is None:
            st.error("Failed to initialize the AI agent. Please check your OpenAI API key.")
        else:
            # Task prompts
            appeal_prompt = f"""
            Generate a professional appeal letter based on these inputs:
            1. Explanation of Benefits (EOB):
            {eob_text}

            2. Medical Records:
            {medical_text}

            3. Denial Letter:
            {denial_text}

            The appeal letter should:
            - Use a polite and professional tone.
            - Clearly state the reason for the appeal.
            - Explain the medical necessity of the procedures.
            - Suggest why the denial reason should be reconsidered.
            """
            
            summarize_prompt = f"""
            Summarize the key details from the following medical records:
            {medical_text}
            """

            rebuttal_prompt = f"""
            The following denial reason was provided:
            {denial_text}

            Suggest improvements to the rebuttal to strengthen the appeal argument.
            """

            with st.spinner("Generating outputs with GPT-4..."):
                # Generate results
                try:
                    appeal_letter = agent.run(appeal_prompt)
                    medical_summary = agent.run(summarize_prompt)
                    rebuttal_suggestions = agent.run(rebuttal_prompt)

                    # Display results
                    st.subheader("Generated Appeal Letter:")
                    st.text_area("Appeal Letter", appeal_letter, height=400)

                    st.subheader("Medical Records Summary:")
                    st.text_area("Summary", medical_summary, height=200)

                    st.subheader("Rebuttal Improvement Suggestions:")
                    st.text_area("Suggestions", rebuttal_suggestions, height=200)

                    # Allow user to download the appeal letter
                    st.download_button(
                        label="Download Appeal Letter",
                        data=appeal_letter,
                        file_name="appeal_letter.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating outputs: {e}")
    else:
        st.error("Please upload all required documents.")
