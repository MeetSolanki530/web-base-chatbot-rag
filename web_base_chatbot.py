from dotenv import load_dotenv
import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()


## Document Loader
def document_loader(web_path):
    """Load documents from a web URL and split them into chunks."""
    loaded_docs = WebBaseLoader(web_path=web_path).load()
    splitted_docs = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(loaded_docs)
    return splitted_docs


## Embedding and VectorStore Load
def load_vectorstore(web_path, embeddings):
    """Create a FAISS vector store from web documents."""
    docs = document_loader(web_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


## Create Retriever
def create_retriever(vectorstore):
    """Create a retriever from the vector store."""
    return vectorstore.as_retriever()


## Create RAG Chain
def create_rag_chain(llm, retriever):
    """Create a retrieval-augmented generation chain."""
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you do not know the answer, say you do not know. "
        "Use a maximum of three sentences and keep the answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


## Streamlit App
def main():
    st.set_page_config(
        page_title="Web-Based Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Web-Based Chatbot")
    st.markdown("Ask questions about any website content.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        web_url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        model_name = st.selectbox(
            "Select Model",
            [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "moonshotai/kimi-k2-thinking",
                "meta/llama-3.3-70b-instruct"
            ],
            index=3
        )

        load_button = st.button("🔄 Load Website", type="primary")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Load website
    if load_button:
        api_key = os.getenv("NVIDIA_API_KEY")

        if not api_key:
            st.error("⚠️ NVIDIA_API_KEY not found in .env file.")
        elif not web_url:
            st.error("⚠️ Please enter a website URL.")
        else:
            try:
                with st.spinner("Loading website content..."):
                    os.environ["NVIDIA_API_KEY"] = api_key

                    embeddings = NVIDIAEmbeddings(
                        model="nvidia/llama-3.2-nemoretriever-300m-embed-v2"
                    )
                    llm = ChatNVIDIA(model=model_name)

                    vectorstore = load_vectorstore(web_url, embeddings)
                    retriever = create_retriever(vectorstore)
                    st.session_state.rag_chain = create_rag_chain(llm, retriever)

                    st.success("✅ Website loaded successfully.")
                    st.session_state.messages = []

            except Exception as e:
                st.error(f"❌ Error loading website: {e}")

    # Chat display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the website..."):
        if st.session_state.rag_chain is None:
            st.warning("⚠️ Load a website first.")
        else:
            # Add and display user message
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke(
                            {"input": prompt}
                        )
                        answer = response["answer"]
                        st.markdown(answer)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        
                        # Optional: Show source documents
                        with st.expander("📚 View Source Documents"):
                            for i, doc in enumerate(response.get("context", []), 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.markdown("---")
                                
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )


if __name__ == "__main__":
    main()
