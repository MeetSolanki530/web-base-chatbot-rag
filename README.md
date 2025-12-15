# 🤖 Web-Based RAG Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://web-based-rag.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent chatbot that answers questions about any website content using **RAG (Retrieval-Augmented Generation)** with NVIDIA AI endpoints and FAISS vector database.

## 🌟 Features

- **🌐 Web Content Analysis**: Load and analyze any website's content
- **🧠 RAG Architecture**: Uses Retrieval-Augmented Generation for accurate responses
- **⚡ NVIDIA AI Integration**: Powered by NVIDIA AI endpoints for fast inference
- **💾 Vector Database**: FAISS for efficient similarity search
- **💬 Interactive Chat Interface**: Built with Streamlit for seamless user experience
- **📚 Source Citations**: View source documents used to generate answers
- **🎨 Modern UI**: Clean and intuitive interface with real-time feedback

## 🚀 Live Demo

Try the live application: **[Web-Based RAG Chatbot](https://web-based-rag.streamlit.app/)**

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **NVIDIA AI Endpoints**: High-performance LLM inference
- **FAISS**: Vector database for semantic search
- **Streamlit**: Web application framework
- **BeautifulSoup4**: Web scraping and parsing
- **Python 3.8+**: Core programming language

## 📋 Prerequisites

- Python 3.8 or higher
- NVIDIA API Key ([Get one here](https://build.nvidia.com/))
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MeetSolanki530/web-base-chatbot-rag.git
   cd web-base-chatbot-rag
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

## 🎯 Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run web_base_chatbot.py
   ```

2. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

3. **Start chatting**
   - Enter a website URL in the sidebar
   - Select your preferred AI model
   - Click "Load Website"
   - Ask questions about the website content!

## 📖 How It Works

1. **Document Loading**: The application uses `WebBaseLoader` to fetch and parse website content
2. **Text Splitting**: Content is split into manageable chunks using `RecursiveCharacterTextSplitter`
3. **Embedding**: Text chunks are converted to vector embeddings using NVIDIA's embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Response Generation**: Retrieved context is sent to the LLM to generate accurate answers

## 🎨 Features in Detail

### Supported AI Models
- OpenAI GPT-OSS 20B
- OpenAI GPT-OSS 120B
- Moonshot AI Kimi K2 Thinking
- Meta Llama 3.3 70B Instruct (Default)

### RAG Pipeline
```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Response
```

### Source Document Viewer
- Expandable section showing source documents
- Preview of relevant content used for answers
- Transparency in AI responses

## 📁 Project Structure

```
web-base-chatbot-rag/
├── web_base_chatbot.py    # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEY` | Your NVIDIA API key for AI endpoints | Yes |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Meet Solanki**
- Portfolio: [meetsolanki.vercel.app](https://www.aimlmeetsolanki.engineer/)
- GitHub: [@MeetSolanki530](https://github.com/MeetSolanki530)
- LinkedIn: [Meet Solanki](https://linkedin.com/in/meet-solanki-b96a78230)
- Email: solankimeet530@gmail.com

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the RAG framework
- [NVIDIA](https://build.nvidia.com/) for AI endpoints
- [Streamlit](https://streamlit.io/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
