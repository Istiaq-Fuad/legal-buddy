# LLM Engineering Learning Path

This codebase guides you from fundamental API usage through advanced model deployment, optimization, and code generation.

## Overview

This learning path covers essential LLM engineering skills:

- **API Integration**: Working with different LLM providers
- **Web Data Processing**: Web scraping and intelligent link extraction
- **Multi-Provider Workflows**: Comparing and combining different models
- **UI Development**: Building interactive applications with Gradio
- **Transformer Models**: Using pre-trained models from Hugging Face
- **Tokenization**: Understanding and working with tokenizers
- **Model Optimization**: Quantization and efficient inference

## Curriculum

### 📘 Day 1: Introduction to LLMs and Web Scraping

**File:** `day1_intro.ipynb`

Learn the fundamentals of working with LLMs and build your first practical application.

**Topics:**

- Setting up API credentials (Mistral AI)
- Making basic chat API calls
- Web scraping with BeautifulSoup
- Building a website content summarizer
- Combining web data with LLM generation

**Key Skills:** API integration, web scraping, prompt engineering

---

### 📘 Day 2: Extracting Links with JSON Response Format

**File:** `day2_parser.ipynb`

Extract and intelligently filter links from websites using structured JSON responses.

**Topics:**

- Extending the Website class with link extraction
- JSON response formatting from LLMs
- Building system prompts for intelligent filtering
- Filtering irrelevant links (privacy, ToS, email)
- Complete workflow for multi-page analysis

**Key Skills:** JSON API responses, semantic link filtering, prompt design

---

### 📘 Day 3: Working with Multiple LLM Providers

**File:** `day3_adv_conv.ipynb`

Compare different LLM providers and create interactions between models.

**Topics:**

- Setting up multiple API clients (Mistral, OpenRouter/GPT-4)
- Comparing model responses on the same prompts
- Designing distinct system prompts for different personalities
- Implementing adversarial conversations between models
- Understanding how prompts shape model behavior

**Key Skills:** Multi-provider integration, comparative analysis, personality design

---

### 📘 Day 4: Building Interactive Web UIs with Gradio

**File:** `day4_gradio.ipynb`

Create web interfaces for LLM applications with real-time streaming.

**Topics:**

- Environment setup and API credential management
- Implementing streaming responses for real-time feedback
- Building Gradio interfaces with text input/markdown output
- Creating user-friendly chat interfaces
- Deploying interactive demos

**Key Skills:** UI development, streaming APIs, user experience design

---

### 📘 Day 5: Exploring Transformers and Hugging Face Hub

**File:** `day5_hf_pipelines.ipynb`

Work with pre-trained transformer models for various NLP tasks. (Requires Google Colab)

**Topics:**

- Setting up Google Colab with GPU acceleration
- Hugging Face authentication and hub access
- Using transformer pipelines for sentiment analysis
- Named Entity Recognition (NER)
- Question answering systems
- Working with the Datasets library

**Key Skills:** Transformer pipelines, GPU utilization, NLP task automation

---

### 📘 Day 6: Tokenization and Chat Templates

**File:** `day6_tokenization.ipynb`

Master tokenizers and learn how to format conversations for different models. (Requires Google Colab)

**Topics:**

- Loading pre-trained tokenizers from Hugging Face
- Applying chat templates to format messages
- Comparing tokenization across models (DeepSeek, Phi-4, Qwen)
- Understanding token-to-text conversion
- Preparing prompts for code-focused models

**Key Skills:** Tokenizer mechanics, chat template formatting, cross-model compatibility

---

### 📘 Day 7: Model Quantization and Local LLM Deployment

**File:** `day7_hf_models.ipynb`

Run large language models efficiently on resource-constrained devices. (Requires Google Colab)

**Topics:**

- Installing quantization libraries (bitsandbytes, accelerate)
- Loading models with 4-bit quantization
- Comparing different open-source models:
  - Llama 3.2 1B Instruct
  - Phi-4 Mini Instruct
  - Qwen 3 4B Instruct
  - DeepSeek R1 Distill Qwen 1.5B
- Implementing streaming text generation
- Memory optimization and footprint analysis
- Creating reusable generation functions

**Key Skills:** Model quantization, memory management, inference optimization

---

### 📘 Day 8: Code Generation and Language Porting

**File:** `day8_code_generator.ipynb`

Build a Gradio tool that ports Python code to high-performance Rust or C++ using LLMs, then compiles and runs the generated code.

**Topics:**

- Selecting LLM backends (Groq, optional Gemini)
- Crafting system/user prompts for code translation
- Stripping fenced code blocks from responses
- Compiling and executing generated Rust/C++ code from Python
- Displaying interactive results with Gradio

**Key Skills:** Code generation, prompt design for compilers, Gradio UI, execution safety

---

## Prerequisites

### General Requirements

- Python 3.8+
- Basic understanding of Python
- Familiarity with Jupyter notebooks

### API Keys Required

- **Mistral AI**: Get from [console.mistral.ai](https://console.mistral.ai)
- **OpenRouter** (Day 3 only): Get from [openrouter.ai](https://openrouter.ai)
- **Groq** (Day 8): Get from [console.groq.com](https://console.groq.com)
- **Gemini** (Day 8, optional): Get from [ai.google.dev](https://ai.google.dev)

### Environment Setup

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Package Dependencies

Install required packages:

```bash
pip install python-dotenv requests beautifulsoup4 mistralai openai gradio torch transformers datasets huggingface-hub bitsandbytes accelerate
```

**Note:** Days 5-7 require running in Google Colab with GPU access.

---

## Getting Started

1. **Clone or download this repository**
2. **Set up your environment variables** (create `.env` file)
3. **Start with Day 1** and progress sequentially
4. **For Days 5-7:** Use the "Open in Colab" button in each notebook

Each notebook is self-contained and includes:

- Clear section headers explaining each step
- Standalone code cells that can be run independently
- Sample output from previous executions
- Hands-on exercises and experiments

---

## Project Structure

```
llm_engineering/
├── day1_intro.ipynb          # LLMs & Web Scraping
├── day2_parser.ipynb         # Link Extraction with JSON
├── day3_adv_conv.ipynb       # Multiple Providers
├── day4_gradio.ipynb         # Gradio UI Development
├── day5_hf_pipelines.ipynb   # Transformers & NLP Tasks
├── day6_tokenization.ipynb   # Tokenization & Chat Templates
├── day7_hf_models.ipynb      # Quantization & Local Deployment
├── day8_code_generator.ipynb # Code Generation & Porting
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

---

## Key Concepts Covered

### LLM Fundamentals

- Understanding different LLM providers and their APIs
- Prompt engineering and system prompts
- Streaming vs. batch responses
- JSON response formatting

### Data Processing

- Web scraping and content extraction
- Intelligent filtering and extraction
- Handling multiple data sources

### Model Usage

- Running inference with different models
- Comparing model outputs
- Optimizing model performance
- Understanding tokenization

### Practical Applications

- Building web interfaces
- Creating multi-step workflows
- Deploying efficient inference systems

---

## Tips for Success

1. **Execute cells sequentially** - Each notebook builds on previous setup steps
2. **Experiment freely** - Modify prompts and inputs to see different behaviors
3. **Monitor API usage** - Days 1-4 make actual API calls (you may incur costs)
4. **Use GPU in Colab** - Enable GPU runtime for Days 5-7 for better performance
5. **Check outputs** - Each notebook includes example outputs to compare against

---

## Troubleshooting

**API Key Issues:**

- Verify `.env` file is in the root directory
- Check that API keys are valid and not expired
- Ensure environment variables are loaded with `load_dotenv()`

**Colab Issues (Days 5-7):**

- Enable GPU: Runtime → Change runtime type → GPU
- Authenticate with HF token when prompted
- Ensure sufficient storage for model downloads

**Out of Memory:**

- Reduce batch sizes in sentiment analysis
- Run memory cleanup cells explicitly
- Use smaller quantized models

---

## Resources

- [Mistral AI Documentation](https://docs.mistral.ai)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [bitsandbytes Quantization](https://github.com/TimDettmers/bitsandbytes)

---

## Learning Outcomes

By completing this course, you'll be able to:

✅ Integrate with multiple LLM APIs  
✅ Build LLM-powered applications with web UIs  
✅ Process and extract data from web sources  
✅ Work with pre-trained transformer models  
✅ Understand and optimize tokenization  
✅ Deploy models efficiently with quantization  
✅ Compare and combine different LLM providers  
✅ Implement streaming and real-time features

---

## Contributing

Feel free to fork this repository and add improvements, additional examples, or corrections.

---

## License

This project is open source and available under the MIT License.

---

**Happy Learning! 🚀**

For questions or issues, please open an issue in the repository.
