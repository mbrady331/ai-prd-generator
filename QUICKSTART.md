# 🚀 Quick Start Guide

Get your AI-Powered PRD Generator up and running in minutes!

## 📋 Prerequisites

- Python 3.12+
- OpenAI API Key (or OpenRouter API Key)

## ⚡ Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-prd-generator.git
   cd ai-prd-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp env-template.txt .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Start the server**
   ```bash
   cd backend
   python main.py
   ```

5. **Open your browser**
   ```
   http://localhost:8000
   ```

## 🎯 First PRD

1. Fill out the form with your product details
2. Click "Generate PRD"
3. Watch the AI agents work their magic!
4. Get a professional PRD in seconds

## 🔧 Troubleshooting

- **Port 8000 in use?** The server will show an error - just kill any existing Python processes
- **API key issues?** Make sure your `.env` file is in the `backend/` directory
- **Import errors?** Run `pip install -r requirements.txt` again

## 📚 What's Next?

- Add Tavily API key for enhanced web research
- Customize the AI agents for your specific needs
- Deploy to production with your preferred hosting platform

Happy PRD generating! 🎉
