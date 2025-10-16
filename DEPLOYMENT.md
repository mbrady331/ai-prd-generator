# 🚀 Deployment Guide

Deploy your AI-Powered PRD Generator to Render.com with zero configuration!

## 🎯 One-Click Deployment

### Option 1: Deploy from GitHub (Recommended)

1. **Push to GitHub** (if you haven't already)
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ai-prd-generator.git
   git branch -M main
   git push -u origin main
   ```

2. **Go to Render.com** and sign in
3. **Click "New +"** → **"Web Service"**
4. **Connect your GitHub repository**
5. **Render will auto-detect** the `render.yaml` configuration
6. **Add your environment variables** (see below)
7. **Click "Create Web Service"**

### Option 2: Manual Configuration

If you prefer manual setup:

- **Name:** `ai-prd-generator`
- **Environment:** `Python`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `cd backend && python main.py`
- **Plan:** `Starter` (free tier available)

## 🔑 Environment Variables

Add these in your Render dashboard:

### Required
- `OPENAI_API_KEY` - Your OpenAI API key
- `PORT` - Set to `8000` (Render will override this)

### Optional (for enhanced features)
- `TAVILY_API_KEY` - For web search research
- `SERPAPI_API_KEY` - Alternative search provider
- `OPENROUTER_API_KEY` - Alternative LLM provider
- `ARIZE_SPACE_ID` - For observability (optional)
- `ARIZE_API_KEY` - For observability (optional)

## 🏥 Health Checks

The application includes a health endpoint at `/health` that Render uses to verify the service is running.

## 🌐 Custom Domain (Optional)

1. **In your Render dashboard** → **Settings** → **Custom Domains**
2. **Add your domain** (e.g., `yourdomain.com`)
3. **Configure DNS** as instructed by Render

## 📊 Monitoring

- **Render Dashboard** - View logs, metrics, and deployment status
- **Health Endpoint** - Monitor at `https://your-app.onrender.com/health`
- **Application Logs** - Available in Render dashboard

## 🔄 Auto-Deploy

The `render.yaml` is configured for:
- ✅ **Auto-deploy** on main branch pushes
- ✅ **Pull Request previews** for testing changes
- ✅ **Health checks** for reliable deployment

## 🚨 Troubleshooting

### Common Issues:

1. **Build Failures**
   - Check `requirements.txt` syntax
   - Verify Python version compatibility

2. **Runtime Errors**
   - Check environment variables are set
   - Review application logs in Render dashboard

3. **API Key Issues**
   - Ensure `OPENAI_API_KEY` is set correctly
   - Verify API key has sufficient credits

4. **Port Issues**
   - Render automatically sets the `PORT` environment variable
   - Your app should use `os.environ.get("PORT", 8000)`

## 💰 Cost Optimization

- **Free Tier:** 750 hours/month (perfect for development)
- **Paid Plans:** Start at $7/month for always-on service
- **Sleep Mode:** Free tier apps sleep after 15 minutes of inactivity

## 🔐 Security Best Practices

- ✅ Never commit API keys to git
- ✅ Use environment variables for all secrets
- ✅ Enable HTTPS (automatic on Render)
- ✅ Monitor usage and set spending limits

Your PRD Generator will be live at: `https://your-app-name.onrender.com`

Happy deploying! 🎉
