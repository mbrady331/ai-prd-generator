# 🚀 Deployment Guide

Deploy your AI-Powered PRD Generator to multiple platforms with zero configuration!

## 🆓 Free Hosting Options (No Payment Required)

## 🎯 Free Platform Options

### Option 1: Railway.app (No Payment Required) ⭐ RECOMMENDED

1. **Go to [railway.app](https://railway.app)** and sign in with GitHub
2. **Click "New Project"** → **"Deploy from GitHub repo"**
3. **Select your repository**: `mbrady331/ai-prd-generator`
4. **Railway will auto-detect** the `railway.toml` configuration
5. **Add environment variables** in the Variables tab:
   - `OPENAI_API_KEY`: Your OpenAI API key
6. **Deploy automatically** - Railway handles everything else

**Benefits**: 
- ✅ No payment method required
- ✅ $5/month free credits (more than enough)
- ✅ Auto-deploy on git push
- ✅ Custom domains available

### Option 2: Render.com (Payment Method Required)

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

### Option 3: Fly.io (No Payment Required)

1. **Install Fly CLI**: Follow instructions at [fly.io/docs](https://fly.io/docs/)
2. **Create fly.toml** (already included in repo)
3. **Deploy**:
   ```bash
   fly auth login
   fly launch
   fly secrets set OPENAI_API_KEY=your_key_here
   fly deploy
   ```

### Option 4: Heroku (No Payment Required for Hobby Tier)

1. **Install Heroku CLI**
2. **Deploy**:
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your_key_here
   git push heroku main
   ```

### Option 5: Render.com (Payment Method Required)

**Note**: Render requires a payment method even for free tier usage.

1. **Go to [render.com](https://render.com)** and sign in
2. **Click "New +"** → **"Web Service"**
3. **Connect your GitHub repository**
4. **Render will auto-detect** the `render.yaml` configuration
5. **Add your environment variables** (see below)
6. **Add payment method** (required even for free tier)
7. **Click "Create Web Service"**

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
