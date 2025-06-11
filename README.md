# 🤖 Bluesky AI Bot with Google Gemini Integration

A sophisticated AI-powered bot for the Bluesky social network that uses Google's Gemini, Imagen, and Veo models to provide intelligent responses, image generation, and video creation.

## ✨ Features

- **Real-time Processing**: Monitors Bluesky posts via Jetstream WebSocket API
- **AI-Powered Responses**: Uses Google Gemini for intelligent conversation
- **Media Generation**: Creates images with Imagen 3 and videos with Veo 2
- **Smart Thread Management**: Prevents spam and manages conversation depth
- **Memory Safety**: Implements memory monitoring and cleanup
- **Rate Limiting**: Built-in API rate limiting for reliability
- **Thread Safety**: Concurrent processing with proper locking
- **Google Search Integration**: Grounded responses with search capabilities

## 🔧 Setup

### Prerequisites

- Python 3.8 or higher
- A Bluesky account with an app password
- Google Gemini API access
- Basic knowledge of environment variables

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd gemini-botsky
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and fill in your credentials:
   ```bash
   BLUESKY_HANDLE=your-handle.bsky.social
   BLUESKY_PASSWORD=your-app-password
   GEMINI_API_KEY=your-gemini-api-key
   ```

4. **Run the bot**:
   ```bash
   python bot.py
   ```

## 🔐 Security & Safety

### Gemini Safety Settings
The bot uses `BLOCK_NONE` safety settings by default. You can adjust these in the code if needed.

### Rate Limiting
- Gemini API: Minimum 1 second between calls
- Bluesky API: Minimum 0.5 seconds between calls

### Memory Management
- Automatic memory monitoring during media processing
- Cleanup of downloaded media after processing
- Cache size limits to prevent memory leaks

## 📊 Configuration Options

All configuration is done via environment variables. See `.env.example` for all available options.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONVERSATION_THREAD_DEPTH` | 50 | Max posts before bot stops replying |
| `MAX_REPLY_THREAD_DEPTH` | 10 | Max posts bot will create in one thread |
| `MENTION_CHECK_INTERVAL_SECONDS` | 15 | How often to check for new mentions |
| `MAX_THREAD_DEPTH_FOR_CONTEXT` | 25 | How much thread history to include |

## 🚀 Deployment

### Cloud Platforms

The bot is designed to work well on various cloud platforms:

- **Heroku**: Add environment variables in dashboard
- **Railway**: Connect GitHub repo and set environment variables  
- **DigitalOcean App Platform**: Use the included requirements.txt
- **AWS/GCP/Azure**: Deploy as container or serverless function

### Environment Setup for Production

1. Set `LOG_LEVEL=WARNING` to reduce log verbosity
2. Consider using external services for persistence
3. Monitor memory usage and adjust limits as needed

## 🔍 Monitoring & Debugging

### Developer Notifications
The bot automatically sends direct messages to @symm.social for:
- **Critical Errors**: Startup failures, API initialization problems
- **Connection Issues**: Persistent Jetstream connection problems  
- **Startup Notifications**: Successful bot initialization with status

Messages are sent via DM when possible, with fallback to public mentions if DMs fail.

### Logs
The bot provides comprehensive logging:
- `INFO`: General operation and successful actions
- `WARNING`: Non-critical issues and fallbacks
- `ERROR`: Failed operations and exceptions
- `DEBUG`: Detailed execution information

### Memory Monitoring
Built-in memory monitoring tracks usage during media processing and automatically cleans up resources.

## 🛠 Customization

### Bot Personality
Edit the `BOT_SYSTEM_INSTRUCTION` in `bot.py` to customize the bot's personality and behavior.

### Media Generation
- Images: Triggered by "IMAGE_PROMPT:" in responses
- Videos: Triggered by "VIDEO_PROMPT:" in responses
- Automatic alt-text generation for accessibility

### Thread Management
The bot automatically:
- Prevents duplicate replies
- Manages conversation depth
- Handles user-to-user vs bot conversations appropriately

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This bot uses AI models that may generate unexpected content. Always monitor its behavior and adjust safety settings as needed for your use case. The bot is provided as-is without warranties.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the existing GitHub issues
2. Create a new issue with detailed information
3. Include relevant log outputs and configuration (without credentials)

---

**Built with ❤️ by symmetricalboy (@symm.social)** 