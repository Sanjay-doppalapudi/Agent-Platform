# 🤖 Agent Platform

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]() [![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.11%2B-blue)]() [![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

**Open-source autonomous AI agent platform with secure code execution and multi-tool orchestration.**

Build intelligent agents that can use tools, execute code safely, and accomplish complex tasks autonomously—all while maintaining enterprise-grade security.

## ✨ Features

- **🧠 Autonomous Reasoning**: Multi-step problem solving with LLM-powered decision making
- **🛠️ Extensible Tool System**: File operations, code execution, web search, git, and custom tools
- **🔒 Secure Sandboxing**: Docker-based isolation with seccomp, resource limits, and gVisor support
- **🚀 Multi-Model Support**: OpenAI, Anthropic, Google, Ollama via LiteLLM
- **📊 Cost Tracking**: Detailed token usage and cost analytics
- **🌊 Streaming Responses**: Real-time output for better UX
- **🔌 RESTful API**: FastAPI-powered endpoints with OpenAPI docs
- **🧪 Production Ready**: Comprehensive testing, logging, and monitoring

## 🚀 Quick Start

### Installation

```bash
# Using pip
pip install agent-platform

# Using poetry
poetry add agent-platform

# From source
git clone https://github.com/yourusername/agent-platform.git
cd agent-platform
poetry install
```

### Basic Usage

```python
from agent_platform.agent_core.factory import create_coding_agent

# Create an agent with file tools
agent = create_coding_agent(workspace_path="./workspace")

# Run a task
result = agent.run("Create a Python script that prints 'Hello, World!'")
print(result)
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 📚 Documentation

- **[User Guide](docs/user-guide.md)** - Comprehensive usage documentation
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Architecture](docs/architecture.md)** - System design and internals
- **[Examples](examples/)** - Code examples and tutorials

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Platform                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   Agent  │────│   LLM    │────│  Tools   │          │
│  │   Core   │    │ Adapter  │    │ Registry │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│        │              │                 │                │
│        └──────────────┴─────────────────┘                │
│                       │                                  │
│        ┌──────────────┴──────────────┐                  │
│        │                              │                  │
│  ┌──────────┐                  ┌──────────┐            │
│  │ Sandbox  │                  │   API    │            │
│  │ Manager  │                  │  Layer   │            │
│  └──────────┘                  └──────────┘            │
│        │                              │                  │
│  ┌──────────┐                  ┌──────────┐            │
│  │  Docker  │                  │ FastAPI  │            │
│  └──────────┘                  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

- **Code Generation**: Generate, test, and execute code automatically
- **File Manipulation**: Read, write, and analyze files programmatically
- **Research Automation**: Web search and information synthesis
- **DevOps Tasks**: Git operations, deployment automation
- **Data Processing**: Multi-step data analysis workflows

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...

# Sandbox Configuration
SANDBOX_IMAGE=agent-sandbox:latest
SANDBOX_MEMORY_LIMIT=512m
SANDBOX_CPU_LIMIT=1.0
SANDBOX_TIMEOUT=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

### Docker Setup

```bash
# Build sandbox image
docker build -t agent-sandbox:latest -f docker/Dockerfile.sandbox .

# Run with security settings
docker run -d \
  --read-only \
  --memory=512m \
  --cpus=1.0 \
  --network=none \
  --security-opt=seccomp=docker/seccomp-profile.json \
  agent-sandbox:latest
```

## 🧪 Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agent-platform.git
cd agent-platform

# Install dependencies
poetry install

# Set up pre-commit hooks
poetry run pre-commit install
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires API keys)
export OPENAI_API_KEY=sk-...
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

## 🗺️ Roadmap

- [x] Core agent orchestrator
- [x] LLM multi-provider support
- [x] Secure sandbox execution
- [x] Tool system with registry
- [x] RESTful API
- [ ] WebSocket streaming
- [ ] React frontend demo
- [ ] Plugin marketplace
- [ ] Multi-agent collaboration
- [ ] Advanced memory systems

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help

- Additional tool implementations
- LLM provider integrations
- Documentation improvements
- Example applications
- Testing and bug reports

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM support
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework
- [Docker](https://www.docker.com/) for containerization
- The open-source AI community

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/agent-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agent-platform/discussions)
- **Email**: support@agent-platform.dev

---

**Built with ❤️ by the Agent Platform team**
