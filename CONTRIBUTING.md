# Contributing to Customer Churn Prediction System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**: `python -m venv venv`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Copy environment template**: `cp .env.example .env`
5. **Train the model**: `python src/main.py`

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines

### Commit Messages
Use conventional commit format:
```
feat: add new feature
fix: correct bug in prediction
docs: update README
test: add unit tests
refactor: improve code structure
```

### Testing
Before submitting, ensure all tests pass:
```bash
cd backend
python manage.py test predictor -v 2
```

## 🔀 Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Ensure tests pass and add new tests if needed
4. Update documentation if required
5. Submit a pull request with a clear description

## 📋 Areas for Contribution

- [ ] Add more ML models (XGBoost, LightGBM)
- [ ] Implement model retraining endpoint
- [ ] Add Docker support
- [ ] Create GitHub Actions CI/CD
- [ ] Add more comprehensive tests
- [ ] Improve UI/UX design

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
