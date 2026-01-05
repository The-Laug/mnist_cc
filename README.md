# mnist_cc

A simple MNIST digit classification project built with PyTorch, following MLOps best practices using cookiecutter template.

## Description

This project trains a convolutional neural network to classify handwritten digits from the Corrupted MNIST dataset. It includes data preprocessing, model training, evaluation, and visualization components, all structured following MLOps principles.

## Installation

This project uses `uv` for dependency management. To set up the environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd mnist_cc

# Install dependencies using uv
uv sync

# Or if you prefer pip
pip install -e .
```

## Data

Place your Corrupted MNIST dataset in `data/raw/` with the following structure:
- `train_images_0.pt` through `train_images_5.pt`
- `train_target_0.pt` through `train_target_5.pt`
- `test_images.pt`
- `test_target.pt`

## Usage

### Using Invoke Tasks

The project includes convenient invoke tasks defined in `tasks.py`:

```bash
# Preprocess the data
uvx invoke preprocess-data

# Train the model
uvx invoke train

# Run tests
uvx invoke test

# Build Docker images
uvx invoke docker-build

# Build documentation
uvx invoke build-docs

# Serve documentation locally
uvx invoke serve-docs
```

### Direct Script Execution

You can also run scripts directly:

```bash
# Preprocess data
uv run src/mnist_cc/data.py data/raw data/processed

# Train model
uv run src/mnist_cc/train.py --lr 0.001 --batch-size 32 --epochs 10

# Evaluate model
uv run src/mnist_cc/evaluate.py models/model.pth

# Visualize embeddings
uv run src/mnist_cc/visualize.py models/model.pth
```

## Model Architecture

The model (`MyAwesomeModel`) is a CNN with:
- 2 convolutional layers (32 and 64 filters)
- MaxPooling layers
- 2 fully connected layers (128 and 10 neurons)
- ReLU activation and Dropout (0.25)

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── mnist_cc/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
