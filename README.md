# Email Spam Detector
CS 410 - Text Information Systems Final Project, Spring 2026

## Project Description
This project develops an automated email spam detection system that uses machine learning and natural language processing to classify incoming emails as spam or ham (legitimate). The system implements a multi-stage pipeline that includes text preprocessing, TF-IDF feature extraction, and multiple supervised learning classifiers to accurately filter spam emails.

## Team Members
- Meera Aleem
- Elijah John
- Rabia Khan
- Hannah McGowan
- Isra Mohamed

## Project Structure
```
sp26-cs410-group15-email-spam-detector/
├── data/             # Raw and cleaned dataset files
├── notebooks/        # Jupyter notebooks for EDA and modeling
├── reports/          # Project proposal and milestone report PDFs
├── results/          # Confusion matrices, metrics, and visualizations
├── src/              # Python source code and helper scripts
└── README.md         # Project documentation
```

## Technologies Used
- **Language:** Python
- **Libraries:** scikit-learn, NLTK, pandas, NumPy, matplotlib
- **Feature Extraction:** TF-IDF Vectorization
- **Models:** Naive Bayes, Logistic Regression, SVM
- **Tools:** Jupyter Notebook, GitHub

## Dataset
This project uses the following publicly available datasets:

- **SpamAssassin Public Corpus:** A collection of thousands of labeled spam and ham emails, widely used for spam filtering research.
- **Enron Email Dataset:** A large dataset containing hundreds of thousands of real emails, used to test and validate the spam detection system.

## How to Run
> **Note:** Setup and installation instructions will be updated as the project is finalized.

### Prerequisites
- Python 3.x
- Jupyter Notebook

### Installation
1. Clone the repository:
```
    git clone https://github.com/hannahm99/sp26-cs410-group15-email-spam-detector.git
```
2. Install the required dependencies:
```
    pip install -r requirements.txt
```

### Running the Project
3. Navigate to the `notebooks/` directory and open the Jupyter notebooks to run the pipeline.
