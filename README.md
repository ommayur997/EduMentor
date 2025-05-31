### ✅ **EduMentor**

**Overview**
EduMentor is an intelligent academic performance dashboard that helps educators identify at-risk students and provides actionable insights to improve learning outcomes. The application uses machine learning models to predict student risk levels and offers detailed academic analytics across multiple performance categories.

**Key Features**
### 🚀 **Predictive Analytics**
* **Risk Score Prediction**: Regression model predicts student risk scores (0-100)
* **Risk Status Classification**: Binary classifier identifies at-risk students
* **Visual Indicators**: Color-coded metrics and emoji-based status alerts

### 📊 **Performance Analysis**
* **Academic Performance**: Subject grades and overall performance evaluation
* **Attendance & Participation**: Lecture attendance and login frequency metrics
* **Learning Behavior**: Session duration, engagement scores, and lesson completion
* **Learning Preferences**: Visual, auditory, and kinesthetic learning styles
* **Qualitative Feedback**: Teacher comments and observations

### 🛠 **Actionable Insights**
* Automated warnings for concerning metrics
* Performance improvement suggestions
* Risk-based intervention recommendations
* Technical Implementation


### 🔧 **Technologies Used**
* Python (Primary programming language)
* Streamlit (Web application framework)
* Pandas & NumPy (Data manipulation)
* XGBoost (Machine learning models)
* Joblib (Model serialization)

### 🧠 **Machine Learning Models**
* **Regression Model:**
Predicts risk scores (0-100)
XGBRegressor implementation

* **Classification Model:**
Identifies at-risk students (binary classification)
XGBClassifier implementation

## 📁 **Project Structure**
```
EduMentor/
├── app.py                 # Main Streamlit application
├── regression_model.pkl   # Trained risk score prediction model
├── classification_model.pkl  # Trained risk classification model
├── edu_mentor_dataset_final(5000).csv  # Sample dataset
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Usage
* Enter student ID and name
* View predicted risk score and status
* Explore academic performance through category tabs
* Review warnings and improvement suggestions

## Future Enhancements
* Implement historical performance tracking
* Add comparative analysis with peer performance
* Develop personalized intervention plans
* Create administrator dashboard for institution-level insights
* Add data export functionality for reports
