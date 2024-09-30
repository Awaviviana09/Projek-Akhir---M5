import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Mushroom Classification App", page_icon="🍄", layout="centered")

# Apply custom CSS for better UI design
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stApp {
        background: linear-gradient(to right, #bdc3c7, #2c3e50);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    h1 {
        color: #ffffff;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    h2, h3, p {
        color: #ffffff;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: #fff;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        padding: 12px 25px;
        border-radius: 15px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 8px #1e8449;
        cursor: pointer;
        transition: all 0.2s;
        margin-top: 15px;
    }
    .stButton>button:hover {
        background-color: #2ecc71;
        transform: scale(1.02);
    }
    .result-box {
        background-color: #ecf0f1;
        color: #2c3e50;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 30px;
    }
    .result-title {
        font-weight: bold;
        font-size: 28px;
        color: #27ae60;
        margin-bottom: 20px;
    }
    .result-detail {
        font-size: 22px;
        margin-top: 10px;
    }
    .about-box {
        background-color: #2c3e50;
        color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
        text-align: justify; /* Align text to justify */
    }
    footer {
        margin-top: 50px;
        text-align: center;
        color: #ffffff;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title('🍄 Mushroom Classification App')

st.write("""
Welcome to the **Mushroom Classification App**! Use this app to classify mushrooms as **edible** or **poisonous** based on their features.
""")

# Sidebar with info and instructions
st.sidebar.title("📊 Mushroom Classifier")
st.sidebar.write("""
### Steps to classify mushrooms:
1. Upload your dataset.
2. Select the classification algorithm.
3. See the classification result instantly!
""")

# Styled "About the App" box
st.sidebar.markdown("""
<div class="about-box">
💡 About the App<br><br>
This app is designed for mushroom classification using machine learning models like Random Forest, Logistic Regression, and Decision Trees. 
Upload your dataset, select features, and run the classification model!
</div>
""", unsafe_allow_html=True)

# Creator Credit
st.sidebar.markdown("<footer>Created by: Zahwa</footer>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📂 Upload your mushroom dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.write(data.head())

    # Select features and target
    st.subheader("⚙️ Select Features and Target")
    features = st.multiselect("Select features", data.columns.tolist(), default=data.columns[:-1])
    target = st.selectbox("Select target", data.columns.tolist(), index=len(data.columns) - 1)

    # Encode categorical features and target
    label_encoder = LabelEncoder()
    for col in features + [target]:
        if data[col].dtype == 'object':
            data[col] = label_encoder.fit_transform(data[col])

    # Split the dataset
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Algorithm selection and classification
    st.subheader("🧠 Choose Algorithm for Classification")
    algorithm = st.radio(
        "Select the algorithm you want to use:",
        ("Random Forest", "Logistic Regression", "Decision Tree")
    )

    # Button for classification
    if st.button('🔍 Classify Data'):
        # Model selection
        if algorithm == "Random Forest":
            model = RandomForestClassifier(max_depth=2, random_state=0)
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = DecisionTreeClassifier()

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predicted_class = label_encoder.inverse_transform([y_pred[0]])[0]

        # Display classification result in an organized and visually appealing box
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("<p class='result-title'>🎯 Classification Result</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result-detail'>The selected algorithm **{algorithm}** has classified the mushroom as:</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result-detail'>🍄 **{predicted_class}**</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result-detail'>This mushroom is predicted to be **{'Edible' if predicted_class == 'e' else 'Poisonous'}**.</p>", unsafe_allow_html=True)
        st.success("✅ Classification was successful!")  # Success notification
        
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Please upload a CSV file to start!")


