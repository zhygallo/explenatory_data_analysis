# from __future__ import absolute_import, unicode_literals, division, print_function


from matplotlib import pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

st.set_option("deprecation.showPyplotGlobalUse", False)


def build_analyser(df):
    st.sidebar.subheader("3. Data Profiling")
    st.sidebar.write("It is optional, so feel free to skip it")
    show_profiling = st.sidebar.radio(label="Profile?", options=["Yes", "No"], index=1)
    if show_profiling == "Yes":
        pr = ProfileReport(df, explorative=True)
        profiling_load_state = st.text("Profiling data...")
        st.header("**Pandas Profiling Report**")
        st_profile_report(pr)
        profiling_load_state.text("Profiling is ready!")

    st.sidebar.subheader("4. Plot Histogram")
    x_axis_var = st.sidebar.selectbox("Variable", df.columns,)
    # y_axis_var = st.sidebar.selectbox("Y Axis Variable", list(df.columns))

    hist_values = np.histogram(df[x_axis_var])[0]

    st.bar_chart(hist_values)
    # fig.show()


def build_classifier(df):
    st.sidebar.subheader("2. Choose target variable")
    st.sidebar.write("This is the variable we try to predict")
    target = st.sidebar.selectbox("Target Variable", list(df.columns),)

    st.sidebar.subheader("3. Choose predictive feature variables")
    st.sidebar.write(
        "The list of these variables will be used to predict your target variable"
    )
    features = st.sidebar.multiselect(
        "Predictive Features",
        list(df.drop(target, axis=1).columns),
        default=list(df.drop(target, axis=1).columns),
    )

    @st.cache(persist=True)
    def filter_data(target, features):
        filtered_df = df[set(features + [target])]
        return filtered_df

    filtered_df = filter_data(target, features)

    st.header("**Filtered DataFrame**")
    st.subheader(
        "Prefiltered dataset that contains only predictive feature variables and the target variable:"
    )
    st.write(filtered_df.head(20))
    st.write("---")

    @st.cache(persist=True)
    def preprocess_data(df):
        labelencoder = LabelEncoder()
        process_df = df.copy()
        process_df.dropna(axis=0, how="any", inplace=True)
        for col in process_df.columns:
            if process_df[col].dtype == "object":
                process_df[col] = labelencoder.fit_transform(process_df[col])
        return process_df

    st.sidebar.subheader("4. Preprocess Dataset?")
    st.sidebar.write(
        "This step will apply some common preprocessing of the data, like encoding catecorical variables"
    )
    is_preprocess = st.sidebar.radio(label="Preprosess?", options=["Yes", "No"])

    if is_preprocess == "Yes":
        prep_df = preprocess_data(filtered_df)
        st.header("**Preprocessed DataFrame**")
        st.write(prep_df.head(20))
        st.write("---")
    else:
        prep_df = filtered_df

    @st.cache(persist=True)
    def split(df):
        y = df[target]
        x = df.drop(columns=[target])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = split(prep_df)

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    st.sidebar.header("**5. Choose Classifier**")
    st.sidebar.write(
        "Note. This is a classification and not regression problem, so your target variable has to be categorical"
    )
    classifiers = [
        "Support Vector Machine (SVM)",
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
    ]
    classifier = st.sidebar.selectbox("Classifier", classifiers)

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        # choose parameters
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_SVM"
        )
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma"
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        model = SVC(C=C, kernel=kernel, gamma=gamma)

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key="max_iter"
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        model = LogisticRegression(C=C, penalty="l2", max_iter=max_iter)

    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 5000, step=10, key="n_estimators",
        )
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key="max_depth"
        )
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", ("True", "False"), key="bootstrap",
        )
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            n_jobs=-1,
        )

    elif classifier == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, value=5, key="max_depth"
        )
        min_samples_split = st.sidebar.number_input(
            "The min_samples_split the tree",
            2,
            20,
            step=1,
            value=2,
            key="min_samples_split",
        )
        min_samples_leaf = st.sidebar.number_input(
            "The min_samples_leaf of the tree",
            1,
            20,
            step=1,
            value=1,
            key="min_samples_leaf",
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        )

        model = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    if st.sidebar.button("Classify", key="classify") and (classifier in classifiers):
        st.subheader("Results")

        train_model_state = st.text("Training model...")
        model.fit(x_train, y_train)
        train_model_state.text("Training model...done!")

        result_gen_state = st.text("Generating results...")
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        result_gen_state.text("Generating results...done!")
        st.write("Accuracy: ", accuracy.round(2))
        st.write(
            "Precision: ", precision_score(y_test, y_pred, average="micro").round(2),
        )
        st.write("Recall: ", recall_score(y_test, y_pred, average="micro").round(2))

        if classifier == "Decision Tree":
            st.write("Decision Tree Plot:")
            fig = plt.figure(figsize=(12, 12))
            tree.plot_tree(model, filled=True, fontsize=10)
            st.pyplot(fig)

        plot_metrics(metrics)


def main():
    st.title("Data Analysis for Dummies")
    st.write(
        "This Web Application will help you understand your data and even apply some classical classification "
        "algorithms with the minimal knowledge of Data Science \n\n"
        "Just follow the steps provided in the sidebar and enjoy\n\n"
    )
    st.sidebar.title("Pick the action")
    st.sidebar.header("1. What is the origin of your Data?")
    data_origin = st.sidebar.radio(
        label="Data Origin", options=["Upload Data", "Generate Random Data"]
    )

    df = None
    nrows = None
    print(data_origin)
    if data_origin == "Upload Data":
        st.sidebar.header("2. Upload Data")
        try:
            uploaded_file = st.sidebar.file_uploader(
                "Upload input file", type=["csv", "xlsx"]
            )
            print(uploaded_file)
        except Exception as e:
            print(e)

        if uploaded_file is not None:

            @st.cache(persist=True)
            def load_data(nrows=None):
                if uploaded_file.name.endswith(".xlsx"):
                    data = pd.read_excel(uploaded_file, nrows=nrows)
                elif uploaded_file.name.endswith(".csv"):
                    data = pd.read_csv(uploaded_file, nrows=nrows)

                return data

            data_load_state = st.text("Loading data...")
            df = load_data(nrows)
            data_load_state.text("Loading data...done!")

    elif data_origin == "Generate Random Data":

        @st.cache(persist=True)
        def generate_data(nrows=100):
            random_df = pd.DataFrame(
                np.random.rand(nrows, 5), columns=["a", "b", "c", "d", "e"]
            )
            return random_df

        data_load_state = st.text("Loading data...")
        df = generate_data()
        data_load_state.text("Geneating data...done!")

    if df is not None:
        st.header("**Input DataFrame**")
        st.subheader("This is how your original Data looks like:")
        st.write(df.head(20))
        st.write("---")

        type_of_analysis = st.sidebar.radio(
            label="Build Classifier or Analyse Data?",
            options=["Analyse Data", "Build Classifier"],
        )

        if type_of_analysis == "Analyse Data":
            build_analyser(df)
        elif type_of_analysis == "Build Classifier":
            build_classifier(df)


if __name__ == "__main__":
    main()
