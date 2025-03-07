# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,classification_report,roc_auc_score,matthews_corrcoef,mean_absolute_error,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error,explained_variance_score,r2_score,roc_curve,precision_recall_curve,auc
def create_heatmap(df):
     
     df_encoded = df.copy()
     label_encoder = LabelEncoder()
     for column in df_encoded.columns:
      df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
     corr_matrix=df_encoded.corr()
     heatmap=ff.create_annotated_heatmap(
          z=corr_matrix.values,
          x=list(corr_matrix.columns),
          y=list(corr_matrix.index),
          annotation_text=corr_matrix.round(2).values,
          colorscale="Viridis",
          showscale=True
     )
     heatmap.update_layout(
          title="correaltion heatmap",
          xaxis_title="Features",
          template="plotly_white"
     )
     return heatmap
     


class ClassificationMetrics:
    def __init__(self, y_test, predictions, probabilities):
        """
        Initialize the class with true labels, predicted labels, and predicted probabilities.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        :param probabilities: Predicted probabilities (for binary: 1D array, for multiclass: 2D array).
        """
        self.y_test = y_test
        self.predictions = predictions
        self.probabilities = probabilities
        self.num_classes = len(np.unique(y_test))  # Check number of classes

        # Compute metrics
        self.metrics = self.calculate_metrics()
        self.conf_matrix = confusion_matrix(y_test, predictions)
        self.report = classification_report(y_test, predictions, output_dict=True)

    def calculate_metrics(self):
        """Calculate classification metrics and return them as a dictionary."""
        acc = accuracy_score(self.y_test, self.predictions)

        # Set average method based on number of classes
        average_method = "binary" if self.num_classes == 2 else "weighted"

        precision = precision_score(self.y_test, self.predictions, average=average_method)
        recall = recall_score(self.y_test, self.predictions, average=average_method)
        f1 = f1_score(self.y_test, self.predictions, average=average_method)

        # ROC-AUC handling
        if self.num_classes == 2:
            roc_auc = roc_auc_score(self.y_test, self.probabilities)  # Binary case
        else:
            roc_auc = roc_auc_score(self.y_test, self.probabilities, multi_class="ovr", average="weighted")  # Multiclass case

        mcc = matthews_corrcoef(self.y_test, self.predictions)

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Matthews Correlation Coefficient": mcc
        }

    def display_metrics(self):
        """Display metrics as Streamlit components."""
        st.title("Classification Model Evaluation")
        st.metric("Accuracy", f"{self.metrics['Accuracy']:.2f}")

        st.subheader("Precision, Recall, and F1-Score")
        st.write(f"**Precision:** {self.metrics['Precision']:.2f}")
        st.write(f"**Recall:** {self.metrics['Recall']:.2f}")
        st.write(f"**F1-Score:** {self.metrics['F1-Score']:.2f}")

        st.subheader("ROC-AUC and Matthews Correlation Coefficient")
        st.write(f"**ROC-AUC:** {self.metrics['ROC-AUC']:.2f}")
        st.write(f"**Matthews Correlation Coefficient (MCC):** {self.metrics['Matthews Correlation Coefficient']:.2f}")
        st.markdown("---")

    def plot_confusion_matrix(self):
        """Plot the confusion matrix as a heatmap."""
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    def display_classification_report(self):
        """Display the classification report as a DataFrame."""
        st.subheader("Classification Report")
        report_df = pd.DataFrame(self.report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

    def plot_roc_curve(self):
        """Plot the Receiver Operating Characteristic (ROC) curve."""
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()

        if self.num_classes == 2:
            # Binary classification case
            fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
            ax.plot(fpr, tpr, label=f'AUC = {self.metrics["ROC-AUC"]:.2f}')
        else:
            # Multiclass classification case
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(self.y_test == i, self.probabilities[:, i])
                ax.plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc_score(self.y_test == i, self.probabilities[:, i]):.2f}')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

    def plot_precision_recall_curve(self):
        """Plot the Precision-Recall curve."""
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()

        if self.num_classes == 2:
            # Binary classification case
            precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, self.probabilities)
            ax.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
        else:
            # Multiclass classification case
            for i in range(self.num_classes):
                precision_vals, recall_vals, _ = precision_recall_curve(self.y_test == i, self.probabilities[:, i])
                ax.plot(recall_vals, precision_vals, label=f'Class {i}')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')
        st.pyplot(fig)

    def run_all_visualizations(self):
        """Run all visualizations."""
        col1,col2=st.columns(2)
        with col1:

            self.plot_confusion_matrix()
        with col2:
            self.display_classification_report()
        col3,col4=st.columns(2)
        with col3:
            self.plot_roc_curve()
        with col4:
            self.plot_precision_recall_curve()


class RegressionMetrics:
    def __init__(self, y_test, predictions, X_test=None):
        """
        Initialize the class with true values, predicted values, and optionally the feature set.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        :param X_test: Feature set used for testing, required for adjusted R².
        """
        self.y_test = y_test
        self.predictions = predictions
        self.X_test = X_test
        self.metrics = self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate regression metrics and return them as a dictionary."""
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        mape = mean_absolute_percentage_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        evs = explained_variance_score(self.y_test, self.predictions)
        
        n = len(self.y_test)
        k = self.X_test.shape[1] if self.X_test is not None else 0
        adjusted_r2_score = 1 - (1 - r2) * (n - 1) / (n - k - 1) if k > 0 else None

        metrics = {
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Percentage Error (MAPE)": mape,
            "R² Score": r2,
            "Adjusted R² Score": adjusted_r2_score,
            "Explained Variance Score": evs
        }
        return metrics
    
    def display_metrics(self):
        """Display metrics as a Streamlit table."""
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value']).reset_index()
        metrics_df = metrics_df.rename(columns={'index': 'Metric'})
        st.header("📝 Regression Metrics")
        st.table(metrics_df.style.format({"Value": "{:.4f}"}))
        st.markdown("---")
    
    def plot_predicted_vs_actual(self):
        """Plot Predicted vs Actual values."""
        st.header("📊 Predicted vs. Actual Values")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.y_test, y=self.predictions, alpha=0.6, ax=ax)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')  # Ideal line
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs. Actual Values")
        st.pyplot(fig)
    
    def plot_residuals(self):
        """Plot residuals vs predicted values."""
        st.header("📉 Residuals Plot")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.predictions, y=residuals, alpha=0.6, ax=ax)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Predicted Values")
        st.pyplot(fig)
    
    def plot_residuals_distribution(self):
        """Plot the distribution of residuals."""
        st.header("📈 Residuals Distribution")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)
    
    def run_all_visualizations(self):
        """Run all visualizations."""
        col1,col2=st.columns([0.45,0.45])
        with col1:
            self.plot_predicted_vs_actual()
        with col2:
            self.plot_residuals()
        self.plot_residuals_distribution()
# Config

def plot_auc_roc_curve(model, X_train, y_train, X_test, y_test):
    """
    Plots the AUC-ROC curve for the given model using train and test data.
    
    Parameters:
        model : Trained classifier with a `predict_proba` method.
        X_train : Training features.
        y_train : Training target labels.
        X_test : Testing features.
        y_test : Testing target labels.
    """
    # Get predicted probabilities for the positive class (class 1)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Compute ROC Curve and AUC Score
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    train_auc = auc(fpr_train, tpr_train)
    test_auc = auc(fpr_test, tpr_test)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_train, tpr_train, color="blue", lw=2, label=f"Train AUC = {train_auc:.2f}")
    ax.plot(fpr_test, tpr_test, color="red", lw=2, label=f"Test AUC = {test_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal line for random guess
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC-ROC Curve")
    ax.legend()
    ax.grid(True)

    # Display plot in Streamlit
    st.pyplot(fig)

# Initial State
def initial_state():
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    if 'X_train' not in st.session_state:
        st.session_state['X_train'] = None

    if 'X_test' not in st.session_state:
        st.session_state['X_test'] = None

    if 'y_train' not in st.session_state:
        st.session_state['y_train'] = None

    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = None

    if 'X_val' not in st.session_state:
        st.session_state['X_val'] = None

    if 'y_val' not in st.session_state:
        st.session_state['y_val'] = None

    if "model" not in st.session_state:
        st.session_state['model'] = None

    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = False

    if "trained_model_bool" not in st.session_state:
        st.session_state['trained_model_bool'] = False

    if "problem_type" not in st.session_state:
        st.session_state['problem_type'] = None

    if "metrics_df" not in st.session_state:
        st.session_state['metrics_df'] = pd.DataFrame()

    if "is_train" not in st.session_state:
        st.session_state['is_train'] = False

    if "is_test" not in st.session_state:
        st.session_state['is_test'] = False

    if "is_val" not in st.session_state:
        st.session_state['is_val'] = False

    if "show_eval" not in st.session_state:
        st.session_state['show_eval'] = False

    if "all_the_process" not in st.session_state:
        st.session_state['all_the_process'] = """"""

    if "all_the_process_predictions" not in st.session_state:
        st.session_state['all_the_process_predictions'] = False

    if 'y_pred_train' not in st.session_state:
        st.session_state['y_pred_train'] = None

    if 'y_pred_test' not in st.session_state:
        st.session_state['y_pred_test'] = None

    if 'y_pred_val' not in st.session_state:
        st.session_state['y_pred_val'] = None

    if 'uploading_way' not in st.session_state:
        st.session_state['uploading_way'] = None

    if "lst_models" not in st.session_state:
        st.session_state["lst_models"] = []

    if "lst_models_predctions" not in st.session_state:
        st.session_state["lst_models_predctions"] = []

    if "models_with_eval" not in st.session_state:
        st.session_state["models_with_eval"] = dict()

    if "reset_1" not in st.session_state:
        st.session_state["reset_1"] = False

initial_state()

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

# Load Data
st.cache_data()
def load_data(upd_file):
    df = pd.read_csv(upd_file)
    return df

# Progress Bar
def progress_bar():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0002)
        my_bar.progress(percent_complete + 1)


# # Logo 
# col1, col2, col3 = st.columns([0.25,1,0.25])
# col2.image("logo_path.jpg")
# new_line(2)



def Machine_Learning():
    # Dataframe selection
    st.markdown("<h2 align='center'> <b> Upload the Dataset", unsafe_allow_html=True)

    new_line(1)



    # Uploading Way
    uploading_way = st.session_state.uploading_way
    col1, col3 = st.columns(2,gap='large')

    # Upload
    def upload_click(): st.session_state.uploading_way = "upload"
    col1.markdown("<h5 align='center'> Upload File", unsafe_allow_html=True)
    col1.button("Upload File", key="upload_file", on_click=upload_click,use_container_width=True)

    # # Select    
    # def select_click(): st.session_state.uploading_way = "select"
    # col2.markdown("<h5 align='center'> Select from Ours", unsafe_allow_html=True)
    # col2.button("Select from Ours", key="select_from_ours", use_container_width=True, on_click=select_click)
            
    # URL
    def url_click(): st.session_state.uploading_way = "url"
    col3.markdown("<h5 align='center'> Write URL", unsafe_allow_html=True)
    col3.button("Write URL", key="write_url", use_container_width=True, on_click=url_click)

    # No Data
    if st.session_state.df is None:

        # Upload
        if uploading_way == "upload":
            uploaded_file = st.file_uploader("Upload the Dataset", type="csv")
            if uploaded_file:
                df = load_data(uploaded_file)
                st.session_state.df = df

        elif uploading_way == "url":
            url = st.text_input("Enter URL")
            if url:
                df = load_data(url)
                st.session_state.df = df


        
        
    # Dataframe
    if st.session_state.df is not None:


        df = st.session_state.df
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        X_val = st.session_state.X_val
        y_val = st.session_state.y_val
        trained_model = st.session_state.trained_model
        is_train = st.session_state.is_train
        is_test = st.session_state.is_test
        is_val = st.session_state.is_val
        model = st.session_state.model
        show_eval = st.session_state.show_eval
        y_pred_train = st.session_state.y_pred_train
        y_pred_test = st.session_state.y_pred_test
        y_pred_val = st.session_state.y_pred_val
        metrics_df = st.session_state.metrics_df

        st.divider()
        new_line()


        # EDA
        st.markdown("Exploratory Data Analysis", unsafe_allow_html=True)
        new_line()
        with st.expander("Show EDA"):
            new_line()

            # Head
            head = st.checkbox("Data", value=False)    
            new_line()
            if head:
                st.dataframe(st.session_state.df)

            

            # Shape
            shape = st.checkbox("Show Shape", value=False)
            new_line()
            if shape:
                st.write(f"This DataFrame has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
                new_line()

            # Columns
            columns = st.checkbox("Show Columns", value=False)
            new_line()
            if columns:
                st.write(pd.DataFrame(df.columns, columns=['Columns']).T)
                new_line()

                
            # Describe Numerical
            describe = st.checkbox("Show Description **(Numerical Features)**", value=False)
            new_line()
            if describe:
                st.dataframe(df.describe(), use_container_width=True)
                new_line()

            # Describe Categorical
            describe_cat = st.checkbox("Show Description **(Categorical Features)**", value=False)
            new_line()
            if describe_cat:
                if df.select_dtypes(include=object).columns.tolist():
                    st.dataframe(df.describe(include=['object']), use_container_width=True)
                    new_line()
                else:
                    st.info("There is no Categorical Features.")
                    new_line()

            # Correlation Matrix using heatmap seabron
            # corr = st.checkbox("Show Correlation", value=False)
            # new_line()
            # if corr:

            #     if df.corr().columns.tolist():
            #         fig=create_heatmap(df)
            #         st.plotly_chart(fig)
            #     else:
            #         st.info("There is no Numerical Features.")
                

            # Missing Values


            missing = st.checkbox("Missing Values", value=False)
            new_line()
            if missing:

                col1, col2 = st.columns([0.4,1])
                with col1:
                    st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                    st.dataframe(df.isnull().sum().sort_values(ascending=False),height=350)

                with col2:
                    st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                    null_values = df.isnull().sum()
                    null_values = null_values[null_values > 0]
                    null_values = null_values.sort_values(ascending=False)
                    null_values = null_values.to_frame()
                    null_values.columns = ['Count']
                    null_values.index.names = ['Feature']
                    null_values['Feature'] = null_values.index
                    fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                    st.plotly_chart(fig, use_container_width=True)

                new_line()
                    

            # Delete Columns
            delete = st.checkbox("Delete Columns", value=False)
            new_line()
            if delete:
                col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
                new_line()
                
                col1, col2, col3 = st.columns([1,0.7,1])
                if col2.button("Delete", use_container_width=True):
                    st.session_state.all_the_process += f"""
    # Delete Columns
    df.drop(columns={col_to_delete}, inplace=True)
    \n """
                    progress_bar()
                    df.drop(columns=col_to_delete, inplace=True)
                    st.session_state.df = df
                    st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


            # Show DataFrame Button
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Show DataFrame", use_container_width=True):
                st.dataframe(df, use_container_width=True)
            

        # Missing Values
        new_line()
        st.markdown("Missing Values", unsafe_allow_html=True)
        new_line()
        with st.expander("Show Missing Values"):

            # Further Analysis
            new_line()
            missing = st.checkbox("Further Analysis", value=False, key='missing')
            new_line()
            if missing:

                col1, col2 = st.columns(2, gap='medium')
                with col1:
                    # Number of Null Values
                    st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                    st.dataframe(df.isnull().sum().sort_values(ascending=False), height=300, use_container_width=True)

                with col2:
                    # Percentage of Null Values
                    st.markdown("<h6 align='center'> Percentage of Null Values", unsafe_allow_html=True)
                    null_percentage = pd.DataFrame(round(df.isnull().sum()/df.shape[0]*100, 2))
                    null_percentage.columns = ['Percentage']
                    null_percentage['Percentage'] = null_percentage['Percentage'].map('{:.2f} %'.format)
                    null_percentage = null_percentage.sort_values(by='Percentage', ascending=False)
                    st.dataframe(null_percentage, height=300, use_container_width=True)

                # Heatmap
                col1, col2, col3 = st.columns([0.1,1,0.1])
                with col2:
                    new_line()
                    st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                    null_values = df.isnull().sum()
                    null_values = null_values[null_values > 0]
                    null_values = null_values.sort_values(ascending=False)
                    null_values = null_values.to_frame()
                    null_values.columns = ['Count']
                    null_values.index.names = ['Feature']
                    null_values['Feature'] = null_values.index
                    fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                    st.plotly_chart(fig, use_container_width=True)


            # INPUT
            col1, col2 = st.columns(2)
            with col1:
                missing_df_cols = df.columns[df.isnull().any()].tolist()
                if missing_df_cols:
                    add_opt = ["Numerical Features", "Categorical Feature"]
                else:
                    add_opt = []
                fill_feat = st.multiselect("Select Features",  missing_df_cols + add_opt ,  help="Select Features to fill missing values")

            with col2:
                strategy = st.selectbox("Select Missing Values Strategy", ["Select", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode (Most Frequent)"], help="Select Missing Values Strategy")


            if fill_feat and strategy != "Select":

                new_line()
                col1, col2, col3 = st.columns([1,0.5,1])
                if col2.button("Apply", use_container_width=True, key="missing_apply", help="Apply Missing Values Strategy"):

                    progress_bar()
                    
                    # All Numerical Features
                    if "Numerical Features" in fill_feat:
                        fill_feat.remove("Numerical Features")
                        fill_feat += df.select_dtypes(include=np.number).columns.tolist()

                    # All Categorical Features
                    if "Categorical Feature" in fill_feat:
                        fill_feat.remove("Categorical Feature")
                        fill_feat += df.select_dtypes(include=object).columns.tolist()

                    
                    # Drop Rows
                    if strategy == "Drop Rows":
                        st.session_state.all_the_process += f"""
    # Drop Rows
    df[{fill_feat}] = df[{fill_feat}].dropna(axis=0)
    \n """
                        df[fill_feat] = df[fill_feat].dropna(axis=0)
                        st.session_state['df'] = df
                        st.success(f"Missing values have been dropped from the DataFrame for the features **`{fill_feat}`**.")


                    # Drop Columns
                    elif strategy == "Drop Columns":
                        st.session_state.all_the_process += f"""
    # Drop Columns
    df[{fill_feat}] = df[{fill_feat}].dropna(axis=1)
    \n """
                        df[fill_feat] = df[fill_feat].dropna(axis=1)
                        st.session_state['df'] = df
                        st.success(f"The Columns **`{fill_feat}`** have been dropped from the DataFrame.")


                    # Fill with Mean
                    elif strategy == "Fill with Mean":
                        st.session_state.all_the_process += f"""
    # Fill with Mean
    from sklearn.impute import SimpleImputer
    num_imputer = SimpleImputer(strategy='mean')
    df[{fill_feat}] = num_imputer.fit_transform(df[{fill_feat}])
    \n """
                        from sklearn.impute import SimpleImputer
                        num_imputer = SimpleImputer(strategy='mean')
                        df[fill_feat] = num_imputer.fit_transform(df[fill_feat])

                        null_cat = df[missing_df_cols].select_dtypes(include=object).columns.tolist()
                        if null_cat:
                            st.session_state.all_the_process += f"""
    # Fill with Mode
    from sklearn.impute import SimpleImputer
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
    \n """
                            cat_imputer = SimpleImputer(strategy='most_frequent')
                            df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                        st.session_state['df'] = df
                        if df.select_dtypes(include=np.object).columns.tolist():
                            st.success(f"The Columns **`{fill_feat}`** has been filled with the mean. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                        else:
                            st.success(f"The Columns **`{fill_feat}`** has been filled with the mean.")
                        

                    # Fill with Median
                    elif strategy == "Fill with Median":
                        st.session_state.all_the_process += f"""
    # Fill with Median
    from sklearn.impute import SimpleImputer
    num_imputer = SimpleImputer(strategy='median')
    df[{fill_feat}] = pd.DataFrame(num_imputer.fit_transform(df[{fill_feat}]), columns=df[{fill_feat}].columns)
    \n """
                        from sklearn.impute import SimpleImputer
                        num_imputer = SimpleImputer(strategy='median')
                        df[fill_feat] = pd.DataFrame(num_imputer.fit_transform(df[fill_feat]), columns=df[fill_feat].columns)

                        null_cat = df[missing_df_cols].select_dtypes(include=object).columns.tolist()
                        if null_cat:
                            st.session_state.all_the_process += f"""
    # Fill with Mode
    from sklearn.impute import SimpleImputer
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[{null_cat}] = cat_imputer.fit_transform(df[{null_cat}])
    \n """
                            cat_imputer = SimpleImputer(strategy='most_frequent')
                            df[null_cat] = cat_imputer.fit_transform(df[null_cat])

                        st.session_state['df'] = df
                        if df.select_dtypes(include=object).columns.tolist():
                            st.success(f"The Columns **`{fill_feat}`** has been filled with the Median. And the categorical columns **`{null_cat}`** has been filled with the mode.")
                        else:
                            st.success(f"The Columns **`{fill_feat}`** has been filled with the Median.")


                    # Fill with Mode
                    elif strategy == "Fill with Mode (Most Frequent)":
                        st.session_state.all_the_process += f"""
    # Fill with Mode
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    df[{fill_feat}] = imputer.fit_transform(df[{fill_feat}])
    \n """
                        from sklearn.impute import SimpleImputer
                        imputer = SimpleImputer(strategy='most_frequent')
                        df[fill_feat] = imputer.fit_transform(df[fill_feat])

                        st.session_state['df'] = df
                        st.success(f"The Columns **`{fill_feat}`** has been filled with the Mode.")


            # Show DataFrame Button
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([0.9, 0.6, 1])
            with col2:
                show_df = st.button("Show DataFrame", key="missing_show_df")
            if show_df:
                st.dataframe(df)


        # Encoding
        new_line()
        st.markdown("Categorical Data", unsafe_allow_html=True)
        new_line()
        with st.expander("Encoding"):
            

            
            
            # INFO
            show_cat = st.checkbox("Show Categorical Features", value=False, key='show_cat')
            # new_line()
            if show_cat:
                col1, col2 = st.columns(2)
                col1.dataframe(df.select_dtypes(include=object), height=250)
                if len(df.select_dtypes(include=object).columns.tolist()) > 1:
                    tmp = df.select_dtypes(include=object)
                    tmp = tmp.apply(lambda x: x.unique())
                    tmp = tmp.to_frame()
                    tmp.columns = ['Unique Values']
                    col2.dataframe(tmp, height=250)
                
            # Further Analysis
            # new_line()
            further_analysis = st.checkbox("Further Analysis", value=False, key='further_analysis')
            if further_analysis:

                col1, col2 = st.columns([0.5,1])
                with col1:
                    # Each categorical feature has how many unique values as dataframe
                    new_line()
                    st.markdown("<h6 align='left'> Number of Unique Values", unsafe_allow_html=True)
                    unique_values = pd.DataFrame(df.select_dtypes(include=object).nunique())
                    unique_values.columns = ['# Unique Values']
                    unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                    st.dataframe(unique_values, width=200, height=300)

                with col2:
                    # Plot for the count of unique values for the categorical features
                    new_line()
                    st.markdown("<h6 align='center'> Plot for the Count of Unique Values ", unsafe_allow_html=True)
                    unique_values = pd.DataFrame(df.select_dtypes(include=object).nunique())
                    unique_values.columns = ['# Unique Values']
                    unique_values = unique_values.sort_values(by='# Unique Values', ascending=False)
                    unique_values['Feature'] = unique_values.index
                    fig = px.bar(unique_values, x='Feature', y='# Unique Values', color='# Unique Values', height=350)
                    st.plotly_chart(fig)




            # INPUT
            col1, col2 = st.columns(2)
            from sklearn.preprocessing import LabelEncoder
            with col1:
                enc_feat = st.multiselect("Select Features", df.select_dtypes(include=object).columns.tolist(), key='encoding_feat', help="Select the categorical features to encode.")

            with col2:
                encoding = st.selectbox("Select Encoding", ["Select",  "Label Encoding","Ordinal Encoding","One Hot Encoding"], key='encoding', help="Select the encoding method.")


            if enc_feat and encoding != "Select":
                new_line()
                col1, col2, col3 = st.columns([1,0.5,1])
                if col2.button("Apply", key='encoding_apply',use_container_width=True ,help="Click to apply encoding."):
                    progress_bar()
                    # Ordinal Encoding
                    new_line()
                    if encoding == "Ordinal Encoding":
                        st.session_state.all_the_process += f"""
    # Ordinal Encoding
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    cat_cols = {enc_feat}
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    \n """
                        from sklearn.preprocessing import OrdinalEncoder
                        encoder = OrdinalEncoder()
                        cat_cols = enc_feat
                        df[cat_cols] = encoder.fit_transform(df[cat_cols])
                        st.session_state['df'] = df
                        st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Ordinal Encoding.")
                        
                    # One Hot Encoding
                    elif encoding == "One Hot Encoding":
                        # Append the process details to the session state log
                        st.session_state.all_the_process += f"""
                        # One Hot Encoding
                        df = pd.get_dummies(df, columns={enc_feat})
                        \n
                        """

                        # Apply one hot encoding to the dataframe
                        df = pd.get_dummies(df, columns=enc_feat)
                        st.session_state['df'] = df

                        # Notify the user
                        st.success(f"The categories of the features **`{enc_feat}`** have been encoded using One Hot Encoding.")
                    elif encoding == "Label Encoding":
                        label_encoders = {}
                        for feature in enc_feat:
                            le = LabelEncoder()
                            df[feature] = le.fit_transform(df[feature])
                            label_encoders[feature] = le  # Store the encoders for inverse transformation if needed
                            
                        st.session_state['df'] = df
                        st.session_state['label_encoders'] = label_encoders  # Save encoders in session state
                        st.success(f"The Categories of the features **`{enc_feat}`** have been encoded using Label Encoding.")
            new_line()       
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([1, 0.7, 1])
            with col2:
                show_df = st.button("DataFrame", key="cat_show_df", help="Click to show the DataFrame.")
            if show_df:
                st.dataframe(df)


        # Scaling
        new_line()
        st.markdown("### ⚖️ Scaling", unsafe_allow_html=True)
        new_line()
        with st.expander("Show Scaling"):
            new_line()






            # Scaling Methods
            
            feat_range = st.checkbox("Further Analysis", value=False, key='feat_range')
            if feat_range:
                new_line()
                st.write("The Ranges for the numeric features:")
                col1, col2, col3 = st.columns([0.05,1, 0.05])
                with col2:
                    st.dataframe(df.describe().T, width=700)
                
                new_line()

            # INPUT
            new_line()
            new_line()
            col1, col2 = st.columns(2)
            with col1:
                scale_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features to be scaled.")

            with col2:
                scaling = st.selectbox("Select Scaling", ["Select", "Standard Scaling", "MinMax Scaling"], help="Select the scaling method.")


            if scale_feat and scaling != "Select":       
                    new_line()
                    col1, col2, col3 = st.columns([1, 0.5, 1])
                    
                    if col2.button("Apply", key='scaling_apply',use_container_width=True ,help="Click to apply scaling."):

                        progress_bar()
        
                        # Standard Scaling
                        if scaling == "Standard Scaling":
                            st.session_state.all_the_process += f"""
    # Standard Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
    \n """
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                            st.session_state['df'] = df
                            st.success(f"The Features **`{scale_feat}`** have been scaled using Standard Scaling.")
        
                        # MinMax Scaling
                        elif scaling == "MinMax Scaling":
                            st.session_state.all_the_process += f"""
    # MinMax Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[{scale_feat}] = pd.DataFrame(scaler.fit_transform(df[{scale_feat}]), columns=df[{scale_feat}].columns)
    \n """
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            df[scale_feat] = pd.DataFrame(scaler.fit_transform(df[scale_feat]), columns=df[scale_feat].columns)
                            st.session_state['df'] = df
                            st.success(f"The Features **`{scale_feat}`** have been scaled using MinMax Scaling.")
        
                        

            # Show DataFrame Button
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([0.9, 0.6, 1])
            with col2:
                show_df = st.button("DataFrame", key="scaling_show_df", help="Click to show the DataFrame.")
            if show_df:
                st.dataframe(df)


        # Data Transformation
        new_line()
        st.markdown("Data Transformation", unsafe_allow_html=True)
        new_line()
        with st.expander("Data Transformation"):
            new_line()
            


            # Transformation Methods
            trans_methods = st.checkbox("Explain Transformation Methods", key="trans_methods", value=False)
            if trans_methods:
                new_line()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h6 align='center'> Log <br> Transformation</h6>", unsafe_allow_html=True)
                    st.latex(r'''z = log(x)''')

                with col2:
                    st.markdown("<h6 align='center'> Square Root Transformation </h6>", unsafe_allow_html=True)
                    st.latex(r'''z = \sqrt{x}''')

                with col3:
                    st.markdown("<h6 align='center'> Cube Root Transformation </h6>", unsafe_allow_html=True)
                    st.latex(r'''z = \sqrt[3]{x}''')
                from scipy.stats import skew
                skewness_before = df.apply(skew).sort_values(ascending=False)

                st.markdown("### **Skewness**")
                st.write(skewness_before)



            # INPUT
            new_line()
            col1, col2 = st.columns(2)
            with col1:
                trans_feat = st.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), help="Select the features you want to transform.", key="transformation features")

            with col2:
                trans = st.selectbox("Select Transformation", ["Select", "Log Transformation", "Square Root Transformation", "Cube Root Transformation", "Exponential Transformation"],
                                    help="Select the transformation you want to apply.", 
                                    key= "transformation")
            

            if trans_feat and trans != "Select":
                new_line()
                col1, col2, col3 = st.columns([1, 0.5, 1])
                if col2.button("Apply", key='trans_apply',use_container_width=True ,help="Click to apply transformation."):

                    progress_bar()

                    # new_line()
                    # Log Transformation
                    if trans == "Log Transformation":
                        st.session_state.all_the_process += f"""
    #Log Transformation
    df[{trans_feat}] = np.log1p(df[{trans_feat}])
    \n """
                        df[trans_feat] = np.log1p(df[trans_feat])
                        st.session_state['df'] = df
                        st.success("Numerical features have been transformed using Log Transformation.")

                    # Square Root Transformation
                    elif trans == "Square Root Transformation":
                        st.session_state.all_the_process += f"""
    #Square Root Transformation
    df[{trans_feat}] = np.sqrt(df[{trans_feat}])
    \n """
                        df[trans_feat] = np.sqrt(df[trans_feat])
                        st.session_state['df'] = df
                        st.success("Numerical features have been transformed using Square Root Transformation.")

                    # Cube Root Transformation
                    elif trans == "Cube Root Transformation":
                        st.session_state.all_the_process += f"""
    #Cube Root Transformation
    df[{trans_feat}] = np.cbrt(df[{trans_feat}])
    \n """
                        df[trans_feat] = np.cbrt(df[trans_feat])
                        st.session_state['df'] = df
                        st.success("Numerical features have been transformed using Cube Root Transformation.")

                    

            # Show DataFrame Button
            # new_line()
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([0.9, 0.6, 1])
            with col2:
                show_df = st.button("DataFrame", key="trans_show_df", help="Click to show the DataFrame.")
            
            if show_df:
                st.dataframe(df)


        # Feature Engineering
        new_line()
        st.markdown("### ⚡ Feature Engineering", unsafe_allow_html=True)
        new_line()
        with st.expander("Show Feature Engineering"):

            # Feature Extraction
            new_line()
            st.markdown("#### Feature Extraction", unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            from sklearn.decomposition import PCA
            with col1:
                n_components = st.slider("Number of Principal Components", min_value=1, max_value=min(df.shape[1], 10), value=2, help="Select the number of principal components to extract.")

                col1, col2, col3 = st.columns([1, 0.6, 1])
                new_line()
                if col2.button("Apply PCA"):
                    pca = PCA(n_components=n_components)
                    principal_components = pca.fit_transform(df.select_dtypes(include=np.number))

                    for i in range(n_components):
                        df[f'PC{i+1}'] = principal_components[:, i]

                    st.session_state['df'] = df
                    st.session_state.all_the_process += f"""
                    # Feature Extraction - PCA
                    pca = PCA(n_components={n_components})
                    principal_components = pca.fit_transform(df.select_dtypes(include=np.number))
                    for i in range({n_components}):
                    df[f'PC{{i+1}}'] = principal_components[:, i]

                    """

                    st.success(f"PCA applied with {n_components} components. New features added to the dataset.")




        
            st.divider()
            st.markdown("#### Feature Selection", unsafe_allow_html=True)
            new_line()

            feat_sel = st.multiselect("Select Feature/s", df.columns.tolist(), key='feat_sel', help="Select the Features you want to keep in the dataset")
            new_line()

            if feat_sel:
                col1, col2, col3 = st.columns([1, 0.7, 1])
                if col2.button("Select Features"):
                    st.session_state.all_the_process += f"""
    # Feature Selection\ndf = df[{feat_sel}]
    \n """
                    progress_bar()
                    new_line()
                    df = df[feat_sel]
                    st.session_state['df'] = df
                    st.success(f"The Features **`{feat_sel}`** have been selected.")
            
            # Show DataFrame Button
            col1, col2, col3 = st.columns([0.15,1,0.15])
            col2.divider()
            col1, col2, col3 = st.columns([0.9, 0.6, 1])
            with col2:
                show_df = st.button("Show DataFrame", key="feat_eng_show_df", help="Click to show the DataFrame.")
            
            if show_df:
                st.dataframe(df)


        # Data Splitting
        st.markdown(" Data Splitting", unsafe_allow_html=True)
        new_line()
        with st.expander("Data Splitting"):

            new_line()
            train_size, val_size, test_size = 0,0,0
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
                st.session_state['target_variable'] = target
            with col2:
                sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model. ")
                st.session_state['split_sets'] = sets

            if sets != "Select" and target:
            
            
                if sets == "Train and Test":

                    new_line()
                    col1, col2 = st.columns(2)
                    with col1:
                        train_size = st.number_input("Train Size", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key='train_size')
                        train_size = round(train_size, 2)
                    with col2:
                        test_size = st.number_input("Test Size", min_value=0.0, max_value=1.0, value=0.30, step=0.05, key='val_size')
                        test_size = round(test_size, 2)

                    if float(train_size + test_size) != 1.0:
                        new_line()
                        st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **test** = **{train_size}** + **{test_size}** = **{sum([train_size, test_size])}**" )
                        new_line()

                    else:
                        split_button = ""
                        col1, col2, col3 = st.columns([1, 0.5, 1])
                        with col2:
                            new_line()
                            split_button = st.button("Split Data")

                            if split_button:
                                st.session_state.all_the_process += f"""
    # Data Splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
    \n """
                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                                st.session_state['X_train'] = X_train
                                st.session_state['X_test'] = X_test
                                st.session_state['y_train'] = y_train
                                st.session_state['y_test'] = y_test

                        
                        
                        col1, col2 = st.columns(2)
                        if split_button:
                            st.success("Data Splitting Done!")
                            with col1:
                                st.write("Train Set")
                                st.write("X Train Shape: ", X_train.shape)
                                st.write("Y Train Shape: ", y_train.shape)

                                train = pd.concat([X_train, y_train], axis=1)
                                train_csv = train.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Train Set", train_csv, "train.csv", key='train2')

                            with col2:
                                st.write("Test Set")
                                st.write("X test Shape: ", X_test.shape)
                                st.write("Y test Shape: ", y_test.shape)

                                test = pd.concat([X_test, y_test], axis=1)
                                test_csv = test.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Test Set", test_csv, "test.csv", key='test2')


        # Building the model
        new_line()
        st.markdown("Model Building")
        new_line()
        problem_type = ""
        with st.expander(" Model Building"):    
            
            target, problem_type, model = "", "", ""
            col1, col2, col3 = st.columns(3)

            with col1:
                target = st.selectbox("Target Variable", [st.session_state['target_variable']] , key='target_ml', help="The target variable is the variable that you want to predict")
                new_line()

            with col2:
                problem_type = st.selectbox("Problem Type", ["Select", "Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")

            with col3:

                if problem_type == "Classification":
                    model = st.selectbox("Model", ["Select", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
                                        key='model', help="The model is the algorithm that you want to use to solve the problem")
                    new_line()

                elif problem_type == "Regression":
                    model = st.selectbox("Model", ["Linear Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
                                        key='model', help="The model is the algorithm that you want to use to solve the problem")
                    new_line()


            if target != "Select" and problem_type and model:
                
                if problem_type == "Classification":
                    
                    if model == "Logistic Regression":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            penalty = st.selectbox("Penalty (Optional)", ["l2", "l1", "none", "elasticnet"], key='penalty')

                        with col2:
                            solver = st.selectbox("Solver (Optional)", ["lbfgs", "newton-cg", "liblinear", "sag", "saga"], key='solver')

                        with col3:
                            C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')

                        
                        col1, col2, col3 = st.columns([1,1,1])
                        if col2.button("Train Model"):
                            
                            
                            progress_bar()

                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Logistic Regression
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='{penalty}', solver='{solver}', C={C}, random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from sklearn.linear_model import LogisticRegression
                            model = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True,  key='save_model')

                    if model == "K-Nearest Neighbors":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_neighbors = st.number_input("N Neighbors **Required**", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')

                        with col2:
                            weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')

                        with col3:
                            algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')

                        
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model", use_container_width=True):
                            progress_bar()

                            st.session_state['trained_model_bool'] = True

                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
    model.fit(X_train, y_train)
    \n """
                            from sklearn.neighbors import KNeighborsClassifier
                            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "Support Vector Machine":
                            
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            kernel = st.selectbox("Kernel (Optional)", ["rbf", "poly", "linear", "sigmoid", "precomputed"], key='kernel')
        
                        with col2:
                            degree = st.number_input("Degree (Optional)", min_value=1, max_value=100, value=3, step=1, key='degree')
        
                        with col3:
                            C = st.number_input("C (Optional)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key='C')
        
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model", use_container_width=True):

                            progress_bar()
                            st.session_state['trained_model_bool'] = True
        
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Support Vector Machine
    from sklearn.svm import SVC
    model = SVC(kernel='{kernel}', degree={degree}, C={C}, random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from sklearn.svm import SVC
                            model = SVC(kernel=kernel, degree=degree, C=C, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")
        
                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "Decision Tree":
                                
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
            
                        with col2:
                            splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
            
                        with col3:
                            min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                                
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model", use_container_width=True):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
            
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split}, random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from sklearn.tree import DecisionTreeClassifier
                            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "Random Forest":
                                    
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
                
                        with col2:
                            criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
                
                        with col3:
                            min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=100, value=2, step=1, key='min_samples_split')
                                    
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model", use_container_width=True):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Random Forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split}, random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "XGBoost":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
                
                        with col2:
                            learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
                
                        with col3:
                            booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> XGBoost
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}', random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from xgboost import XGBClassifier
                            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == 'LightGBM':

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
                
                        with col2:
                            learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
                
                        with col3:
                            boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss", "rf"], key='boosting_type')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> LightGBM
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from lightgbm import LGBMClassifier
                            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", key='save_model')

                    if model == 'CatBoost':

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
                
                        with col2:
                            learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
                
                        with col3:
                            boosting_type = st.selectbox("Boosting Type (Optional)", ["Ordered", "Plain"], key='boosting_type')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> CatBoost
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
    model.fit(X_train, y_train)
    \n """
                            from catboost import CatBoostClassifier
                            model = CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')      

                if problem_type == "Regression":
                    
                    if model == "Linear Regression":
                    
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            fit_intercept = st.selectbox("Fit Intercept (Optional)", [True, False], key='normalize')
                
                        with col2:
                            positive = st.selectbox("Positve (Optional)", [True, False], key='positive')
                
                        with col3:
                            copy_x = st.selectbox("Copy X (Optional)", [True, False], key='copy_x')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept={fit_intercept}, positive={positive}, copy_X={copy_x})
    model.fit(X_train, y_train)
    \n """
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression(fit_intercept=fit_intercept, positive=positive, copy_X=copy_x)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "K-Nearest Neighbors":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_neighbors = st.number_input("N Neighbors (Optional)", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')
                
                        with col2:
                            weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')
                
                        with col3:
                            algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
    model.fit(X_train, y_train)
    \n """
                            from sklearn.neighbors import KNeighborsRegressor
                            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "Support Vector Machine":
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            kernel = st.selectbox("Kernel (Optional)", ["linear", "poly", "rbf", "sigmoid", "precomputed"], key='kernel')
                
                        with col2:
                            degree = st.number_input("Degree (Optional)", min_value=1, max_value=10, value=3, step=1, key='degree')
                
                        with col3:
                            gamma = st.selectbox("Gamma (Optional)", ["scale", "auto"], key='gamma')
                            
                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Support Vector Machine
    from sklearn.svm import SVR
    model = SVR(kernel='{kernel}', degree={degree}, gamma='{gamma}')
    model.fit(X_train, y_train)
    \n """
                            from sklearn.svm import SVR
                            model = SVR(kernel=kernel, degree=degree, gamma=gamma)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "Decision Tree":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
                
                        with col2:
                            splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
                
                        with col3:
                            min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Decision Tree
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split})
    model.fit(X_train, y_train)
    \n """
                            from sklearn.tree import DecisionTreeRegressor
                            model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')
                    
                    if model == "Random Forest":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
                
                        with col2:
                            criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
                
                        with col3:
                            min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=1, max_value=10, value=2, step=1, key='min_samples_split')

                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> Random Forest
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split})
    model.fit(X_train, y_train)
    \n """
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "XGBoost":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
                
                        with col2:
                            learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0001, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
                
                        with col3:
                            booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')

                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> XGBoost
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}')
    model.fit(X_train, y_train)
    \n """
                            from xgboost import XGBRegressor
                            model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model')

                    if model == "LightGBM":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
                
                        with col2:
                            learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
                
                        with col3:
                            boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss", "rf"], key='boosting_type')

                        col1, col2, col3 = st.columns([1,0.7,1])
                        if col2.button("Train Model"):
                            progress_bar()
                            st.session_state['trained_model_bool'] = True
                
                            # Train the model
                            st.session_state.all_the_process += f"""
    # Model Building --> LightGBM
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')
    model.fit(X_train, y_train)
    \n """
                            from lightgbm import LGBMRegressor
                            model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type)
                            model.fit(X_train, y_train)
                            st.session_state['trained_model'] = model
                            st.success("Model Trained Successfully!")

                            # save the model
                            import joblib
                            joblib.dump(model, 'model.pkl')

                            # Download the model
                            model_file = open("model.pkl", "rb")
                            model_bytes = model_file.read()
                            col2.download_button("Download Model", model_bytes, "model.pkl", use_container_width=True, key='save_model') 

                    
        # Evaluation
        if st.session_state['trained_model_bool']:
            st.markdown("### 📈 Evaluation")
            new_line()
            with st.expander("Model Evaluation"):
                # Load the model
                import joblib
                model = joblib.load('model.pkl')
                

                if str(model) not in st.session_state.lst_models_predctions:
                    
                    st.session_state.lst_models_predctions.append(str(model))
                    st.session_state.lst_models.append(str(model))
                    if str(model) not in st.session_state.models_with_eval.keys():
                        st.session_state.models_with_eval[str(model)] = []


                    

                    # Predictions
                    if st.session_state["split_sets"] == "Train, Validation, and Test":
                            
                            st.session_state.all_the_process += f"""
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    \n """
                            y_pred_train = model.predict(X_train)
                            st.session_state.y_pred_train = y_pred_train
                            y_pred_val = model.predict(X_val)
                            st.session_state.y_pred_val = y_pred_val
                            y_pred_test = model.predict(X_test)
                            st.session_state.y_pred_test = y_pred_test


                    elif st.session_state["split_sets"] == "Train and Test":
                        
                        st.session_state.all_the_process += f"""
    # Predictions 
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    \n """  
                        
                        y_pred_train = model.predict(X_train)
                        st.session_state.y_pred_train = y_pred_train
                        y_pred_test = model.predict(X_test)
                        st.session_state.y_pred_test = y_pred_test

                # Choose Evaluation Metric
                if st.session_state['problem_type'] == "Classification":
                    evaluation_metric = st.multiselect("Evaluation Metric", ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"], key='evaluation_metric')

                elif st.session_state['problem_type'] == "Regression":
                    evaluation_metric = st.multiselect("Evaluation Metric", ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2 Score"], key='evaluation_metric')

                
                col1, col2, col3 = st.columns([1, 0.6, 1])
                
                st.session_state.show_eval = True
                    
                
                if evaluation_metric != []:
                    

                    for metric in evaluation_metric:


                            if metric == "Accuracy":

                                # Check if Accuary is element of the list of that model
                                if "Accuracy" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("Accuracy")

                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - Accuracy 
    from sklearn.metrics import accuracy_score
    print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
    print("Accuracy Score on Validation Set: ", accuracy_score(y_val, y_pred_val))
    print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import accuracy_score
                                        train_acc = accuracy_score(y_train, y_pred_train)
                                        val_acc = accuracy_score(y_val, y_pred_val)
                                        test_acc = accuracy_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_acc, val_acc, test_acc]
                                        st.session_state['metrics_df'] = metrics_df


                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - Accuracy
    from sklearn.metrics import accuracy_score
    print("Accuracy Score on Train Set: ", accuracy_score(y_train, y_pred_train))
    print("Accuracy Score on Test Set: ", accuracy_score(y_test, y_pred_test))
    \n """

                                        from sklearn.metrics import accuracy_score
                                        train_acc = accuracy_score(y_train, y_pred_train)
                                        test_acc = accuracy_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_acc, test_acc]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "Precision":
                                
                                if "Precision" not in st.session_state.models_with_eval[str(model)]:
                                    
                                    st.session_state.models_with_eval[str(model)].append("Precision")

                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - Precision
    from sklearn.metrics import precision_score
    print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
    print("Precision Score on Validation Set: ", precision_score(y_val, y_pred_val))
    print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import precision_score
                                        train_prec = precision_score(y_train, y_pred_train)
                                        val_prec = precision_score(y_val, y_pred_val)
                                        test_prec = precision_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_prec, val_prec, test_prec]
                                        st.session_state['metrics_df'] = metrics_df
                                        
                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - Precision
    from sklearn.metrics import precision_score
    print("Precision Score on Train Set: ", precision_score(y_train, y_pred_train))
    print("Precision Score on Test Set: ", precision_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import precision_score
                                        train_prec = precision_score(y_train, y_pred_train)
                                        test_prec = precision_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_prec, test_prec]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "Recall":

                                if "Recall" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("Recall")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - Recall
    from sklearn.metrics import recall_score
    print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
    print("Recall Score on Validation Set: ", recall_score(y_val, y_pred_val))
    print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import recall_score
                                        train_rec = recall_score(y_train, y_pred_train)
                                        val_rec = recall_score(y_val, y_pred_val)
                                        test_rec = recall_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_rec, val_rec, test_rec]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - Recall
    from sklearn.metrics import recall_score
    print("Recall Score on Train Set: ", recall_score(y_train, y_pred_train))
    print("Recall Score on Test Set: ", recall_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import recall_score
                                        train_rec = recall_score(y_train, y_pred_train)
                                        test_rec = recall_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_rec, test_rec]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "F1 Score":

                                if "F1 Score" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("F1 Score")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - F1 Score
    from sklearn.metrics import f1_score
    print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
    print("F1 Score on Validation Set: ", f1_score(y_val, y_pred_val))
    print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import f1_score
                                        train_f1 = f1_score(y_train, y_pred_train)
                                        val_f1 = f1_score(y_val, y_pred_val)
                                        test_f1 = f1_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_f1, val_f1, test_f1]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - F1 Score
    from sklearn.metrics import f1_score
    print("F1 Score on Train Set: ", f1_score(y_train, y_pred_train))
    print("F1 Score on Test Set: ", f1_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import f1_score
                                        train_f1 = f1_score(y_train, y_pred_train)
                                        test_f1 = f1_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_f1, test_f1]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "AUC Score":

                                if "AUC Score" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("AUC Score")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - AUC Score
    from sklearn.metrics import roc_auc_score
    print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
    print("AUC Score on Validation Set: ", roc_auc_score(y_val, y_pred_val))
    print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import roc_auc_score
                                        train_auc = roc_auc_score(y_train, y_pred_train)
                                        val_auc = roc_auc_score(y_val, y_pred_val)
                                        test_auc = roc_auc_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_auc, val_auc, test_auc]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - AUC Score
    from sklearn.metrics import roc_auc_score
    print("AUC Score on Train Set: ", roc_auc_score(y_train, y_pred_train))
    print("AUC Score on Test Set: ", roc_auc_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import roc_auc_score
                                        train_auc = roc_auc_score(y_train, y_pred_train)
                                        test_auc = roc_auc_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_auc, test_auc]
                                        st.session_state['metrics_df'] = metrics_df
                                

                            elif metric == "Mean Absolute Error (MAE)":

                                if "Mean Absolute Error (MAE)" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("Mean Absolute Error (MAE)")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - MAE
    from sklearn.metrics import mean_absolute_error
    print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
    print("MAE on Validation Set: ", mean_absolute_error(y_val, y_pred_val))
    print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import mean_absolute_error
                                        train_mae = mean_absolute_error(y_train, y_pred_train)
                                        val_mae = mean_absolute_error(y_val, y_pred_val)
                                        test_mae = mean_absolute_error(y_test, y_pred_test)

                                        metrics_df[metric] = [train_mae, val_mae, test_mae]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:
                                        st.session_state.all_the_process += f"""
    # Evaluation - MAE
    from sklearn.metrics import mean_absolute_error
    print("MAE on Train Set: ", mean_absolute_error(y_train, y_pred_train))
    print("MAE on Test Set: ", mean_absolute_error(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import mean_absolute_error
                                        train_mae = mean_absolute_error(y_train, y_pred_train)
                                        test_mae = mean_absolute_error(y_test, y_pred_test)

                                        metrics_df[metric] = [train_mae, test_mae]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "Mean Squared Error (MSE)":

                                if "Mean Squared Error (MSE)" not in st.session_state.models_with_eval[str(model)]:
                                    
                                    st.session_state.models_with_eval[str(model)].append("Mean Squared Error (MSE)")

                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - MSE
    from sklearn.metrics import mean_squared_error
    print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
    print("MSE on Validation Set: ", mean_squared_error(y_val, y_pred_val))
    print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import mean_squared_error
                                        train_mse = mean_squared_error(y_train, y_pred_train)
                                        val_mse = mean_squared_error(y_val, y_pred_val)
                                        test_mse = mean_squared_error(y_test, y_pred_test)

                                        metrics_df[metric] = [train_mse, val_mse, test_mse]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:

                                        st.session_state.all_the_process += f"""
    # Evaluation - MSE
    from sklearn.metrics import mean_squared_error
    print("MSE on Train Set: ", mean_squared_error(y_train, y_pred_train))
    print("MSE on Test Set: ", mean_squared_error(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import mean_squared_error
                                        train_mse = mean_squared_error(y_train, y_pred_train)
                                        test_mse = mean_squared_error(y_test, y_pred_test)

                                        metrics_df[metric] = [train_mse, test_mse]
                                        st.session_state['metrics_df'] = metrics_df


                            elif metric == "Root Mean Squared Error (RMSE)":

                                if "Root Mean Squared Error (RMSE)" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("Root Mean Squared Error (RMSE)")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - RMSE
    from sklearn.metrics import mean_squared_error
    print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print("RMSE on Validation Set: ", np.sqrt(mean_squared_error(y_val, y_pred_val)))
    print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    \n """
                                        from sklearn.metrics import mean_squared_error
                                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                        metrics_df[metric] = [train_rmse, val_rmse, test_rmse]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:

                                        st.session_state.all_the_process += f"""
    # Evaluation - RMSE
    from sklearn.metrics import mean_squared_error
    print("RMSE on Train Set: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print("RMSE on Test Set: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    \n """
                                        from sklearn.metrics import mean_squared_error
                                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                        metrics_df[metric] = [train_rmse, test_rmse]
                                        st.session_state['metrics_df'] = metrics_df

                                
                            elif metric == "R2 Score":

                                if "R2 Score" not in st.session_state.models_with_eval[str(model)]:

                                    st.session_state.models_with_eval[str(model)].append("R2 Score")
                                
                                    if st.session_state["split_sets"] == "Train, Validation, and Test":

                                        st.session_state.all_the_process += f"""
    # Evaluation - R2 Score
    from sklearn.metrics import r2_score
    print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
    print("R2 Score on Validation Set: ", r2_score(y_val, y_pred_val))
    print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import r2_score
                                        train_r2 = r2_score(y_train, y_pred_train)
                                        val_r2 = r2_score(y_val, y_pred_val)
                                        test_r2 = r2_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_r2, val_r2, test_r2]
                                        st.session_state['metrics_df'] = metrics_df

                                    else:

                                        st.session_state.all_the_process += f"""
    # Evaluation - R2 Score
    from sklearn.metrics import r2_score
    print("R2 Score on Train Set: ", r2_score(y_train, y_pred_train))
    print("R2 Score on Test Set: ", r2_score(y_test, y_pred_test))
    \n """
                                        from sklearn.metrics import r2_score
                                        train_r2 = r2_score(y_train, y_pred_train)
                                        test_r2 = r2_score(y_test, y_pred_test)

                                        metrics_df[metric] = [train_r2, test_r2]
                                        st.session_state['metrics_df'] = metrics_df



                    # Show Evaluation Metric
                    if show_eval:
                        new_line()
                        col1, col2, col3 = st.columns([0.5, 1, 0.5])
                        st.markdown("### Evaluation Metric")

                    
                        if st.session_state["split_sets"] == "Train and Test":
                            st.session_state['metrics_df'].index = ['Train', 'Test']
                            st.write(st.session_state['metrics_df'])

                        


                        # Show Evaluation Metric Plot
                        new_line()
                        st.markdown("### Evaluation Metric Plot")
                        st.line_chart(st.session_state['metrics_df'])

                        # Show ROC Curve as plot
                        if "AUC Score" in evaluation_metric:
                            from sklearn.metrics import roc_curve
                            st.markdown("### ROC Curve")
                            new_line()
                            
                            

                            if st.session_state["split_sets"] == "Train and Test":
                                

                                # Show the ROC curve plot without any columns
                                # plot_auc_roc_curve(model, X_train, y_train, X_test, y_test)
                                # col1, col2, col3 = st.columns([0.2, 1, 0.2])
                                fig, ax = plt.subplots()
                                # roc_curve(model, X_train, y_train, ax=ax)
                                # roc_curve(model, X_test, y_test, ax=ax)
                                # ax.legend(['Train', 'Test'])
                                # col2.pyplot(fig, legend=True)

                                

                        # Show Confusion Matrix as plot
                        if st.session_state['problem_type'] == "Classification":
                            # from sklearn.metrics import plot_confusion_matrix
                            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                            st.markdown("### Confusion Matrix")
                            new_line()
                            probabilities = model.predict_proba(X_test)[:,1] 
                            predictions=model.predict(X_test)
                            logistic=ClassificationMetrics(y_test,predictions,probabilities)
                            logistic.display_metrics()
                            logistic.run_all_visualizations()

                            # cm = confusion_matrix(y_test, y_pred_test)
                            # col1, col2, col3 = st.columns([0.2,1,0.2])
                            # fig, ax = plt.subplots()
                            # ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax)
                            # col2.pyplot(fig)
                            
                            # Show the confusion matrix plot without any columns
                            # col1, col2, col3 = st.columns([0.2, 1, 0.2])
                            # fig, ax = plt.subplots()
                            # plot_confusion_matrix(model, X_test, y_test, ax=ax)
                            # col2.pyplot(fig)
                        elif st.session_state['problem_type'] == "Regression":
                            predictions= model.predict(X_test)
                            Decison=RegressionMetrics(y_test,predictions,X_test)
                            Decison.display_metrics()
                            Decison.run_all_visualizations() 

                        
        st.divider()          
        col1, col2, col3, col4= st.columns(4, gap='small')        

        if col1.button(" df"):
            new_line()
            st.subheader(" The Dataframe")
            st.write("The dataframe is the dataframe that is used on this application to build the Machine Learning model. You can see the dataframe below 👇")
            new_line()
            st.dataframe(df)

        st.session_state.df.to_csv("df.csv", index=False)
        df_file = open("df.csv", "rb")
        df_bytes = df_file.read()
        if col2.download_button(" Download df", df_bytes, "df.csv", key='save_df', use_container_width=True):
            st.success("Downloaded Successfully!")

        if col3.button("Code", use_container_width=True):
            new_line()
            st.subheader("The Code")
            st.write("The code below is the code that is used to build the model. It is the code that is generated by the app. You can copy the code and use it in your own project 😉")
            new_line()
            st.code(st.session_state.all_the_process, language='python')

        if col4.button("Reset", use_container_width=True):
            new_line()
            st.subheader("Reset")
            st.write("Click the button below to reset the app and start over again")
            new_line()
            st.session_state.reset_1 = True

        if st.session_state.reset_1:
            col1, col2, col3 = st.columns(3)
            if col2.button("Reset", use_container_width=True, key='reset'):
                st.session_state.df = None
                st.session_state.clear()
                st.experimental_rerun()
                

if __name__ == "__main__":
    Machine_Learning()