import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings
import datetime
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
import io
import base64
from ydata_profiling import ProfileReport
from fpdf import FPDF
from assets.css import set_custom_style
import tempfile
import markdownify
from sweet import generate_sweetviz_report, get_download_link_sweet
import numpy as np
from ML import Machine_Learning

if 'uploading_way' not in st.session_state:
        st.session_state['uploading_way'] = None

def generate_autoeda_report(df):
    """
    Generates an AutoEDA report using ydata-profiling and returns the report's HTML content.
    """
    # Generate the profile report
    profile = ProfileReport(df, explorative=True)

    # Create a temporary HTML file to store the report
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        temp_file_name = tmp.name
        profile.to_file(temp_file_name)

    # Read the HTML content from the file
    with open(temp_file_name, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content


def get_download_link_autoeda(content, filename, ext):
    """
    Generates a download link for the AutoEDA report.
    """
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/{ext};base64,{b64}" download="{filename}.{ext}">Download Report</a>'


if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning


warnings.filterwarnings("ignore")


def area_chart(df, selected_column, col2):
    fig = px.area(
        df,
        x=col2,
        y=selected_column,
        title=f"{selected_column}",
        labels={col2: col2, selected_column: selected_column},
        template="plotly_white",
    )
    return fig


# Function to create a beautiful heatmap for the dashboard
def create_heatmap(df):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()

    # Encode categorical features
    for column in df_encoded.columns:
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    # Calculate the correlation matrix
    corr_matrix = df_encoded.corr()

    # Create an annotated heatmap
    heatmap = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale="Rainbow",
        showscale=True,
        colorbar=dict(
            title="Correlation",
            thickness=20,
            len=0.75,
            tickmode="array",
        ),
    )

    # Update layout for better UI/UX
    heatmap.update_layout(
        xaxis=dict(
            title="Features",
            tickangle=45,
            tickfont=dict(size=12, color="#ffffff"),
            showgrid=False,
        ),
        yaxis=dict(
            title="Features",
            tickfont=dict(size=12, color="#ffffff"),
            showgrid=False,
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )

    return heatmap


def create_boxplot(df, boxcolumn):
    fig = px.box(
        df,
        x=boxcolumn,  # Swap y with x for horizontal plot
        points="outliers",
        color_discrete_sequence=["#FF5733"],
    )
    fig.update_layout(
        xaxis_title=boxcolumn,  # Update the axis title
        template="plotly_dark",
        hovermode="y",  # Swap hover mode for better interaction
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        xaxis=dict(showgrid=True),  # Show grid on x-axis
        yaxis=dict(showgrid=True),  # Show grid on y-axis
    )
    return fig


def create_scatter(df, xaxis, yaxis):
    fig = px.scatter(
        df,
        x=xaxis,
        y=yaxis,
        color=xaxis,
        color_continuous_scale="Viridis",
        labels={xaxis: xaxis, yaxis: yaxis},
        hover_data={xaxis: True, yaxis: True},
    )
    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        template="plotly_dark",
        hovermode="closest",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
        histnorm="density",
        opacity=0.7,
        color_discrete_sequence=["#60A5FA"],
        hover_data={column: True},
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Density",
        hovermode="x",
        bargap=0.05,
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_pie_plot(df, column_to_plot):
    value_counts = df[column_to_plot].value_counts()
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        legend_title=column_to_plot,
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_bar_chart(df, xaxis, yaxis):
    fig = px.bar(
        df,
        x=xaxis,
        y=yaxis,
        color=xaxis,
        title=f"{xaxis} by {yaxis}",
        hover_data=[yaxis],
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_treemap(df, path_columns, values_column):
    fig = px.treemap(
        df,
        path=path_columns,
        values=values_column,
        title=f"Treemap of {' > '.join(path_columns)} by {values_column}",
        color=values_column,
        color_continuous_scale="Viridis",
        hover_data=[values_column],
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=50, l=25, r=25, b=25),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_bubble_chart(df, x_col, y_col, size_col, color_col=None):
    if color_col is None:
        color_col = size_col

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        title=f"Bubble Chart of {x_col} vs {y_col} (Size: {size_col})",
        hover_name=df.index,
        size_max=50,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_line_chart(df, x_col, y_col, group_col=None):
    if group_col:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=group_col,
            title=f"Trend of {y_col} Over {x_col} Grouped by {group_col}",
            markers=True,
        )
    else:
        fig = px.line(
            df, x=x_col, y=y_col, title=f"Trend of {y_col} Over {x_col}", markers=True
        )

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_sunburst(df, path_columns, values_column):
    fig = px.sunburst(
        df,
        path=path_columns,
        values=values_column,
        title=f"Sunburst of {' > '.join(path_columns)} by {values_column}",
        color=values_column,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=50, l=25, r=25, b=25),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_violin_plot(df, x_col, y_col):
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        box=True,
        points="all",
        title=f"Distribution of {y_col} by {x_col}",
        color=x_col,
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def get_data_summary(df):
    numeric_df = df.select_dtypes(include=["number"])
    summary = pd.DataFrame(
        {
            "Mean": numeric_df.mean(),
            "Median": numeric_df.median(),
            "Std Dev": numeric_df.std(),
            "Min": numeric_df.min(),
            "Max": numeric_df.max(),
            "Missing": df.isnull().sum(),
        }
    )
    return summary


# Report Generation Functions
def generate_autoeda_report(df, filename="autoeda_report"):
    """Generate a report using pandas-profiling (ydata-profiling)"""
    # Create Profile Report
    profile = ProfileReport(
        df, title="Data Analysis Report", explorative=True, dark_mode=True
    )

    # Save report in various formats
    report_html = profile.to_html()

    # Return the HTML report and other formats
    return {"html": report_html, "title": "AutoEDA Data Analysis Report"}


def generate_custom_report(df, selected_sections, filename="custom_report"):
    """Generate a custom report with user-selected sections"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Custom Data Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2C3E50;
            }}
            h1 {{
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 2px solid #3498DB;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #F8F9FA;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #3498DB;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .plot-container {{
                width: 100%;
                margin: 20px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7F8C8D;
            }}
        </style>
    </head>
    <body>
        <h1>Custom Data Analysis Report</h1>
        <div class="section">
            <h2>Report Summary</h2>
            <p>This report was generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Dataset dimensions: {df.shape[0]} rows and {df.shape[1]} columns</p>
            <p>Selected sections: {', '.join(selected_sections)}</p>
        </div>
    """

    # Add selected sections to the report
    if "data_overview" in selected_sections:
        html_content += f"""
        <div class="section">
            <h2>Data Overview</h2>
            <h3>Data Head</h3>
            {df.head(10).to_html(classes='dataframe')}
            
            <h3>Data Types</h3>
            {pd.DataFrame({'Type': df.dtypes}).reset_index().rename(columns={'index': 'Column'}).to_html(classes='dataframe')}
            
            <h3>Missing Values</h3>
            {pd.DataFrame({'Missing Count': df.isnull().sum(), 'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)}).to_html(classes='dataframe')}
        </div>
        """

    if "statistical_summary" in selected_sections:
        # Create a statistical summary
        numeric_df = df.select_dtypes(include=["number"])
        summary = pd.DataFrame(
            {
                "Mean": numeric_df.mean(),
                "Median": numeric_df.median(),
                "Std Dev": numeric_df.std(),
                "Min": numeric_df.min(),
                "Max": numeric_df.max(),
            }
        )

        html_content += f"""
        <div class="section">
            <h2>Statistical Summary</h2>
            {summary.to_html(classes='dataframe')}
        </div>
        """

    if "distribution_plots" in selected_sections:
        html_content += """
        <div class="section">
            <h2>Distribution Plots</h2>
        """

        # For each numerical column, create a histogram and add to report
        for col in df.select_dtypes(include=["number"]).columns:
            # Create histogram
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig.update_layout(width=800, height=400)

            # Save to a temporary file
            temp_fig_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.write_image(temp_fig_file.name)

            # Read image as base64
            with open(temp_fig_file.name, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

            # Add to HTML
            html_content += f"""
            <div class="plot-container">
                <h3>Distribution of {col}</h3>
                <img src="data:image/png;base64,{img_base64}" alt="Distribution of {col}">
            </div>
            """

            # Clean up temp file
            os.unlink(temp_fig_file.name)

        html_content += """
        </div>
        """

    if "correlation_analysis" in selected_sections:
        # Only create correlation analysis if there are numerical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            # Create correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Create heatmap
            fig = px.imshow(
                corr_matrix, color_continuous_scale="RdBu_r", title="Correlation Matrix"
            )
            fig.update_layout(width=800, height=600)

            # Save to a temporary file
            temp_fig_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.write_image(temp_fig_file.name)

            # Read image as base64
            with open(temp_fig_file.name, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

            # Add to HTML
            html_content += f"""
            <div class="section">
                <h2>Correlation Analysis</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{img_base64}" alt="Correlation Matrix">
                </div>
                {corr_matrix.round(2).to_html(classes='dataframe')}
            </div>
            """

            # Clean up temp file
            os.unlink(temp_fig_file.name)

    if "categorical_analysis" in selected_sections:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            html_content += """
            <div class="section">
                <h2>Categorical Analysis</h2>
            """

            for col in categorical_cols:
                value_counts = df[col].value_counts().head(10)

                # Create pie chart
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {col}",
                )
                fig.update_layout(width=700, height=500)

                # Save to a temporary file
                temp_fig_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.write_image(temp_fig_file.name)

                # Read image as base64
                with open(temp_fig_file.name, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()

                # Add to HTML
                html_content += f"""
                <div class="plot-container">
                    <h3>Distribution of {col}</h3>
                    <img src="data:image/png;base64,{img_base64}" alt="Distribution of {col}">
                </div>
                {value_counts.to_frame().to_html(classes='dataframe')}
                """

                # Clean up temp file
                os.unlink(temp_fig_file.name)

            html_content += """
            </div>
            """

    # Close HTML
    html_content += """
        <div class="footer">
            <p>Generated by Advanced Data Analysis Dashboard</p>
        </div>
    </body>
    </html>
    """

    return {"html": html_content, "title": "Custom Data Analysis Report"}


def get_download_link(content, filename, file_format="html"):
    """Generate a download link for various file formats"""
    if file_format == "html":
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">Download HTML Report</a>'
        return href
    elif file_format == "pdf":
        # Convert HTML to PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add basic content to PDF
        # Note: This is a very basic conversion and doesn't preserve HTML formatting
        # For better HTML to PDF conversion, you might want to use a different library like WeasyPrint
        content_text = markdownify.markdownify(content, strip=["script", "style"])
        pdf.multi_cell(
            0, 10, content_text[:5000]
        )  # Limiting content as a simple example

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        b64 = base64.b64encode(pdf_output.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF Report</a>'
        return href
    elif file_format == "markdown":
        # Convert HTML to Markdown
        markdown_text = markdownify.markdownify(content, strip=["script", "style"])
        b64 = base64.b64encode(markdown_text.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}.md">Download Markdown Report</a>'
        return href

    return None


# Main application
def main():
    # Set page layout to wide mode for better laptop screen usage
    st.set_page_config(layout="wide", page_title="Data Analysis Dashboard")

    # Now call set_custom_style() after set_page_config()
    set_custom_style()

    st.markdown(
        '<h1 class="main-header">Data Science Suite</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar for data upload and configuration

    st.markdown('<h2 class="section-header">Upload Data</h2>', unsafe_allow_html=True)
    data = st.file_uploader("📁 Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

    # Removed color theme selector as requested

    # Main content
    if data is not None:
        if data.name.endswith("xlsx"):
            df = pd.read_excel(data)
        else:
            df = pd.read_csv(data)
            for column in df.columns:
                df[column] = df[column].fillna(
                    df[column].mode()[0] if not df[column].mode().empty else 0
                )

        # Data preview section
        st.markdown(
            '<h2 class="section-header">Data Preview</h2>', unsafe_allow_html=True
        )
        st.markdown(
            '<div style="border: 3px solid #ffffff; margin-bottom: 5px;"></div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            # A big white line
            st.dataframe(df.head(len(df)), height=250)

        with col2:
            st.markdown("**Dataset Information**")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing Values: {df.isnull().sum().sum()}")
            st.markdown("**Column Types:**")
            col_types = (
                pd.DataFrame({"Type": df.dtypes})
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            st.dataframe(col_types, height=150)
            st.markdown("</div>", unsafe_allow_html=True)

        # Create lists of available columns by type
        numerical_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        date_cols = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
        all_cols = df.columns.tolist()

        # Tabs for different visualization types with proper spacing
        st.markdown(
            '<h2 class="section-header">Visualizations</h2>', unsafe_allow_html=True
        )
        st.markdown(
            '<div style="border: 3px solid #ffffff; margin-bottom: 5px;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
                .stTabs [data-baseweb="tab-list"] {
                    display: flex;
                    justify-content: center;
                }
                .stTabs [data-baseweb="tab"] {
                    height: 40px;
                    white-space: pre-wrap;
                    width: 250px;
                }
                
            </style>
            """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(
            [
                "DashBoard",
                "Report Generation",
                "Machine Leaning DashBoard",
            ]
        )

        # Tab 1: Basic Analysis
        with tabs[0]:
            numerical_cols = df.columns.to_list()
            hist_c = df.select_dtypes(include=["int", "float"]).columns.tolist()

            col3, col4 = st.columns([2, 2])
            with col3:
                try:
                    xaxis, yaxis = st.multiselect(
                        "", numerical_cols, key="Xaxisselt", default=numerical_cols[:2]
                    )
                    fig = create_scatter(df, xaxis, yaxis)
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(
                        f"Please Choose 2 numerical columns to create scatter plot"
                    )

            with col4:
                try:

                    xaxis, yaxis = st.multiselect(
                        "", numerical_cols, key="Xaxilt", default=numerical_cols[:2]
                    )

                    fig = px.bar(
                        df,
                        x=xaxis,
                        y=yaxis,
                        labels={"TotalSales": "total Sales {$}"},
                        hover_data=[yaxis],
                        template="gridon",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Plese Choose 2 numerical columns to create bar plot")
            # with col5:

            #     a = st.selectbox("selct numerical columns", hist_c)

            #     hist_fig = create_histogram(df, a)
            #     st.plotly_chart(hist_fig)

            col6, col7 = st.columns([2, 2])
            with col6:
                unique_value_counts = df.nunique()
                filtered_columns = unique_value_counts[
                    unique_value_counts < 12
                ].index.tolist()
                all_columns = df.columns.to_list()

                column_to_plot = st.selectbox("", filtered_columns)

                st.markdown(
                    f"<div style='text-align: center; font-size: 18px; font-family: \"Times New Roman\";'>Distribution of  {column_to_plot}</div>",
                    unsafe_allow_html=True,
                )

                # Create the pie chart with increased size
                pie_fig = create_pie_plot(df, column_to_plot)
                pie_fig.update_layout(
                    height=500, width=500
                )  # Set height and width here

                st.plotly_chart(pie_fig)

            with col7:

                boxcolumn = st.selectbox(
                    "Select a numerical column for outlier detection",
                    numerical_cols,
                )
                st.markdown(
                    f"<div style='text-align: center; font-size: 18px; font-family: \"Times New Roman\";'>Outlier detection: {boxcolumn}</div>",
                    unsafe_allow_html=True,
                )
                fig = create_boxplot(df, boxcolumn)
                st.plotly_chart(fig)

            # with col8:
            #     col2, ara_col = st.multiselect(
            #         "select category than numerical value",
            #         numerical_cols,
            #         default=numerical_cols[:2],
            #     )
            #     fig = area_chart(df, ara_col, col2)
            #     st.plotly_chart(fig)

            st.markdown(
                """
                        <div style="
                            text-align: center; 
                            font-size: 26px; 
                            font-family: Arial, sans-serif; 
                            color: #ffffff; 
                            margin-bottom: 20px;
                        ">
                            🔍 Correlation Heatmap
                        </div>
                        """,
                unsafe_allow_html=True,
            )
            fig = create_heatmap(df)
            st.plotly_chart(fig)

        # Tab 6: Report Generation (New Tab)
        with tabs[1]:
            st.markdown(
                '<h2 class="section-header">Generate Analysis Report</h2>',
                unsafe_allow_html=True,
            )

            # Report type selection
            report_type = st.radio(
                "Select report type:",
                [
                    "Create using AutoEDA",
                    "Create using Sweetviz",
                    "Create custom report",
                ],
            )

            if report_type == "Create using AutoEDA":
                st.markdown(
                    """
                    <style>
                        .card {
                            background-color: #f9f9f9;
                            border-radius: 8px;
                            padding: 20px;
                            margin: 20px 0;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        }
                        .card h3 {
                            color: #2c3e50;
                            margin-bottom: 10px;
                        }
                        .card p {
                            font-size: 16px;
                            color: #7f8c8d;
                        }
                    </style>
                    <div class="card">
                        <h3>AutoEDA Report</h3>
                        <p>This report automatically generates an in-depth Exploratory Data Analysis (EDA) with:</p>
                        <ul>
                            <li>Variable distributions</li>
                            <li>Missing values analysis</li>
                            <li>Correlation heatmaps</li>
                            <li>Data types and unique values</li>
                            <li>Interactive HTML output</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Generate AutoEDA Report"):
                    with st.spinner(
                        "Generating detailed AutoEDA report... Please wait."
                    ):
                        try:
                            # Generate the AutoEDA report
                            report_html = generate_autoeda_report(df)

                            # Display the report preview inside an expander
                            with st.expander("Report Preview (Click to expand)"):
                                st.components.v1.html(
                                    report_html, height=800, scrolling=True
                                )

                            # Provide the download link
                            st.markdown("### Download Options")
                            st.markdown(
                                get_download_link_autoeda(
                                    report_html, "autoeda_report", "html"
                                ),
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")

            elif report_type == "Create using Sweetviz":
                st.markdown(
                    """
                        <style>
                            .card {
                                background-color: #f9f9f9;
                                border-radius: 8px;
                                padding: 20px;
                                margin: 20px 0;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            }
                            .card h3 {
                                color: #2c3e50;
                                margin-bottom: 10px;
                            }
                            .card p {
                                font-size: 16px;
                                color: #7f8c8d;
                            }
                            .card ul {
                                list-style: none;
                                padding-left: 0;
                            }
                            .card ul li {
                                background: url("data:image/svg+xml,%3Csvg width='6' height='6' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='3' cy='3' r='3' fill='%23343a40'/%3E%3C/svg%3E") left center no-repeat;
                                padding-left: 10px;
                                margin-bottom: 5px;
                                color: #34495e;
                            }
                        </style>
                        <div class="card">
                            <h3>Sweetviz Report</h3>
                            <p>This report uses Sweetviz to create a beautiful, interactive EDA report featuring:</p>
                            <ul>
                                <li>Automated visualization of relationships</li>
                                <li>Comparative analysis capabilities</li>
                                <li>Summary statistics with visual cues</li>
                                <li>Target feature analysis</li>
                                <li>Interactive HTML output</li>
                            </ul>
                        </div>
                        """,
                    unsafe_allow_html=True,
                )

                # Button to trigger report generation
                if st.button("Generate Sweetviz Report", key="sweetviz"):
                    with st.spinner(
                        "Generating beautiful interactive report... This may take a minute."
                    ):
                        try:
                            # Generate the report using the helper function
                            report_data = generate_sweetviz_report(df)
                            # Display the full report preview with increased height and scrolling enabled
                            with st.expander("Report Preview (Click to expand)"):
                                st.components.v1.html(
                                    report_data["html"], height=800, scrolling=True
                                )
                            # Provide a download link for the report
                            st.markdown("### Download Options")
                            st.markdown(
                                get_download_link_sweet(
                                    report_data["html"], "sweetviz_report", "html"
                                ),
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")

            elif report_type == "Create custom report":
                st.markdown(
                    """
                <div class="card">
                    <h3>Custom Report</h3>
                    <p>Create a tailored report with only the sections you need. Choose from the following options:</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Section selection
                selected_sections = st.multiselect(
                    "Select sections to include in your report:",
                    options=[
                        "data_overview",
                        "statistical_summary",
                        "distribution_plots",
                        "correlation_analysis",
                        "categorical_analysis",
                    ],
                    default=["data_overview", "statistical_summary"],
                    format_func=lambda x: {
                        "data_overview": "Data Overview (head, info, missing values)",
                        "statistical_summary": "Statistical Summary (mean, median, std, etc.)",
                        "distribution_plots": "Distribution Plots for Numerical Columns",
                        "correlation_analysis": "Correlation Analysis & Heatmap",
                        "categorical_analysis": "Categorical Data Analysis",
                    }[x],
                )

                report_name = st.text_input("Report name:", "my_custom_report")

                # Generate button
                if st.button("Generate Custom Report"):
                    if not selected_sections:
                        st.error("Please select at least one section for your report.")
                    else:
                        with st.spinner("Generating custom report..."):
                            try:
                                report_data = generate_custom_report(
                                    df, selected_sections, report_name
                                )

                                # Show preview in an expander
                                with st.expander("Report Preview (Click to expand)"):
                                    st.components.v1.html(
                                        report_data["html"], height=600
                                    )

                                # Download options
                                st.markdown("### Download Options")
                                st.markdown(
                                    get_download_link(
                                        report_data["html"], report_name, "html"
                                    ),
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    get_download_link(
                                        report_data["html"], report_name, "pdf"
                                    ),
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    get_download_link(
                                        report_data["html"], report_name, "markdown"
                                    ),
                                    unsafe_allow_html=True,
                                )
                            except Exception as e:
                                st.error(f"Error generating report: {e}")

        with tabs[2]:
            Machine_Learning()
    else:
        # Welcome screen when no data is uploaded
        st.markdown(
            """
        <div class="welcome-card">
            <h2>Welcome to the Advanced Data Analysis Dashboard</h2>
            <p>Upload a CSV, TXT, or Excel file to get started with your data analysis journey.</p>
            <p>This dashboard allows you to:</p>
            <ul>
                <li>Explore your data with interactive visualizations</li>
                <li>Analyze relationships between variables</li>
                <li>Detect patterns and outliers</li>
                <li>Generate comprehensive reports</li>
                <li>Export insights in multiple formats</li>
            </ul>
            <p>Upload your file using the sidebar to begin!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        #


if __name__ == "__main__":
    main()
