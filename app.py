import streamlit as st
import pandas as pd
from quality_analyzer import analyze_quality

def main():
    st.set_page_config(page_title="DataClean", layout="centered")

    st.title("DataClean")
    st.subheader("Intelligent Data Quality Tool")

    # Initialize session state for DataFrame and issues
    if "df" not in st.session_state:
        st.session_state.df = None
    if "issues" not in st.session_state:
        st.session_state.issues = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Check if a new file was uploaded to reset state and avoid re-reading
        if st.session_state.last_uploaded_file != uploaded_file.file_id:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.last_uploaded_file = uploaded_file.file_id
                st.session_state.issues = None  # Reset issues for the new file
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # If we have a dataframe in session state, display it
    if st.session_state.df is not None:
        df = st.session_state.df

        st.header("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of rows", df.shape[0])
        with col2:
            st.metric("Number of columns", df.shape[1])
        
        st.write("**Column names:**")
        st.write(", ".join(df.columns))

        st.divider()

        if st.button("Analyze Data Quality"):
            with st.spinner("Analyzing data quality..."):
                st.session_state.issues = analyze_quality(df)

        # Display issues if they have been generated
        if st.session_state.issues is not None:
            st.header("Analysis Results")
            issues = st.session_state.issues

            if not issues:
                st.success("No quality issues detected!")
            else:
                st.warning(f"Detected {len(issues)} potential quality issues.")
                
                for i, issue in enumerate(issues):
                    issue_type = issue.get("type", "unknown")
                    title = issue_type.replace('_', ' ').title()
                    
                    if "column" in issue:
                        title += f" in '{issue['column']}'"
                        
                    with st.expander(f"{i + 1}. {title}"):
                        if issue_type == "missing_values":
                            st.write(f"**Missing count:** {issue.get('count')}")
                        elif issue_type == "duplicate_rows":
                            st.write(f"**Duplicate count:** {issue.get('count')}")
                        elif issue_type == "text_inconsistency":
                            st.write("**Variants found:**")
                            st.write(", ".join(issue.get("variants", [])))
                        elif issue_type == "date_format_inconsistency":
                            st.write("**Formats found:**")
                            st.write(", ".join(issue.get("formats", [])))
                        elif issue_type == "embedded_units":
                            st.write("**Values with units found:**")
                            st.write(", ".join(issue.get("values", [])))
                        else:
                            st.json(issue)

if __name__ == "__main__":
    main()
