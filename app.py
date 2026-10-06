import streamlit as st
import tempfile
import os
import pandas as pd

from ingestion import DataIngester
from context_builder import ContextBuilder
from state_manager import HistoryManager
from ai_provider import OllamaProvider
from ai_interpreter import NaturalLanguageInterpreter
from contract_executor import execute_contract

def init_session():
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = None
    if "last_uploaded_file_id" not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if "pending_result" not in st.session_state:
        st.session_state.pending_result = None
    if "pending_contract" not in st.session_state:
        st.session_state.pending_contract = None
    if "pending_preview_df" not in st.session_state:
        st.session_state.pending_preview_df = None
    if "pending_impact" not in st.session_state:
        st.session_state.pending_impact = None

def clear_pending_state():
    st.session_state.pending_result = None
    st.session_state.pending_contract = None
    st.session_state.pending_preview_df = None
    st.session_state.pending_impact = None

def main():
    st.set_page_config(page_title="DataClean-v2", layout="wide")
    init_session()

    st.title("DataClean-v2")
    st.subheader("Intelligent Data Transformation")

    # Upload flow
    is_pending = st.session_state.pending_contract is not None
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], disabled=is_pending)

    if uploaded_file is not None:
        if st.session_state.last_uploaded_file_id != uploaded_file.file_id:
            with st.spinner("Ingesting file..."):
                # Safe bridge to DataIngester
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    ingester = DataIngester()
                    ingest_result = ingester.ingest_csv(tmp_path)
                finally:
                    os.remove(tmp_path)

                if ingest_result.status == "success":
                    st.session_state.history_manager = HistoryManager(ingest_result.dataframe)
                    st.session_state.last_uploaded_file_id = uploaded_file.file_id
                    clear_pending_state()
                    st.rerun()
                else:
                    st.error(f"Ingestion Failed: {ingest_result.message}")
                    if ingest_result.structural_errors:
                        for err in ingest_result.structural_errors:
                            st.write(f"- Row {err.row_index}: {err.issue_type} ({err.message or 'No additional details'})")
                    
                    st.session_state.last_uploaded_file_id = uploaded_file.file_id
                    return

    if st.session_state.history_manager is not None:
        current_df = st.session_state.history_manager.get_current_df()

        st.header("CURRENT DATASET")
        st.dataframe(current_df, use_container_width=True)

        st.divider()
        st.header("Transformation")
        
        # Display clarification if it exists
        if st.session_state.pending_result and st.session_state.pending_result.status == "needs_clarification":
            st.info(f"Clarification needed: {st.session_state.pending_result.message}")

        instruction = st.text_area("Enter transformation instruction:", disabled=is_pending)
        
        if not is_pending:
            if st.button("Interpret & Preview"):
                if instruction.strip():
                    with st.spinner("Interpreting..."):
                        # Ensure context is always built from the current df, not the original upload
                        context = ContextBuilder().build_context(current_df)
                        provider = OllamaProvider()
                        interpreter = NaturalLanguageInterpreter(provider)
                        result = interpreter.interpret(instruction, context, current_df)
                        
                        st.session_state.pending_result = result
                        
                        if result.status == "ready":
                            # Execute Preview on a strictly isolated copy
                            try:
                                preview_df, impact = execute_contract(current_df.copy(), result.contract)
                                st.session_state.pending_contract = result.contract
                                st.session_state.pending_preview_df = preview_df
                                st.session_state.pending_impact = impact
                            except Exception as e:
                                st.error(f"Preview execution failed: {e}")
                                clear_pending_state()
                        elif result.status == "needs_clarification":
                            # Message displayed above text area on rerun
                            pass
                        elif result.status == "unsupported":
                            st.warning(f"Unsupported: {result.message}")
                        elif result.status == "invalid":
                            st.error(f"Invalid instruction or provider error: {result.errors}")
                        
                        st.rerun()
        
        if is_pending:
            st.subheader("Proposed Transformation")
            st.json(st.session_state.pending_contract.model_dump())
            
            st.subheader("Impact")
            st.json(st.session_state.pending_impact)
            
            st.subheader("PREVIEW — NOT YET APPLIED")
            st.dataframe(st.session_state.pending_preview_df, use_container_width=True)
            
            col1, col2, _ = st.columns([1, 1, 4])
            with col1:
                if st.button("Approve", type="primary"):
                    try:
                        st.session_state.history_manager.apply_contract(st.session_state.pending_contract)
                        clear_pending_state()
                        st.success("Transformation applied.")
                        st.rerun()
                    except ValueError as e:
                        # Stale contract safeguard
                        st.error(f"Failed to apply transformation: {e}")
                        clear_pending_state()
                        st.rerun()
            with col2:
                if st.button("Reject"):
                    clear_pending_state()
                    st.rerun()
                    
        st.divider()
        st.header("History")
        history = st.session_state.history_manager.get_history()
        if not history:
            st.info("No transformations applied yet.")
        else:
            for i, contract in enumerate(history, 1):
                st.write(f"**Step {i}:**")
                st.json(contract.model_dump())
                
            if not is_pending:
                if st.button("Undo Last Action"):
                    st.session_state.history_manager.undo()
                    clear_pending_state()
                    st.rerun()

if __name__ == "__main__":
    main()
