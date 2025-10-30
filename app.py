import streamlit as st
try:
    from extractive_summarizer import extractive_summary
    from abstractive_summarizer import abstractive_summary
except Exception as import_err:
    st.error("Failed to import summarizer modules.")
    st.exception(import_err)
else:
    st.title("🧠 Text Summarizer App")

    text_input = st.text_area("Enter your text here:")

    if st.button("Extractive Summary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Generating extractive summary..."):
                summary = extractive_summary(text_input)
            st.subheader("Extractive Summary:")
            st.write(summary)

    if st.button("Abstractive Summary"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Generating abstractive summary (first run may take a while)..."):
                summary = abstractive_summary(text_input)
            st.subheader("Abstractive Summary:")
            st.write(summary)
