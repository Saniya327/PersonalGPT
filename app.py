import streamlit as st
from mygpt import load_documents
from mygpt import split_text
from mygpt import save_to_chroma
from mygpt import initialize_db
from mygpt import get_query_answer


def main():
    if 'db' not in st.session_state:
        documents =  load_documents("data")
        chunks = split_text(documents)
        save_to_chroma(chunks,"chroma")
        db = initialize_db()
        st.session_state['db'] = db

    st.set_page_config(page_title="Personal GPT", page_icon=":books:")
    st.header("Personal GPT")
    with st.form(key="my_form"):
        input = st.text_input("Ask a question about your documents:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            db = st.session_state['db']
            answer = get_query_answer(input,db)
            st.write(answer)


if __name__ == '__main__':
    main()