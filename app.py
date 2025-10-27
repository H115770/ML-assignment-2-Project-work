import streamlit as st

st.set_page_config(page_title="Hello Streamlit", page_icon="🚗")
st.title("🚗 Used Car ML – sanity check")
st.write("Hvis du ser dette i nettleseren, funker Streamlit og repoet ditt. 🎉")

name = st.text_input("Hva heter du?")
if name:
    st.success(f"Heisann, {name}! Repoet ditt er klart for Streamlit.")
