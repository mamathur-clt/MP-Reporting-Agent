"""
Tab render modules for the v2 tabbed Streamlit app.

Each module exposes a `render(ctx: AppContext)` function. The main entry
`streamlit_app.py` builds the sidebar + shared context, then dispatches
to each tab's renderer inside a `st.tabs(...)` block.
"""
