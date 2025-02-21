import streamlit as st
import pandas as pd
import numpy as np
import os

# Print the current working directory
# print(os.getcwd())
#=========================================
# st.write("hello world")
# x = st.text_input("Favorite Movie?")
# st.write(f"Your favorite movie is: {x}")
#=========================================
# Write from csv file.
# data = pd.read_csv("./movies_data.csv")
# st.write(data)
#=========================================

chart_data = pd.DataFrame(
                np.random.randn(20,3),
                columns=["a","b","c"]
                )
st.bar_chart(chart_data)
st.line_chart(chart_data)