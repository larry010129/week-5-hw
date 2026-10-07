import streamlit as st

from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write(
'''
# Titanic Visualization 1

Within each passenger class, did women survive at a higher rate than men in every age group?
'''
)
# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2

Last-name counts do not match family size. last_names() counts everyone who shares a surname (Andersson appears 9 times). family_groups() uses family_size, which is siblings/spouses plus parents/children plus the passenger. The largest family_size is 11, and that row is 7 third-class passengers, not 11 people with one last name. A surname can also mix unrelated passengers.

Does average ticket fare rise with family size, and does that pattern differ by passenger class?
'''
)
# Generate and display the figure
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus

How many passengers are in each family size, and how does that differ by class?
'''
)
# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)
