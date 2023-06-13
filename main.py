import streamlit as st
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
from streamlit_timeline import timeline
from components import sidebar,main_timeline
import json
# outlines 

# Introdution
## my profile / cool cover
# About me
# experience
# Projects
# Contact me


st.title("Hello, I'm Suchat T.")
st.image("src/covers/Main Cover.png", use_column_width=True)
st.write("*A Data Scientist (and more!)*")
# insert cool cover here
# st_lottie('https://assets7.lottiefiles.com/packages/lf20_rrqimc3f.json') # data center animation

st.header("About Me")
st.write("""
Highly skilled and dedicated Machine Learning `Engineer` / `Scientist` with over four years of experience in the industry. 
Proven track record in developing innovative ML solutions across multiple domains. 
Passionate about Robotics Engineering and leveraging tech & data for driving ground-breaking outcomes. 
Key competencies include `ML`, `CV`, `NLP`, `AI`, `Cloud Architecture`, and deep understanding of Mathematics, Statistics, & Programming. 
Excellent adaptability, problem-solving, and leadership abilities honed via project management and collaboration with multi-disciplinary teams. 
Seeking new challenges to contribute exceptional expertise and expand horizons in this cutting-edge sector.
""")
st.header("Timeline")
with open('src/main_timeline.json', "r") as f:
    data = f.read()
    with st.expander("raw", expanded=False):
        st.write(json.loads(data))

timeline(data, height=800)

st.divider()
st.header("Experience")
st.subheader("Data Scientist")
st.write("""(2019-2022)""")
st.image("src/covers/Allianz.png", use_column_width=True)

st.write("""
Worked at Allianz Ayudhya Assurance PCL. (Thailand) as a Senoir/Supervisor Data Scientist for 2+ years.
""")
with st.expander("details", expanded=False):

    st.write("""
### Responsibility
- Conducting data analysis and utilizing statistical methods to identify patterns and trends in insurance claims data.
- Collaborating with cross-functional teams to develop and implement data-driven strategies to mitigate risk and improve operational efficiency.
- Creating and maintaining comprehensive reports and dashboards to monitor key performance indicators and provide insights for decision-making.
- Managing and maintaining large datasets, ensuring data quality and accuracy.
- Contributing to the development and enhancement of predictive models and algorithms to optimize claim handling processes.

### Achievements
- Successfully implemented data-driven fraud detection algorithms, resulting in a 20% reduction in fraudulent claims and saving the company millions in potential losses.
- Led a cross-functional team in the implementation of a new claims management system, resulting in streamlined processes, reduced turnaround time, and improved customer experience.
- Received recognition for outstanding performance and contributions to data analysis and risk management initiatives within the organization.
""")
   
st.subheader("Technologist")
st.write("""(2018-2019)""")
st.image("src/covers/Eatlab.png", use_column_width=True)

st.write("""
Worked at EATLAB Co., Ltd. (Thailand) as a Technologist for 2+ years.
""")
with st.expander("details", expanded=False):
    st.write("""
### Responsibility
- Developing and deploying AI-powered cameras across various restaurants in Singapore to collect customer data for informed decision-making.
- Implementing feature extraction techniques such as sound recognition, object detection, and movement recognition for the smart table project.
- Collaborating with a multidisciplinary team to design and enhance the functionality of the smart table, ensuring seamless integration of AI technologies.
- Conducting research and staying updated with the latest advancements in computer vision, AI, and robotics to drive innovation within the organization.

### Achievements
- Successfully deployed AI cameras in multiple restaurants, enabling data-driven insights and informed decision-making for restaurant management.
- Developed and implemented sound recognition algorithms for the smart table, enhancing the dining experience for customers by providing interactive audio feedback.
- Contributed to the growth and success of EATLAB by continuously improving AI technologies, fostering innovation, and delivering high-quality solutions to customers.

""")
    

st.divider()

st.header("Technical Skills")

with open('src/technical_skills.json', "r") as f:
    skills = f.read()
    skills = json.loads(skills)
    


cols = st.columns(3)
for i,key in enumerate(skills.keys()):
    with cols[i%3]:
        st.subheader(key)
        st.write(",".join(map(lambda x: f"`{x}`",skills[key])))

st.header("Soft Skills")
with open('src/soft_skills.json', "r") as f:
    skills = f.read()
    skills = json.loads(skills)

cols = st.columns(3)
for i,key in enumerate(skills.keys()):
    with cols[i%3]:
        st.subheader(key)
        st.write(skills[key])





sidebar.sidebar_footer()




