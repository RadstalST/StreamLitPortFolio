import streamlit as st
# Import modules
import numpy as np
import pandas as pd
# Import PySwarms
import pyswarms as ps
import plotly.graph_objects as go
import scipy.interpolate
import plotly.express as px

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}# the hyper param for PSO
optimizer = ps.single.GlobalBestPSO(
	n_particles=10, dimensions=2,bounds=(
		[-1.5, -1.5],
		[1.5, 1.5]
	),
	options=options
)
def fitness_function(xy):
    # sin(10(x^2+y^2))/10 + (x^2+y^2)^0.5 
    cost = np.sin(10*(xy[0]**2+xy[1]**2))/10 + (xy[0]**2+xy[1]**2)**0.5
    return cost
@st.cache_data
def get_postions(iteration=100):
    positions = []
    fitness = []

    
    
    def fitness_function_wrapper(x_array):
        positions.append(x_array)
        cost = np.array([fitness_function(xy) for xy in x_array])
        fitness.append(cost)
        return cost
    

    cost,pos = optimizer.optimize(fitness_function_wrapper, iters=100)
    pos_history = np.array(optimizer.pos_history)
    cost_history = np.array(fitness)

    df_pos_history = pd.DataFrame(pos_history.reshape(-1,2), columns=['x','y'])
    df_pos_history['iter'] = np.repeat(np.arange(100),10)
    df_pos_history['particle'] = np.tile(np.arange(10),100)
    df_pos_history['particle'].astype(str)
    df_pos_history['cost'] = cost_history.reshape(-1,1)
    # df_pos_history



        
    return df_pos_history,pos,cost


st.title("Particle Swarm Optimization")
st.write("This is a demo of the Particle Swarm Optimization algorithm.")

st.header("What is Particle Swarm Optimization?")
st.write("""
Particle Swarm Optimization (PSO) is a computational optimization technique inspired by the behavior of bird flocking or fish schooling. It simulates a population of particles that move through a search space to find the optimal solution. Each particle adjusts its position based on its own experience and the information shared by neighboring particles, aiming to converge towards the best solution in the search space.
""")


# what is PSO
# what is the objective function



X = np.linspace(-1.5, 1.5, 100)
Y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(X, Y)

Z = fitness_function(np.array([X,Y]))

st.header("Objective Function Surface")

st.write("The objective function is defined as:")
st.latex(r'''
f(x,y) = \frac{sin(10(x^2+y^2))}{10} + \sqrt{x^2+y^2}
''')
         
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Magma')])
fig.update_layout(title='Ripple And Cone function', autosize=False)
st.plotly_chart(fig, use_container_width=True)


st.header("Particle Swarm Optimization")
st.write("""
the PSO algorithm is a population based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.
and what we are showing here is a 2D version of PSO, where the objective function is a 3D surface that we can visualize with animation over iteration.
""")
st.write("The search process is shown below with size of each partilce as its cost/ fitness value:")

df_pos_history,pos,cost = get_postions()
fig = px.scatter(
    df_pos_history, x='x', y='y',
    size='cost',
    color='particle',animation_frame='iter', 
    range_x=[-1.5,1.5], range_y=[-2,2], 
    title='PSO demo'
    )
fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig.update_layout(title='Swarm Search Animation', autosize=False)
fig.update_layout(transition = {'duration': 0.1})

st.plotly_chart(fig, use_container_width=True)
# df_pos_history,pos,cost = get_postions()
st.header("Particle Swarm Optimization Surface")
st.write("And heres the surface of the objective function obtained with PSO with the minimum position marked in red:")
x = df_pos_history['x'].values
y = df_pos_history['y'].values
z = df_pos_history['cost'].values
X = np.linspace(-1.5, 1.5, 100)
Y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(X, Y)
z = scipy.interpolate.griddata((x, y), z, (X, Y), method='cubic')

fig = go.Figure(data=[go.Surface(z=z, x=X, y=Y, colorscale='Magma')])
# mark the minimum pos
fig.add_trace(go.Scatter3d(
    x=[pos[0]], y=[pos[1]], z=[cost],
    mode='markers',
    marker=dict(
        size=12,
        color='red')
))
fig.update_layout(title='Sphere function', autosize=False,)
st.plotly_chart(fig, use_container_width=True)