import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
# import MDS
from sklearn.manifold import MDS

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
@st.cache_data
def get_data(path='src/data/Mall_Customers.csv'):
    return pd.read_csv(path).set_index('CustomerID')
st.title("Customer Segment Clustering")
st.warning("WARNING: This is a sample data clustering to test the capabilities of streamlit")
raw_data = get_data()
st.write(raw_data.head())


st.header("EDA")
st.subheader("Data Description")
st.write("The data contains 200 rows and 5 columns.")
st.write(raw_data.describe().T)


st.subheader("Pairplots")
st.write("Pairplots are a great way to visualize the relationship between each variable.")
# fig = px.scatter_matrix(
#     raw_data,
#     color="Gender",
#     dimensions=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
#     )
# fig.update_traces(diagonal_visible=False)

fig = ff.create_scatterplotmatrix(raw_data, diag='box', index='Gender',
                                  height=800, width=800,colormap='Portland',
                                  title=""
                                  )

st.plotly_chart(fig,use_container_width=True)


st.subheader("Quantile-Quantile Plot and normality test")
st.write("#### Normality Test")
st.write("The null hypothesis is that the data is normally distributed.")
st.write("If the p-value is less than 0.05, we reject the null hypothesis and conclude that the data is not normally distributed.")
st.write("If the p-value is greater than 0.05, we fail to reject the null hypothesis and conclude that the data is normally distributed.")
st.write("The normality test is performed using the Shapiro-Wilk test.")

tabs = st.tabs(["Age", "Annual Income (k$)", "Spending Score (1-100)"])
for i, cols in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    with tabs[i]:
        # normality test
        st.write(f"##### {cols}")
        st.write("###### Shapiro-Wilk Test")
        test = stats.shapiro(raw_data[cols])
        with st.expander("details", expanded=False):
            st.write(test)
        st.write("p-value: ", test[1])
        if test[1] < 0.05:
            st.write("The data is not normally distributed.")
        else:
            st.write("The data is normally distributed.")
        


        # qq plot
        qq = stats.probplot(raw_data[cols], dist='norm')
    

        x = np.array([qq[0][0][0], qq[0][0][-1]])

        fig = go.Figure()
        # set title
        fig.update_layout(
            title_text=f"QQ Plot for {cols}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Ordered Values",
        )
        fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
        fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
        fig.layout.update(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

st.info("the data is not normally distributed.")

st.subheader("Correlation")
st.write("Correlation is a measure of the strength of the relationship between two variables.")
st.write("The correlation coefficient ranges from -1 to 1.")
st.write("If the correlation coefficient is close to 1, it means that there is a strong positive correlation between the two variables.")
st.write("If the correlation coefficient is close to -1, it means that there is a strong negative correlation between the two variables.")
st.write("If the correlation coefficient is close to 0, it means that there is no correlation between the two variables.")
st.write("The correlation coefficient is calculated using the Pearson correlation coefficient.")

st.write("The correlation matrix is shown below.")
corr = raw_data.drop("Gender",axis=1).corr()

fig = go.Figure(data=go.Heatmap(
                     z=corr,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        reversescale=True,
                        zmin=-1,
                        zmax=1,
                        ))
st.plotly_chart(fig,use_container_width=True)
st.info("There is a negative correlation between age and spending score.")

st.header("Diminsionality Reduction")

st.subheader("Principal Component Analysis")
st.write("Principal Component Analysis (PCA) is a technique for reducing the dimensionality of data.")
st.write("The data is projected onto a lower-dimensional space.")

scaler = RobustScaler()
mds = MDS(n_components=2,random_state=42)
scaled_data = raw_data.drop("Gender",axis=1).copy()
scaled_data[scaled_data.columns] = scaler.fit_transform(scaled_data[scaled_data.columns])
st.write("The data is scaled using MinMaxScaler")
cols = st.columns(2)

with cols[0]:
    st.write(raw_data.describe())
with cols[1]:
    st.write(scaled_data.describe())


pca_data = mds.fit_transform(scaled_data)
pca_data = pd.DataFrame(pca_data, columns=['MDS1', 'MDS2'])

st.write("The data is projected onto a 2-dimensional space using MDS.")

for col,col_name in zip(st.columns(3),["Age", "Annual Income (k$)", "Spending Score (1-100)"]):
   
    fig = px.scatter(
        pca_data, x='MDS1', y='MDS2',
        color=raw_data[col_name],
        color_continuous_scale="Magma",
        )
    # set to be square
    fig.update_layout(
        width=500,
        height=500,
        autosize=False,
    )
    # remove axis labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # set title
    fig.update_layout(
        title_text=f"Scatter plot of MDS colored by {col_name}",
    )
    # set color scale pallette
    # fig.update_layout(coloraxis_showscale=False)


    st.plotly_chart(fig,use_container_width=True)
cols = st.columns(2)
with cols[0]:
    # elbow method for determining number of clusters
    st.subheader("Elbow Method")
    st.write("The elbow method is used to determine the optimal number of clusters. with k-means clustering.")

    sse = []
    K = list(range(1, 11))
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(pca_data)
        sse.append(km.inertia_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K, y=sse,
                        mode='lines+markers',
                        name='lines+markers'))
    fig.update_layout(
        title_text="Elbow Method",
        xaxis_title="Number of Clusters",
        yaxis_title="Sum of Squared Errors",
    )
    st.plotly_chart(fig,use_container_width=True)

with cols[1]:
    st.subheader("Silhouette Method")
    st.write("The silhouette method is used to determine the optimal number of clusters. with k-means clustering.")
    sil = []
    K = list(range(2, 11))
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(pca_data)
        sil.append(silhouette_score(pca_data, km.labels_, metric = 'euclidean'))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K, y=sil,
                        mode='lines+markers',
                        name='lines+markers'))
    fig.update_layout(
        title_text="Silhouette Method",
        xaxis_title="Number of Clusters",
        yaxis_title="Silhouette Score",
    )
    st.plotly_chart(fig,use_container_width=True)


st.info("The optimal number of clusters is 4.")

# plot clustering
st.subheader("Clustering")
st.write("The data is clustered using k-means clustering.")
st.write("The data is clustered into 4 clusters.")
km = KMeans(n_clusters=4)
km.fit(pca_data)
pca_data['cluster'] = list(map(lambda x : f"cluster {x}",km.labels_.astype(str)))

fig = px.scatter(
    pca_data, x='MDS1', y='MDS2',
    color='cluster',

    marginal_x="box",
    marginal_y="box",
    )
# set to be square
fig.update_layout(
    width=800,
    height=800,
    autosize=False,
)

st.plotly_chart(fig,use_container_width=True)