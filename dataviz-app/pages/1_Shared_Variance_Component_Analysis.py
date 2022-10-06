import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os

st.markdown('# Introduction to Dimensionality Reduction in Neuroscience')

st.markdown('Due to impressive advances in the ability to monitor the activity of many neurons at cellular resolution, experimental neuroscientists can now regularly record many hundreds or thousands - and in [some cases](https://www.nature.com/articles/s41592-021-01239-8) up to 1 million(!) - neurons simultaneously! Optical microscopy provides a scalable and high spatiotemporal resolution approach to monitor the activity of neurons that have been induced to express a [Genetically Encodable Calcium Indicator (GECI)](https://en.wikipedia.org/wiki/Calcium_imaging#Genetically_encoded_calcium_indicators), such that the neuron fluoresces when its calcium levels are high, a useful proxy for neuronal spiking. See for example the video below from [Demas et al. 2021](https://www.nature.com/articles/s41592-021-01239-8); the blinking circles are spiking neurons in a 2x2mm region of mouse cortex, and the dark curves across the image are actually blood vessels (scale bar: 200mm, playback sped up 4x). This can be done in an awake, behaving animal as it performs tasks, which has provided huge insights into the neural dynamics underlying a wide range of adaptive behaviors.')

video = open('pages/demas2021_video9.mp4', 'rb')
video_bytes = video.read()
st.video(video_bytes)

st.markdown('While many studies have focused on the properties of single neurons, it is now clear that the population-level neural dynamics can provide insight into the neural mechanisms underlying various behaviors. As such, dimensionality reduction has become a major tool in systems neuroscience in order to quickly visualize and quantify the latent (or "hidden", in the sense that they are not directly measured) variables encoded across neurons.')

st.markdown('### Shared Variance Component Analysis')
st.markdown('[Stringer & Pachitariu et al. 2019](https://www.science.org/doi/10.1126/science.aav7893) developed Shared Variance Component Analysis (SVCA) as a neural dimensionality reduction technique to identify "dimensions of neural nariance that are reliably determined by common underlying signals." SVCA splits the neurons into two sets and identifies the dimensions of each set that maximally covary, called the Shared Variance Components (SVCs). Then, in order to determine whether each SVC contained true robust signal or just noise, the reliability of each SVC is quantified by the covariance between the projections of the two neural sets on held-out testing timepoints. This reliability can be visualized below by comparing the SVC projections of each cell set.')

file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
data = np.load(file)

t = data['t']
projs1 = data['projs1']
projs2 = data['projs2']
toplot = [1,7,31,127]

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

for i in range(len(toplot)):
	fig.append_trace(go.Scatter(
	    x=t,
	    y=-projs1[toplot[i],:]/7+i+1, mode='lines',
	    line=dict(color='rgba(44,160,44,0.8)'),
	   	name='SVC '+str(toplot[i]+1)+', cell set 1'
	), row=1, col=1)


	fig.append_trace(go.Scatter(
	    x=t,
	    y=-projs2[toplot[i],:]/7+i+1, mode='lines',
	    line=dict(color='rgba(148,103,189,0.8)'),
	    name='SVC '+str(toplot[i]+1)+', cell set 2'
	), row=1, col=1)

fig.update_layout(height=600, width=1000, xaxis_title='Time (seconds)', yaxis_title='SVC dimension', showlegend=False,
	              yaxis1_tickvals=np.arange(len(toplot))+1, yaxis1_ticktext=np.asarray(toplot)+1, yaxis_showgrid=False)

xstart=int(np.random.rand(1)[0]*6200)

fig.update_xaxes(range=[xstart,xstart+800], row=1, col=1)
fig.update_yaxes(range=[4.8,0.1], row=1, col=1)


st.plotly_chart(fig, use_container_width=True)

st.markdown('The reliability of each SVC can be further quantified by calculating the percentage of variance in each SVC that is reliably covarying on the held-out testing timepoints. Below, on the left we see that more than 100 SVCs exhibit non-zero reliability, suggesting that this region of visual cortex encodes a high-dimensional signal. However, the total variance in each SVC decays rapidly, as shown in the total variance in each SVC on the right, suggesting that many of these reliable dimensions are very small signals.')

sneur = data['sneur']
varneur = data['varneur']

fig = make_subplots(rows=1, cols=2, shared_xaxes=True)

fig.append_trace(go.Scatter(
	    x=np.arange(len(sneur))+1,
	    y=sneur/varneur*100, mode='lines',
	    name = "% reliable variance"
	), row=1, col=1)

fig.append_trace(go.Scatter(
	    x=np.arange(len(sneur))+1,
	    y=sneur/np.sum(sneur), mode='lines',
	    name = "Normalized variance"
	), row=1, col=2)

fig.update_yaxes(range=[-5,-0.8], row=1, col=2)

fig.update_layout(height=400, width=400, xaxis_title='SVC dimension', yaxis_title='% reliable variance', xaxis_type='log', showlegend=False, 
	              xaxis2_type='log', yaxis2_title='Normalized variance', xaxis2_title='SVC dimension', yaxis2_type='log')
st.plotly_chart(fig, use_container_width=True)


st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')

