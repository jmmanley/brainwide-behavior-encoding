import numpy as np 
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from scipy.stats import zscore
import os


st.markdown('# Example Behavior and Neural Data')

st.markdown('To get a sense of the type of data that are analyzed in systems neuroscience, below I show some example data from [Stringer & Pachitariu et al. 2019](https://www.science.org/doi/10.1126/science.aav7893), who monitored neural activity in populations of about 10,000 neurons in visual cortex of a mouse while it performed spontaneous and uninstructed behaviors. Stringer and Pachitariu\'s major insight was that a large fraction of neural variance within the visual cortex of the mouse (and, as it turns out, across the brain) can be explained just by the animal\'s own behavior. While it is expected that self-motion variables must be combined with sensory information somewhere in the brain in order to enable flexible behavior, it was surprising that this could be done as early as primary sensory cortex!')

st.markdown('As a proxy for the mouse\'s movements and arousal, the top row displays its Running speed (purple) and Pupil area (red). Below in the middle row are the latent variables identified via dimensionality reduction of the full neural data, called the Shared Variance Components (SVCs). Finally, the bottom row shows traces of actual spiking from individual example neurons.')

file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
data = np.load(file)

t = data['t']
run_speed = data['run_speed']
pupil_area = data['pupil_area']
projs1 = data['projs1']
neurons = data['ex_neurons'][:100,:]


fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

fig.append_trace(go.Line(
    x=t,
    y=zscore(run_speed), mode='lines',
    name = "Run speed"
), row=1, col=1)


fig.append_trace(go.Line(
    x=t,  mode='lines',
    y=zscore(pupil_area)-5, name="Pupil area"
), row=1, col=1)

colors = sample_colorscale('Twilight',[(x%10)/10 for x in np.arange(projs1.shape[0])])

for i in range(projs1.shape[0]):
    fig.append_trace(go.Line(
        x=t,  mode='lines',
        y=-projs1[i,:]/7+i+1, name="SVC "+str(i+1),
        line = dict(color=colors[i])
    ), row=2, col=1)

fig.update_yaxes(range=[3.5,0.1], row=2, col=1)

colors = sample_colorscale('IceFire',[(x%10)/10 for x in np.arange(neurons.shape[0])])

for i in range(neurons.shape[0]):
    fig.append_trace(go.Line(
        x=t,  mode='lines',
        y=-neurons[i,:]/7+i+1, name="Neuron "+str(i+1),
        line = dict(color=colors[i])
    ), row=3, col=1)

fig.update_yaxes(range=[3.5,0.1], row=3, col=1)


fig.update_layout(height=1000, width=1000, xaxis3_title='Time (seconds)', yaxis_title='Behavioral variables', 
                  yaxis2_title='Neural SVC dimensions', yaxis3_title='Random example neurons', showlegend=False,
                  yaxis1_tickvals=[-5,1], yaxis1_ticktext=['Pupil area', 'Running'], yaxis1_showgrid=False, yaxis1_zeroline=False, 
                  yaxis2_showgrid=False, yaxis2_zeroline=False, yaxis3_showgrid=False, yaxis3_zeroline=False)


st.plotly_chart(fig)

st.markdown('Note the strong correlations between some of the Neural SVCs (e.g., SVCs 1 & 2) and the behavioral variables. Can you spot any single neuron in the bottom plot that is correlated with behavior? As you may notice, these individual neurons are much noisier and many of them are not clearly correlated with the behavior; however, dimensionality reduction of these noisy neurons reveals the latent representation of behavior within the SVCs.')

st.markdown('Some tips for interacting with these plots: click and drag horizontally within a plot to select a time period. Click and drag on the x or y axes to scroll in time or across neural variables, respectively.  To reset, click the home🏠 on the top right of the plot.')

st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')
