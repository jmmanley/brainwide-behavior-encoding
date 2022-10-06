import numpy as np 
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import os


st.markdown('# Predicting Neural Dynamics from Behavior')

st.markdown('Prediction of neural activity from external variables provides a useful approach to study what information is encoded in neural circuits, while taking advantage of state-of-the-art machine learning techniques for learning complex input-output relationships. The converse of this (predicting behavior, or more generally motor intent, from neural activity) is also very important, for example in the field of brain-computer interfaces.')

st.markdown('## Behavior quantification')

st.markdown('In addition to the variables in the previous examples, Stringer & Pachitariu further quantified the mouse\'s behavior by monitoring its facial movements with videography and performing a principal component (PC) decomposition of the motion energy, or the difference between consecutive video frames. These motion energy PCs provide a multi-dimensional description of the mouse\'s facial movements. They found that these motion energy PCs were better at predicting the neural activity than the basic running and pupil area variables. Below I show a snippet of the running speed, pupil area, and first motion energy PC alongside the motion energy video itself. Drag the slider to view different motion energy frames, with the current timepoint indicated on the other plots with a white vertical line.')


file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M150824_MP019_2016-04-05_ex_behavior.npz')
data = np.load(file)

t = data['t']
run_speed = data['run_speed']
pupil_area = data['pupil_area']
motion_pc1 = data['motion_pc1']
motionVideo_cropped = data['motionVideo_cropped']


fig = make_subplots(rows=3, cols=2,
                    specs=[[{}, {"rowspan": 2}],
                           [{}, None],
                           [{}, {}]])

fig.append_trace(go.Scatter(
    x=t,
    y=run_speed, mode='lines',
    name = "Running speed"
), row=1, col=1)


fig.append_trace(go.Scatter(
    x=t,  mode='lines',
    y=pupil_area, name="Pupil area"
), row=2, col=1)

fig.append_trace(go.Scatter(
    x=[t[0],t[0]],  mode='lines',
    y=[np.min(run_speed)-np.abs(np.min(run_speed*0.1)), np.max(run_speed)+np.abs(np.max(run_speed)*0.1)],
    line_color='white'
), row=1, col=1)

fig.update_yaxes(range=[np.min(run_speed)-np.abs(np.min(run_speed*0.05)), np.max(run_speed)+np.abs(np.max(run_speed)*0.05)], row=1, col=1)


fig.append_trace(go.Scatter(
    x=[t[0],t[0]],  mode='lines',
    y=[np.min(pupil_area)-np.abs(np.min(pupil_area*0.1)), np.max(pupil_area)+np.abs(np.max(pupil_area)*0.1)],
    line_color='white'
), row=2, col=1)

fig.update_yaxes(range=[np.min(pupil_area)-np.abs(np.min(pupil_area*0.05)), np.max(pupil_area)+np.abs(np.max(pupil_area)*0.05)], row=2, col=1)


fig.append_trace(go.Scatter(
    x=t,  mode='lines',
    y=motion_pc1, name="Motion energy PC 1"
), row=3, col=1)

fig.append_trace(go.Scatter(
    x=[t[0],t[0]],  mode='lines',
    y=[np.min(motion_pc1)-np.abs(np.min(motion_pc1*0.1)), np.max(motion_pc1)+np.abs(np.max(motion_pc1)*0.1)],
    line_color='white'
), row=3, col=1)

fig.update_yaxes(range=[np.min(motion_pc1)-np.abs(np.min(motion_pc1*0.05)), np.max(motion_pc1)+np.abs(np.max(motion_pc1)*0.05)], 
                 row=3, col=1)
fig.update_xaxes(range=[5600,5850], row=1, col=1)
fig.update_xaxes(range=[5600,5850], row=2, col=1)
fig.update_xaxes(range=[5600,5850], row=3, col=1)

fig.append_trace(go.Heatmap(z=motionVideo_cropped[0,:], zmin=30, zmax=200, colorscale='gray', showscale=False,
), row=1, col=2)

fig.update_layout(height=600, width=1000, xaxis4_title='Time (seconds)', yaxis_title='Run speed', 
                  yaxis3_title='Pupil area', yaxis4_title='Motion energy PC 1', showlegend=False, yaxis2_showgrid=False, xaxis2_showgrid=False,
                  yaxis2_tickvals=[], xaxis2_tickvals=[], xaxis1_showgrid=False, xaxis3_showgrid=False, xaxis4_showgrid=False)
fig['layout']['yaxis2']['autorange'] = "reversed"


# Define frames
number_frames = 150
frames = [dict(
    name=k,
    data=[go.Scatter(visible=True), 
    	  go.Scatter(visible=True), 
          go.Scatter(x=[t[k*5+16800],t[k*5+16800]]), 
          go.Scatter(x=[t[k*5+16800],t[k*5+16800]]),
          go.Scatter(visible=True), 
          go.Scatter(x=[t[k*5+16800],t[k*5+16800]]),
          go.Heatmap(z=motionVideo_cropped[k,:], zmin=30, zmax=200, colorscale='Greys', showscale=False)  
          ],
    #traces=[0, 1, 2] 
) for k in range(number_frames)]


# Play button
updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[f'{k}' for k in range(number_frames)],
                                        dict(frame=dict(duration=500, redraw=False),
                                             transition=dict(duration=0),
                                             easing='linear',
                                             fromcurrent=True,
                                             mode='immediate'
                                             )]),
                             dict(label='Pause',
                                  method='animate',
                                  args=[[None],
                                        dict(frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0),
                                             mode='immediate'
                                             )])
                             ],
                    direction='left',
                    pad=dict(r=10, t=85),
                    showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
               ]


# Slider
sliders = [{'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {'font': {'size': 16}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.1, 'y': 0,
            'steps': [{'args': [[k], {'frame': {'duration': 0, 'easing': 'linear', 'redraw': True},
                                      'transition': {'duration': 0, 'easing': 'linear'}}],
                       'label': np.round(t[k*5+16800],1), 'method': 'animate'} for k in range(number_frames)
                      ]}]


fig.update(frames=frames)
fig.update_layout(updatemenus=updatemenus,
                  sliders=sliders)

st.plotly_chart(fig)


st.markdown('## Neurobehavioral models')

st.markdown('Finally, let\'s see how well the neural SVCs can be predicted from these motion energy PCs. We focus on the use of two model types: 1. reduced-rank linear regression as described by Stringer & Pachitariu; and 2. LSTM networks for multi-timepoint regression, a novel approach to this dataset.')

st.markdown('### Reduced-rank regression')

st.markdown('Reduced-rank regression predicts the desired variables from some k number of linear combinations of the input variables. By restricting this rank, reduced-rank regression can help particularly to prevent overfitting. Below, you can explore how the predicted neural SVC dynamics (in blue, with the real SVCs in white) change as a function of the rank (the number of linear combinations of the behavioral motion energy PCs) is modified. You can see that while the first SVCs are highly predictable from a very low number of behavioral components, the higher SVCs require many linear combinations of the behavior for accurate prediction.')

file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
data = np.load(file)

t = data['t']
projs1 = data['projs1']
projs1_pred_linear_vsrank = data['projs1_pred_linear_vsrank']

fig = make_subplots(rows=1, cols=1)

toplot = [0,1,4,15]
colors = sample_colorscale('Twilight',[(2*x%10)/10 for x in np.arange(len(toplot))])

for i in range(len(toplot)):

    fig.append_trace(go.Scatter(x=t[:projs1_pred_linear_vsrank.shape[2]],
        y=projs1[toplot[i],:projs1_pred_linear_vsrank.shape[2]]/7 - i, mode='lines',
        line = dict(color=colors[0]), name = "SVC "+str(toplot[i]+1)
    ), row=1, col=1)

    fig.append_trace(go.Scatter(x=t[:projs1_pred_linear_vsrank.shape[2]],
        y=projs1_pred_linear_vsrank[0,toplot[i],:]/7 - i, mode='lines',
        line = dict(color=colors[1]), name = "Predicted SVC "+str(toplot[i]+1)
    ), row=1, col=1)

# Define frames
number_frames = projs1_pred_linear_vsrank.shape[0]
frames = []

for k in range(number_frames):
    
    data = []
    
    for i in range(len(toplot)):
        data.append(go.Scatter(visible=True))
        data.append(go.Scatter(y=projs1_pred_linear_vsrank[k,toplot[i],:]/7 - i))

    frames.append(dict(
        name=k,
        data=data
    ))


# Play button
updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[f'{k}' for k in range(number_frames)],
                                        dict(frame=dict(duration=500, redraw=False),
                                             transition=dict(duration=0),
                                             easing='linear',
                                             fromcurrent=True,
                                             mode='immediate'
                                             )]),
                             dict(label='Pause',
                                  method='animate',
                                  args=[[None],
                                        dict(frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0),
                                             mode='immediate'
                                             )])
                             ],
                    direction='left',
                    pad=dict(r=10, t=85),
                    showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
               ]


# Slider
sliders = [{'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {'font': {'size': 16}, 'prefix': 'Rank of regression: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.1, 'y': 0,
            'steps': [{'args': [[k], {'frame': {'duration': 0, 'easing': 'linear', 'redraw': True},
                                      'transition': {'duration': 0, 'easing': 'linear'}}],
                       'label': 2**k, 'method': 'animate'} for k in range(number_frames)
                      ]}]


fig.update(frames=frames)
fig.update_layout(height=600, width=1000, xaxis_title='Time (seconds)', yaxis_title='SVC dimension', 
                  showlegend=False, yaxis_showgrid=False, yaxis_zeroline=False, yaxis_tickvals=[0,-1,-2,-3],
                  yaxis_ticktext=[x+1 for x in toplot], updatemenus=updatemenus,
                  sliders=sliders)

st.plotly_chart(fig)



st.markdown('### LSTM recurrent neural networks')

st.markdown('While Stringer and Pachitariu et al. only considered linear and instantaneous relationships between the behavior and neural activity, both exhibit dynamic motifs on longer timescales, and it is likely that they may display time-lagged correlations or other time-dependent, non-linear relationships. In order to explore this, I utilized a recurrent neural network, namely a small LSTM network with dropout to prevent overfitting, in order to predict neural activity from a sequence of the motion energy PC activity. As you can see below, the percent of reliable neural variance explained by behavior was significantly increased when utilizing the LSTM network. This result demonstrates the importance of considering longer timescale and nonlinear relationships between neural activity and behavior.')

file = '/Users/jmanley/Desktop/spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz'
data = np.load(file)

cov_res_beh_linear = data['cov_res_beh_linear']
sneur = data['sneur'][:128]
varneur = data['varneur'][:128]
cov_res_beh_lstm = data['cov_res_beh_lstm']

fig = make_subplots(rows=1, cols=1)

varexpl_linear = (sneur - np.min(np.min(cov_res_beh_linear,axis=1),axis=1))/varneur*100
varexpl_lstm = (sneur - np.min(np.min(cov_res_beh_lstm,axis=1),axis=1))/varneur*100

tot_varexpl_linear = np.round(np.sum(sneur - np.min(np.min(cov_res_beh_linear,axis=1),axis=1))/np.sum(varneur)*100,2)
tot_varexpl_lstm = np.round(np.sum(sneur - np.min(np.min(cov_res_beh_lstm,axis=1),axis=1))/np.sum(varneur)*100,2)

fig.append_trace(go.Scatter(
    x=np.arange(len(sneur))+1,
    y=varexpl_linear, mode='lines',
    name = "Linear"
    ), row=1, col=1)

fig.append_trace(go.Scatter(
    x=np.arange(len(sneur))+1,
    y=varexpl_lstm, mode='lines',
    name = "LSTM"
    ), row=1, col=1)

fig.update_layout(width=500, xaxis_type='log', xaxis_title='SVC dimension', yaxis_title='% reliable variance explained by motion PCs')

st.plotly_chart(fig)

st.markdown('While the linear reduced rank regression was able to predict about '+str(tot_varexpl_linear)+'% of the neural variance, the LSTM model predicted '+str(tot_varexpl_lstm)+'%.')


st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs and predictions were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')

