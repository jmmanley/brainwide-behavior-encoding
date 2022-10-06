"""
Source code for various plots utilized in my dataviz-app.

Jason Manley, 2022
jmanley at rockefeller dot edu
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import numpy as np
import os
from scipy.stats import zscore

def plot_example_SVCs():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
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

	return fig


def plot_reliable_variance():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
	data = np.load(file)
	
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

	return fig


def plot_example_behavior_neurons():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
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

	return fig


def plot_example_behavior_with_video():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_ex_behavior.npz')
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

	return fig


def plot_predictions_vs_rank():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
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
	return fig


def plot_linear_vs_lstm_varexpl():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M150824_MP019_2016-04-05_128SVC_Truepredict.npz')
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
	return fig, tot_varexpl_linear, tot_varexpl_lstm


def plot_reconstructed_svcs_vs_latent_dim():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M161025_MP030_2017-06-16_128SVC_VAE.npz')
	data = np.load(file)

	projs1 = zscore(data['projs1'],axis=1)
	projs2_preds = zscore(data['projs2_preds'],axis=2)
	t = np.arange(projs1.shape[1])/3

	fig = make_subplots(rows=1, cols=1)

	toplot = [0,1,15,31,63,127]
	colors = sample_colorscale('Twilight',[(2*x%10)/10 for x in np.arange(len(toplot))])

	for i in range(len(toplot)):

	    fig.append_trace(go.Scatter(x=t,
	        y=projs1[toplot[i],:]/7 - i, mode='lines',
	        line = dict(color=colors[0]), name = "SVC "+str(toplot[i]+1)
	    ), row=1, col=1)

	    fig.append_trace(go.Scatter(x=t,
	        y=projs2_preds[0,toplot[i],:]/7 - i, mode='lines',
	        line = dict(color=colors[1]), name = "Predicted SVC "+str(toplot[i]+1)
	    ), row=1, col=1)

	# Define frames
	number_frames = projs2_preds.shape[0]
	frames = []

	for k in range(number_frames):
	    
	    data = []
	    
	    for i in range(len(toplot)):
	        data.append(go.Scatter(visible=True))
	        data.append(go.Scatter(y=projs2_preds[k,toplot[i],:]/7 - i))

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
	            'currentvalue': {'font': {'size': 16}, 'prefix': 'VAE latent dimensionality: ', 'visible': True, 'xanchor': 'right'},
	            'transition': {'duration': 0, 'easing': 'linear'},
	            'pad': {'b': 10, 't': 50},
	            'len': 0.9, 'x': 0.1, 'y': 0,
	            'steps': [{'args': [[k], {'frame': {'duration': 0, 'easing': 'linear', 'redraw': True},
	                                      'transition': {'duration': 0, 'easing': 'linear'}}],
	                       'label': 2**k, 'method': 'animate'} for k in range(number_frames)
	                      ]}]


	fig.update(frames=frames)
	fig.update_layout(height=800, width=1000, xaxis_title='Time (seconds)', yaxis_title='SVC dimension', 
	                  showlegend=False, yaxis_showgrid=False, yaxis_zeroline=False, yaxis_tickvals=[-x for x in range(len(toplot))],
	                  yaxis_ticktext=[x+1 for x in toplot], updatemenus=updatemenus,
	                  sliders=sliders)
	return fig


def plot_nreliable_vae_embedding():
	file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed-data', 'spont_M161025_MP030_2017-06-16_128SVC_VAE.npz')
	data = np.load(file)
	sneur = data['sneur'][:2048]
	varneur = data['varneur'][:2048]

	sneur_vae = data['sneur_vae'][:2048]
	varneur_vae = data['varneur_vae'][:2048]

	fig = make_subplots(rows=1, cols=1)

	fig.append_trace(go.Scatter(x=np.arange(len(sneur))+1,
	    y=np.cumsum(sneur/varneur>0.05), mode='lines',
	    name = "Linear (SVCA)"
	), row=1, col=1)

	output_dims = 2**np.arange(12)

	fig.append_trace(go.Scatter(x=output_dims,
	    y=np.sum(sneur_vae / varneur_vae > 0.05,axis=1), mode='lines',
	    name = "Nonlinear (VAE)"
	), row=1, col=1)

	fig.update_layout(width=600, xaxis_title='Embedding dimensionality', yaxis_title='Number of encoded SVCs (>5% reliable variance)', 
	                  xaxis_type='log', yaxis_type='log')
	return fig