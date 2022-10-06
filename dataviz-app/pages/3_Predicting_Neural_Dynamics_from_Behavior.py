import streamlit as st
import plots


st.markdown('# Predicting Neural Dynamics from Behavior')

st.markdown('Prediction of neural activity from external variables provides a useful approach to study what information is encoded in neural circuits, while taking advantage of state-of-the-art machine learning techniques for learning complex input-output relationships. The reverse of this (predicting behavior, or more generally motor intent, from neural activity) is also very important, for example in the field of brain-computer interfaces.')

st.markdown('## Behavior quantification')

st.markdown('In addition to the variables in the previous examples, Stringer & Pachitariu further quantified the mouse\'s behavior by monitoring its facial movements with videography and performing a principal component (PC) decomposition of the motion energy, or the difference between consecutive video frames. These motion energy PCs provide a multi-dimensional description of the mouse\'s facial movements. Their key insight was that these motion energy PCs were better at predicting the neural activity than the basic running and pupil area variables. Below I show a snippet of the running speed, pupil area, and first motion energy PC alongside the motion energy video itself. Drag the slider to view different motion energy frames, with the current timepoint indicated on the other plots with a white vertical line.')

fig = plots.plot_example_behavior_with_video()
st.plotly_chart(fig)

st.markdown('## Neurobehavioral models')

st.markdown('Finally, let\'s see how well the neural SVCs can be predicted from these motion energy PCs. We focus on the use of two model types: 1. reduced-rank linear regression as described by Stringer & Pachitariu; and 2. LSTM networks for multi-timepoint regression, a novel approach to this dataset.')

st.markdown('### Reduced-rank regression')

st.markdown('Reduced-rank regression predicts the desired variables from some k number of linear combinations of the input variables. By restricting this rank, reduced-rank regression can help particularly to prevent overfitting. Below, you can explore how the predicted neural SVC dynamics (in blue, with the real SVCs in white) change as a function of the rank (the number of linear combinations of the behavioral motion energy PCs) is modified. You can see that while the first SVCs are highly predictable from a very low number of behavioral components, the higher SVCs require many more linear combinations of the behavior for accurate prediction.')

fig = plots.plot_predictions_vs_rank()
st.plotly_chart(fig)

st.markdown('### LSTM recurrent neural networks')

st.markdown('While Stringer and Pachitariu et al. only considered linear and instantaneous relationships between the behavior and neural activity, both exhibit dynamic motifs on longer timescales, and it is likely that they may display time-lagged correlations or other time-dependent, non-linear relationships. In order to explore this, I utilized a recurrent neural network, namely a [long short-term memory (LSTM) network](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) with dropout to prevent overfitting, in order to predict neural activity from a sequence of the motion energy PC activity. As you can see below, the percent of reliable neural variance explained by behavior was significantly increased when utilizing the LSTM network. This result demonstrates the importance of considering longer timescale and nonlinear relationships between neural activity and behavior.')

fig, tot_varexpl_linear, tot_varexpl_lstm = plots.plot_linear_vs_lstm_varexpl()
st.plotly_chart(fig)

st.markdown('While the linear reduced rank regression was able to predict about '+str(tot_varexpl_linear)+'% of the neural variance, the LSTM model predicted '+str(tot_varexpl_lstm)+'%.')


st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs and predictions were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')

