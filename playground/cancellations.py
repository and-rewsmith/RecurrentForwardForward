import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
sns.set(font_scale=2)
pd.set_option('display.width', None)
sns.set_style(style='white')

df = pd.read_parquet('converted_data_926.parquet', engine='fastparquet')
# df = pd.read_parquet('converted_data_initial.parquet', engine='fastparquet')
df['label'] = df['label'].apply(lambda x: x[0])
df = df[df['neuron index'] < 150]

# df = df[df['data_sample_id']<20]
# df = df[df['label']<20]

# %% Average activations over time per layer
vars = ['image_timestep', 'layer_index', 'is_label_showing', 'label',
        'data_sample_id', 'activation', 'neuron index', 'is_correct']
df2plot = df[vars]

# df2plot = df2plot[(df2plot['layer_index']==2) & (df2plot['data_sample_id']==0)]
df2plot['image_timestep'] = df2plot['image_timestep'].values  # - 9
df2plot = df2plot[df2plot['image_timestep'] >= 10]

# %
# g = sns.relplot(data=df2plot, x='image_timestep', y='activation', hue='label', style='is_correct', col='layer_index', kind='line', errorbar='ci', facet_kws={'sharey':False}, palette='Paired', aspect=1.5)
g = sns.relplot(data=df2plot, x='image_timestep', y='activation', hue='label', style='is_correct',
                col='layer_index', kind='line', errorbar='ci', facet_kws={'sharey': True}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_crosslabels.pdf')
plt.show()

g = sns.relplot(data=df2plot, x='image_timestep', y='activation', hue='label', row='is_correct',
                col='layer_index', kind='line', errorbar='ci', facet_kws={'sharey': True}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_crosslabels_split.pdf')
plt.show()

g = sns.relplot(data=df2plot, x='image_timestep', y='activation', style='is_correct', hue='layer_index',
                kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations.pdf')
plt.show()

g = sns.relplot(data=df2plot, x='image_timestep', y='activation', row='is_correct', hue='layer_index',
                col='label', kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_split.pdf')
plt.show()

# %%
vars = ['image_timestep', 'layer_index', 'is_label_showing', 'label',
        'data_sample_id', 'activation', 'neuron index', 'is_correct']
df2plot = df[vars]
df2plot['activation_squared'] = (df2plot['activation'].values)**2  # * (df2plot['activation'].values>0)
df_L2norm = df2plot.groupby(['image_timestep', 'label', 'is_correct', 'layer_index', 'data_sample_id'])
df_L2norm = df_L2norm['activation'].apply(lambda x: np.sqrt(np.mean(x**2))).reset_index()
df_L2norm = df_L2norm[df_L2norm['image_timestep'] > 10.5]
# df2plot['activation_squared'] = (df2plot['activation'].values)**2# * (df2plot['activation'].values>0)
g = sns.relplot(data=df_L2norm, x='image_timestep', y='activation', row='is_correct', hue='layer_index',
                kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_squared.pdf')
plt.show()

# %%
g = sns.relplot(data=df_L2norm, x='image_timestep', y='activation', row='is_correct', hue='layer_index',
                col='label', kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_squared_split.pdf')
plt.show()


# %%
vars = ['image_timestep', 'layer_index', 'is_label_showing', 'label',
        'data_sample_id', 'backward_activation_component', 'neuron index', 'is_correct']
df2plot = df[vars]

# * (df2plot['activation'].values>0)
df2plot['activation_squared'] = (df2plot['backward_activation_component'].values)**2
g = sns.relplot(data=df2plot, x='image_timestep', y='activation_squared', row='is_correct', hue='layer_index',
                kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_backward_squared.pdf')
plt.show()

# * (df2plot['activation'].values>0)
df2plot['activation_squared'] = (df2plot['backward_activation_component'].values)**2
g = sns.relplot(data=df2plot, x='image_timestep', y='activation_squared', row='is_correct', hue='layer_index',
                col='label', kind='line', errorbar='ci', facet_kws={'sharey': False}, palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
plt.savefig('Layers_activations_backward_squared_split.pdf')
plt.show()


# %% Activations differences
df_match = df[df['is_correct'] == True]
df_mismatch = df[df['is_correct'] == False]
df_match = df_match.sort_values(by=['layer_index', 'image_timestep', 'neuron index', 'label', 'data_sample_id'], axis=0)
df_mismatch = df_mismatch.sort_values(
    by=['layer_index', 'image_timestep', 'neuron index', 'label', 'data_sample_id'], axis=0)
df_diff = df_match
df_diff[['activation_diff', 'forward_activation_component_diff', 'backward_activation_component_diff',
         'lateral_activation_component_diff']] = df_mismatch[['activation', 'forward_activation_component',
                                                              'backward_activation_component', 'lateral_activation_component']].values - df_match[['activation',
                                                                                                                                                   'forward_activation_component', 'backward_activation_component', 'lateral_activation_component']].values

vars2idxs = ['image_timestep', 'layer_index', 'is_label_showing',
             'data_sample_id', 'label', 'neuron index', 'is_correct']
vars2stack = ['activation_diff', 'forward_activation_component_diff',
              'backward_activation_component_diff',  'lateral_activation_component_diff']
df2plot = df_diff.set_index(vars2idxs)[vars2stack]
df2plot = df2plot.stack().reset_index().rename(columns={0: 'value', 'level_'+str(len(vars2idxs)): 'variable'})
df2plot['image_timestep'] = df2plot['image_timestep'].values - 9
df2plot = df2plot[df2plot['image_timestep'] >= -0]
g = sns.relplot(data=df2plot, x='image_timestep', y='value', hue='variable', col='layer_index',
                style='is_correct', kind='line', errorbar='ci', facet_kws={'sharey': False}, aspect=1.5)
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer ' + str(item))
plt.savefig('Layers_activations.pdf')
plt.show()

# %% Cancellation order
# df2plot = df2plot[df2plot['variable']=='activation_diff']
g = sns.relplot(data=df2plot, x='image_timestep', y='value', hue='layer_index', col='variable',
                style='is_correct', kind='line', errorbar='ci', facet_kws={'sharey': False}, aspect=1.5)
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title(item)
plt.savefig('Layers_activations_cancellation_currents.pdf')
plt.show()

# %% Cosyne similarity
df2plot = df.drop(columns=['image', 'dataset'])

df2plot = df2plot.pivot_table(
    index=['layer_index', 'image_timestep', 'is_correct', 'label', 'data_sample_id'],
    values=['activation', 'forward_activation_component', 'backward_activation_component',
            'lateral_activation_component'],
    columns='neuron index')
df2plot = df2plot.sort_index()
comparisons = [('forward_activation_component', 'backward_activation_component'),
               ('forward_activation_component', 'lateral_activation_component'),
               ('backward_activation_component', 'lateral_activation_component')]

df2plot_comparisons = pd.DataFrame()
for i_vars, (var_A, var_B) in enumerate(comparisons):
    vars_A, vars_B = df2plot[var_A].values, df2plot[var_B].values
    sims_AB = np.diag(cosine_similarity(vars_A, vars_B))
    df2plot_vars = df2plot
    df2plot_vars = df2plot_vars.drop(
        columns=['activation', 'forward_activation_component', 'backward_activation_component',
                 'lateral_activation_component'])
    df2plot_vars.columns = df2plot_vars.columns.get_level_values(0)
    df2plot_vars['comparison'] = var_A.split('_')[0]+' '+var_B.split('_')[0]
    df2plot_vars['value'] = sims_AB
    df2plot_comparisons = pd.concat([df2plot_comparisons, df2plot_vars], axis=0)

df2plot_comparisons = df2plot_comparisons.reset_index()
df2plot_comparisons['image_timestep'] = df2plot_comparisons['image_timestep'].values - 9
# df2plot_comparisons = df2plot_comparisons[df2plot_comparisons['image_timestep']>=-2]

g = sns.relplot(
    data=df2plot_comparisons, x='image_timestep', y='value', row='is_correct', hue='comparison', style='is_correct',
    col='layer_index', kind='line', errorbar='ci', facet_kws={'sharey': True},
    palette='Paired', aspect=1.5)
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
    ax.set_ylim([-1., 1.])
plt.savefig('Comparisons_cosine_similarity.pdf')
plt.show()

# %% Cancellation in box plot
df2plot_summ = df2plot_comparisons
df2plot_summ['before/after'] = (df2plot_summ['image_timestep'].values >= -0).astype(float) + (
    df2plot_summ['image_timestep'] >= 5).astype(float) + (df2plot_summ['image_timestep'] >= 10).astype(float) - 1
# df2plot_summ = df2plot_summ[df2plot_summ['before/after']!=-1]
df2plot_summ = df2plot_summ[df2plot_summ['before/after'] != 1]
g = sns.catplot(data=df2plot_summ, x='comparison', y='value', hue='layer_index', col='is_correct',
                row='before/after', palette=sns.color_palette("Blues", as_cmap=True), aspect=2., kind='bar')
sns.despine()
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title('Layer'+' '+str(item))
    ax.set_ylim([-1., 1.])
plt.savefig('Comparisons_cosine_similarity_boxplot.pdf')
plt.show()


# %% Plot layers activation over time
vars2idxs = ['image_timestep', 'layer_index', 'is_label_showing',
             'data_sample_id', 'label', 'neuron index', 'is_correct']
vars2stack = ['activation', 'forward_activation_component',
              'backward_activation_component',  'lateral_activation_component']
df2plot = df.set_index(vars2idxs)[vars2stack]
df2plot = df2plot.stack().reset_index().rename(columns={0: 'value', 'level_'+str(len(vars2idxs)): 'variable'})
# df2plot = df2plot[df2plot['label']==2]
df2plot['image_timestep'] = df2plot['image_timestep'].values - 9
df2plot = df2plot[df2plot['image_timestep'] >= -0]
g = sns.relplot(data=df2plot, x='image_timestep', y='value', hue='variable', row='is_correct',
                col='layer_index', kind='line', errorbar='ci', facet_kws={'sharey': False}, aspect=1.5)
plt.savefig('Layers_activations_temporal.pdf')
plt.show()

# %% Cosine and CCA similarity across layers analysis
vars2idxs = ['image_timestep', 'is_label_showing', 'data_sample_id',
             'label', 'is_correct', 'layer_index', 'neuron index']
vars2stack = ['activation', 'forward_activation_component',
              'backward_activation_component',  'lateral_activation_component']
df2plot = df.set_index(vars2idxs)[vars2stack]
df2plot = df2plot.stack().reset_index().rename(columns={0: 'value', 'level_'+str(len(vars2idxs)): 'variable'})
# df2plot = df2plot[df2plot['label']==2]
df2plot['image_timestep'] = df2plot['image_timestep'].values - 9
# df2plot = df2plot[df2plot['image_timestep']>=-0]
df2plot = df2plot[df2plot['variable'] == 'activation']
vars2idxs = ['is_label_showing', 'data_sample_id', 'label', 'is_correct', 'layer_index', 'image_timestep']
df2plot_piv = df2plot.set_index(vars2idxs).drop(columns='variable').reset_index()
df2plot_piv = df2plot_piv.pivot_table(index=vars2idxs, columns='neuron index', values='value')
df2plot_piv['activations'] = df2plot_piv.values.tolist()
df2plot_pivcol = df2plot_piv[['activations']]
df2plot_pivcol = df2plot_pivcol['activations'].apply(lambda x: np.array(x))
# df2plot_norm = df2plot_pivcol.reset_index()
df2plot_norm = df2plot_pivcol.apply(lambda x: x/np.sqrt((np.sum(x**2)+float(np.sum(x**2) == 0))))
df2plot_norm = df2plot_norm.reset_index()

df2plot_cos = pd.DataFrame()
layer_pairs = [[0, 1], [1, 2], [2, 3], [3, 4]]
tvalues = np.unique(df2plot_norm['image_timestep'].values)
timesteps_pairs = [(x, y) for x, y in itertools.product(tvalues, tvalues)]
# df2plot_norm_act = df2plot_norm[df2plot_norm['variable']=='activation']
for is_correct in [True, False]:
    df2plot_norm_dat = df2plot_norm[df2plot_norm['is_correct'] == is_correct]
    for layer_pair in layer_pairs:
        df2plot_norm_A, df2plot_norm_B = df2plot_norm_dat[df2plot_norm_dat['layer_index'] ==
                                                          layer_pair[0]], df2plot_norm_dat[df2plot_norm_dat['layer_index'] == layer_pair[1]]
        # for t_time in tvalues:
        for t_pair in timesteps_pairs:
            df2plot_norm_At = df2plot_norm_A[df2plot_norm_A['image_timestep'] == t_pair[0]]
            df2plot_norm_Bt = df2plot_norm_B[df2plot_norm_B['image_timestep'] == t_pair[1]]
            df2plot_norm_At, df2plot_norm_Bt = df2plot_norm_At.sort_values(
                by=vars2idxs), df2plot_norm_Bt.sort_values(by=vars2idxs)
            df2plot_cos_pair = df2plot_norm_At[vars2idxs]
            df2plot_cos_pair['layer_index_1'] = df2plot_norm_Bt['layer_index'].values
            df2plot_cos_pair['image_timestep_0'] = t_pair[0]
            df2plot_cos_pair['image_timestep_1'] = t_pair[1]
            df2plot_cos_pair['is_correct'] = is_correct
            df2plot_cos_pair['layers'] = ''.join([str(l) for l in layer_pair])
            df2plot_cos_pair['cos'] = np.sum(np.stack(df2plot_norm_At['activations'].values)
                                             * np.stack(df2plot_norm_Bt['activations'].values), axis=1)
            cca = CCA(n_components=10)
            A = np.stack(df2plot_norm_At['activations'].values)
            B = np.stack(df2plot_norm_Bt['activations'].values)
            try:
                cca.fit_transform(A, B)
                rsquared = cca.score(A, B)
                score = np.mean(
                    np.diag(
                        np.corrcoef(cca._x_scores, cca._y_scores, rowvar=False)
                        [: cca.n_components, cca.n_components:]))
            except:
                rsquared, score = np.nan, np.nan
            df2plot_cos_pair['CCA_rsquared'] = rsquared
            df2plot_cos_pair['CCA_corr'] = score
            df2plot_cos = pd.concat([df2plot_cos, df2plot_cos_pair])

# %% Plotting CCA through timesteps
df2plot = df2plot_cos[df2plot_cos['image_timestep_0'] == df2plot_cos['image_timestep_1']]
# df2plot = df2plot[df2plot['image_timestep']>=0]
# df2plot = df2plot[df2plot['variable']=='activation']
sns.relplot(data=df2plot, x='image_timestep', y='CCA_corr', hue='layers',
            col='is_correct', kind='line', errorbar='ci', facet_kws={'sharey': True})
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title(item)
plt.savefig('Layers_activations_cascade_cca.pdf')
plt.show()

sns.relplot(data=df2plot, x='image_timestep', y='cos', hue='layers',
            col='is_correct', kind='line', errorbar='ci', facet_kws={'sharey': True})
for item, ax in g.axes_dict.items():
    ax.grid(False, axis='x')
    ax.set_title(item)
plt.savefig('Layers_activations_cascade_cos.pdf')
plt.show()

# %% Plot CCA all temporal dependencies
# vars2idxs = ['is_label_showing', 'data_sample_id', 'label', 'is_correct', 'variable', 'layers', 't_0']
df2plot_all = df2plot_cos.reset_index()
# df2plot_all = df2plot_all[df2plot_all['variable']=='activation']

fig, axs = plt.subplots(4, 4, figsize=(20, 17))
for i_lp, layer_pair in enumerate(['01', '12', '23', '34']):
    for i_cor, is_correct in enumerate([True, False]):
        df2plot = df2plot_all
        df2plot = df2plot[df2plot['layers'] == layer_pair]
        df2plot = df2plot[df2plot['is_correct'] == is_correct]
        df2plot_piv = df2plot.pivot_table(index='image_timestep_0', columns='image_timestep_1', values='CCA_corr')
        df2plot_piv_diff = df2plot_piv-df2plot_piv.transpose()
        sns.heatmap(data=df2plot_piv, ax=axs[i_lp, i_cor*2], cmap='jet')  # 'crest')
        sns.heatmap(data=df2plot_piv_diff, ax=axs[i_lp, i_cor*2+1], cmap='jet')

plt.tight_layout()
sns.despine()
plt.savefig('Layers_activations_temporal_cascade.pdf')
plt.show()
