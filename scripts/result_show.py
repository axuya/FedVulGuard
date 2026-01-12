# scripts/result_show.py
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

# 1. 读数据
df = pd.read_csv('/home/xu/FedVulGuard/results/experiments/result_all.csv')

# 2. 解析任务与模式
def parse_run(run: str):
    m = re.match(r'(.*?)_(\d+)_(.*?)_seed\d+', run)
    return m.groups()[:3:2] if m else (run, 'unknown')

df[['task', 'mode']] = df['run'].apply(lambda x: pd.Series(parse_run(x)))

# 3. 画布：左三曲线，右一柱状
fig = plt.figure(figsize=(14, 4))
palette = {'stats': '#1f77b4', 'llm': '#ff7f0e'}

# 3.1 三条曲线
metrics = ['train_loss', 'test_f1', 'test_auc']
for idx, metric in enumerate(metrics, 1):
    ax = fig.add_subplot(1, 4, idx)
    sns.lineplot(data=df, x='epoch', y=metric, hue='mode', style='task',
                 markers=True, dashes=False, palette=palette, ax=ax)
    ax.set_title(metric.replace('_', ' ').title())
    ax.legend_.remove()          # 大图里图例太乱，统一放到空白处

# 3.2 最佳 F1 柱状图
best_f1 = (df.groupby(['task', 'mode'])['test_f1'].max()
             .reset_index()
             .rename(columns={'test_f1': 'best_f1'}))
ax4 = fig.add_subplot(1, 4, 4)
sns.barplot(data=best_f1, x='task', y='best_f1', hue='mode',
            palette=palette, ax=ax4)
ax4.set_title('Best Test F1')
ax4.set_ylim(0, 1)
for p in ax4.patches:
    ax4.text(p.get_x() + p.get_width()/2., p.get_height() + 0.01,
             f'{p.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# 4. 统一图例（取第一根线的 handles & labels）
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(rect=[0, 0.1, 1, 1])   # 给底部图例留空
plt.savefig('all_in_one.png', dpi=300, bbox_inches='tight')
plt.close()
print('saved -> all_in_one.png')