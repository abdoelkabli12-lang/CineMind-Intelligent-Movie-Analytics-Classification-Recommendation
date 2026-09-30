import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from connection_db import mycol


def load_data():
    data_raw = mycol.find({})
    df = pd.DataFrame(data_raw)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    print(f"Loaded {len(df)} movies from MongoDB")
    return df


def plot_01_vote_avg(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    m = df['vote_average'].median()
    sns.histplot(df['vote_average'], kde=True, bins=30, color='steelblue', alpha=0.4, ax=ax)
    ax.axvline(m, color='red', linestyle='--', linewidth=1.5, label=f'Median: {m:.2f}')
    ax.set_xlabel('Vote Average')
    ax.set_ylabel('Frequency')
    ax.set_title('Vote Average per Movie', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_02_popularity(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    p = df['popularity'].dropna()
    pc = p[p <= p.quantile(0.95)]
    sns.histplot(pc, bins=30, kde=True, ax=ax, color='#FF5722', line_kws={'linewidth': 2})
    ax.axvline(p.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {p.median():.2f}')
    ax.set_title('Popularity Distribution (95th percentile)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Popularity')
    ax.set_ylabel('Number of Movies')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_03_genres(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import Counter
    gc = Counter()
    for gs in df['genres'].dropna():
        for g in str(gs).split(', '):
            gc[g] += 1
    top = gc.most_common(15)
    n = [t[0] for t in top][::-1]
    v = [t[1] for t in top][::-1]
    sns.barplot(x=v, y=n, ax=ax, palette='viridis')
    ax.set_title('Top 15 Genres (by movie count)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Movies')
    ax.set_ylabel('Genre')
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_04_year(df, sp=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    yc = df['release_year'].value_counts().sort_index()
    ax.fill_between(yc.index, yc.values, alpha=0.3, color='#2196F3')
    ax.plot(yc.index, yc.values, marker='o', color='#2196F3', linewidth=2)
    ax.set_title('Number of Movies Released per Year', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Movies')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_05_runtime(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    rt = df['runtime'].dropna()
    rtc = rt[(rt >= 30) & (rt <= 240)]
    sns.histplot(rtc, bins=40, kde=True, ax=ax, color='#4CAF50', line_kws={'linewidth': 2})
    ax.axvline(rt.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {rt.median():.0f} min')
    ax.set_title('Runtime Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Runtime (minutes)')
    ax.set_ylabel('Number of Movies')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_06_budget_rev(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    f = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    sns.scatterplot(data=f, x='budget', y='revenue', ax=ax, alpha=0.5, color='#9C27B0', s=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Budget vs Revenue (log-log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Budget ($)')
    ax.set_ylabel('Revenue ($)')
    mv = max(f['budget'].max(), f['revenue'].max())
    ax.plot([1, mv], [1, mv], 'r--', label='Budget = Revenue')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_07_votes_pop(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    v = df.dropna(subset=['vote_count', 'popularity', 'vote_average'])
    v = v[v['vote_count'] > 0]
    sc = ax.scatter(v['vote_count'], v['popularity'], c=v['vote_average'], cmap='RdYlGn', alpha=0.6, s=30)
    ax.set_title('Votes vs Popularity (color = vote average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Votes')
    ax.set_ylabel('Popularity')
    ax.set_xscale('log')
    plt.colorbar(sc, label='Vote Average')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_08_decade(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    d = df.copy()
    d['decade'] = (d['release_year'] // 10) * 10
    d['decade_str'] = d['decade'].astype(str) + 's'
    o = sorted(d['decade_str'].unique())
    sns.boxplot(data=d, x='decade_str', y='vote_average', ax=ax, palette='Set2', order=o)
    ax.set_title('Ratings Distribution by Decade', fontsize=14, fontweight='bold')
    ax.set_xlabel('Decade')
    ax.set_ylabel('Vote Average')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_09_corr(df, sp=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime', 'release_year']
    av = [c for c in cs if c in df.columns]
    corr = df[av].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap — Numeric Variables', fontsize=14, fontweight='bold')
    fig.tight_layout()
    if sp: fig.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def run_eda(od='notebooks/eda_figs'):
    import os
    df = load_data()
    os.makedirs(od, exist_ok=True)
    print("\nGenerating all 9 EDA charts...")
    plot_01_vote_avg(df, f'{od}/01_vote_avg.png'); print("  [1/9] Vote average")
    plot_02_popularity(df, f'{od}/02_popularity.png'); print("  [2/9] Popularity")
    plot_03_genres(df, f'{od}/03_genres.png'); print("  [3/9] Genres")
    plot_04_year(df, f'{od}/04_year.png'); print("  [4/9] Year")
    plot_05_runtime(df, f'{od}/05_runtime.png'); print("  [5/9] Runtime")
    plot_06_budget_rev(df, f'{od}/06_budget_rev.png'); print("  [6/9] Budget/Revenue")
    plot_07_votes_pop(df, f'{od}/07_votes_pop.png'); print("  [7/9] Votes/Popularity")
    plot_08_decade(df, f'{od}/08_decade.png'); print("  [8/9] Decade")
    plot_09_corr(df, f'{od}/09_corr.png'); print("  [9/9] Correlation")
    print(f"\nAll charts saved to: {os.path.abspath(od)}/")
    return df


if __name__ == '__main__':
    run_eda()
